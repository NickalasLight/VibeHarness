"""Web toolset: a stateful browser exposed through discrete, first-class tools.

Wraps the Playwright **Agent CLI** (`playwright-cli`, from `@playwright/cli`),
which keeps a browser alive between calls within a named session, so navigation,
clicks, and content extraction all share state.

Each basic browser operation the CLI supports is its own ``Tool`` subclass
(``goto``, ``click``, ``fill``, ``type``, ``press_key``, ``select_option``,
``hover``, ``navigate_back``, …) — see :class:`WebToolset`. There is
An ``evaluate`` tool allows the agent to run read-only JS to inspect an element
(e.g. its tagName, type, or option list) before choosing how to interact. This
replaces the old monolithic ``browse(action=...)`` dispatcher: every operation is
now its own named tool with its own typed parameters and description, so the model
chooses a tool the same way it chooses ``read_file`` vs ``write_file``.

The agent does NOT request the page contents. There is no agent-callable
``snapshot`` tool: the live page is captured automatically every turn and injected
into the system prompt under '# Current page (live snapshot — provided
automatically)'. The internal capture helpers (:func:`capture_page_snapshot`,
:func:`capture_page_snapshot_raw`, :func:`make_snapshot_provider`,
:func:`make_raw_snapshot_provider`) drive that auto-injection and are NOT tools.

Install the backend with:  npm install -g @playwright/cli@latest
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Callable

from .config import Config
from .toolset import Toolset
from .tools import Param, Tool, ToolResult

BINARY = "playwright-cli"

# The Config default for ``web_session``. When the resolved config still carries this
# sentinel, the session name was NOT explicitly chosen (CLI/settings), so each run is
# given a fresh unique name instead (issues #111/#112) — see :func:`resolve_web_session`.
DEFAULT_WEB_SESSION = "vibe"


def resolve_web_session(config: Config) -> str:
    """The per-RUN Playwright session name (issues #111/#112).

    The persistent playwright-cli is a client/daemon keyed by session NAME: every
    wrapper bound to the same name talks to (and tears down) the SAME daemon. When
    ``config.web_session`` was a single constant (``"vibe"``), every concurrent run
    shared ONE daemon, so one run's teardown (``close``) killed another run's browser
    mid-run — the root cause of the "browser is not open" daemon deaths (#101/#111).

    To make runs isolated, we mint a UNIQUE name per run — ``vibe-<short-guid>`` — so
    concurrent runs never collide on the daemon. An explicitly-set name (anything other
    than the :data:`DEFAULT_WEB_SESSION` sentinel, e.g. from saved settings) is honoured
    verbatim as an override; only the default triggers a generated name. Resolve ONCE
    per run and thread the result through ``config.web_session`` so every web tool and
    both snapshot providers share the exact same name.
    """
    name = config.web_session
    if name and name != DEFAULT_WEB_SESSION:
        return name
    return f"{DEFAULT_WEB_SESSION}-{uuid.uuid4().hex[:8]}"

# A snapshot ref token, as it appears in the auto-injected live page view, e.g.
# "[e163] button ...". We accept it bare (``e163``) or bracketed (``[e163]``).
_REF_RE = re.compile(r"\be(\d+)\b")
_REF_TOKEN_RE = re.compile(r"\[?\be(\d+)\b\]?")

# Roles/patterns that make a snapshot line interactable for text or selection.
_INTERACTABLE_ROLES = frozenset({
    "textbox", "combobox", "listbox", "checkbox", "radio", "spinbutton",
    "searchbox", "button", "link", "menuitem", "option", "switch",
})

# Playwright error substring that signals a non-interactable element was targeted.
_NOT_INTERACTABLE_PHRASES = (
    "element is not an <input>",
    "element is not a <textarea>",
    "does not have a role allowing",
    "element is not an <input>, <textarea>",
)

# Pattern to extract the HTML snippet from Playwright's "locator resolved to <...>" error.
_RESOLVED_TO_RE = re.compile(
    r"locator resolved to (<[^>]+>[^<]{0,120}(?:</[^>]+>)?|<[^>]+/>|<[^>]+>)",
    re.DOTALL,
)


def _nearby_interactable_refs(failed_ref: str, snapshot: str, window: int = 12) -> str:
    """Return a short description of interactable refs near ``failed_ref`` in the snapshot.

    Searches ±``window`` ref-numbers from the failed ref and returns the first 5
    interactable lines found, formatted as "eN (description)".
    """
    if not failed_ref or not snapshot:
        return "(could not determine nearby refs)"
    m = re.match(r"e(\d+)$", failed_ref)
    if not m:
        return "(could not determine nearby refs)"
    center = int(m.group(1))
    candidates = []
    for line in snapshot.splitlines():
        rm = _REF_RE.search(line)
        if not rm:
            continue
        ref_num = int(rm.group(1))
        if abs(ref_num - center) > window:
            continue
        ref = f"e{ref_num}"
        if ref == failed_ref:
            continue
        line_lower = line.lower()
        if any(role in line_lower for role in _INTERACTABLE_ROLES):
            candidates.append(f"{ref} ({line.strip()[:80]})")
    candidates.sort(key=lambda s: abs(int(s[1:s.index(" ")]) - center))
    return ", ".join(candidates[:5]) or "(none found nearby)"

# Substrings playwright-cli emits in its (exit-0) output when a target/ref did NOT
# resolve to a real element. The CLI exits 0 and embeds the failure as text, so the
# only way to detect a no-match is to scan the output (issue #73). Matched
# case-insensitively.
_NO_MATCH_MARKERS: tuple[str, ...] = (
    "does not match any elements",
    "no elements match",
    "ref not found",
    "no element found",
    "element not found",
    "could not find element",
    "waiting for locator",  # playwright timeout phrasing when a locator never appears
    "timeout",
    "modal state",          # upload without an open OS file-picker (issue #144)
)


def annotate_filled_snapshot(snapshot: str, filled: dict[str, str]) -> str:
    """Append filled-value markers to snapshot lines whose ref is in ``filled``.

    Each matching line gets:  ``  [ALREADY FILLED WITH 'value' — DO NOT FILL AGAIN]``
    appended, so the model knows which elements are already handled.
    """
    if not snapshot or not filled:
        return snapshot
    lines = []
    for line in snapshot.splitlines():
        m = _REF_RE.search(line)
        if m:
            ref = f"e{m.group(1)}"
            if ref in filled:
                line = f"{line}  [ALREADY FILLED WITH '{filled[ref]}' — DO NOT FILL AGAIN]"
        lines.append(line)
    return "\n".join(lines)


def parse_snapshot_refs(snapshot: str) -> set[str]:
    """Extract the set of element refs (``e163`` …) present in a live page snapshot.

    The snapshot is the same auto-injected page view the model sees, where each
    actionable node is prefixed with a ``[eN]`` token (e.g. ``[e163] button
    "Accept …"``). We return the bare, normalized ref strings (``{"e163", …}``)
    for membership checks. Lines that carry no ref (``[-] tooltip``, ``text: …``)
    simply contribute nothing.
    """
    return {f"e{m.group(1)}" for m in _REF_RE.finditer(snapshot or "")}


def normalize_ref(target: str) -> str | None:
    """Normalize a user-supplied ``target`` to a bare ref (``e163``) if it *is* a
    ref in one of the snapshot's accepted forms (``e163`` or ``[e163]``).

    Returns ``None`` for anything that is not a clean ref token — e.g. a CSS
    selector (``.ytd-play-button``), an id (``#play``), or a tag — so the caller
    can reject guessed selectors outright (issue #73).
    """
    if not target:
        return None
    t = target.strip()
    m = _REF_TOKEN_RE.fullmatch(t)
    if m:
        return f"e{m.group(1)}"
    return None


# An ARIA line for a selectable option in an OPEN custom listbox/menu: a selectable
# role + a quoted accessible name. Used by the select_option combobox fallback (#125).
_OPTION_LINE_RE = re.compile(
    r'\b(?:option|menuitem|menuitemradio|menuitemcheckbox|listitem|treeitem|radio)\b'
    r'\s+"(?P<name>(?:[^"\\]|\\.)*)"',
    re.IGNORECASE,
)


def _snapshot_line_for_ref(snapshot: str, ref: str | None) -> str | None:
    """Return the snapshot line carrying ``[ref=<ref>]`` (stripped), or None."""
    if not ref:
        return None
    for line in (snapshot or "").splitlines():
        if f"[ref={ref}]" in line:
            return line.strip()
    return None


def target_is_open_listbox(snapshot: str, target: str) -> bool:
    """True when ``target``'s snapshot line is an OPEN listbox/option container.

    A custom combobox's TRIGGER is rendered as ``button [role=combobox]``; once opened, the
    overlay it spawns is a separate ``listbox`` node (with ``option`` children). When the
    agent mistakenly calls select_option on that LISTBOX (or an OPTION) ref instead of the
    trigger, the popup is ALREADY open — pressing Escape (the normal trigger path) would CLOSE
    it without selecting (iter-3 turn 5: select_option(e188=listbox) -> Escape closed it ->
    Country never set). Detecting this lets the caller skip Escape+click and click the matching
    option directly in the already-open list."""
    line = _snapshot_line_for_ref(snapshot, normalize_ref(target))
    if not line:
        return False
    low = line.lower()
    # The target's own line is the listbox container, or is itself a selectable option.
    if "listbox" in low:
        return True
    return bool(_OPTION_LINE_RE.search(line))


def find_option_ref_by_text(snapshot: str, value: str) -> str | None:
    """Find the ref of an OPEN-listbox option whose visible text matches ``value``.

    Matching is tolerant, in priority order: exact (case-insensitive) -> startswith ->
    substring. Returns the bare ref (e.g. ``e90``) or ``None`` if nothing matches (the
    caller then leaves the dropdown OPEN so the model can pick the option from the next
    snapshot). Used by :class:`SelectOptionTool` to drive a custom `<div role="listbox">`
    combobox that Playwright's native ``select`` cannot operate (#125)."""
    value_l = (value or "").strip().lower()
    if not value_l:
        return None
    exact = starts = contains = None
    for line in (snapshot or "").splitlines():
        m = _OPTION_LINE_RE.search(line)
        if not m:
            continue
        mr = re.search(r"ref=(e\d+)", line)
        if not mr:
            continue
        ref, name = mr.group(1), m.group("name").replace('\\"', '"').strip().lower()
        if name == value_l:
            exact = exact or ref
        elif name.startswith(value_l):
            starts = starts or ref
        elif value_l in name:
            contains = contains or ref
    return exact or starts or contains


# An ISO date value (yyyy-mm-dd) the task supplies for a date field.
_ISO_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
# A calendar day cell button carries its ISO date as the accessible name/aria-label.
# e.g. ``- button "2026-07-21" [ref=e221] [cursor=pointer]: "21"``
_DAY_BUTTON_RE = re.compile(r'button\s+"(\d{4}-\d{2}-\d{2})"[^\n]*?ref=(e\d+)', re.IGNORECASE)
# The calendar header label, e.g. ``generic [ref=e193]: January 2020`` (month + year).
_CAL_MONTHYEAR_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{4})", re.IGNORECASE)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]


def find_day_button_ref(snapshot: str, iso_date: str) -> str | None:
    """Ref of the calendar day-cell button whose ISO aria-label == ``iso_date``, or None."""
    for m in _DAY_BUTTON_RE.finditer(snapshot or ""):
        if m.group(1) == iso_date.strip():
            return m.group(2)
    return None


def find_nav_button_ref(snapshot: str, label: str) -> str | None:
    """Ref of a calendar nav button by its accessible name (e.g. 'Next year')."""
    pat = re.compile(rf'button\s+"{re.escape(label)}"[^\n]*?ref=(e\d+)', re.IGNORECASE)
    m = pat.search(snapshot or "")
    return m.group(1) if m else None


def calendar_view_month(snapshot: str) -> tuple[int, int] | None:
    """(year, month-1) currently shown in an open calendar, or None if not a calendar."""
    m = _CAL_MONTHYEAR_RE.search(snapshot or "")
    if not m:
        return None
    month = _MONTHS.index(m.group(1).lower())
    return int(m.group(2)), month


_SNAPSHOT_FILELINK_RE = re.compile(
    r"\n?###\s+Snapshot\s*\n\s*-\s*\[Snapshot\]\([^)]+\.yml\)\s*\n?",
    re.IGNORECASE,
)

def _strip_snapshot_filelink(text: str) -> str:
    """Remove the '### Snapshot\\n- [Snapshot](.playwright-cli/page-*.yml)' block
    that playwright-cli emits in action stdout. The file-path link is useless to
    the model — the real ARIA content is delivered through the auto-injected per-turn
    '# Current page' snapshot and DOM-delta blocks; passing a dangling .yml path only
    confuses the model into thinking the snapshot IS the file reference."""
    return _SNAPSHOT_FILELINK_RE.sub("\n", text).strip()


def output_signals_no_match(output: str) -> bool:
    """True when playwright-cli's (exit-0) output text indicates the action hit no
    element — a ref/selector that resolved to nothing. The CLI reports these as an
    embedded ``### Error`` rather than a non-zero exit, so the web tool used to
    record them as ``ok=true`` (the issue #73 status bug). Scanning the text lets us
    flip the ToolResult to ok=False.
    """
    low = (output or "").lower()
    if "### error" in low or "error:" in low:
        if any(marker in low for marker in _NO_MATCH_MARKERS):
            return True
    return False


# ---------------------------------------------------------------------------
# DOM-change detection helpers — appended to interaction tool observations.
# After any successful interactive action (click, fill, type, press_key,
# select_option, check, uncheck) the snapshot is diffed against a pre-action
# baseline and newly-appeared elements are reported inline so the model can
# immediately use their refs without needing an explicit snapshot call.
# ---------------------------------------------------------------------------

def _diff_snapshot_refs(before: str, after: str) -> list[str]:
    """Return refs in ``after`` that were not in ``before`` (new elements).

    Extracts all ``[ref=eN]`` patterns from both snapshots and returns the
    sorted list of refs that are new in ``after``. Returns an empty list when
    nothing changed or snapshots are empty.
    """
    _REF_ATTR_RE = re.compile(r"\[ref=(e\d+)\]")
    before_refs = set(_REF_ATTR_RE.findall(before or ""))
    after_refs = set(_REF_ATTR_RE.findall(after or ""))
    new_refs = after_refs - before_refs
    return sorted(new_refs, key=lambda r: int(r[1:]))


def _extract_ref_lines(refs: list[str], snapshot: str) -> list[str]:
    """Return the snapshot lines that contain each ref in ``refs``.

    Returns up to 20 lines (stripped) to keep the delta message concise.
    """
    results: list[str] = []
    for ref in refs:
        for line in (snapshot or "").splitlines():
            if f"[ref={ref}]" in line:
                results.append(line.strip())
                break
    return results[:20]


def _interactable_ref_lines(snapshot: str, limit: int = 40) -> list[str]:
    """Return concise 'eN: <role> "<label>"' lines for each interactable control.

    Filters the snapshot to lines carrying an interactable role (textbox, button,
    combobox, …) AND a ref, so an error message can show the model the REAL fields and
    buttons on the current page instead of a meaningless dump of every ref. Used by the
    invalid-target guard so a model that guessed a non-existent ref (iter-1: e208/e209
    after a page advanced) is shown the actual refs to choose from."""
    out: list[str] = []
    for line in (snapshot or "").splitlines():
        low = line.lower()
        if not any(role in low for role in _INTERACTABLE_ROLES):
            continue
        m = re.search(r"\[ref=(e\d+)\]", line)
        if not m:
            continue
        out.append(f"{m.group(1)}: {line.strip().lstrip('- ')[:90]}")
        if len(out) >= limit:
            break
    return out


def _extract_validation_alerts(snapshot: str) -> list[str]:
    """Return the text of any ``alert`` nodes in ``snapshot`` (client-side validation errors).

    A wizard step that rejects a Continue click renders one ``alert [ref=eN]: <message>``
    line per invalid field (iter-2 Step 2: "Invalid enum value... received ''" for the
    unset work arrangement, "Please choose a valid date"). Surfacing these verbatim lets
    the steer name the EXACT fields still blocking advancement instead of leaving the model
    to re-click Continue blindly (iter-2 turns 16-19: 4 blind Continue clicks, never read
    the errors, never advanced)."""
    alerts: list[str] = []
    for line in (snapshot or "").splitlines():
        # Snapshot shape: '- alert [ref=e189]: Invalid enum value. Expected ...'
        m = re.search(r"\balert\b[^:]*\[ref=e\d+\][^:]*:\s*(.+?)\s*$", line)
        if m:
            text = m.group(1).strip()
            if text and text not in alerts:
                alerts.append(text)
    return alerts


def _check_dom_delta(cli: "PlaywrightCli", before_snapshot: str,
                     result: "ToolResult") -> "ToolResult":
    """Append a DOM-change summary to ``result`` when new elements appeared.

    Captures a fresh snapshot, computes the ref diff against
    ``before_snapshot``, and — if any new refs exist — appends a
    ``DOM CHANGE DETECTED`` paragraph to the observation so the model sees
    the new elements' refs immediately without needing to call ``snapshot``.
    Returns the original ``result`` unchanged when the action failed or no
    new elements appeared.
    """
    if not result.ok:
        return result
    after = capture_page_snapshot_raw(cli) or ""
    new_refs = _diff_snapshot_refs(before_snapshot, after)
    # NOTE (iter-3 fix): do NOT early-return when ``new_refs`` is empty. A blind Continue
    # re-click on a step whose validation errors are ALREADY rendered produces NO new refs
    # (the alert nodes were created by the FIRST rejected Continue and persist), yet the form
    # is still stuck — the model needs the "FORM NOT ADVANCED" steer EVERY time it re-clicks
    # Continue while errors are visible, not only the first time (iter-3 turns 6-9: 4 blind
    # Continue clicks, Country alert pre-existing -> no new refs -> steer never re-fired).
    # PAGE-ADVANCE detection (iter-2): a wizard Continue/Back click swaps the ENTIRE step
    # — most old refs vanish and a fresh batch appears. Distinguish that from a small
    # in-place change (a dropdown opening, an inline error). When the page substantially
    # turned over, say so explicitly and list the new step's INTERACTABLE controls, so the
    # model re-reads the new refs instead of guessing sequential ones (iter-1: e208/e209
    # invented after Continue advanced the form).
    before_set = set(re.findall(r"\[ref=(e\d+)\]", before_snapshot or ""))
    after_set = set(re.findall(r"\[ref=(e\d+)\]", after))
    gone = before_set - after_set
    page_changed = bool(before_set) and len(gone) >= max(5, len(before_set) // 2)
    if page_changed:
        controls = _interactable_ref_lines(after)
        delta_msg = (
            "\n\nPAGE CHANGED: this click loaded a NEW page/step — the previous refs are "
            "GONE and the refs below are the ONLY valid ones now. Do NOT reuse or increment "
            "old refs; read these fresh refs and act on them:\n"
            + "\n".join(f"  {line}" for line in controls[:40])
        )
        return ToolResult(result.ok, (result.observation or "") + delta_msg)
    # VALIDATION-REJECT detection (iter-2 fix; iter-3 extension): the page did NOT turn over
    # (no page advance), but `alert` nodes are visible — i.e. a Continue/Submit click was
    # REJECTED by client-side validation. Tell the model the EXACT errors and that the form did
    # not advance, so it fixes the named fields instead of re-clicking Continue blindly.
    #
    # ITER-3 FIX: surface ALL currently-visible alerts, not just NEWLY-appeared ones. After the
    # FIRST rejected Continue, the alert nodes persist in the DOM. A subsequent Continue re-click
    # that leaves a field still unset produces NO new alert (it was already there) — under the
    # old new-alerts-only check the steer went silent and the agent looped Continue blindly
    # (iter-3 turns 6-9: Country alert pre-existing from turn 2, never re-surfaced, 4 wasted
    # clicks). Any alert visible after a Continue means the form is still rejecting, so we
    # always re-surface them.
    alerts = _extract_validation_alerts(after)
    if alerts:
        errs = "\n".join(f"  • {a}" for a in alerts)
        reject_msg = (
            "\n\nFORM NOT ADVANCED — the page did NOT move to the next step. The form is still "
            "REJECTED by validation. Fix these errors BEFORE clicking Continue again:\n"
            + errs
            + "\nFor each error: find the matching field in the snapshot above and set it "
            "(select_option for dropdowns, the calendar for dates). Do NOT click Continue "
            "again until every error above is resolved — re-clicking it without fixing them "
            "does nothing."
        )
        return ToolResult(result.ok, (result.observation or "") + reject_msg)
    if not new_refs:
        return result
    new_lines = _extract_ref_lines(new_refs, after)
    delta_msg = (
        f"\n\nDOM CHANGE DETECTED: {len(new_refs)} new element(s) appeared:\n"
        + "\n".join(f"  {line}" for line in new_lines[:20])
    )
    return ToolResult(result.ok, (result.observation or "") + delta_msg)


# Substrings playwright-cli emits (in its error text) when the named session has no
# live daemon/browser to act on — i.e. the persistent session has died or was never
# opened. The CLI's exact phrasing is "The browser '<name>' is not open, please run
# open first ...". When a per-turn web action hits this we can transparently reopen
# the session and retry, rather than dead-ending the agent (issues #101, #75).
_SESSION_CLOSED_MARKERS: tuple[str, ...] = (
    "is not open",
    "please run open first",
    "browser is not open",
    "no browser is open",
    "session closed",
    "session is closed",
    "not open, please run open",
)


def output_signals_session_closed(output: str) -> bool:
    """True when playwright-cli's output indicates the *whole session/daemon* is
    gone (not merely a missing element) — the "browser 'vibe' is not open" class of
    failure seen when the persistent daemon dies mid-run (issue #101). Distinct from
    :func:`output_signals_no_match`, which is a per-element miss on a live page.
    Matched case-insensitively so callers can decide to reopen + retry (issue #75).
    """
    low = (output or "").lower()
    return any(marker in low for marker in _SESSION_CLOSED_MARKERS)


# Commands that are themselves part of session lifecycle/recovery — they must NEVER
# trigger the auto-resume path (that would recurse: an `open` failing must not try to
# `open` again from inside the resume). Everything else (goto/click/fill/snapshot/…)
# is a normal command that resume protects.
_RECOVERY_COMMANDS: frozenset[str] = frozenset({"open", "close"})


class SessionState:
    """Run-scoped, SHARED across every ``PlaywrightCli`` a single run creates (the
    discrete tools and both snapshot providers each build their own wrapper bound to
    the same session name). Holds exactly the state the self-healing resume needs so
    that a daemon death detected by ANY of them heals the session for ALL of them
    (issue #102, composing with #101/#75):

    - ``open_flags``: the ``--headed``/``--browser …`` flags ``setup`` opened with, so
      a reopen restores the same browser;
    - ``last_url``: the last URL successfully navigated to, so a reopen can re-navigate
      back to the page the agent was on (refs are re-derived from the fresh snapshot);
    - ``resumes``/``max_resumes``: a hard bound so a genuinely broken environment
      terminates instead of looping forever on reopen.
    """

    def __init__(self, open_flags: "list[str] | None" = None, max_resumes: int = 5):
        self.open_flags: list[str] = list(open_flags or [])
        self.last_url: str | None = None
        self.resumes: int = 0
        self.max_resumes: int = max_resumes

    def may_resume(self) -> bool:
        return self.resumes < self.max_resumes


# Run-scoped registry of the ONE ``SessionState`` per session name (issue #113). The
# three places a run builds a ``PlaywrightCli`` for its session — ``create_tools`` (every
# discrete web tool), ``setup`` (the run's lifecycle CLI), and the two snapshot providers
# — are created at different points in cli.py and cannot easily be handed a single object
# at construction. Keying the shared state by session NAME instead means every wrapper
# bound to the run's (now unique, issue #112) name transparently gets the SAME state, so
# recovery bookkeeping (open_flags/last_url/resumes) written by any tool is visible to all
# of them. Because each run mints a unique name, entries never collide across runs;
# ``WebToolset.teardown`` drops the run's entry so the registry does not accumulate.
_SESSION_STATES: "dict[str, SessionState]" = {}


def shared_session_state(session: str, open_flags: "list[str] | None" = None) -> SessionState:
    """Return the ONE run-scoped :class:`SessionState` for ``session`` (issue #113).

    The first caller for a given session name creates the state (seeding ``open_flags``);
    every later caller for the SAME name gets that same instance, so all web tools and
    both snapshot providers in one run share recovery bookkeeping. ``open_flags`` is only
    applied when the state is first created (or was empty), never overwriting flags an
    earlier caller already seeded.
    """
    state = _SESSION_STATES.get(session)
    if state is None:
        state = SessionState(open_flags)
        _SESSION_STATES[session] = state
    elif open_flags and not state.open_flags:
        state.open_flags = list(open_flags)
    return state


def drop_session_state(session: str) -> None:
    """Forget the shared :class:`SessionState` for ``session`` (called on teardown) so
    the per-run registry does not grow unbounded across the process's lifetime."""
    _SESSION_STATES.pop(session, None)


# Pull the "Page URL: <url>" line the CLI prints in snapshots / nav results so a
# resume can re-navigate to where the agent was (issue #102).
_PAGE_URL_RE = re.compile(r"Page URL:\s*(\S+)", re.IGNORECASE)


def parse_page_url(output: str) -> str | None:
    m = _PAGE_URL_RE.search(output or "")
    url = m.group(1) if m else None
    if url and url.lower() != "about:blank":
        return url
    return None


class PlaywrightCli:
    """Thin, injectable wrapper around the stateful `playwright-cli` binary."""

    def __init__(self, session: str, timeout: int, open_flags: "list[str] | None" = None,
                 state: "SessionState | None" = None):
        self._session = session
        self._timeout = timeout
        self._binary = shutil.which(BINARY)
        # Run-scoped shared state for the self-healing resume (issue #102). When a
        # caller threads one ``SessionState`` through every wrapper, a daemon death
        # seen by any tool/snapshot heals the session for all of them; standalone
        # wrappers (tests, ad-hoc use) get their own private state. ``open_flags``
        # passed explicitly seed the state so a reopen restores the same browser.
        self._state = state if state is not None else SessionState(open_flags)
        if open_flags is not None and state is not None and not self._state.open_flags:
            self._state.open_flags = list(open_flags)
        # Handle to the most recent child this wrapper spawned. Retained so that a
        # later ``close()`` can tree-kill any browser daemon/grandchildren that the
        # session left alive (issue #15) — not just the direct CLI child.
        self._last_proc: "subprocess.Popen | None" = None

    @property
    def session(self) -> str:
        return self._session

    @property
    def state(self) -> "SessionState":
        return self._state

    @property
    def open_flags(self) -> list[str]:
        return list(self._state.open_flags)

    @property
    def available(self) -> bool:
        return self._binary is not None

    def open(self, *flags: str) -> tuple[bool, str]:
        """(Re)open the persistent browser for this session.

        Idempotent from the agent's view: ``playwright-cli open`` on an
        already-open session simply re-opens it. Uses the session's captured
        ``open_flags`` (``--headed``/``--browser …``) when no explicit flags are
        passed so an automatic recovery reopen restores the original browser
        (issues #101, #75). Returns ``(ok, output)`` like :meth:`run`. Goes through
        ``_run_once`` (NOT the self-healing ``run``) so it can never recurse into the
        resume path.
        """
        use = list(flags) if flags else self._state.open_flags
        return self._run_once("open", *use)

    def run(self, *args: str) -> tuple[bool, str]:
        """Run one CLI command, self-healing the session if the daemon has died.

        Thin wrapper over :meth:`_run_once` that adds the issue-#102 clean-resume:
        every discrete tool, the target-ref guard, and both per-turn snapshot
        providers funnel through here, so wrapping at this single seam makes the
        WHOLE toolset resilient with no change to any tool subclass.

        Behaviour:
        - track ``last_url`` on a successful ``goto`` (and from any "Page URL:" line)
          so a resume can re-navigate back;
        - if a NORMAL command (not ``open``/``close``) fails with the daemon-death
          signature, reopen the session (same flags), re-navigate to ``last_url`` if
          known, and RETRY the command once — transparently, so the agent never sees
          a dead-end "not open" loop;
        - bound the number of resumes so an unrecoverable environment still stops.
        """
        ok, out = self._run_once(*args)
        cmd = args[0] if args else ""
        if ok:
            url = parse_page_url(out)
            if cmd == "goto" and len(args) > 1:
                self._state.last_url = url or args[1]
            elif url:
                self._state.last_url = url
        # Only normal commands self-heal; open/close are the recovery primitives.
        if ok or cmd in _RECOVERY_COMMANDS or not output_signals_session_closed(out):
            return ok, out
        if not self._state.may_resume():
            return ok, out
        self._resume()
        return self._run_once(*args)

    def snapshot(self) -> tuple[bool, str]:
        """Capture the live page WITHOUT triggering session resume (issue #102).

        Per-turn snapshot capture runs on every turn and through the target-ref
        guard; it must be DEATH-TOLERANT, not session-fatal — a snapshot that hits a
        dead daemon should simply return empty so the caller renders no page section,
        and let the next ACTION (which the agent intends) drive the reopen. Going
        through ``_run_once`` (not the self-healing ``run``) also keeps a stray
        snapshot from spawning a browser when none is wanted (e.g. in tests).
        """
        return self._run_once("snapshot")

    def _resume(self) -> None:
        """Bring a dead session back: best-effort reap, reopen with the run's flags,
        re-navigate to the last known URL. Best-effort and never raises — a failed
        step just leaves the retry to surface the real error (issue #102)."""
        self._state.resumes += 1
        try:
            self._run_once("close")           # reap a zombie daemon if any
        except Exception:
            pass
        self._run_once("open", *self._state.open_flags)
        if self._state.last_url:
            self._run_once("goto", self._state.last_url)

    def _run_once(self, *args: str) -> tuple[bool, str]:
        """Run one CLI command in this session. Returns (ok, combined_output).

        Every command is HARD-BOUNDED by ``self._timeout``. On timeout the whole
        process tree is killed and a clear error is returned — the call never
        hangs (issue #4).

        We drive the child with Popen + ``communicate(timeout=...)`` rather than
        ``subprocess.run(timeout=...)`` on purpose: ``playwright-cli`` is a Node
        process that spawns a *browser grandchild* which inherits the captured
        stdout/stderr pipe handles. ``subprocess.run``'s timeout path kills only
        the direct child and then re-reads the pipes to drain them; while the
        browser grandchild is still alive holding the write-end, those pipes
        never reach EOF and that post-kill read blocks *forever* — so the
        ``TimeoutExpired`` is never delivered and the agent turn wedges. Killing
        the whole tree first lets the drain complete (or be skipped) promptly.
        """
        if not self._binary:
            return False, f"{BINARY} is not installed"
        cmd = self._command(*args)
        # Put the child in its own process group/session so that, on timeout, we
        # can signal the WHOLE tree (its browser grandchildren too) without
        # touching the harness's own group.
        popen_kw = {}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kw["start_new_session"] = True
        # Force UTF-8 decoding: page snapshots contain emoji/unicode that the
        # default Windows codec (cp1252) cannot decode, which would otherwise
        # crash the reader thread and return empty output.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", **popen_kw)
        # Remember this child so teardown/close can reap its whole tree (#15).
        self._last_proc = proc
        try:
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire tree (child + browser grandchildren) so the pipe
            # write-ends close; only then can we drain without re-blocking.
            self._kill_tree(proc)
            try:
                proc.communicate(timeout=self._kill_grace)
            except subprocess.TimeoutExpired:
                pass  # drained best-effort; never block the agent turn on it
            return False, f"command timed out after {self._timeout}s"
        out = ((stdout or "") + (stderr or "")).strip()
        return proc.returncode == 0, out

    def _command(self, *args: str) -> list[str]:
        """Build the argv for one CLI invocation. Isolated as a seam so the
        bounded-execution path in ``run`` can be exercised against a stand-in
        command (e.g. a deliberately slow process) without a live browser."""
        return [self._binary, f"-s={self._session}", *args]

    # Grace period to reap a killed process tree before giving up the drain.
    _kill_grace = 5

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen") -> None:
        """Best-effort kill of ``proc`` and every descendant it spawned.

        ``proc.kill()`` only terminates the direct child; the Node-launched
        browser would survive and keep the captured pipes open. We use the OS
        process-tree killers (``taskkill /T`` on Windows, process-group signal
        on POSIX) and always fall back to a plain ``proc.kill()``.
        """
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                import os
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def close(self) -> None:
        """Tear the session down for good — gracefully if possible, forcibly always.

        Two layers, both best-effort and exception-safe (issue #15):

        1. Ask ``playwright-cli`` to ``close`` the session so the browser shuts down
           cleanly. This itself goes through bounded ``run`` (kills its own tree on
           timeout), so a wedged close can never hang teardown.
        2. Regardless of whether ``close`` succeeded, tree-kill the LAST child this
           wrapper spawned (the ``open``/``close`` invocation and, on POSIX, its whole
           process group — the Node daemon + chrome grandchildren). A crashed run may
           have skipped a clean close, or ``close`` may have orphaned the daemon; this
           guarantees no ``chrome``/``node`` tree is left leaking.

        Idempotent: calling it again after the tree is already reaped is a harmless
        no-op. Never raises — teardown must always complete.
        """
        try:
            self.run("close")
        except Exception:
            pass
        proc = self._last_proc
        if proc is not None:
            self._kill_tree(proc)
            self._last_proc = None


# Shared element-targeting guidance, woven into every tool that takes a `target`.
# NOTE: deliberately does NOT embed the page-section heading text verbatim — that
# heading is a structural marker tests use to detect the auto-injected page section,
# and repeating it inside a tool description would create false matches.
_REF_NOTE = (
    "You MUST target an element by the stable ref shown for it in the live page "
    "view (e.g. 'e163') — the page view is provided to you automatically each "
    "turn. NEVER guess a CSS selector, class, id or tag: only refs that appear "
    "in the current snapshot are accepted; anything else is rejected before the "
    "browser is touched."
)


class _WebTool(Tool):
    """Base for every discrete browser tool.

    Each concrete subtool declares its ``name``, ``description``, ``parameters``,
    a past-tense ``_verb`` for the narrative observation, and a ``_build`` that maps
    validated args to the ``playwright-cli`` argv. The base owns the shared run
    machinery: missing-param guard, bounded CLI invocation, error/success
    observation phrasing, and output truncation. New operations = one tiny subclass.
    """

    _verb: str = "acted on"          # past-tense narrative verb
    _required: tuple[str, ...] = ()  # arg names that must be present + non-empty
    # When True, this tool's ``target`` arg must be a ref present in the CURRENT
    # live snapshot; a guessed CSS selector / id / unknown ref is rejected without
    # ever calling playwright (issue #73). Tools without a targeted element
    # (goto/type/press_key/screenshot-whole-page/navigate_*/reload) leave this off.
    _validate_target: bool = False
    # When True, ``run()`` captures a DOM snapshot before the action and appends a
    # DOM CHANGE DETECTED summary to the observation when new refs appear. Set on
    # interactive tools (click, type, press_key, check, uncheck) via the class body.
    # FillTool and SelectOptionTool manage the delta themselves because they override
    # run() with additional logic that must run first.
    _detect_dom_change: bool = False

    def __init__(self, cli: PlaywrightCli, observation_limit: int):
        self._cli = cli
        self._limit = observation_limit

    # ---- subclasses implement these two ----
    def _build(self, args: dict) -> list[str]:
        """Map validated args to the playwright-cli argv for this operation."""
        raise NotImplementedError

    def _subject(self, args: dict) -> str:
        """The thing acted on, for the narrative observation. Override as needed."""
        return args.get("url") or args.get("target") or "the page"

    def _run_impl(self, args: dict) -> ToolResult:
        """Core run logic: validate params, guard target ref, invoke CLI, phrase result.

        Extracted from the original ``run()`` so that the public ``run()`` entry point
        can wrap it with optional DOM-change detection without subclass complications.
        Subclasses that override ``run()`` call ``super().run()`` which now goes through
        this wrapper; they should call ``super()._run_impl()`` if they need the raw
        result before the DOM delta is appended (see FillTool, SelectOptionTool).
        """
        missing = [p for p in self._required if not args.get(p)]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")
        if self._validate_target:
            guard = self._guard_target(args)
            if guard is not None:
                return guard
        ok, output = self._cli.run(*self._build(args))
        # AUTO-RECOVERY (issues #101, #75, #102): the persistent browser daemon can
        # die mid-run (a shared session torn down by a concurrent run, an external
        # reaper, a renderer crash). The CLI layer (PlaywrightCli.run) already
        # detects that signature, reopens the session, re-navigates to the last URL
        # and retries the command transparently — so a recoverable death never
        # reaches here. If the death signature STILL survives that retry, resume was
        # exhausted/failed: surface a clear, actionable recovery path to the agent
        # (which, being web-only, must be told how to get unstuck), never a bare
        # error. A real per-element miss (issue #73) is a different case handled
        # below by the no-match path.
        if not ok and output_signals_session_closed(output):
            return ToolResult(
                False,
                f"you tried to {self._subject(args)} but the browser session is closed and "
                f"could not be restored automatically. Call `open_browser` to reopen it, then "
                f"`goto` your target URL and continue.")
        subject = self._subject(args)
        # The CLI exits 0 even when the ref/selector matched no element, embedding
        # the failure as text. Treat that as a real failure (issue #73): a no-match
        # must be ok=False so the agent (and the validator) sees it failed.
        if ok and output_signals_no_match(output):
            ok = False
        if not ok:
            return ToolResult(False, f"you tried to {self._verb} {subject} but it failed: "
                              f"{self._trim(_strip_snapshot_filelink(output))}")
        return ToolResult(True, f"you {self._verb} {subject}. Result:\n{self._trim(_strip_snapshot_filelink(output))}")

    def run(self, args: dict) -> ToolResult:
        """Public entry point. Wraps ``_run_impl`` with optional DOM-change detection.

        When ``_detect_dom_change`` is True, captures a before-snapshot, runs the
        action, then appends a DOM CHANGE DETECTED message listing any new element
        refs that appeared. Subclasses that override ``run()`` call
        ``super()._run_impl()`` directly so they retain control of when (and if) the
        delta is appended.
        """
        if not self._detect_dom_change:
            return self._run_impl(args)
        before = capture_page_snapshot_raw(self._cli) or ""
        result = self._run_impl(args)
        return _check_dom_delta(self._cli, before, result)

    def _guard_target(self, args: dict) -> ToolResult | None:
        """Reject a ``target`` that is not a ref present in the current snapshot.

        Returns a ready ok=False :class:`ToolResult` (listing the available refs)
        when the target is invalid, or ``None`` to let the action proceed. We
        source the ref set from a FRESH snapshot captured through the SAME session
        the tool drives (:func:`capture_page_snapshot_raw`), so it reflects the
        exact page the model is acting on. If the snapshot can't be captured we
        fail open (proceed) rather than block a legitimate action on a flaky read.
        """
        target = args.get("target") or ""
        ref = normalize_ref(target)
        snapshot = capture_page_snapshot_raw(self._cli)
        if not snapshot:
            return None  # no snapshot to validate against: don't block
        refs = parse_snapshot_refs(snapshot)
        if ref is not None and ref in refs:
            return None  # valid ref present on the page
        # Prefer to show the INTERACTABLE controls (textbox/button/combobox/…) with their
        # labels rather than the raw ref dump — a small model that guessed a sequential ref
        # (iter-1: clicked e208/e209/e210 that never existed after a page advance) needs to
        # see the REAL field/button refs on the now-current page, not "e1..e999". Falls back
        # to the full ref list only when no interactable line could be extracted.
        interactable = _interactable_ref_lines(snapshot)
        if interactable:
            avail_desc = "Interactable elements on the CURRENT page:\n" + "\n".join(
                f"  {ln}" for ln in interactable)
        else:
            avail_desc = ("Available refs on the current page: "
                          + (", ".join(sorted(refs, key=lambda r: int(r[1:]))) or "(none)"))
        return ToolResult(
            False,
            f"you tried to {self._verb} '{target}' but '{target}' does NOT exist on the "
            f"current page. The page snapshot changed (a Continue/Back click loads a NEW "
            f"step whose refs are completely DIFFERENT) — refs are NOT sequential and you "
            f"must NEVER guess or increment them (e.g. e208 -> e209). READ the live page "
            f"snapshot above and copy a ref EXACTLY as shown.\n" + avail_desc,
        )

    def _trim(self, text: str) -> str:
        if len(text) <= self._limit:
            return text
        return text[:self._limit] + f"\n…[+{len(text) - self._limit} chars truncated]"


# ---------------------------------------------------------------------------
# Discrete, first-class browser tools — one per basic playwright-cli operation.
# Each is registered by WebToolset.create_tools and appears in the prompt + the
# codec call-schema in its own right. There is NO `snapshot` tool: the page is
# auto-injected every turn (see the module docstring + capture_* helpers).
# ---------------------------------------------------------------------------


class GotoTool(_WebTool):
    name = "goto"
    description = (
        "Navigate the browser to a URL. The page, cookies and history persist. "
        "If no browser session is open yet, this will open one automatically — "
        "use goto as your FIRST action to open the target page."
    )
    _verb = "navigated to"
    _required = ("url",)

    @property
    def parameters(self):
        return [Param("url", "string", "The URL to open, e.g. 'https://example.com'.")]

    def _build(self, args):
        return ["goto", args["url"]]


# A wizard "go to the PREVIOUS step" control — matched on its accessible name. Clicking
# it abandons the current step's progress, so for a forward fill-and-submit task it is
# almost always a mistake (iter-2: the model clicked "← Back" instead of setting the date,
# bouncing page2->page1 and oscillating forever).
_BACK_BUTTON_RE = re.compile(r'button\s+"[^"]*\b(?:back|previous|prev)\b[^"]*"', re.IGNORECASE)


class ClickTool(_WebTool):
    name = "click"
    description = "Click an element (a link, button, checkbox, …). " + _REF_NOTE
    _verb = "clicked"
    _required = ("target",)
    _validate_target = True
    _detect_dom_change = True

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref (e.g. 'e163') from the current page snapshot to "
                      "click. Must be a ref from the snapshot — never a guessed CSS selector/class/id.")]

    def _build(self, args):
        return ["click", args["target"]]

    def run(self, args: dict) -> ToolResult:
        # BACK-BUTTON GUARD (iter-2): block a click on the wizard "← Back"/"Previous"
        # control. The task is forward-only (fill every step, then Continue/Submit); going
        # back discards the current step and caused a page2->page1->page2 oscillation when
        # the model used Back instead of filling the date and clicking Continue. Steer it to
        # the right action (set any remaining field — especially the date picker — then click
        # Continue). Fail open if we can't read the snapshot.
        target = normalize_ref(args.get("target", "") or "")
        if target is not None:
            snap = capture_page_snapshot_raw(self._cli) or ""
            for line in snap.splitlines():
                if f"[ref={target}]" in line and _BACK_BUTTON_RE.search(line):
                    date_ref = SelectOptionTool._find_date_combobox_ref(snap)
                    date_hint = (f" If the 'Earliest available start date' is not set yet, "
                                 f"call select_option(target='{date_ref}', "
                                 f"value='2026-07-21') first." if date_ref else "")
                    return ToolResult(
                        False,
                        f"I did NOT click '{target}' — it is the BACK/Previous button, which "
                        f"would discard this step's progress and send you to an earlier step. "
                        f"Do not go back. Instead, fill any remaining required field on THIS "
                        f"step, then click the 'Continue →' button to advance.{date_hint}")
        return super().run(args)


class FillTool(_WebTool):
    name = "fill"
    description = ("Set a text input / textarea to an exact value, clearing it first. " + _REF_NOTE)
    _verb = "filled"
    _required = ("target", "text")
    _validate_target = True

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot of the field "
                  "to fill — never a guessed CSS selector/class/id."),
            Param("text", "string", "The exact text to put in the field."),
        ]

    def _build(self, args):
        return ["fill", args["target"], args["text"]]

    def run(self, args: dict) -> ToolResult:
        # Capture the DOM before calling the action so we can detect new elements.
        before = capture_page_snapshot_raw(self._cli) or ""
        result = self._run_impl(args)
        if not result.ok:
            obs_lower = (result.observation or "").lower()
            if any(p in obs_lower for p in _NOT_INTERACTABLE_PHRASES):
                target = args.get("target", "")
                # Extract the HTML snippet Playwright gives us.
                m = _RESOLVED_TO_RE.search(result.observation or "")
                html_snippet = m.group(1).strip() if m else "(unknown element type)"
                # Find nearby interactable refs from a fresh snapshot.
                snapshot = capture_page_snapshot_raw(self._cli) or ""
                nearby = _nearby_interactable_refs(target, snapshot)
                # If the element is a combobox/select/listbox, fill will NEVER work — the
                # correct tool is select_option on the SAME ref (it opens a custom combobox
                # and clicks the matching option for you). Make that the explicit, primary
                # instruction so the 3B model stops retrying fill (iter-1: State/Country).
                snip_l = html_snippet.lower()
                is_dropdown = ('role="combobox"' in snip_l or 'role="listbox"' in snip_l
                               or "<select" in snip_l or 'haspopup="listbox"' in snip_l)
                if is_dropdown:
                    obs = (
                        f"ERROR: '{target}' is a DROPDOWN (combobox/select), not a text box — "
                        f"fill/type will NEVER work on it. Use select_option on the SAME ref: "
                        f"select_option(target='{target}', value='<the exact option text>'). "
                        f"That opens the dropdown and clicks the matching option for you. "
                        f"Do NOT call fill/type on '{target}' again. "
                        f"Playwright resolved it to: {html_snippet}"
                    )
                else:
                    obs = (
                        f"ERROR: '{target}' is NOT interactable for text input. "
                        f"Playwright resolved it to: {html_snippet}  "
                        f"This element cannot accept fill/type — it is a label, div, span, "
                        f"or other non-input element. "
                        f"Nearby interactable elements: {nearby}. "
                        f"Pick one of those refs instead, or use select_option/click if "
                        f"it is a dropdown or combobox."
                    )
                return ToolResult(False, obs)
            return result
        # Action succeeded — append DOM change delta if any new elements appeared.
        return _check_dom_delta(self._cli, before, result)


class TypeTool(_WebTool):
    name = "type"
    description = ("Type text into the currently focused element, keystroke by keystroke "
                  "(use `fill` to set a field's value directly).")
    _verb = "typed into"
    _required = ("text",)
    _detect_dom_change = True

    def _subject(self, args):
        return "the focused element"

    @property
    def parameters(self):
        return [Param("text", "string", "The text to type into the focused element.")]

    def _build(self, args):
        return ["type", args["text"]]


class PressKeyTool(_WebTool):
    name = "press_key"
    description = "Press a single keyboard key, e.g. 'Enter', 'Tab', 'ArrowLeft', 'Escape'."
    _verb = "pressed a key on"
    _required = ("key",)
    _detect_dom_change = True

    @property
    def parameters(self):
        return [Param("key", "string", "The key to press, e.g. 'Enter' or 'Tab'.")]

    def _build(self, args):
        return ["press", args["key"]]


class SelectOptionTool(_WebTool):
    name = "select_option"
    description = ("Choose an option in a dropdown — a native <select> OR a custom "
                   "listbox/combobox. " + _REF_NOTE)
    _verb = "selected an option in"
    _required = ("target", "value")
    _validate_target = True

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot of the "
                  "dropdown — never a guessed CSS selector/class/id."),
            Param("value", "string", "The option value or visible label to select."),
        ]

    def _build(self, args):
        return ["select", args["target"], args["value"]]

    def run(self, args: dict) -> ToolResult:
        # Capture DOM before the action so we can detect new elements afterward.
        before = capture_page_snapshot_raw(self._cli) or ""
        # ISO-DATE TARGET GUARD (iter-2): when the value is an ISO date the model means to
        # drive a DATE PICKER. If the targeted ref is NOT the date field (a stale/shifted
        # ref pointing at the Back button or another combobox — refs shift after each
        # combobox interaction), clicking it would navigate away (iter-1: clicked "← Back",
        # bounced to page 1) and wreck the run. Verify the target's own snapshot line looks
        # like a date field; if not, point the model at the REAL date field's ref instead of
        # acting on the wrong element.
        if _ISO_DATE_RE.match(args.get("value") or ""):
            misfire = self._guard_date_target(args.get("target", ""), before)
            if misfire is not None:
                return misfire
        # SPINBUTTON GUARD (iter-2 fix): a number stepper (spinbutton) is not a dropdown
        # and cannot be interacted with via select_option. Its value is set by clicking the
        # adjacent Increase/Decrease buttons (not the spinbutton container itself). The guard
        # detects a spinbutton target and returns a message pointing to the correct Increase
        # button ref so the agent can batch-click it to reach the target value.
        # Must run BEFORE _target_is_plain_button because "spinbutton" contains "button" and
        # the plain-button guard would otherwise fire with a less specific 'use click(eN)'
        # message that points to the spinbutton container — not the Increase button.
        sb = self._guard_spinbutton(args.get("target", ""), before)
        if sb is not None:
            return sb
        # PLAIN-BUTTON GUARD (iter-2): the model sometimes calls select_option on a real
        # <button> (e.g. "Continue →"/"Submit") instead of click. The custom-combobox
        # fallback would then just CLICK the button — which "works" once but records the
        # button as a successful select_option; the anti-loop guard then BLOCKS every later
        # select_option on the next page's Continue button (same signature), so the form
        # gets stuck (iter-2: stuck on Step 2 re-emitting select_option e81). A button is
        # NOT a dropdown: steer the model to use click, and do NOT act, so nothing is
        # recorded and the model switches tools.
        btn = self._target_is_plain_button(args.get("target", ""), before)
        if btn is not None:
            return btn
        # Try the native <select> path first (the base behaviour).
        result = self._run_impl(args)
        if result.ok:
            return _check_dom_delta(self._cli, before, result)
        # FALLBACK (#125): a CUSTOM combobox (<div role="listbox">) is not a native
        # <select>, so Playwright's `select` fails ("Element is not a <select> element")
        # and the discrete subtools alone never reach the (closed) options. Detect that
        # specific failure and drive the combobox the human way: click the trigger to
        # OPEN it, then click the option whose text matches `value`. If the option can't
        # be auto-matched, we still leave the list OPEN so its options appear in the next
        # snapshot for the model to click — strictly better than the old hard failure.
        obs = (result.observation or "").lower()
        if "not a <select>" in obs or "is not a select" in obs or "<select> element" in obs:
            combo = self._select_via_combobox(args["target"], args["value"])
            if combo is not None:
                return _check_dom_delta(self._cli, before, combo)
        return result

    # How many times to re-snapshot (and how long to wait between) while a freshly
    # opened custom popup renders its content. ~5 × 0.4s ≈ 2s, well under the per-tool
    # timeout, covers the observed open animation/portal-mount delay (iter-2).
    _POPUP_SETTLE_TRIES = 5
    _POPUP_SETTLE_DELAY = 0.4

    def _await_popup(self, value: str) -> str:
        """Re-snapshot until a just-opened popup's content appears, then return it.

        A custom combobox/date-picker mounts its listbox or calendar dialog AFTER the
        trigger click (React portal + open animation), so the immediate snapshot can be
        the pre-open page. Poll a few times until the snapshot shows EITHER a calendar
        header (date path) OR a matching option / any open listbox (option path); return
        the last snapshot read once content is detected or the budget is spent. Always
        returns a string (never blocks indefinitely)."""
        want_date = bool(_ISO_DATE_RE.match(value or ""))
        snapshot = capture_page_snapshot_raw(self._cli)
        for _ in range(self._POPUP_SETTLE_TRIES):
            if want_date and calendar_view_month(snapshot) is not None:
                return snapshot
            if not want_date and (
                    find_option_ref_by_text(snapshot, value) is not None
                    or _OPTION_LINE_RE.search(snapshot or "")):
                return snapshot
            time.sleep(self._POPUP_SETTLE_DELAY)
            snapshot = capture_page_snapshot_raw(self._cli)
        return snapshot

    def _guard_date_target(self, target: str, snapshot: str) -> "ToolResult | None":
        """Reject a date select_option whose target is NOT a date field (iter-2).

        Returns a steering :class:`ToolResult` when ``target`` does not look like the
        page's date field (so we DON'T click the wrong element — e.g. the Back button —
        and navigate away), or ``None`` to let the normal date path proceed. Identifies
        the date field by scanning the snapshot for a combobox whose accessible name
        mentions 'date'. If the target IS that combobox we proceed; otherwise we name the
        correct ref. Fails OPEN (returns None) if no date combobox can be found, so a
        page with an unusual date markup is not blocked."""
        ref = normalize_ref(target)
        date_ref = self._find_date_combobox_ref(snapshot)
        if date_ref is None:
            return None  # can't identify a date field -> don't block
        if ref == date_ref:
            return None  # targeting the real date field -> proceed
        # Is the model's chosen target plausibly a date field by its own line?
        for line in (snapshot or "").splitlines():
            if ref and f"[ref={ref}]" in line and "date" in line.lower():
                return None  # the target's own label says 'date' -> trust it
        return ToolResult(
            False,
            f"'{target}' is NOT the date field, so I did not click it (clicking the wrong "
            f"element here can navigate the form backwards). The date picker on this page is "
            f"'{date_ref}'. Call select_option(target='{date_ref}', value='<yyyy-mm-dd>') — "
            f"the harness will open the calendar and pick the day for you. Re-read the live "
            f"snapshot and use '{date_ref}' for the date.",
        )

    @staticmethod
    def _guard_spinbutton(target: str, snapshot: str) -> "ToolResult | None":
        """Reject select_option on a spinbutton, pointing to its Increase button (iter-2).

        A number stepper (spinbutton) cannot be interacted with via select_option or fill
        — the value is changed by clicking its adjacent Increase/Decrease BUTTONS. When the
        agent calls select_option on a spinbutton ref, return a targeted message naming the
        correct Increase button ref so the agent clicks the right thing immediately. Must
        run BEFORE _target_is_plain_button, since spinbutton lines contain 'button' as a
        substring and would otherwise be caught by the plain-button guard with a less
        specific 'use click(eN)' message that points to the spinbutton container (wrong)."""
        ref = normalize_ref(target)
        if ref is None:
            return None
        lines = (snapshot or "").splitlines()
        for i, line in enumerate(lines):
            if f"[ref={ref}]" not in line:
                continue
            if "spinbutton" not in line.lower():
                return None  # not a spinbutton -> let normal path handle it
            # Found the spinbutton. Search forward for the Increase button.
            increase_ref = None
            for j in range(i + 1, min(i + 8, len(lines))):
                sib = lines[j]
                if "increase" in sib.lower() and "[ref=" in sib:
                    m = re.search(r"\[ref=(e\d+)\]", sib)
                    if m:
                        increase_ref = m.group(1)
                        break
            if increase_ref:
                return ToolResult(
                    False,
                    f"'{target}' is a NUMBER STEPPER (spinbutton) — select_option/fill do "
                    f"NOT apply. To increment it click the INCREASE BUTTON '{increase_ref}' "
                    f"(aria-label 'Increase …') — NOT the spinbutton container '{target}' "
                    f"which only focuses the field without changing the value. "
                    f"To set the value to 9: click(target='{increase_ref}') nine times. "
                    f"You can batch all 9 clicks in one turn.",
                )
            return ToolResult(
                False,
                f"'{target}' is a NUMBER STEPPER (spinbutton) — select_option/fill do not "
                f"apply. Find the 'Increase …' button sibling in the snapshot (it is a "
                f"button whose aria-label starts with 'Increase') and click it once per "
                f"unit (9 times to reach 9). Do NOT click the spinbutton container "
                f"itself — only the Increase/Decrease sibling buttons change the value.",
            )
        return None  # target not in snapshot -> fail open

    @staticmethod
    def _target_is_plain_button(target: str, snapshot: str) -> "ToolResult | None":
        """Steer to `click` when ``target`` is a plain <button> (not a dropdown).

        Returns a steering :class:`ToolResult` when the target's snapshot line is a button
        WITHOUT any combobox/listbox/select affordance (a real action button like
        "Continue →"/"Submit"), so select_option does not accidentally click it and poison
        the anti-loop guard (iter-2). Returns ``None`` (proceed) for combobox-buttons
        (`button` lines that ALSO say combobox/listbox/haspopup) and when the target can't
        be found (fail open). Spinbuttons are excluded here — they are handled earlier by
        _guard_spinbutton, which gives a more targeted message about the Increase button."""
        ref = normalize_ref(target)
        if ref is None:
            return None
        for line in (snapshot or "").splitlines():
            if f"[ref={ref}]" not in line:
                continue
            low = line.lower()
            if "button" not in low:
                return None  # not a button line -> let normal path handle it
            # A combobox/listbox rendered as a button IS a dropdown -> proceed normally.
            # Spinbuttons contain 'button' as a substring but are handled by _guard_spinbutton.
            if any(k in low for k in ("combobox", "listbox", "haspopup", "select", "expanded",
                                      "spinbutton")):
                return None
            return ToolResult(
                False,
                f"'{target}' is a BUTTON, not a dropdown — select_option does not apply. "
                f"Use click(target='{target}') to press it. If this is the "
                f"Continue/Next/Submit button, click it to advance to the next step.",
            )
        return None  # target not in snapshot -> fail open

    @staticmethod
    def _find_date_combobox_ref(snapshot: str) -> str | None:
        """Ref of the combobox whose accessible name mentions 'date' (the date picker)."""
        for line in (snapshot or "").splitlines():
            low = line.lower()
            if "combobox" in low and "date" in low:
                m = re.search(r"\[ref=(e\d+)\]", line)
                if m:
                    return m.group(1)
        return None

    def _select_via_combobox(self, target: str, value: str) -> ToolResult | None:
        """Open a custom combobox by clicking ``target`` and click the matching option.

        Returns a :class:`ToolResult`, or ``None`` to fall back to the original error
        (e.g. the trigger click itself failed for a non-combobox reason)."""
        # ITER-4 FIX (root cause 2): the agent sometimes calls select_option on the OPEN
        # listbox container (or one of its option nodes) instead of the combobox TRIGGER —
        # i.e. the dropdown is ALREADY open. The normal path below presses Escape first to
        # clear stray overlays, but here that Escape would CLOSE the very listbox we want and
        # the selection would never happen (iter-3 turn 5: select_option(e188=listbox,
        # 'United States') -> Escape closed it -> click on now-invisible e188 -> Country never
        # set -> 4 blind Continue clicks). When the target is itself an open listbox/option,
        # skip Escape+trigger-click entirely and click the matching option directly from the
        # already-visible options.
        pre_snapshot = capture_page_snapshot_raw(self._cli) or ""
        if target_is_open_listbox(pre_snapshot, target):
            ref = find_option_ref_by_text(pre_snapshot, value)
            if ref is None:
                return ToolResult(
                    True,
                    f"the '{value}' option's dropdown is already OPEN. Its options are shown in "
                    f"the page snapshot — click the one matching '{value}' by its ref (do not "
                    f"call select_option on the listbox container itself).")
            ok, out = self._cli.run("click", ref)
            if ok and output_signals_no_match(out):
                ok = False
            if not ok:
                return ToolResult(
                    False,
                    f"the dropdown was already open but clicking the '{value}' option ({ref}) "
                    f"failed: {self._trim(out)}")
            return ToolResult(
                True,
                f"you selected '{value}' by clicking option {ref} in the already-open dropdown. "
                f"Result:\n{self._trim(out)}")
        # ITER-3 FIX: if another dropdown/listbox is currently open (e.g. the agent called
        # select_option on two comboboxes in the same turn), its overlay intercepts the click
        # on ``target`` and prevents this combobox from opening — which makes the fallback
        # click return ok=True but the calendar/listbox never appears, causing _select_via_combobox
        # to return None and propagate the native "not a <select>" error (root cause of the
        # work-arrangement + date-picker interference on Step 2 in iter-2). Press Escape first
        # to close any open popup/listbox so the click reaches the actual trigger.
        self._cli.run("press", "Escape")
        time.sleep(0.15)
        ok, out = self._cli.run("click", target)
        if not ok and output_signals_session_closed(out):
            return ToolResult(False, f"you tried to open the '{target}' dropdown but the browser "
                              f"session is closed; call `open_browser`, then `goto` your URL.")
        if not ok or output_signals_no_match(out):
            return None  # couldn't even open it -> let the native error stand
        # SETTLE-RETRY (iter-2 fix): a custom popup (listbox OR calendar dialog) renders/
        # animates ASYNCHRONOUSLY after the trigger click. The FIRST snapshot taken
        # immediately after the click frequently shows the page WITHOUT the popup yet — so
        # calendar_view_month()/find_option_ref_by_text() both return nothing, the calendar
        # branch is skipped, and the date/option is never set (iter-1 blocker: page 2 date
        # never set -> form stuck on Step 2/8). Re-snapshot a few times until the popup's
        # content (a calendar header OR a matching option) actually appears.
        snapshot = self._await_popup(value)
        # CALENDAR date picker (iter-1): a DatePicker opens a calendar dialog, not a list of
        # options. A 3B model cannot blindly navigate from the default month to the target by
        # clicking month-nav arrows, so drive it deterministically here: parse the requested
        # ISO date, click Next/Previous year+month the exact number of times to reach the
        # target month, then click the day cell whose aria-label == the ISO date.
        if _ISO_DATE_RE.match(value or ""):
            cal = self._select_calendar_date(target, value, snapshot)
            if cal is not None:
                return cal
        ref = find_option_ref_by_text(snapshot, value)
        if ref is None:
            # Opened but no auto-match: leave it open and tell the model to click the option.
            return ToolResult(True, f"you opened the '{target}' dropdown (it is a custom combobox, "
                              f"not a native <select>). Its options are now shown in the page "
                              f"snapshot — click the one matching '{value}' by its ref.")
        ok2, out2 = self._cli.run("click", ref)
        if ok2 and output_signals_no_match(out2):
            ok2 = False
        if not ok2:
            return ToolResult(False, f"you opened the '{target}' dropdown but clicking the '{value}' "
                              f"option ({ref}) failed: {self._trim(out2)}")
        return ToolResult(True, f"you selected '{value}' in the '{target}' combobox (opened it and "
                          f"clicked option {ref}). Result:\n{self._trim(out2)}")

    def _select_calendar_date(self, target: str, iso_date: str,
                              snapshot: str) -> ToolResult | None:
        """Drive an OPEN custom calendar to ``iso_date`` (yyyy-mm-dd) and click the day.

        Returns a ToolResult on success/explicit failure, or ``None`` if the open popup is
        NOT a calendar (so the caller falls back to the normal option-matching path). Steps
        the month/year nav buttons the exact number of times needed, then clicks the day
        cell whose ISO aria-label matches. Deterministic — no model month-navigation needed."""
        view = calendar_view_month(snapshot)
        if view is None:
            return None  # not a calendar -> let the caller try option matching
        m = _ISO_DATE_RE.match(iso_date)
        target_y, target_m = int(m.group(1)), int(m.group(2)) - 1  # 0-based month
        snap = snapshot
        cur_y, cur_m = view
        # Bounded loop: navigate to the target month. 700 steps covers ~58 years either way;
        # the cap purely prevents an infinite loop if the calendar markup is unexpected.
        for _ in range(700):
            # Signed month delta from the currently-shown month to the target.
            delta = (target_y - cur_y) * 12 + (target_m - cur_m)
            if delta == 0:
                break
            # Jump by year when >=12 months off (faster), else step a single month.
            if delta >= 12:
                label = "Next year"
            elif delta <= -12:
                label = "Previous year"
            elif delta > 0:
                label = "Next month"
            else:
                label = "Previous month"
            nav = find_nav_button_ref(snap, label)
            if nav is None:
                return None  # unexpected markup -> let caller fall back
            self._cli.run("click", nav)
            # Re-read the calendar header after the nav click to get the new view. The
            # month label updates ASYNCHRONOUSLY, so poll briefly until it actually moves
            # off (cur_y, cur_m) — otherwise a too-fast re-read sees the OLD month and the
            # loop double-steps or stalls (iter-2: calendar nav race).
            prev_y, prev_m = cur_y, cur_m
            view = None
            for _ in range(4):
                snap = capture_page_snapshot_raw(self._cli)
                view = calendar_view_month(snap)
                if view is not None and view != (prev_y, prev_m):
                    break
                time.sleep(0.15)
            if view is None:
                break
            cur_y, cur_m = view
        # Now click the day cell carrying the exact ISO date. Poll briefly in case the
        # final month grid is still rendering.
        day_ref = None
        for _ in range(4):
            snap = capture_page_snapshot_raw(self._cli)
            day_ref = find_day_button_ref(snap, iso_date)
            if day_ref is not None:
                break
            time.sleep(0.15)
        if day_ref is None:
            return ToolResult(True, f"you opened the '{target}' date picker and navigated to "
                              f"{iso_date[:7]}. Click the day cell whose label is exactly "
                              f"'{iso_date}' by its ref (use the calendar nav buttons if the "
                              f"month is still wrong).")
        ok, out = self._cli.run("click", day_ref)
        if ok and output_signals_no_match(out):
            ok = False
        if not ok:
            return ToolResult(False, f"you opened the '{target}' date picker and reached "
                              f"{iso_date[:7]}, but clicking the day '{iso_date}' ({day_ref}) "
                              f"failed: {self._trim(out)}")
        return ToolResult(True, f"you set the '{target}' date to {iso_date} (opened the calendar, "
                          f"navigated to {iso_date[:7]}, and clicked day {day_ref}).")


# A spinbutton's adjacent display generic shows its current value as "<number> yrs"
# (e.g. "0 yrs", "9 yrs"). Used to read the current value when no aria-valuenow is
# captured in the snapshot. Captures the first integer on/near the spinbutton line.
_SPINBUTTON_YRS_RE = re.compile(r"(\d+)\s*yrs?\b", re.IGNORECASE)
# aria-valuenow is the authoritative current value when the snapshot exposes it on the
# spinbutton line (e.g. "spinbutton ... [valuenow=2]" / "aria-valuenow=\"2\"").
_VALUENOW_RE = re.compile(r'(?:aria-)?valuenow[=:]\s*"?(\d+)"?', re.IGNORECASE)


def spinbutton_current_value(snapshot: str, ref: str) -> int | None:
    """Best-effort current integer value of the spinbutton ``ref`` from ``snapshot``.

    Reads, in priority order: an ``aria-valuenow``/``valuenow`` on the spinbutton's own
    line, then a ``<number> yrs`` display on the spinbutton line or the line immediately
    after it (the adjacent display generic, e.g. ``generic [ref=eN]: "0 yrs"``). Returns
    ``None`` when the ref isn't present or no value can be read (the caller then refuses
    to act rather than guess)."""
    if not ref:
        return None
    lines = (snapshot or "").splitlines()
    for i, line in enumerate(lines):
        if f"[ref={ref}]" not in line:
            continue
        m = _VALUENOW_RE.search(line)
        if m:
            return int(m.group(1))
        m = _SPINBUTTON_YRS_RE.search(line)
        if m:
            return int(m.group(1))
        # Fall back to the adjacent display generic on the next couple of lines.
        for j in range(i + 1, min(i + 3, len(lines))):
            m = _SPINBUTTON_YRS_RE.search(lines[j])
            if m:
                return int(m.group(1))
        return None
    return None


def find_stepper_button_ref(snapshot: str, ref: str, direction: str) -> str | None:
    """Ref of the Increase/Decrease button adjacent to spinbutton ``ref`` in ``snapshot``.

    ``direction`` is 'increase' or 'decrease'. Searches the lines following the
    spinbutton (its sibling stepper buttons, e.g. ``button "Increase Total years…"``)
    for a button whose accessible name starts with the direction word, returning its
    ref. Returns ``None`` when no such sibling button is found."""
    want = direction.lower()
    lines = (snapshot or "").splitlines()
    for i, line in enumerate(lines):
        if f"[ref={ref}]" not in line:
            continue
        for j in range(i + 1, min(i + 10, len(lines))):
            sib = lines[j]
            low = sib.lower()
            if want in low and "button" in low:
                m = re.search(r"\[ref=(e\d+)\]", sib)
                if m:
                    return m.group(1)
        return None
    return None


class SetSpinbuttonTool(_WebTool):
    """Set a number stepper (role="spinbutton") to a target value in ONE tool call.

    qwen3:4b makes exactly one LLM tool call per spinbutton click, so reaching "9 years"
    used to cost nine LLM roundtrips per field (iter-4: stalled on page 3 with
    totalExperienceYears stuck at 2). This tool reads the spinbutton's current value from
    the snapshot, finds its adjacent Increase/Decrease sibling buttons, and clicks the
    right one ``|target - current|`` times INSIDE Python — so the model sets the value in
    a single roundtrip instead of N."""

    name = "set_spinbutton"
    description = (
        "Set a spinbutton (numeric stepper) to a specific integer value by clicking its "
        "Increase/Decrease buttons. Use this instead of clicking the Increase button many "
        "times. Target the element whose role is 'spinbutton' (NOT the Increase/Decrease "
        "buttons). " + _REF_NOTE
    )
    _verb = "set the spinbutton"
    _required = ("target", "value")
    _validate_target = True

    # Refuse to click a stepper more than this many times in one call — a runaway delta
    # almost certainly means a misread current value, and 50+ clicks would wedge the turn.
    _MAX_CLICKS = 50

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref (e.g. 'e730') of the spinbutton (the numeric "
                  "stepper container, role='spinbutton') — never the Increase/Decrease button, and "
                  "never a guessed CSS selector/class/id."),
            Param("value", "integer", "The integer value to set the spinbutton to (e.g. 9)."),
        ]

    def _subject(self, args):
        return f"spinbutton '{args.get('target')}'"

    def _build(self, args):  # pragma: no cover - run() is overridden, never reached
        return ["snapshot"]

    def run(self, args: dict) -> ToolResult:
        # Standard missing-param + ref-on-page guards (issues #51/#73).
        missing = [p for p in self._required if args.get(p) in (None, "")]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")
        guard = self._guard_target(args)
        if guard is not None:
            return guard
        target = normalize_ref(args.get("target", "") or "")
        try:
            value = int(args["value"])
        except (TypeError, ValueError):
            return ToolResult(False, f"you called `{self.name}` with a non-integer value "
                              f"'{args.get('value')}'. Pass an integer, e.g. value=9.")

        snapshot = capture_page_snapshot_raw(self._cli) or ""
        # Confirm the target really is a spinbutton; if not, steer to the right tool so a
        # mistargeted call doesn't blindly click a sibling button N times.
        sb_line = _snapshot_line_for_ref(snapshot, target)
        if sb_line is not None and "spinbutton" not in sb_line.lower():
            return ToolResult(
                False,
                f"'{args.get('target')}' is not a spinbutton (its snapshot line is: {sb_line[:90]}). "
                f"set_spinbutton only applies to a numeric stepper (role='spinbutton'). Use fill "
                f"for text inputs, select_option for dropdowns, or click for buttons.")

        current = spinbutton_current_value(snapshot, target)
        if current is None:
            current = 0  # display not parseable -> assume the stepper starts at 0
        delta = value - current
        if delta == 0:
            return ToolResult(True, f"spinbutton '{args.get('target')}' is already {value} — "
                              f"no clicks needed.")
        direction = "increase" if delta > 0 else "decrease"
        clicks = abs(delta)
        if clicks > self._MAX_CLICKS:
            return ToolResult(
                False,
                f"refusing to click the stepper {clicks} times to go from {current} to {value} "
                f"(over the {self._MAX_CLICKS}-click safety limit). Re-check the target value; "
                f"if it really needs that many steps, set it in smaller increments.")

        button_ref = find_stepper_button_ref(snapshot, target, direction)
        if button_ref is None:
            return ToolResult(
                False,
                f"could not find the '{direction.capitalize()}' button next to spinbutton "
                f"'{args.get('target')}' in the snapshot. The stepper's Increase/Decrease buttons "
                f"are siblings of the spinbutton — re-read the live page snapshot and use the "
                f"button whose label starts with '{direction.capitalize()}'.")

        before = capture_page_snapshot_raw(self._cli) or ""
        for n in range(clicks):
            ok, out = self._cli.run("click", button_ref)
            if ok and output_signals_no_match(out):
                ok = False
            if not ok:
                if output_signals_session_closed(out):
                    return ToolResult(
                        False,
                        f"you tried to set spinbutton '{args.get('target')}' but the browser "
                        f"session is closed and could not be restored. Call `open_browser`, then "
                        f"`goto` your URL and continue.")
                return ToolResult(
                    False,
                    f"set spinbutton '{args.get('target')}' partway: clicked {direction.capitalize()} "
                    f"({button_ref}) {n} of {clicks} times before a click failed: {self._trim(out)}")
        result = ToolResult(
            True,
            f"you set spinbutton '{args.get('target')}' to {value} (clicked {direction.capitalize()} "
            f"button {button_ref} {clicks} time(s), from {current}).")
        return _check_dom_delta(self._cli, before, result)


class DrawSignatureTool(_WebTool):
    """Draw a signature stroke on a canvas drawing pad (SignaturePad component).

    The Review & Submit page has a canvas element (role='img', aria-label='Signature')
    that requires drawn pointer-event strokes — fill/click cannot activate it.
    This tool dispatches pointerdown + pointermove + pointerup via JS evaluate to
    simulate a drawn stroke, satisfying the 'Please provide your signature' validation.

    The SignaturePad React component sets drawing.current=true and calls ctx.beginPath/
    moveTo BEFORE setPointerCapture, so the stroke lands even if setPointerCapture
    throws for synthetic events. We wrap that dispatchEvent in try/catch to absorb it.
    """

    name = "draw_signature"
    description = (
        "Draw a signature stroke on the canvas signature pad on the Review & Submit page. "
        "Use when the form shows 'Please provide your signature'. "
        "Pass the ref of the signature canvas (role='img', aria-label='Signature'). "
        "Do NOT use this on the 'Type your full legal name' textbox — fill that separately first. "
        + _REF_NOTE
    )
    _verb = "drew signature on"
    _required = ("target",)
    _validate_target = True

    @property
    def parameters(self):
        return [
            Param("target", "string",
                  "Element ref of the signature canvas (role='img', aria-label='Signature') "
                  "from the current snapshot. NOT the 'Type your full legal name' textbox."),
        ]

    def _subject(self, args):
        return f"signature canvas '{args.get('target')}'"

    def _build(self, args):  # pragma: no cover — run() is overridden
        return ["snapshot"]

    def run(self, args: dict) -> ToolResult:
        missing = [p for p in self._required if args.get(p) in (None, "")]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")
        guard = self._guard_target(args)
        if guard is not None:
            return guard

        # Dispatch pointer events on the signature canvas via JS evaluate.
        # setPointerCapture in onPointerDown may throw for synthetic events, but
        # drawing.current=true, ctx.beginPath, and ctx.moveTo all execute BEFORE it,
        # so the stroke still registers. Wrap that dispatchEvent in try/catch.
        js = (
            "() => {"
            "  const c = document.querySelector('canvas[aria-label=\"Signature\"]');"
            "  if (!c) return 'ERROR: no canvas[aria-label=Signature] found';"
            "  const r = c.getBoundingClientRect();"
            "  const y  = r.top  + r.height * 0.5;"
            "  const x1 = r.left + r.width  * 0.15;"
            "  const xm = r.left + r.width  * 0.50;"
            "  const x2 = r.left + r.width  * 0.85;"
            "  const pe = (t, x) => new PointerEvent(t, {"
            "    bubbles:true, cancelable:true,"
            "    clientX:x, clientY:y,"
            "    pointerId:1, pointerType:'mouse', isPrimary:true,"
            "    button:0, buttons: t==='pointermove' ? 1 : 0"
            "  });"
            "  try { c.dispatchEvent(pe('pointerdown', x1)); } catch(e) {}"
            "  c.dispatchEvent(pe('pointermove', xm));"
            "  c.dispatchEvent(pe('pointermove', x2));"
            "  c.dispatchEvent(pe('pointerup',   x2));"
            "  return 'OK: stroke drawn across canvas (' + Math.round(r.width) + 'x' + Math.round(r.height) + ')';"
            "}"
        )

        before = capture_page_snapshot_raw(self._cli) or ""
        ok, output = self._cli.run("eval", js)
        if not ok:
            return ToolResult(
                False,
                f"you tried to draw a signature on canvas '{args.get('target')}' but the "
                f"JS evaluate call failed: {self._trim(output)}. "
                f"Make sure you are on the Review & Submit page with the signature canvas visible."
            )
        out_text = (output or "").strip()
        if out_text.startswith("ERROR:"):
            return ToolResult(
                False,
                f"you tried to draw a signature but: {out_text}. "
                f"Confirm the signature canvas (role='img', aria-label='Signature') is on the page."
            )
        result = ToolResult(
            True,
            f"you drew a signature stroke on the canvas '{args.get('target')}'. "
            f"Result: {out_text}"
        )
        return _check_dom_delta(self._cli, before, result)


class CheckTool(_WebTool):
    name = "check"
    description = "Check a checkbox or radio button (no-op if already checked). " + _REF_NOTE
    _verb = "checked"
    _required = ("target",)
    _validate_target = True
    _detect_dom_change = True

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot of the "
                      "checkbox/radio — never a guessed CSS selector/class/id.")]

    def _build(self, args):
        return ["check", args["target"]]


class UncheckTool(_WebTool):
    name = "uncheck"
    description = "Uncheck a checkbox (no-op if already unchecked). " + _REF_NOTE
    _verb = "unchecked"
    _required = ("target",)
    _validate_target = True
    _detect_dom_change = True

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot of the "
                      "checkbox — never a guessed CSS selector/class/id.")]

    def _build(self, args):
        return ["uncheck", args["target"]]


class HoverTool(_WebTool):
    name = "hover"
    description = "Move the mouse over an element (e.g. to reveal a menu). " + _REF_NOTE
    _verb = "hovered over"
    _required = ("target",)
    _validate_target = True

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot to hover "
                      "over — never a guessed CSS selector/class/id.")]

    def _build(self, args):
        return ["hover", args["target"]]


class DragTool(_WebTool):
    name = "drag"
    description = "Drag one element and drop it onto another. " + _REF_NOTE
    _verb = "dragged"
    _required = ("target", "end")
    _validate_target = True

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref (e.g. 'e6') from the current snapshot to drag FROM."),
            Param("end", "string", "Element ref (e.g. 'e9') from the current snapshot to drop ONTO."),
        ]

    def _guard_target(self, args):
        """Both endpoints must be refs on the current page. Validate each against
        the same fresh snapshot; reject (listing available refs) if either is not a
        known ref, never calling playwright on a guessed selector (issue #73)."""
        snapshot = capture_page_snapshot_raw(self._cli)
        if not snapshot:
            return None
        refs = parse_snapshot_refs(snapshot)
        for arg_name in ("target", "end"):
            raw = args.get(arg_name) or ""
            ref = normalize_ref(raw)
            if ref is None or ref not in refs:
                available = ", ".join(sorted(refs, key=lambda r: int(r[1:]))) or "(none)"
                return ToolResult(
                    False,
                    f"you tried to drag but '{raw}' (the {arg_name}) is not a valid target: "
                    f"both endpoints must be refs shown in the current page snapshot "
                    f"(e.g. 'e6'); never guess a CSS selector, class or id. "
                    f"Available refs on the current page: {available}.",
                )
        return None

    def _build(self, args):
        return ["drag", args["target"], args["end"]]


class UploadTool(_WebTool):
    name = "upload"
    description = (
        "Upload a file to a file input. "
        "TWO-STEP SEQUENCE REQUIRED: "
        "(1) First CLICK the file-upload trigger element (a button or file input ref) in the "
        "snapshot — this opens the OS file-picker dialog ('modal state'). "
        "(2) Then immediately call `upload` with the ABSOLUTE file path. "
        "Calling upload before clicking the trigger fails with a 'modal state' error. "
        "Always pass the path EXACTLY as given in the task (absolute, e.g. "
        "C:\\\\path\\\\to\\\\file.docx) — never shorten it to a relative path."
    )
    _verb = "uploaded a file to"
    _required = ("file",)

    def _subject(self, args):
        return "the file input"

    @property
    def parameters(self):
        return [Param("file", "string",
                      "Absolute path of the file to upload. Must be an absolute path "
                      "(e.g. C:\\\\Users\\\\...\\\\file.docx), exactly as provided in the task. "
                      "Never use a relative path.")]

    def _build(self, args):
        return ["upload", args["file"]]

    def run(self, args: dict) -> ToolResult:
        result = super().run(args)
        # The Playwright-CLI upload tool is named 'browser_file_upload' internally.
        # When the OS file-picker is not open it exits 0 with "modal state" in the
        # output — which _run_impl now correctly marks ok=False via _NO_MATCH_MARKERS
        # (issue #144). Clean up the error text so the agent sees 'upload' (the tool
        # it called) instead of the internal 'browser_file_upload' name.
        if not result.ok:
            msg = (result.observation or "").replace("browser_file_upload", "upload")
            if "modal state" in msg.lower():
                msg = (
                    "upload failed: the OS file-picker is not open (no 'modal state'). "
                    "You must FIRST click the file-upload trigger button/input ref in the "
                    "snapshot to open the picker, THEN call upload with the absolute path."
                )
            return ToolResult(False, msg)
        return result


class ScreenshotTool(_WebTool):
    name = "screenshot"
    description = "Save a PNG screenshot of the current page (or of one element if `target` is given)."
    _verb = "screenshotted"

    @property
    def parameters(self):
        return [Param("target", "string", "Optional element ref/selector to screenshot just that "
                      "element instead of the whole page.", required=False)]

    def _build(self, args):
        return ["screenshot"] + ([args["target"]] if args.get("target") else [])


class NavigateBackTool(_WebTool):
    name = "navigate_back"
    description = "Go back to the previous page in the browser history."
    _verb = "went back on"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["go-back"]


class NavigateForwardTool(_WebTool):
    name = "navigate_forward"
    description = "Go forward to the next page in the browser history."
    _verb = "went forward on"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["go-forward"]


class ReloadTool(_WebTool):
    name = "reload"
    description = "Reload the current page."
    _verb = "reloaded"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["reload"]


class EvaluateTool(_WebTool):
    """Run JavaScript on the page (or on one element) to inspect its state.

    Use this when you need to understand what kind of element a ref points to before
    deciding which interaction tool to use — e.g. inspect whether a combobox is a
    native <select> or a custom widget, read its current value, list its <option>
    children, or check a data attribute. Returns the JavaScript return value as text.
    Does NOT modify the page. Never use it to set values — use fill/select_option/click.
    """

    name = "evaluate"
    description = (
        "Run a JavaScript snippet on the page (or on a specific element) to inspect "
        "its state. Use to identify element types, list dropdown options, read current "
        "values, or check attributes before choosing the right interaction tool. "
        "Example: expression='el => el.tagName + \" \" + el.type' with target='e6' "
        "inspects element e6. Without a target, the expression runs on the whole page "
        "(use 'document.title' or 'document.readyState' etc.). "
        "Does NOT modify the page."
    )
    _verb = "evaluated JS on"

    @property
    def parameters(self):
        return [
            Param("expression", "string",
                  "JavaScript expression or arrow function to evaluate. "
                  "If target is given, the function receives the DOM element as its first "
                  "argument: 'el => el.tagName'. Without a target, it runs on the page "
                  "as a plain expression: 'document.title'."),
            Param("target", "string",
                  "Optional element ref (e.g. 'e6') from the current snapshot to pass "
                  "to the expression as the first argument. Omit to run on the whole page.",
                  required=False),
        ]

    def _subject(self, args):
        return args.get("target") or "the page"

    def _build(self, args):
        parts = ["eval", args["expression"]]
        if args.get("target"):
            parts.append(args["target"])
        return parts

    def run(self, args: dict) -> ToolResult:
        if "expression" not in args:
            return ToolResult(False, "evaluate requires an 'expression' argument.")
        ok, output = self._cli.run(*self._build(args))
        subject = self._subject(args)
        if not ok:
            return ToolResult(False,
                              f"you tried to evaluate JS on {subject} but it failed: "
                              f"{self._trim(output)}")
        return ToolResult(True,
                          f"you evaluated JS on {subject}. Result:\n{self._trim(output)}")


class OpenBrowserTool(_WebTool):
    """(Re)open the persistent browser session (issues #101, #75).

    The harness opens the browser once at run start, but the persistent daemon can
    die mid-run (a shared session torn down by a concurrent run, an external
    chrome/node reaper, etc.). The other web tools auto-recover by reopening on a
    "not open" failure, but the agent also needs an explicit lever: this tool
    ensures the session is alive so the page can be navigated again. Calling it on an
    already-open session is harmless (it simply re-opens). After opening, the page is
    blank — `goto` your target URL next."""

    name = "open_browser"
    description = ("Open (or re-open) the browser session. Use this if a web action reports the "
                   "browser is not open / the session was closed, or when there is no current "
                   "page to act on. After opening, the page is blank — call `goto` with your "
                   "target URL next.")
    _verb = "opened the browser"

    @property
    def parameters(self):
        return []

    def _subject(self, args):
        return "the browser session"

    def run(self, args: dict) -> ToolResult:
        ok, output = self._cli.open()
        if not ok:
            return ToolResult(False, f"you tried to open the browser session but it failed: "
                              f"{self._trim(output)}")
        return ToolResult(True, "you opened the browser session (the page is now blank — "
                          f"`goto` your target URL next). Result:\n{self._trim(output)}")

    def _build(self, args):  # pragma: no cover - run() is overridden, never reached
        return ["open"]


class SnapshotTool(_WebTool):
    """Explicitly request a fresh page snapshot.

    The page snapshot is auto-injected into the system prompt every turn, but when
    you interact with a dynamic element (e.g. click opens a dropdown, a page section
    expands, a field validates), calling snapshot lets you see the CURRENT DOM in the
    tool result sequence — useful when you need to find the refs of newly appeared
    options or changed elements mid-turn BEFORE deciding the next action.

    When you call snapshot this turn, the auto-injected snapshot is SUPPRESSED from
    the next turn's system prompt (since your explicit snapshot IS the fresh state).
    Use it after clicking a combobox/dropdown to find the option refs you need to click.
    """

    name = "snapshot"
    description = (
        "Capture a fresh page snapshot now and return it as a tool result. "
        "Use this after clicking a dropdown, combobox, or dynamic element to see the "
        "updated DOM (including new option refs) before deciding your next action. "
        "The auto-injected page snapshot in the system prompt is suppressed on the "
        "following turn since your explicit snapshot is the fresh state. "
        "Do NOT call this every turn — it is only needed when the page changed mid-turn "
        "and you need to find newly appeared refs."
    )
    _verb = "took a snapshot of"

    # Set to True when run() is called this turn; read by cli.py's snapshot provider
    # to suppress the auto-injected snapshot on the NEXT turn (no duplication).
    _called: bool = False

    @property
    def parameters(self):
        return []

    def _build(self, args):  # pragma: no cover
        return ["snapshot"]

    def run(self, args: dict) -> ToolResult:
        SnapshotTool._called = True
        ok, output = self._cli.run("snapshot")
        if not ok:
            return ToolResult(False, f"snapshot failed: {self._trim(output)}")
        return ToolResult(True, f"Current page snapshot:\n{self._trim(output)}")


# The full, ordered set of discrete web tools the toolset exposes.
# ``open_browser`` lets the agent restore a dead persistent session.
# ``snapshot`` lets the agent explicitly request a fresh page view mid-turn.
_WEB_TOOL_CLASSES: tuple[type[_WebTool], ...] = (
    GotoTool, ClickTool, FillTool, TypeTool, PressKeyTool, SelectOptionTool,
    SetSpinbuttonTool, DrawSignatureTool,
    CheckTool, UncheckTool, HoverTool, DragTool, UploadTool,
    ReloadTool,
    # Excluded: OpenBrowserTool (goto opens browser automatically),
    # EvaluateTool (not needed), SnapshotTool (auto-injected into system prompt),
    # NavigateBackTool, NavigateForwardTool (not needed), ScreenshotTool (model is not visual).
)


def capture_page_snapshot(cli: PlaywrightCli, char_limit: int) -> str:
    """Capture a fresh `snapshot` of the live page from an EXISTING session and
    return its text, truncated to ``char_limit`` (issue #24).

    Reuses the supplied :class:`PlaywrightCli` — i.e. the SAME named session the
    agent's discrete browser tools (goto/click/fill/…) drive — so the captured
    snapshot reflects the actual page the model is acting on; it never launches a
    second browser. This is internal auto-injection, NOT an agent tool. On any failure
    (no session open yet, CLI error, timeout) it returns "" so the caller simply
    renders no page section that turn rather than crashing the run.

    ``cli`` is the injectable seam: tests pass a stand-in whose ``run`` returns
    canned snapshot text, so the per-turn injection can be exercised with no browser.
    """
    try:
        ok, output = _capture(cli)
    except Exception:
        return ""
    if not ok:
        return ""
    text = (output or "").strip()
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + f"\n…[+{len(text) - char_limit} chars truncated]"


def _snapshot_is_blank(text: str) -> bool:
    """True when the snapshot captured about:blank or yielded no meaningful ARIA content.

    Happens when a stale orphaned browser process makes the playwright-cli snapshot
    subprocess attach to a blank context/tab instead of the navigated page. We detect
    it by looking for the about:blank URL marker in the output, or for an empty YAML
    body (the snapshot section exists but has no element lines).
    """
    if not text:
        return True
    lower = text.lower()
    if "about:blank" in lower:
        return True
    # An empty snapshot has the page header but zero ref lines (no [ref=eN]).
    if "[ref=" not in text and "- " not in text:
        return True
    return False


def capture_page_snapshot_raw(cli: PlaywrightCli) -> str:
    """Capture the COMPLETE, UNTRUNCATED `snapshot` of the live page (issues #37, #43).

    Same session-sharing and never-raises contract as :func:`capture_page_snapshot`,
    but applies NO char cap. #37 uses it for ground-truth diagnostic logging; #43
    uses it so the per-turn injection can apply the DYNAMIC context-budget truncation
    (truncate only as much as the context window requires). Returns "" on any failure
    (no session, CLI error, timeout) so callers record nothing rather than crash.

    Blank-snapshot recovery: if the first capture returns about:blank or empty ARIA
    (stale orphaned browser attaches snapshot subprocess to wrong context), wait 1.5s
    and retry once. Warns to stderr if still blank so the anomaly is diagnosable.
    """
    import time as _time
    import sys as _sys
    for attempt in range(2):
        try:
            ok, output = _capture(cli)
        except Exception:
            return ""
        if not ok:
            return ""
        text = (output or "").strip()
        if not _snapshot_is_blank(text):
            return text
        if attempt == 0:
            _sys.stderr.write(
                "\nwarning: snapshot returned about:blank / empty — retrying in 1.5s "
                "(possible stale browser context)\n"
            )
            _time.sleep(1.5)
    _sys.stderr.write("\nwarning: snapshot still blank after retry — model will have no page context this turn\n")
    return text


def make_raw_snapshot_provider(config: Config) -> Callable[[], str]:
    """Build a per-turn provider of the RAW, untruncated page snapshot (issues #37, #43).

    Mirrors :func:`make_snapshot_provider` but returns the full snapshot with no char
    cap — for #37 diagnostic ground-truth sizing and #43's dynamic-budget truncation.
    Uses the run's existing session (same name/timeout from ``config``) so it reflects
    the page the model acts on. Seeded with the run's open flags so a snapshot that
    hits a dead session can self-heal it too (issue #102) — snapshots run every turn
    and are a major resume surface. Shares the run-scoped :class:`SessionState` (keyed
    by session name) with every web tool, so a resume the snapshot triggers heals the
    session for the discrete tools too (issue #113).
    """
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout,
                        open_flags=open_flags_for(config),
                        state=shared_session_state(config.web_session, open_flags_for(config)))
    return lambda: capture_page_snapshot_raw(cli)


def make_snapshot_provider(config: Config) -> Callable[[], str]:
    """Build a per-turn page-snapshot provider bound to the run's web session.

    Mirrors cli.py's ``render_workspace`` seam: returns a zero-arg callable that,
    each time it is called (once per turn, at prompt-build time), captures a FRESH
    snapshot from the run's existing Playwright session and returns it truncated to
    ``config.web_snapshot_char_limit``. The session name and timeout come from
    ``config`` so the snapshot CLI shares the exact session the discrete browser
    tools use.

    NOTE (#43): this fixed-cap provider is retained for backward compatibility and
    tests; the live run now uses :func:`make_raw_snapshot_provider` plus the dynamic
    budget so the snapshot is sized against the full message each turn.
    """
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout,
                        open_flags=open_flags_for(config),
                        state=shared_session_state(config.web_session, open_flags_for(config)))
    return lambda: capture_page_snapshot(cli, config.web_snapshot_char_limit)


def _capture(cli: "PlaywrightCli") -> tuple[bool, str]:
    """Run a death-tolerant page snapshot: prefer the wrapper's resume-free
    ``snapshot()`` seam (issue #102) so per-turn capture never reopens a browser;
    fall back to ``run('snapshot')`` for stand-ins that only implement ``run``."""
    snap = getattr(cli, "snapshot", None)
    if callable(snap):
        return snap()
    return cli.run("snapshot")


def open_flags_for(config: Config) -> list[str]:
    """The ``playwright-cli open`` flags implied by ``config`` (headed/browser).

    Single source of truth so the run's initial open (``setup``) and any automatic
    or agent-driven REOPEN (issues #101/#75) launch the SAME browser — headed by
    default so a human can watch, on the configured channel.
    """
    flags: list[str] = []
    if not config.web_headless:
        flags.append("--headed")
    if config.web_browser:
        flags += ["--browser", config.web_browser]
    return flags


class WebToolset(Toolset):
    name = "web"
    description = ("Browse the web with a stateful browser: navigate, click, fill forms, "
                   "select options, hover, press keys, upload files, and screenshot. "
                   "The page is shown to you automatically each turn.")

    def system_guidance(self) -> str | None:
        return (
            "Each turn the current page is shown to you automatically under "
            "'# Current page (live snapshot — provided automatically)' — you do NOT and CANNOT "
            "request it; just read it before deciding what to do next. "
            "There is no tool to fetch the page; use the discrete browser tools (goto, click, "
            "fill, type, select_option, hover, press_key, navigate_back, …) to ACT. "
            "If there is NO current page shown, or a web action reports the browser is not open / "
            "the session was closed, call `open_browser` to (re)open the session, then `goto` "
            "your target URL again before continuing — never give up because the page is missing. "
            "SNAPSHOT IS GROUND TRUTH — the page_snapshot result is the COMPLETE current state "
            "of the page. Only elements listed there exist right now. NEVER guess, invent, or "
            "increment refs (e.g. trying e308 because you saw e307): if a ref is not in the "
            "latest snapshot it does not exist. Only act on refs you can SEE in the most recent "
            "page_snapshot observation. "
            "When a tool takes a target element you MUST pass the element's ref (e.g. 'e163') "
            "exactly as it appears in the current snapshot. NEVER guess a CSS selector, class, "
            "id or tag (e.g. '.ytd-play-button'): such targets are rejected before the browser "
            "is touched, and the rejection lists the refs that are actually available. "
            "If a cookie/consent banner or modal dialog blocks what you need, clear it first: "
            "locate its Accept / Agree / Reject / Dismiss / Continue control in the snapshot and "
            "click that ref before doing anything else. "
            "FILE UPLOADS — two-step sequence: (1) click the upload trigger button or file-input "
            "ref in the snapshot to open the OS file-picker, then (2) immediately call `upload` "
            "with the ABSOLUTE path exactly as given in the task. Never use a relative path. "
            "If the upload trigger button is NOT visible in the snapshot, the upload section is "
            "not on the current page yet — scroll or click something to reveal it first; do NOT "
            "call click or upload with a ref that is not in the snapshot. "
            "Calling `upload` before the picker is open fails with a 'modal state' error. "
            "FORM VALIDATION ERRORS — if after clicking a 'Next'/'Weiter'/'Submit'/'Absenden' "
            "button the form still shows (page did not advance, or a field is marked [invalid] / "
            "shows 'Eingabe erforderlich'), do NOT click the button again. Instead locate the "
            "field(s) marked [invalid] in the snapshot, fill or select the correct value, and "
            "only then click the advance/submit button again. "
            "FAILED TOOL CALLS — never repeat the EXACT same tool call that just failed. "
            "If an action fails, change your approach: use a different ref, open a prerequisite "
            "dialog, or take a different action entirely."
        )

    def create_tools(self, config: Config) -> list[Tool]:
        # Carry the run's open flags so a tool that has to reopen a dead session
        # (auto-recovery or `open_browser`, issues #101/#75) restores the SAME
        # headed/channel browser the run started with — not a default one. Bind the
        # run-scoped shared SessionState (keyed by the run's unique session name) so
        # recovery bookkeeping is shared across every web tool and both snapshot
        # providers, not private per wrapper (issue #113).
        cli = PlaywrightCli(config.web_session, config.web_cli_timeout,
                            open_flags=open_flags_for(config),
                            state=shared_session_state(config.web_session, open_flags_for(config)))
        limit = config.web_observation_char_limit
        return [cls(cli, limit) for cls in _WEB_TOOL_CLASSES]

    def check_prerequisites(self) -> list[str]:
        if shutil.which(BINARY) is None:
            return [f"'{BINARY}' not found on PATH. Install it with: "
                    f"npm install -g @playwright/cli@latest"]
        return []

    def __init__(self) -> None:
        # One CLI wrapper for the whole run so the ``open`` child handle survives to
        # teardown and its tree can be reaped (issue #15). Created lazily in setup().
        self._cli: PlaywrightCli | None = None
        self._atexit_hook: Callable[[], None] | None = None

    def setup(self, config: Config) -> None:
        # Open the browser once for the run. Headed by default so a human can watch.
        # The same flags are stored on the CLI so any reopen restores this browser.
        # Bind the run-scoped shared SessionState (issue #113) so the lifecycle CLI and
        # every web tool/snapshot provider share one recovery state for this session.
        self._cli = PlaywrightCli(config.web_session, config.web_cli_timeout,
                                  open_flags=open_flags_for(config),
                                  state=shared_session_state(config.web_session, open_flags_for(config)))
        # Defensive last-resort reaper: if the run is hard-killed (Ctrl-C escaping the
        # cli finally, os._exit, an unhandled signal) the toolset's teardown may never
        # run. Registering close() with atexit ensures the browser tree is still reaped
        # on interpreter shutdown. teardown() unregisters it once it has run cleanly so
        # we never double-reap on a normal exit (close() is idempotent regardless).
        import atexit
        self._atexit_hook = self._cli.close
        atexit.register(self._atexit_hook)
        self._cli.open()

    def teardown(self, config: Config) -> None:
        """Close the session AND reap its whole browser process tree. Must never raise
        (the cli run path swallows teardown errors, but we are defensive here too)."""
        try:
            if self._atexit_hook is not None:
                import atexit
                atexit.unregister(self._atexit_hook)
                self._atexit_hook = None
            cli = self._cli
            if cli is None:
                # teardown without a prior setup (e.g. setup failed early): still make
                # a best-effort close of any session left over by name.
                cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
            cli.close()
        except Exception:
            pass
        finally:
            self._cli = None
            # Forget this run's shared SessionState so the per-run registry (keyed by
            # the run's unique session name) does not accumulate across runs (#113).
            try:
                drop_session_state(config.web_session)
            except Exception:
                pass
