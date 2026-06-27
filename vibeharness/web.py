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

import json
import logging
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Callable

_log = logging.getLogger(__name__)

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

    ``filled`` MUST be the live per-snapshot control-value map from
    :func:`live_control_values` (issue #205) — i.e. derived from the page's ACTUAL current
    DOM values this turn, never a cache of intended action args. A control that holds no
    committed value is simply absent from ``filled`` and gets no marker.
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


# Input controls whose committed value is the inline ARIA scalar (text after the colon).
_VALUE_INPUT_ROLES = frozenset({
    "textbox", "searchbox", "combobox", "spinbutton", "slider",
})
# Option-container roles whose value is its [selected] option children (native/multi select).
_OPTION_CONTAINER_ROLES = frozenset({"listbox", "select", "menu", "combobox"})
# Toggle controls whose "filled" state is a checked/selected flag, not a scalar value.
_CHECKED_ROLES = frozenset({
    "checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio",
})


def _prop_is_set(props: list[str], key: str) -> bool:
    """True when ARIA state ``key`` is present and not explicitly false (``checked`` /
    ``checked=true`` -> True; ``checked=false`` / ``checked=mixed`` -> False)."""
    for p in props:
        name, _, val = p.partition("=")
        if name.strip().lower() == key:
            return val.strip().lower() not in ("false", "mixed")
    return False


def _selected_option_text(node) -> str | None:
    """Joined accessible names of a container's ``[selected]`` option children, or None.

    Handles native single-select (one selected option) and multi-select (several). Used
    when a listbox/combobox exposes its value via selected option nodes rather than an
    inline scalar.
    """
    from .snapshot_prose import _iter_nodes  # local import: avoids any import cycle
    names = []
    for child in _iter_nodes(node):
        if child.role.lower() == "option" and _prop_is_set(child.props, "selected"):
            txt = (child.name or "").strip()
            if txt:
                names.append(txt)
    return ", ".join(names) if names else None


def live_control_values(raw_aria_snapshot: str) -> dict[str, str]:
    """Map ``{ref: committed_value}`` for controls that ACTUALLY hold a value right now.

    Issue #205: the single source of truth for filled-state is the live page, read from the
    raw Playwright ARIA snapshot captured every turn — NEVER a cache of intended action
    args. A control is included ONLY when its current DOM value/checked-state is non-empty:
      * text inputs / spinbuttons / sliders / native single-selects: the inline ARIA scalar
        (``- textbox "Name" [ref=e1]: Jason``); an empty field has no scalar -> excluded.
      * listbox / multi-select / native select: the joined ``[selected]`` option names.
      * checkbox / radio / switch: ``"(checked)"`` when ``[checked]`` is set; unchecked -> excluded.
    A popup that was only OPENED (no commit), a cleared field, and a ref absent from the
    snapshot all yield NO entry, so :func:`annotate_filled_snapshot` never falsely marks them.
    Parsing reuses the deterministic ARIA tree builder in :mod:`vibeharness.snapshot_prose`.
    """
    if not raw_aria_snapshot or not raw_aria_snapshot.strip():
        return {}
    try:
        from .snapshot_prose import _build_tree, _iter_nodes, _split_header_and_yaml
        _, _, body = _split_header_and_yaml(raw_aria_snapshot)
        root = _build_tree(body)
    except Exception:
        return {}
    filled: dict[str, str] = {}
    for node in _iter_nodes(root):
        if not node.ref:
            continue
        role = node.role.lower()
        value = (node.value or "").strip()
        if role in _CHECKED_ROLES:
            if _prop_is_set(node.props, "checked") or _prop_is_set(node.props, "selected"):
                filled[node.ref] = "(checked)"
            continue
        if role in _VALUE_INPUT_ROLES and value:
            filled[node.ref] = value
            continue
        if role in _OPTION_CONTAINER_ROLES:
            sel = _selected_option_text(node)
            if sel:
                filled[node.ref] = sel
    return filled


def parse_snapshot_refs(snapshot: str) -> set[str]:
    """Extract the set of element refs (``e163`` …) present in a live page snapshot.

    The snapshot is the same auto-injected page view the model sees, where each
    actionable node is prefixed with a ``[eN]`` token (e.g. ``[e163] button
    "Accept …"``). We return the bare, normalized ref strings (``{"e163", …}``)
    for membership checks. Lines that carry no ref (``[-] tooltip``, ``text: …``)
    simply contribute nothing.
    """
    return {f"e{m.group(1)}" for m in _REF_RE.finditer(snapshot or "")}


# ---------------------------------------------------------------------------
# Pre-compaction VISIBILITY FILTER (issue #223) — drop aria-hidden + zero-size
# (honeypot/trap) elements from the RAW ARIA snapshot BEFORE prose compaction and
# before live_control_values/annotate, so the model only ever sees controls a human
# could see. Split into a PURE text transform (filter_hidden_snapshot_refs) that is
# unit-testable without a browser, and a single-DOM-pass detector (compute_hidden_refs).
#
# DETECTION RULE (an element is HIDDEN if ANY holds), justified:
#   * aria-hidden semantics — el.closest('[aria-hidden="true"]') is non-null. The classic
#     visually-hidden honeypot (FlashTec ApplyWizard) wraps the trap inputs in an
#     aria-hidden="true" subtree; a human never sees them but the a11y tree (hence the
#     Playwright aria snapshot) still emits them. This single signal catches the whole
#     trap subtree (wrapper + every descendant) deterministically.
#   * computed visibility — getComputedStyle display:none | visibility:hidden/collapse |
#     opacity≈0 (<=0.01). The standard "truly not painted" signals.
#   * zero-size — getBoundingClientRect() width<=EPS OR height<=EPS (EPS=2px), i.e. the
#     1px clipped honeypot (width:1px;height:1px;clip:rect(0 0 0 0)).
# Position-alone (off-screen left:-9999px) and tabIndex<0-alone are deliberately NOT
# treated as hidden: scroll-reachable / virtualized real content sits off-screen with a
# real size and aria-hidden=false and MUST be kept. We never test position, so such
# content is preserved; the honeypot is still caught because it is ALSO aria-hidden and
# 1px-sized. (Ground-truthed live: the e75 trap wrapper is aria-hidden + 1x1 + at
# -10000,-10000 while e41/e69/e73 real fields are aria-hidden=false at full size.)
#
# REF→ELEMENT RESOLUTION (load-bearing, ground-truthed): playwright-core registers an
# "aria-ref" selector engine whose queryAll resolves a ref token (e.g. "e77") through the
# injected script's _lastAriaSnapshotForQuery.elements map — the ref→element map built by
# the LAST `snapshot`. That map lives in the page's injected world and PERSISTS across
# separate playwright-cli subprocess invocations within one session (exactly how the
# EvaluateTool's `eval <expr> <ref>` resolves a ref captured by an earlier snapshot). So
# we reuse the single per-turn raw capture: ONE `run-code` pass resolves every ref via
# page.locator('aria-ref=' + ref) and computes the hidden set in a single page.evaluate —
# no second snapshot. (Verified: @playwright/cli node_modules/playwright-core/lib/
# coreBundle.js `_createAriaRefEngine` / `aria-ref=${param.target}`.)
# ---------------------------------------------------------------------------

# A raw-snapshot line carries its element ref as ``[ref=eN]``.
_REF_ATTR_LINE_RE = re.compile(r"\[ref=(e\d+)\]")

# The single-pass DOM probe. ``__REFS_JSON__`` is replaced with a JSON array of the refs
# present in the current raw snapshot; it resolves each via the aria-ref selector engine
# and returns the subset that is hidden by the detection rule above. Robust per-element:
# a ref that no longer resolves (stale/detached) is simply skipped, never marked hidden.
# Kept on ONE LINE on purpose: playwright-cli is invoked as a Windows ``.CMD`` shim, whose
# batch arg-parsing mangles a multi-line argv (newlines -> a JS SyntaxError) — verified.
_VISIBILITY_PROBE_JS = (
    "async page => { "
    "const refs = __REFS_JSON__; const handles = []; const valid = []; "
    "for (const ref of refs) { try { "
    "const h = await page.locator('aria-ref=' + ref).elementHandle({ timeout: 250 }); "
    "if (h) { handles.push(h); valid.push(ref); } } catch (e) {} } "
    "const flags = await page.evaluate((els) => { const EPS = 2; return els.map(el => { try { "
    "if (el.closest('[aria-hidden=\"true\"]')) return true; "
    "const s = getComputedStyle(el); "
    "if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') return true; "
    "if (parseFloat(s.opacity) <= 0.01) return true; "
    "const r = el.getBoundingClientRect(); if (r.width <= EPS || r.height <= EPS) return true; "
    "return false; } catch (e) { return false; } }); }, handles); "
    "return valid.filter((ref, i) => flags[i]); }"
)


def _raw_snapshot_refs(raw: str) -> list[str]:
    """Ordered, de-duplicated list of element refs (``[ref=eN]``) in a raw snapshot."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _REF_ATTR_LINE_RE.finditer(raw or ""):
        ref = m.group(1)
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _parse_run_code_result(output: str) -> list:
    """Extract the JSON value playwright-cli emits under its ``### Result`` heading.

    ``run-code`` reports the function's return value as ``### Result\\n<json>\\n###  …``.
    Returns the parsed JSON (a list of hidden refs here), or ``[]`` if absent/unparseable.
    """
    if not output:
        return []
    lines = output.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "### Result")
    except StopIteration:
        return []
    body: list[str] = []
    for ln in lines[start + 1:]:
        if ln.startswith("### "):
            break
        body.append(ln)
    blob = "\n".join(body).strip()
    if not blob:
        return []
    try:
        value = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def compute_hidden_refs(cli: "PlaywrightCli", raw: str) -> set[str]:
    """Refs in ``raw`` whose live DOM element is HIDDEN (aria-hidden / zero-size /
    display:none / visibility:hidden / opacity≈0) — issue #223.

    ONE evaluate pass per turn: extracts the refs already present in the per-turn raw
    capture and resolves them in a single ``run-code`` invocation via the aria-ref
    selector engine (see the module note above). Never raises — any failure (no session,
    CLI error, malformed result) yields an empty set so the caller keeps the unfiltered
    snapshot. ``cli`` is an injectable seam: tests pass a stand-in whose ``run`` returns a
    canned ``### Result`` payload.
    """
    refs = _raw_snapshot_refs(raw)
    if not refs:
        return set()
    js = _VISIBILITY_PROBE_JS.replace("__REFS_JSON__", json.dumps(refs))
    try:
        ok, output = cli.run("run-code", js)
    except Exception:
        return set()
    if not ok:
        return set()
    present = set(refs)
    return {r for r in _parse_run_code_result(output)
            if isinstance(r, str) and r in present}


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


class _RawLineNode:
    """One raw-snapshot line plus its indentation-nested children (for the pure filter)."""
    __slots__ = ("idx", "indent", "ref", "children")

    def __init__(self, idx: int, indent: int, ref: str | None):
        self.idx = idx
        self.indent = indent
        self.ref = ref
        self.children: list[_RawLineNode] = []


def _subtree_has_ref(node: _RawLineNode) -> bool:
    """True when any DESCENDANT (not the node itself) still carries a ref."""
    for child in node.children:
        if child.ref is not None or _subtree_has_ref(child):
            return True
    return False


def filter_hidden_snapshot_refs(raw: str, hidden_refs: "set[str]") -> str:
    """PURE: drop the lines for ``hidden_refs`` (and their subtrees) from a raw ARIA
    snapshot, then prune any wrapper left with NO visible (ref-bearing) descendant —
    while keeping visible siblings. Issue #223.

    Byte-identical passthrough when ``hidden_refs`` is empty/false. Pruning is SCOPED to
    ancestors of a removed element: a wrapper is dropped only when it lost a descendant to
    filtering AND nothing ref-bearing remains beneath it (so an orphaned label like
    ``text: Company website`` left behind by the removed ``textbox [ref=e77]`` goes too),
    never an untouched text/disclaimer block. Total: any parse surprise degrades to
    returning ``raw`` unchanged (mirrors aria_yaml_to_prose's guard) — never blanks.
    """
    if not raw or not hidden_refs:
        return raw
    try:
        hidden = set(hidden_refs)
        lines = raw.split("\n")
        # Build the indentation forest (parent has strictly smaller indent than child).
        roots: list[_RawLineNode] = []
        stack: list[_RawLineNode] = []
        for i, line in enumerate(lines):
            m = _REF_ATTR_LINE_RE.search(line)
            node = _RawLineNode(i, _indent_of(line), m.group(1) if m else None)
            while stack and stack[-1].indent >= node.indent:
                stack.pop()
            (stack[-1].children if stack else roots).append(node)
            stack.append(node)

        def _keep(node: _RawLineNode) -> "tuple[_RawLineNode | None, bool]":
            # Returns (kept_node_or_None, removed_something_beneath_or_self).
            if node.ref is not None and node.ref in hidden:
                return None, True          # drop this ref's whole subtree
            removed_below = False
            kept: list[_RawLineNode] = []
            for child in node.children:
                kc, rb = _keep(child)
                if kc is None:
                    removed_below = True
                else:
                    kept.append(kc)
                    removed_below = removed_below or rb
            node.children = kept
            if removed_below and not _subtree_has_ref(node):
                # This wrapper lost its actionable content — prune it (and any orphan
                # label scaffolding it now holds). Untouched blocks (removed_below False)
                # are always kept, so pure text/disclaimer regions survive.
                return None, True
            return node, removed_below

        kept_roots: list[_RawLineNode] = []
        for root in roots:
            kr, _ = _keep(root)
            if kr is not None:
                kept_roots.append(kr)

        kept_idx: list[int] = []

        def _emit(node: _RawLineNode) -> None:
            kept_idx.append(node.idx)
            for child in node.children:
                _emit(child)

        for root in kept_roots:
            _emit(root)
        kept_idx.sort()
        return "\n".join(lines[i] for i in kept_idx)
    except Exception:
        return raw


def filter_hidden_snapshot(raw: str, cli: "PlaywrightCli") -> str:
    """Drop hidden (aria-hidden / zero-size) elements from a raw ARIA snapshot — the
    wired entry point for the #223 pre-compaction stage. Computes the hidden set in ONE
    DOM pass (:func:`compute_hidden_refs`) and applies the pure text filter
    (:func:`filter_hidden_snapshot_refs`). Robust: on ANY error, or when nothing is
    hidden, returns the UNFILTERED ``raw`` (never blanks the snapshot). Logs the
    dropped-ref count into the run log for #37 diagnostic observability.
    """
    if not raw or not raw.strip():
        return raw
    try:
        hidden = compute_hidden_refs(cli, raw)
        if not hidden:
            return raw
        filtered = filter_hidden_snapshot_refs(raw, hidden)
        _log.info("visibility filter (#223) dropped %d hidden ref(s): %s",
                  len(hidden), ",".join(sorted(hidden)))
        return filtered
    except Exception:
        return raw


def make_visibility_filter(config: Config) -> Callable[[str], str]:
    """Build the per-turn ``raw -> filtered_raw`` visibility filter bound to the run's web
    session (issue #223). Reuses the run's shared :class:`SessionState` (same session name)
    so the ``run-code`` aria-ref resolution targets the SAME live page / snapshot map the
    per-turn raw capture produced — no second browser, no second snapshot."""
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout,
                        open_flags=open_flags_for(config),
                        state=shared_session_state(config.web_session, open_flags_for(config)))
    return lambda raw: filter_hidden_snapshot(raw, cli)


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

# --- New-tab auto-redirect ---------------------------------------------------
# playwright-cli reports every open tab after interactive actions:
#   ### Open tabs
#   - 0: (current) [Title](original_url)
#   - 1: [Title](new_url)            <- new tab opened by target="_blank" click
# We silently navigate to the newest non-current tab and strip the block so
# the agent never sees tab metadata.
_OPEN_TABS_BLOCK_RE = re.compile(
    r"###\s+Open\s+tabs\b.*?(?=###|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NON_CURRENT_TAB_RE = re.compile(
    r"-\s+\d+:\s+(?!\(current\))\[([^\]]*)\]\(([^)]+)\)",
)


def _auto_redirect_new_tabs(cli: "PlaywrightCli", output: str) -> str:
    """Navigate to any new tab that opened and strip the Open-tabs section.

    When a click opens a page in a new tab (target="_blank"), playwright-cli
    includes an "### Open tabs" block listing all open tabs. This function
    detects non-current tabs, navigates to the newest one silently via goto,
    and returns the output with the block removed so the agent sees only the
    resulting page — not tab metadata.
    """
    new_tabs = _NON_CURRENT_TAB_RE.findall(output)
    if new_tabs:
        _title, url = new_tabs[-1]   # last = most recently opened tab
        if url and not url.lower().startswith("about:"):
            try:
                cli.run("goto", url)
            except Exception:
                pass
    return _OPEN_TABS_BLOCK_RE.sub("", output).strip()

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
    """DOM-delta detection disabled.

    Mid-turn partial snapshots were causing ref conflicts: the PAGE CHANGED summary
    listed refs from a snapshot taken 2s post-click, while the end-of-turn
    'Latest page state' captured refs from a LATER snapshot — the same button could
    appear as e144 in one and e65 in the other, confusing the model into hallucinating
    refs. The end-of-turn 'Latest page state' is now the SOLE snapshot the agent
    receives. Returning result unchanged here; call sites are preserved for easy
    re-enabling.
    """
    return result


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
        # Silently redirect to any new tab that opened and strip the Open-tabs
        # section so the agent sees only the destination page, not tab metadata.
        output = _auto_redirect_new_tabs(self._cli, output)
        return ToolResult(True, f"you {self._verb} {subject}. Result:\n{self._trim(_strip_snapshot_filelink(output))}")

    def run(self, args: dict) -> ToolResult:
        """Public entry point. Runs the action and returns the result.

        DOM-change detection removed: the end-of-turn 'Latest page state' is the
        sole snapshot the model receives (see _check_dom_delta).
        """
        return self._run_impl(args)

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


class ClickTool(_WebTool):
    name = "click"
    description = "Click an element (a link, button, checkbox, …). " + _REF_NOTE
    _verb = "clicked"
    _required = ("target",)
    _validate_target = True
    _detect_dom_change = True

    # Hard safety ceiling on `repeat`: a single call clicking more than this many times
    # almost certainly reflects a misparsed count and would wedge the whole turn (each
    # click carries a 2s settle). Covers real multi-click widgets (a stepper, a small
    # date-picker nudge) while tightly bounding a runaway.
    _MAX_REPEAT = 9
    # ISSUE #222 — multi-target ceilings (only reachable on the flag-ON `targets` path):
    #   * _MAX_TARGETS  — cap on how many distinct refs one `targets` list may hold, so a
    #     misparsed list can't enumerate the whole page.
    #   * _MAX_TOTAL_CLICKS — cap on the SUM of all per-item repeats, so even a small list of
    #     big counts can't wedge the turn (every physical click carries the 2s settle).
    _MAX_TARGETS = 20
    _MAX_TOTAL_CLICKS = 200

    def __init__(self, cli, observation_limit):
        super().__init__(cli, observation_limit)
        # ISSUE #222 — per-MODEL multi-target flag. False by default → the advertised schema
        # AND run() behaviour are byte-identical to the single-target tool. cli._run_locked
        # sets this from the ACTIVE base model (resolve_model_click_multi_target) and the
        # agent re-resolves it on escalation take-over, so the `targets` array is advertised
        # ONLY to the capable API models (GLM/DeepSeek) and tracks the active model.
        self._multi_target: bool = False

    # ISSUE #222 — the nested object-array `items` sub-schema for `targets`: each item is one
    # {target, repeat?} click. This is the standard JSON-Schema / OpenAI / Anthropic shape for
    # a list-of-objects (type:array + items:{type:object, properties, required}); it flows
    # unchanged into the json-codec decode constraint, the hermes <tools> block, and the
    # native Ollama tools: envelope (all built from Param.schema via _args_schema).
    _TARGETS_ITEM_SCHEMA = {
        "type": "object",
        "properties": {
            "target": {"type": "string",
                       "description": "snapshot ref to click, e.g. 'e163' (never a guessed "
                                      "CSS selector/class/id)."},
            "repeat": {"type": "integer", "minimum": 1, "default": 1,
                       "description": "times to click THIS ref in succession (default 1)."},
        },
        "required": ["target"],
    }
=======
    # click carries a 2s settle). Covers real multi-click widgets (a stepper, a small
    # date-picker nudge) while tightly bounding a runaway.
    _MAX_REPEAT = 9
>>>>>>> 2ea9959 (chore(click): cap repeat at 9)

    @property
    def parameters(self):
        # Single-target params: identical to today. When the per-model multi-target flag is
        # OFF, ONLY these are advertised, so the schema is byte-identical to the pre-#222 tool.
        target = Param(
            "target", "string",
            "Element ref (e.g. 'e163') from the current page snapshot to click. Must be a ref "
            "from the snapshot — never a guessed CSS selector/class/id.",
            required=not self._multi_target)
        repeat = Param(
            "repeat", "integer",
            "How many times to click this SAME target in succession, in one call, with the "
            "standard ~2s settle after each click. Default 1. Use a count > 1 to drive a "
            "multi-click widget in a single call — e.g. a date-picker 'Previous month'/'Next "
            "month' button or a numeric stepper — instead of issuing many separate click "
            "calls. Stops early if a click fails (e.g. the target disappears).",
            required=False, default=1)
        if not self._multi_target:
            return [target, repeat]
        # ISSUE #222 — flag ON: also advertise a `targets` ARRAY so the model can click MANY
        # refs in ONE call, each its own count, IN ORDER. The worked example lives in the
        # description (best practice for tool schemas — JSON-Schema docs / OpenAI function
        # calling): models that read only the Markdown tool docs still see the exact shape.
        targets = Param(
            "targets", "array",
            "OPTIONAL list of clicks to perform IN ORDER, each item one snapshot ref clicked N "
            "times — use this to click SEVERAL different controls (and/or the same control "
            "several times) in ONE call, e.g. several fields of a form. Each item is "
            "{\"target\": <ref>, \"repeat\": <int, default 1>}. Example: "
            "targets=[{\"target\":\"e10\",\"repeat\":8},{\"target\":\"e22\",\"repeat\":2}] "
            "clicks e10 eight times then e22 twice (the standard ~2s settle after EVERY "
            "individual click). `target` must be a snapshot ref, never a guessed selector. "
            "When `targets` is given it takes precedence over the single `target`/`repeat` "
            "above. Stops early and reports progress if any click fails.",
            required=False, items=self._TARGETS_ITEM_SCHEMA)
        return [target, repeat, targets]

    def _build(self, args):
        return ["click", args["target"]]

    # Post-click settle: overlay panels and async renders (e.g. fill.co.at "Bewirb dich"
    # form) take ~1-2 s to appear. Sleeping here ensures the end-of-turn 'Latest page
    # state' snapshot (taken ~1 s after all actions) sees a fully-rendered page. When
    # repeat > 1 this settle is applied AFTER EACH click (commit 6a131bf; issue #206).
    _POST_CLICK_SETTLE_S: float = 2.0

    def _parse_repeat(self, raw) -> int | ToolResult:
        """Coerce the optional ``repeat`` arg to a positive int (default 1).

        0/negative collapse to a single click; a non-integer is a clear user error.
        Clamped to :attr:`_MAX_REPEAT` so a misparsed count can't wedge the turn.
        """
        if raw is None or raw == "":
            return 1
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return ToolResult(False, f"you called `{self.name}` with a non-integer repeat "
                              f"'{raw}'. Pass a positive integer, e.g. repeat=3.")
        if n < 1:
            return 1
        return min(n, self._MAX_REPEAT)

    def _parse_targets(self, raw) -> "list[tuple[str, int]] | ToolResult":
        """Validate the multi-target ``targets`` arg into an ordered ``[(ref, repeat), …]``
        plan, or a clear user ToolResult on any structural error (issue #222).

        Each item must be an object with a non-empty string ``target`` and an optional
        positive-int ``repeat`` (default 1, clamped to :attr:`_MAX_REPEAT`). The list itself
        must be non-empty and at most :attr:`_MAX_TARGETS` long, and the TOTAL of all repeats
        at most :attr:`_MAX_TOTAL_CLICKS` (every physical click carries the 2s settle). Refs
        are NOT checked against the live snapshot here — that happens per click in
        ``_run_impl`` (so a stale ref fails at its own item with the standard guidance)."""
        if not isinstance(raw, list):
            return ToolResult(False, f"you called `{self.name}` with `targets` that is not a "
                              f"list. Pass a list of objects, e.g. "
                              f'targets=[{{"target":"e10","repeat":8}},{{"target":"e22"}}].')
        if not raw:
            return ToolResult(False, f"you called `{self.name}` with an EMPTY `targets` list. "
                              f"Provide at least one {{\"target\": <ref>}} item, or use the "
                              f"single `target` arg.")
        if len(raw) > self._MAX_TARGETS:
            return ToolResult(False, f"you called `{self.name}` with {len(raw)} targets, more "
                              f"than the {self._MAX_TARGETS} allowed in one call. Split it "
                              f"across several click calls.")
        plan: list[tuple[str, int]] = []
        total = 0
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                return ToolResult(False, f"targets[{i}] is not an object — each item must be "
                                  f'{{"target": <ref>, "repeat": <optional int>}}.')
            tgt = item.get("target")
            if not tgt or not isinstance(tgt, str):
                return ToolResult(False, f"targets[{i}] is missing a string `target` ref. Each "
                                  f'item must name a snapshot ref, e.g. {{"target":"e10"}}.')
            rep = self._parse_repeat(item.get("repeat"))
            if isinstance(rep, ToolResult):
                # Re-phrase the per-item non-integer repeat error with its index.
                return ToolResult(False, f"targets[{i}] ('{tgt}'): {rep.observation}")
            plan.append((tgt, rep))
            total += rep
        if total > self._MAX_TOTAL_CLICKS:
            return ToolResult(False, f"you called `{self.name}` with {total} total clicks across "
                              f"the list, more than the {self._MAX_TOTAL_CLICKS} allowed in one "
                              f"call. Reduce the per-item repeats or split the call.")
        return plan

    def _run_multi(self, args: dict) -> ToolResult:
        """Click a LIST of refs IN ORDER, each its own count (issue #222, flag ON).

        Runs each physical click through the SAME ``_run_impl`` as the single-target path (so
        per-ref validation, no-match handling and auto-recovery are identical), with the
        ``_POST_CLICK_SETTLE_S`` settle after EVERY individual click — between repeats of one
        ref AND between different refs. Stops at the first failing click and reports how far it
        got (which item / iteration). On success the summary carries the END-DOM observation
        of the last click so the model sees the resulting page state."""
        plan = self._parse_targets(args.get("targets"))
        if isinstance(plan, ToolResult):
            return plan
        last: ToolResult | None = None
        done = 0  # physical clicks that landed
        for i, (tgt, rep) in enumerate(plan):
            for k in range(rep):
                last = self._run_impl({"target": tgt})
                if not last.ok:
                    if done:
                        return ToolResult(False, f"clicked {done} time(s) across "
                                          f"{i} target(s), then targets[{i}] '{tgt}' click "
                                          f"{k + 1} failed: {last.observation}")
                    return ToolResult(False, f"targets[{i}] '{tgt}' failed on the first "
                                      f"click: {last.observation}")
                done += 1
                # PRESERVE THE DELAY (#206/#222): settle after EVERY physical click, between
                # repeats of one ref AND between different refs — one flat settle per click.
                time.sleep(self._POST_CLICK_SETTLE_S)
        summary = ", ".join(f"{t}×{r}" for t, r in plan)
        return ToolResult(True, f"clicked {done} time(s) across {len(plan)} target(s) "
                          f"({summary}). Last result:\n{last.observation if last else ''}")

    def run(self, args: dict) -> ToolResult:
        # ISSUE #222 — multi-target path: when the per-MODEL flag is ON and a `targets` list
        # is supplied, click each ref its own count IN ORDER (targets WINS over the single
        # `target`/`repeat`). Flag OFF (or no `targets`) falls through to the byte-identical
        # single-target path below. Re-checking `_multi_target` here means a model that emits
        # `targets` while the flag is off is simply ignored (single-target behaviour kept).
        if self._multi_target and args.get("targets") is not None:
            return self._run_multi(args)
        # repeat (default 1): click the SAME target N times in succession in ONE tool call,
        # with the 2s settle AFTER EACH click (#206). Lets a capable model drive multi-click
        # widgets (date-picker "Previous month", steppers) in one call, and sidesteps the
        # consecutive-duplicate dedup (#201/#204) which would collapse N identical calls.
        repeat = self._parse_repeat(args.get("repeat"))
        if isinstance(repeat, ToolResult):
            return repeat
        last: ToolResult | None = None
        done = 0
        for _ in range(repeat):
            last = self._run_impl(args)
            if not last.ok:
                if done:
                    # Some clicks landed before one failed mid-sequence — report both.
                    return ToolResult(False, f"clicked '{args.get('target')}' {done} time(s), then "
                                      f"click {done + 1} failed: {last.observation}")
                return last
            done += 1
            time.sleep(self._POST_CLICK_SETTLE_S)
        if repeat > 1 and last is not None:
            return ToolResult(True, f"clicked '{args.get('target')}' {done} times. Last "
                              f"result:\n{last.observation}")
        return last


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
        # Validate required params BEFORE touching the browser: a missing-param call
        # is rejected without ever snapshotting (don't capture a DOM state for an
        # invalid call — issue #166). The base _run_impl re-checks, so valid calls
        # behave identically; only the wasteful pre-snapshot on an invalid call is
        # avoided. DOM-delta detection stays intact for valid fills below.
        missing = [p for p in self._required if not args.get(p)]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")
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
        "Upload a file to a file-upload field in one step. "
        "Provide the `target` ref of the upload trigger (button or file-input shown in the snapshot) "
        "and the absolute `file` path. The tool clicks the trigger and uploads the file automatically "
        "— do NOT click the trigger separately first. "
        "Always pass the path EXACTLY as given in the task (absolute, e.g. "
        "C:\\\\path\\\\to\\\\file.docx) — never shorten it to a relative path. "
        "If the form hides its file upload fields from the snapshot (common on some sites), "
        "use target='js:0' for the first hidden file input or 'js:1' for the second."
    )
    _verb = "uploaded a file via"
    _required = ("target", "file")
    _validate_target = True

    def _subject(self, args):
        return args.get("target", "the file input")

    @property
    def parameters(self):
        return [
            Param("target", "string",
                  "Ref of the upload trigger element (button or file-input) from the current page "
                  "snapshot, e.g. 'e163'. Or use 'js:0' / 'js:1' to target hidden file inputs "
                  "by index when no upload trigger appears in the snapshot."),
            Param("file", "string",
                  "Absolute path of the file to upload, e.g. C:\\\\Users\\\\...\\\\file.docx. "
                  "Never use a relative path."),
        ]

    def _build(self, args):  # pragma: no cover — run() overrides the dispatch
        return ["upload", args["file"]]

    def _upload_via_js(self, idx: int, file_path: str, before: str) -> "ToolResult":
        """Click the Nth hidden <input type=file> via JS eval, then supply the file."""
        eval_js = f"document.querySelectorAll('input[type=\"file\"]')[{idx}]?.click()"
        self._cli.run("eval", eval_js)
        time.sleep(0.5)
        upload_ok, upload_out = self._cli.run("upload", file_path)
        upload_msg = self._trim(_strip_snapshot_filelink(upload_out)).replace(
            "browser_file_upload", "upload")
        if not upload_ok:
            if "modal state" in upload_msg.lower():
                return ToolResult(
                    False,
                    f"upload(js:{idx}) failed: no file picker opened after clicking hidden file "
                    f"input [{idx}]. The form may not have a file input at index {idx}, or it "
                    f"may be disabled. Try a different index (e.g. 'js:{idx+1}').",
                )
            return ToolResult(False, upload_msg)
        result = ToolResult(
            True,
            f"you uploaded '{file_path}' via hidden file input [js:{idx}]. Result:\n{upload_msg}",
        )
        return _check_dom_delta(self._cli, before, result)

    def run(self, args: dict) -> ToolResult:
        # Validate required params.
        missing = [p for p in self._required if not args.get(p)]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")

        target = args["target"]
        file_path = args["file"]

        # JS-index mode: target="js:0" / "js:1" clicks hidden file inputs by DOM index.
        if target.startswith("js:"):
            try:
                js_idx = int(target[3:])
            except ValueError:
                return ToolResult(False,
                                  f"invalid js-index target '{target}': use 'js:N' where N is "
                                  f"0, 1, 2… (e.g. 'js:0' for CV, 'js:1' for cover letter).")
            before = capture_page_snapshot_raw(self._cli) or ""
            return self._upload_via_js(js_idx, file_path, before)

        # Standard ARIA-ref path: guard, click, upload.
        guard = self._guard_target(args)
        if guard is not None:
            return guard

        # Capture DOM before the sequence for delta detection afterward.
        before = capture_page_snapshot_raw(self._cli) or ""

        # Step 1: click the upload trigger to open the OS file picker.
        click_ok, click_out = self._cli.run("click", target)
        if not click_ok or output_signals_no_match(click_out):
            return ToolResult(
                False,
                f"upload failed: could not click trigger '{target}': "
                f"{self._trim(_strip_snapshot_filelink(click_out))}",
            )

        # Step 2: brief pause so the file-picker dialog has time to open.
        time.sleep(0.5)

        # Step 3: supply the file to the open picker.
        upload_ok, upload_out = self._cli.run("upload", file_path)
        upload_msg = self._trim(_strip_snapshot_filelink(upload_out)).replace(
            "browser_file_upload", "upload")
        if not upload_ok:
            if "modal state" in upload_msg.lower():
                upload_msg = (
                    f"upload failed after clicking '{target}': the file picker did not open. "
                    f"Ensure the target is the correct upload trigger button from the snapshot. "
                    f"If no upload trigger is visible in the snapshot, use target='js:0' for the "
                    f"first file input or 'js:1' for the second."
                )
            return ToolResult(False, upload_msg)

        result = ToolResult(
            True,
            f"you uploaded '{file_path}' via {target}. Result:\n{upload_msg}",
        )
        return _check_dom_delta(self._cli, before, result)


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
    """Run JavaScript on the page (or on one element) to inspect OR set its state.

    Use this to understand what kind of element a ref points to before deciding which
    interaction tool to use (inspect whether a combobox is a native <select> or a custom
    widget, read its current value, list its <option> children, check a data attribute),
    AND — for complex widgets where clicking is unreliable (date pickers/calendars, custom
    dropdowns, sliders) — to SET a value directly on the underlying input instead of many
    blind clicks. When setting a value, dispatch the events the page listens for
    (``input``/``change``) so frameworks pick it up. Returns the JS return value as text.

    ISSUE #203: this tool is exposed ONLY to capable API models (GLM/DeepSeek); the small
    local model never sees it (its per-model toolset omits it), preserving #67's guarantee
    that the 3B agent cannot execute arbitrary JavaScript.
    """

    name = "evaluate"
    description = (
        "Run a JavaScript snippet on the page (or on a specific element) to INSPECT or SET "
        "its state. Inspect: identify element types, list dropdown options, read current "
        "values/attributes — e.g. expression='el => el.tagName + \" \" + el.type' with "
        "target='e6'. SET (for complex widgets like date pickers where clicking loops): "
        "target the underlying input and assign + dispatch events, e.g. "
        "expression=\"el => { el.value = '2026-06-27'; "
        "el.dispatchEvent(new Event('input', {bubbles:true})); "
        "el.dispatchEvent(new Event('change', {bubbles:true})); }\" with target='e6'. "
        "Without a target, the expression runs on the whole page (e.g. 'document.title'). "
        "Prefer this over many blind clicks for date pickers / custom dropdowns / sliders."
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
    # ISSUE #203: EvaluateTool (run-JS) is LOADED again so capable API models (GLM/DeepSeek)
    # can set values directly on complex widgets (date pickers, custom dropdowns) instead of
    # looping on blind clicks. Issue #67's guarantee that the limited 3B local model must
    # NEVER execute arbitrary JavaScript is preserved PER-MODEL: qwen3:4b's per-model toolset
    # (config.MODEL_TOOL_POLICIES) OMITS evaluate, so its view is unchanged; only the capable
    # models — which are also PROMPT-GUIDED to prefer it (config._EVALUATE_GUIDANCE) — see it.
    EvaluateTool,
    # Excluded: OpenBrowserTool (goto opens browser automatically),
    # SnapshotTool (auto-injected into system prompt), ScreenshotTool (model is not visual).
    # NavigateBackTool / NavigateForwardTool are added (loaded) by #206 (it removes the
    # back-button guard); qwen3:4b's per-model toolset OMITS them (forward-compatible here).
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
    and retry once. Silently returns the blank text if still blank after retry — the
    caller's observation path handles this gracefully with no error output.
    """
    import time as _time
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
            _time.sleep(1.5)
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
        return """\
# Web Agent

## Reading the page
Each turn the current page is shown to you automatically under '# Current page (live snapshot — provided automatically)'. Read it before every action — there is no tool to request it.

SNAPSHOT IS GROUND TRUTH. Only refs (e.g. `e12`) listed in the latest snapshot exist. Never guess, invent, or increment a ref, and never use CSS selectors or class names.

## Acting on the page
- Open a page with `goto <url>` — it starts the browser automatically if needed.
- Identify a control by its visible label, placeholder, aria-label, role, or nearby text, then act on that control's ref.
- Use the most specific tool for the control: `fill` for text fields, `select_option` for dropdowns/comboboxes, `check`/`uncheck` for checkboxes and radios, `set_spinbutton` for numeric steppers, `upload` (pass the trigger ref AND the absolute file path in one call) for file inputs, and `click` for buttons and links.
- Batch independent actions in one turn (e.g. fill several fields, then click the control that advances); use a single action when you must see its result before deciding the next step.

## Recovering from problems
- If a cookie or consent banner blocks the page, click its Accept/Agree/Dismiss control first.
- Never repeat the exact same failing call — change the ref or the approach.
- If an action does not change the page (still the same view, or fields marked [invalid]), do not retry it blindly: fix the flagged fields, then try once more.
- If the page shows a CAPTCHA or other human-verification challenge, do not attempt to bypass it — call `validate` and describe the blocker.

## Finishing
- Use only the information the task provides; do not invent values, credentials, or facts.
- Work step by step and confirm each action from the next snapshot before assuming it worked.
- Call `validate` only when the task's stated goal is actually met (the page shows the expected result or confirmation), or when a genuine blocker prevents further progress — including the current URL, what is blocking you, and what you already tried.\
"""

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
