"""Deterministic ARIA-YAML snapshot -> WebArena-style prose (issue #64).

VibeThinker-3B (and small models generally) struggle to reason over Playwright's
raw AI-optimized ARIA-YAML accessibility snapshot: the deep nesting, the swarm of
semantically-empty ``generic`` / ``img`` wrapper nodes, the verbose ARIA labels, and
the YAML punctuation drown the few controls the agent actually needs to act on. This
module renders that tree into the **field-standard text observation for web agents** —
the rule-based accessibility-tree linearization used by **WebArena** — which prunes the
noise and emits one compact, depth-indented line per *interesting* node.

Design ground-truthed from WebArena (do NOT invent):
  ``web-arena-x/webarena`` — ``browser_env/processors.py``,
  ``TextObervationProcessor.parse_accessibility_tree`` and ``clean_accesibility_tree``
  (https://github.com/web-arena-x/webarena/blob/main/browser_env/processors.py).

What we copied / adapted from WebArena:
  * **Per-node line format** ``[<id>] <role> "<name>" <props>`` — a bracketed leading
    identifier, then the role, then the quoted accessible name, then any kept state
    properties. (WebArena uses ``[<obs_node_id>] <role> '<name>'`` with single quotes
    and an integer id; see the identifier decision below for our deviation.)
  * **Depth indentation** — one indent unit per tree level.
  * **Pruning of uninteresting nodes** — a node with *no accessible name and no kept
    properties* whose role is in a noise set (``generic, img, list, strong, paragraph,
    banner, navigation, listitem, ...``) is dropped, EXACTLY mirroring WebArena's
    "empty generic node" skip list.
  * **Depth collapse on drop** — when a node is pruned, its children are re-parented up
    one level (rendered at the *parent's* depth) rather than orphaned. WebArena does
    this with ``child_depth = depth + 1 if valid_node else depth``.
  * **Ignored properties** — ``cursor`` is dropped (it is Playwright chrome, not an ARIA
    state), mirroring WebArena's ``IGNORED_ACTREE_PROPERTIES``.
  * **StaticText de-duplication** — a text leaf whose content already appears in the
    immediately preceding kept lines is dropped, mirroring ``clean_accesibility_tree``.

### IDENTIFIER DECISION (issue #64, step 3) — preserve the native ``ref``
WebArena mints its OWN integer ids (``[5]``) and keeps a side ``obs_nodes_info`` map back
to the real backend DOM node, because the model can only *say* a number. Our agent does
NOT have that limitation: the discrete web subtools (#51 ``click`` / ``fill`` / ``hover`` /
… in :mod:`vibeharness.web`) take a Playwright **ref** (``e1300``) or a CSS selector as
their ``target``. So the cleanest, lowest-risk choice is to **keep the native ref inline**
as the leading bracketed token: ``[e1300] button "Accept all"``. The agent reads a line and
already knows the exact string to pass to ``click`` — no second mapping table to build,
keep in sync, or get wrong, and the existing tools work UNCHANGED.

We therefore deviate from WebArena's integer scheme deliberately and document why: a
synthetic integer id would *require* a ref-mapping back-channel to stay actionable, adding
a failure mode (drift / stale map) for zero benefit, since our ref is already a stable,
tool-resolvable handle. Every actionable node in the prose carries its ``[eN]`` ref. Nodes
the tree gives no ref (pure decorative/text) are still shown for context but, having no
ref, are simply not clickable — which is correct.

This module is a pure string/tree transform: no model call, no GPU, deterministic, ~0 ms.
It is wired behind a config seam (``Config.web_snapshot_prose``) so it can be A/B'd against
the raw ARIA injection without removing it (see :mod:`vibeharness.cli`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- WebArena parity: roles that are pure noise when they carry no name + no state.
# Adapted from WebArena's "empty generic node" skip list in
# TextObervationProcessor.parse_accessibility_tree. We add Playwright's extra wrapper
# roles seen in our real snapshots (``image`` alias, ``tooltip`` without a name, etc.).
_NOISE_ROLES_WHEN_EMPTY = frozenset({
    "generic", "img", "image", "list", "strong", "paragraph", "banner",
    "navigation", "section", "labeltext", "legend", "listitem", "main",
    "complementary", "contentinfo", "group", "tooltip", "emphasis", "time",
    "separator", "presentation", "none",
})

# --- WebArena parity: ARIA-snapshot attributes that are NOT semantic state and should
# never be rendered as a property. ``cursor`` is Playwright chrome (e.g. ``cursor=pointer``);
# mirrors WebArena's IGNORED_ACTREE_PROPERTIES (which drops layout/hidden bookkeeping).
_IGNORED_PROPERTIES = frozenset({"cursor"})

# A snapshot line, e.g.  ``  - button "Accept all" [ref=e1300] [cursor=pointer]:``
# We parse: indent (-> depth), role, optional quoted name, bracketed attrs, trailing colon.
# The whole "- ... " bullet may itself be wrapped in single quotes when the YAML value
# contains a colon (Playwright does this for e.g. ``'button "Language: English" [ref=e1219]'``).
_INDENT_UNIT = 2  # our raw snapshot indents children by two spaces per level

# role token: a leading word (letters) right after the dash / opening quote.
_ROLE_RE = re.compile(r"""^(?P<role>[A-Za-z][\w-]*)""")
# quoted accessible name immediately following the role
_NAME_RE = re.compile(r'^\s*"(?P<name>(?:[^"\\]|\\.)*)"')
# a bracketed attribute like [ref=e1300], [active], [level=2], [checked], [expanded]
_ATTR_RE = re.compile(r"\[(?P<body>[^\]]*)\]")
# leaf nodes the ARIA YAML emits as ``key: value`` rather than ``- role``
_URL_RE = re.compile(r"^/url:\s*(?P<url>.+)$")
_TEXT_RE = re.compile(r"^text:\s*(?P<text>.*)$")


@dataclass
class _Node:
    role: str
    name: str = ""
    ref: str | None = None
    props: list[str] = field(default_factory=list)   # kept ARIA state, e.g. ["active", "level=2"]
    url: str | None = None                            # for links
    text: str | None = None                           # raw text leaf content
    depth: int = 0
    children: list["_Node"] = field(default_factory=list)


def _strip_yaml_bullet(line: str) -> tuple[int, str]:
    """Return (depth, content) for one raw snapshot line.

    Depth is the leading-space count // 2. Content is the line past the ``- `` bullet,
    with the optional surrounding single quotes (Playwright adds them when the value
    contains a colon) removed. Lines that are not list items (rare) are treated as
    depth-by-indent content as-is.
    """
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    depth = indent // _INDENT_UNIT
    if stripped.startswith("- "):
        content = stripped[2:]
    elif stripped == "-":
        content = ""
    else:
        content = stripped
    content = content.rstrip()
    if content.endswith(":"):
        content = content[:-1].rstrip()
    # Unwrap Playwright's defensive single-quoting of a whole bullet value.
    if len(content) >= 2 and content[0] == "'" and content[-1] == "'":
        content = content[1:-1]
    return depth, content


def _parse_attrs(content: str) -> tuple[str | None, list[str]]:
    """Extract (ref, kept_props) from the bracketed attributes in a content string."""
    ref: str | None = None
    props: list[str] = []
    for m in _ATTR_RE.finditer(content):
        body = m.group("body").strip()
        if body.startswith("ref="):
            ref = body[len("ref="):].strip()
            continue
        key = body.split("=", 1)[0].strip().lower()
        if key in _IGNORED_PROPERTIES:
            continue
        props.append(body)
    return ref, props


def _parse_line(content: str) -> _Node | None:
    """Parse one unwrapped content string into a _Node, or None if it is empty.

    Handles the three shapes our ARIA YAML emits:
      * ``role "name" [attrs]``           -> element node
      * ``/url: https://...``             -> url leaf (attached to nearest link)
      * ``text: some words``              -> raw text leaf
    """
    if not content:
        return None
    murl = _URL_RE.match(content)
    if murl:
        return _Node(role="__url__", url=murl.group("url").strip())
    mtext = _TEXT_RE.match(content)
    if mtext:
        return _Node(role="__text__", text=mtext.group("text").strip())

    mrole = _ROLE_RE.match(content)
    if not mrole:
        return None
    role = mrole.group("role")
    rest = content[mrole.end():]
    mname = _NAME_RE.match(rest)
    name = ""
    if mname:
        name = mname.group("name").replace('\\"', '"')
    ref, props = _parse_attrs(rest)
    return _Node(role=role, name=name, ref=ref, props=props)


def _build_tree(raw: str) -> _Node:
    """Parse the ARIA-YAML body of a raw snapshot into a _Node tree.

    Only the lines inside the ```` ```yaml ... ``` ```` fence are treated as the tree;
    the textual header (Page URL / Title / Console) is returned separately by the caller.
    A synthetic root holds the top-level nodes. ``/url:`` and ``text:`` leaves are folded
    into their owning element rather than kept as standalone nodes.
    """
    root = _Node(role="__root__", depth=-1)
    # stack of (depth, node) for parent resolution
    stack: list[_Node] = [root]
    for line in raw.splitlines():
        if not line.strip():
            continue
        depth, content = _strip_yaml_bullet(line)
        node = _parse_line(content)
        if node is None:
            continue
        # Pop until the top of the stack is a strict ancestor (smaller depth).
        while len(stack) > 1 and depth <= stack[-1].depth:
            stack.pop()
        parent = stack[-1]
        if node.role == "__url__":
            # attach url to the nearest enclosing element (the parent)
            if parent is not root and parent.url is None:
                parent.url = node.url
            continue
        if node.role == "__text__":
            # represent a raw text leaf as a child text node so de-dup can see it
            node.depth = depth
            parent.children.append(node)
            continue
        node.depth = depth
        parent.children.append(node)
        stack.append(node)
    return root


def _is_interesting(node: _Node) -> bool:
    """WebArena's keep/skip rule, adapted to ref-keyed nodes.

    Keep a node if it has an accessible name, OR a kept state property, OR a ref to an
    *actionable* role, OR it is a text leaf with content. Drop a node that has no name
    and no properties when its role is in the noise set (WebArena's "empty generic node"
    skip). A named or stated node is always kept regardless of role.
    """
    if node.role == "__text__":
        return bool(node.text and node.text.strip())
    has_name = bool(node.name and node.name.strip())
    has_props = bool(node.props)
    if has_name or has_props:
        return True
    if node.role.lower() in _NOISE_ROLES_WHEN_EMPTY:
        return False
    # An unnamed, unstated node of a non-noise role (e.g. an unlabeled ``textbox``) is
    # still potentially actionable if it has a ref; keep it so the agent can target it.
    return node.ref is not None


# --- Role -> human label + tool affordance (issue #70) --------------------------------
# Each actionable role gets a stable human-readable LABEL plus an AFFORDANCE suffix whose
# verb names the *exact* web subtool (:mod:`vibeharness.web`) the agent should reach for.
# This is derived from the ROLE ALONE — Playwright's ARIA-YAML here exposes no
# ``editable``/``value`` props, so we cannot inspect them. Tools (web.py): ``fill`` (set a
# text field), ``click`` (button/link), ``select_option`` (a real <select>/listbox),
# ``check``/``uncheck`` (checkbox/radio).
_AFF_FILL = "type a value with fill"
_AFF_CLICK = "click"
_AFF_SELECT = "pick an option with select_option"
_AFF_TOGGLE = "toggle with check/uncheck"
# A bare ``combobox`` is ambiguous in the ARIA tree: it may be an EDITABLE text field
# (e.g. a search box) OR a button-style option-picker (the job-form State/Country/etc.
# dropdowns, which are <button role="combobox" aria-haspopup="listbox"> — fill FAILS on
# them). Playwright's snapshot does NOT expose aria-haspopup, so we cannot tell them apart.
# We cue select_option FIRST because the select_option tool has a built-in fallback (#125):
# on a custom combobox it CLICKS the trigger to open the listbox and CLICKS the matching
# option — so select_option works for BOTH the picker case and (harmlessly) leaves an
# editable box for fill. This reverses the old fill-first default, which made the 3B model
# fail on all 12 of the job form's comboboxes (iter-1).
_AFF_COMBO = "pick an option with select_option (it opens the dropdown); if it is editable, use fill"

# role (lowercased) -> (label, affordance). ``None`` affordance = surfaced for context but
# not itself a primary action target (e.g. ``option`` inside a listbox, ``tab``).
#
# COMBOBOX (#70, revised iter-1): a bare ``combobox`` is ambiguous — it can be an editable
# text field (YouTube search) OR a button option-picker (the job form's State/Country/...
# dropdowns). The #70 default cued ``fill``, which FAILS on every button-combobox. Since the
# ``select_option`` tool now has an open-then-click fallback for custom comboboxes (#125),
# we cue ``select_option`` (works for pickers; falls through for editable). Genuine native
# pickers are ``listbox``/``select``/``menu``. See ``_AFF_COMBO`` for the full rationale.
_ROLE_INFO: dict[str, tuple[str, str | None]] = {
    "link": ("link", _AFF_CLICK),
    "button": ("button", _AFF_CLICK),
    "tab": ("tab", _AFF_CLICK),
    "menuitem": ("menuitem", _AFF_CLICK),
    "menuitemcheckbox": ("checkbox", _AFF_TOGGLE),
    "menuitemradio": ("radio", _AFF_TOGGLE),
    # fillable text inputs -> fill
    "textbox": ("text field", _AFF_FILL),
    "searchbox": ("search field", _AFF_FILL),
    "combobox": ("dropdown", _AFF_COMBO),     # ambiguous combobox -> select_option (has open+pick fallback)
    "spinbutton": ("number field", _AFF_FILL),
    # genuine option pickers -> select_option
    "listbox": ("dropdown list", _AFF_SELECT),
    "select": ("dropdown list", _AFF_SELECT),
    "menu": ("menu", _AFF_SELECT),
    # toggles -> check/uncheck
    "checkbox": ("checkbox", _AFF_TOGGLE),
    "radio": ("radio", _AFF_TOGGLE),
    "switch": ("switch", _AFF_TOGGLE),
    # surfaced for context, no single primary tool verb
    "option": ("option", None),
    "slider": ("slider", None),
}

# Actionable roles (have a tool affordance). Used to disambiguate the [-] placeholder: an
# actionable role with NO ref is targetable-in-principle but the tree gave no handle.
_ACTIONABLE_ROLES = frozenset(r for r, (_, aff) in _ROLE_INFO.items() if aff is not None)


def _role_info(role: str) -> tuple[str, str | None]:
    """(label, affordance) for a role, falling back to the raw role + no affordance."""
    return _ROLE_INFO.get(role.lower(), (role, None))


def _render_node_line(node: _Node, depth: int) -> str:
    """Render a single kept node into a WebArena-style line, ref-keyed.

    Format:  ``<indent>[<ref>] <label> "<name>"<state>[ -> <url>] — <affordance>``
    The leading ``[<ref>]`` is the tool-resolvable identifier (issue #64 step 3). The
    trailing ``— <affordance>`` (issue #70 step 2) names the exact web subtool verb for
    the role so a small model is not left to guess (e.g. a text field cues ``fill``, a
    button cues ``click``).

    The ``[-]`` placeholder is disambiguated (#70 step 3): a *decorative* refless node
    just gets ``[-]``; an *actionable-role* refless node is flagged ``(no ref) … not
    directly targetable`` so the agent does not waste a tool call trying to act on a
    handle that does not exist.
    """
    indent = "  " * depth
    if node.role == "__text__":
        return f'{indent}text: {node.text}'
    role_l = node.role.lower()
    label, affordance = _role_info(node.role)
    is_actionable = role_l in _ACTIONABLE_ROLES

    if node.ref:
        ident = f"[{node.ref}]"
    elif is_actionable:
        # actionable role but the tree gave no ref -> say so explicitly
        ident = "(no ref)"
    else:
        ident = "[-]"

    line = f"{indent}{ident} {label}"
    if node.name and node.name.strip():
        line += f' "{node.name}"'
    elif role_l == "link" and node.url:
        # unnamed link (#70 step 4): hint its destination so it is not a mystery target
        line += f' "{_link_hint(node.url)}"'
    if node.props:
        line += " " + " ".join(f"[{p}]" for p in node.props)
    if node.url and role_l == "link":
        line += f" -> {node.url}"

    if node.ref and affordance:
        line += f" — {affordance}"
    elif is_actionable and not node.ref:
        line += " — not directly targetable"
    return line


def _link_hint(url: str) -> str:
    """A short human hint for an unnamed link, derived from its href (#70 step 4)."""
    u = url.strip()
    if u.startswith("javascript:") or u in ("", "#"):
        return "link"
    # keep the last meaningful path segment (or the query's v= for YouTube watch links)
    m = re.search(r"[?&]v=([^&]+)", u)
    if m:
        return f"watch {m.group(1)}"
    tail = u.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0]
    return tail or u


def _walk(node: _Node, depth: int, out: list[str], recent: list[str]) -> None:
    """Depth-first WebArena linearization with depth-collapse-on-prune + text de-dup.

    ``recent`` holds the last few rendered lines so a StaticText leaf already covered by a
    nearby line is dropped (WebArena ``clean_accesibility_tree``).
    """
    keep = _is_interesting(node)
    if keep:
        # text de-duplication (clean_accesibility_tree parity)
        if node.role == "__text__":
            t = (node.text or "").strip()
            if t and any(t in r for r in recent[-3:]):
                keep = False
        if keep:
            line = _render_node_line(node, depth)
            out.append(line)
            recent.append(line)
    # WebArena: children render one deeper only if THIS node was kept; otherwise they
    # collapse up to this node's depth (child_depth = depth + 1 if valid_node else depth).
    child_depth = depth + 1 if keep else depth
    for child in node.children:
        _walk(child, child_depth, out, recent)


# --- header parsing -------------------------------------------------------------------
_PAGE_URL_RE = re.compile(r"Page URL:\s*(?P<v>.+)")
_PAGE_TITLE_RE = re.compile(r"Page Title:\s*(?P<v>.+)")


def _split_header_and_yaml(raw: str) -> tuple[str, str, str]:
    """Return (page_url, page_title, yaml_body) from a raw snapshot string.

    The raw snapshot from :func:`vibeharness.web.capture_page_snapshot_raw` contains a
    small textual header (Page URL / Title / Console) and a fenced ```` ```yaml ```` block.
    We isolate the YAML body for tree parsing and lift URL/title for the prose preamble.
    If no fence is present we treat the whole thing as YAML (robustness).
    """
    page_url = ""
    page_title = ""
    mu = _PAGE_URL_RE.search(raw)
    if mu:
        page_url = mu.group("v").strip()
    mt = _PAGE_TITLE_RE.search(raw)
    if mt:
        page_title = mt.group("v").strip()
    # extract the ```yaml ... ``` body if fenced
    fence = re.search(r"```(?:yaml)?\s*\n(?P<body>.*?)\n```", raw, re.DOTALL)
    if fence:
        body = fence.group("body")
    else:
        # no fence: drop obvious header lines, keep the indented tree
        lines = [ln for ln in raw.splitlines()
                 if not ln.lstrip().startswith("#")
                 and "Page URL:" not in ln and "Page Title:" not in ln
                 and "Console:" not in ln and ln.strip() not in ("### Page", "### Snapshot")]
        body = "\n".join(lines)
    return page_url, page_title, body


def aria_yaml_to_prose(raw: str) -> str:
    """Convert one raw Playwright ARIA-YAML page snapshot into WebArena-style prose.

    This is the public entry point wired behind the ``web_snapshot_prose`` config seam.
    It is total: any unparseable input degrades to returning the input unchanged (so a
    surprising snapshot shape can never blank out the page section the agent relies on).

    The output is a compact, depth-indented list of the interesting (named / stated /
    actionable) elements, each led by its tool-resolvable ``[ref]`` identifier, preceded
    by a one-line page preamble and a foreground-dialog note when an ``[active]`` dialog is
    present.
    """
    if not raw or not raw.strip():
        return raw
    try:
        page_url, page_title, body = _split_header_and_yaml(raw)
        root = _build_tree(body)
        if not root.children:
            return raw
        out: list[str] = []
        recent: list[str] = []
        for child in root.children:
            _walk(child, 0, out, recent)
        if not out:
            return raw
        preamble: list[str] = []
        if page_title:
            preamble.append(f'Page: "{page_title}"')
        if page_url:
            preamble.append(f"URL: {page_url}")
        # Surface the single piece of true "foreground" the ARIA tree has: an [active]
        # modal dialog. This is the consent-banner pain point #61/#64 cares about most.
        active_dialog = _find_active_dialog(root)
        if active_dialog is not None:
            label = f' "{active_dialog.name}"' if active_dialog.name else ""
            ref = f" [{active_dialog.ref}]" if active_dialog.ref else ""
            preamble.append(
                f"A dialog{label}{ref} is OPEN IN FRONT of the page and is blocking it; "
                f"clear it (click its Accept/Reject/Dismiss control) before acting on "
                f"anything behind it."
            )
        head = "\n".join(preamble)
        prose = "\n".join(out)
        return (head + "\n\n" + prose) if head else prose
    except Exception:
        # Never let a parsing surprise blank the page section: fall back to raw ARIA.
        return raw


def _find_active_dialog(root: _Node) -> _Node | None:
    """Depth-first search for the first ``dialog`` node carrying the ``active`` state."""
    stack = list(root.children)
    while stack:
        node = stack.pop(0)
        if node.role.lower() == "dialog" and any(
            p.split("=", 1)[0].strip().lower() == "active" for p in node.props
        ):
            return node
        stack[:0] = node.children
    return None
