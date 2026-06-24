"""RESEARCH PROTOTYPE (issue #42) — NOT production wiring.

Post-process pruner for the playwright-cli ARIA-yaml page snapshot.

Goal: shrink the auto-injected snapshot (vibeharness/web.py::capture_page_snapshot)
to interactive-/visible-/text-bearing nodes while KEEPING every actionable element
(its role, accessible name, and [ref=...] so the agent can still click it).

The playwright-cli ARIA snapshot is an indented YAML tree, e.g.:

    - generic [ref=e64]:
      - banner [ref=e66]:
        - button "Guide" [ref=e71] [cursor=pointer]:
          - generic [ref=e89]:
            - img

Strategy (structure-preserving prune):
  KEEP a line if it is "interesting":
    - has an interactive role (button/link/textbox/combobox/checkbox/radio/
      menuitem/tab/option/searchbox/slider/switch/dialog/...), OR
    - carries an accessible name in quotes ("...") , OR
    - is a leaf text node (`- generic [ref=eN]: some visible text`), OR
    - is a heading / has [cursor=pointer].
  DROP "filler" container lines: bare `- generic [ref=eN]:` (or img/none/
    paragraph wrappers) that have NO name and NO text — these are pure layout
    nesting that the model never targets.
  But ALWAYS keep a dropped container if one of its descendants is kept, so the
  indentation/ancestry of kept nodes stays valid (we re-emit a compact ancestor).

This is deliberately conservative: when unsure, keep. Measured reductions are
reported by __main__.
"""
from __future__ import annotations

import re
import sys

# Roles that are directly actionable or structurally important to the agent.
INTERACTIVE_ROLES = (
    "button", "link", "textbox", "searchbox", "combobox", "listbox", "option",
    "checkbox", "radio", "switch", "slider", "spinbutton", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "dialog", "alertdialog",
    "menu", "menubar", "heading", "alert",
)
_ROLE_RE = re.compile(r"^\s*-\s+([a-zA-Z]+)")
_NAME_RE = re.compile(r'"[^"]+"')
_TEXT_LEAF_RE = re.compile(r"^\s*-\s+\w+(?:\s+\[[^\]]+\])*:\s+\S")  # `- generic [ref=e]: text`
_INDENT_RE = re.compile(r"^(\s*)")


def _indent(line: str) -> int:
    return len(_INDENT_RE.match(line).group(1))


def _is_interesting(line: str) -> bool:
    m = _ROLE_RE.match(line)
    role = m.group(1) if m else ""
    if role in INTERACTIVE_ROLES:
        return True
    if "cursor=pointer" in line:
        return True
    if _NAME_RE.search(line):           # accessible name present
        return True
    if _TEXT_LEAF_RE.match(line):       # leaf with visible text after the colon
        return True
    return False


def prune_aria_yaml(yaml_block: str) -> str:
    """Prune the YAML snapshot body (lines after '```yaml'), keeping interesting
    nodes and the minimal ancestor chain needed to keep them well-formed."""
    lines = yaml_block.splitlines()
    keep = [False] * len(lines)

    # Pass 1: mark interesting lines.
    for i, ln in enumerate(lines):
        if ln.strip() and _is_interesting(ln):
            keep[i] = True

    # Pass 2: keep ancestors of any kept line (a shallower-indented line that is
    # the nearest enclosing parent).
    for i in range(len(lines)):
        if not keep[i]:
            continue
        ind = _indent(lines[i])
        # walk upward to mark the ancestor chain
        cur = ind
        for j in range(i - 1, -1, -1):
            if not lines[j].strip():
                continue
            jind = _indent(lines[j])
            if jind < cur:
                keep[j] = True
                cur = jind
                if cur == 0:
                    break
    return "\n".join(ln for i, ln in enumerate(lines) if keep[i])


def prune_snapshot(full: str) -> str:
    """Prune a full playwright-cli snapshot (### Page header + ```yaml block```).
    Keeps the Page header verbatim; prunes only the YAML body."""
    if "```yaml" not in full:
        return full
    head, _, rest = full.partition("```yaml")
    body, _, tail = rest.partition("```")
    pruned = prune_aria_yaml(body.strip("\n"))
    return f"{head}```yaml\n{pruned}\n```{tail}"


if __name__ == "__main__":
    path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path, encoding="utf-8") as f:
        full = f.read()
    pruned = prune_snapshot(full)
    sys.stderr.write(
        f"{path}: full={len(full)} chars -> pruned={len(pruned)} chars "
        f"({100*(1-len(pruned)/max(1,len(full))):.0f}% smaller)\n"
    )
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pruned)
    else:
        sys.stdout.buffer.write(pruned.encode("utf-8"))
