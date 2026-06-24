# Prose Labeling Analysis — Interactable Elements (issue #70 spec)

**Branch:** `research/prose-labeling-analysis` (off `beta` @ `e561c12`)
**Scope:** ANALYSIS ONLY. This document defines the #70 implementation
("clearly point out interactables"). No transform code is changed here.

**Transform under test:** `vibeharness/snapshot_prose.py` (`aria_yaml_to_prose`),
the issue-#64 WebArena-style ARIA → prose linearizer wired behind
`Config.web_snapshot_prose`.

**Ground truth used (real captured runs, 2026-06-24):**

| Run | Raw ARIA (`turn-*-snapshot-*.txt`) | Notes |
|-----|-----------------------------------|-------|
| `C:\git\vh-ashley38\.vibe\20260624_074859-diagnostics\` | turn-002 (YouTube home), turn-003 (watch page) | **true raw ARIA** with `[ref=eN]` — the source for re-running the transform |
| `C:\git\vibetestruns\cat-search-ff74a90\.vibe\20260624_102737-diagnostics\` | turns 2-8 | snapshot files already hold **transformed prose** (a prose A/B run) — used to confirm real-world output |
| `C:\git\vh-catvids-prose\.vibe\20260624_101333-diagnostics\` | turn 1 | prose run; system-prompt files show injected prose |

All "current prose" below was reproduced by importing
`vibeharness.snapshot_prose.aria_yaml_to_prose` and running it on the raw ARIA
from the ashley run (verified to byte-match the prose captured in the cat-search
prose run, e.g. `[e30] dropdown "Search" [expanded]`).

---

## TL;DR — the top labeling fixes

1. **The search box is the headline bug.** Raw `combobox "Search" [expanded]`
   renders as `dropdown "Search"`. "dropdown" tells the agent (and our own
   `select_option` tool docs say "Choose an option in a `<select>` dropdown") to
   pick a pre-existing option — but YouTube's search box is a **fillable text
   field** you `fill` then submit. The single most common control in every run
   is mislabeled toward the wrong tool.
2. **No affordance / no "how to act" anywhere.** Every line is a static
   role+name. Nothing says *fill* vs *click* vs *select*. A 3B model has to
   infer the tool from the role word, and the role words don't map cleanly to
   our tool names (`textbox`→`fill`, `combobox`→`fill`/`type`, `button`→`click`,
   `link`→`click`, `select`→`select_option`, `checkbox`→`check`).
3. **`[-]` is ambiguous.** It's used for BOTH "decorative, not actionable" and
   "actionable but Playwright gave no ref" (e.g. `[-] slider "Volume"`). The
   agent can't tell "ignore this" from "you want this but you can't address it".
4. **Unnamed actionable elements look like noise.** Search-result video links
   render as bare `[e138] link -> /watch?v=...` with no name — the agent can't
   tell what it links to without parsing a URL.
5. **The blocking-dialog buttons are correct and clear** (`[e168] button
   "Reject..."`, `[e175] button "Accept..."`) and the preamble note is good —
   keep that pattern; extend the same clarity to fillable fields.

---

## How the transform labels today (mechanics)

- Line format: `<indent>[<ref>] <role> "<name>" [<state>] [-> <url>]`
  (`_render_node_line`, lines 254-274).
- Role is passed through `_ROLE_LABEL` (lines 237-251), a cosmetic rename map.
  The **only** semantic remap is `combobox → dropdown` and
  `searchbox → searchbox`; everything else prints near-verbatim
  (`textbox`→`textbox`, `button`→`button`, `link`→`link`, `slider`→`slider`).
- There is **no notion of "actionable"** beyond keeping a ref. `_is_interesting`
  (lines 215-233) keeps named/stated nodes and any ref'd non-noise node, but the
  rendered line never states that the element can be acted on or *how*.
- Refless nodes get the literal token `[-]` (line 265).

### Important environmental fact (drives the scheme)

Playwright's ARIA-YAML in these real snapshots emits **no** `editable`,
`focusable`, `value`, or `url=` state properties — verified:
`grep -c 'editable|focusable|\[value|\[url=' = 0` across both raw snapshots. The
search combobox carries only `[expanded]`. **Therefore the scheme cannot key off
an `editable` ARIA property — affordance must be derived from the ARIA role.**
The only state tokens actually present in the corpus are: `active`, `disabled`,
`expanded`, `level=N`. The recommended scheme below is role-driven, with
`editable`/`focusable`/`readonly`/`checked`/`selected`/`required` handled
opportunistically *if/when* present (forward-compatible) but never required.

---

## Problem catalogue — real examples (raw → current → why → proposed)

### 1. Search box — mislabeled as a dropdown (THE headline)

**Raw ARIA** (ashley turn-002, line 29; identical on watch page turn-003 line 29,
and in the live cat-search prose run as `[e30]`):
```
- combobox "Search" [expanded] [ref=e34]
```
**Current prose:**
```
[e34] dropdown "Search" [expanded]
```
**Why it's unclear / wrong:**
- "dropdown" is the visible cue for our `select_option` tool, whose own
  description is *"Choose an option in a `<select>` dropdown."* The agent is
  steered to pick a pre-existing option that does not exist.
- The actual correct action is `fill(e34, "cat videos")` then submit (`press_key
  Enter` or `click` the adjacent `[e35] button "Search"`). Nothing in the line
  says "type here".
- `[expanded]` (an autocomplete-popup state) reinforces the false "dropdown"
  read.
- This is not a corner case: it is the top-of-page control on **every** YouTube
  turn in all three runs.

**Proposed prose:**
```
[e34] textfield "Search" — type a value here with fill, then submit (Enter or click the Search button)
```
(Concretely, role `combobox` with no `multiselectable`/options → treat as an
editable text field. See scheme §A.)

### 2. Adjacent submit button — fine, but the pairing is invisible

**Raw** (turn-002 line 30): `- button "Search" [ref=e35] [cursor=pointer]`
**Current:** `[e35] button "Search"`
**Why:** Correct, but the agent doesn't know this is how you submit the field
above. Low priority; the field's own hint (§1) should name the submit path.
**Proposed:** keep `[e35] button "Search" — click` (affordance suffix per §A).

### 3. Refless actionable controls — `[-]` overloaded (sliders)

**Raw** (watch page turn-003 lines 70, 78):
```
- slider "Click or scroll the panel for the precise seeking." [disabled]
- slider "Volume"
```
**Current prose** (confirmed in cat-search prose turns 4-5):
```
[-] slider "Click or scroll the panel for the precise seeking." [disabled]
[-] slider "Volume"
```
**Why it's unclear:** `[-]` is the same token used for decorative nodes. The
agent cannot distinguish "this is a control you'd want but it has no ref so you
can't target it" from "ignore this". A disabled slider and an un-ref'd-but-live
control look identical.
**Proposed:**
```
(no ref) slider "Volume" — not directly targetable
```
i.e. replace `[-]` with an explicit "(no ref)" marker and a "not targetable"
note for would-be interactables, so the missing handle is a stated fact, not a
silent dash. (Disabled state still shown: `[disabled]`.)

### 4. Unnamed links read as noise

**Current prose** (cat-search results turns 2-8, e.g. line 28):
```
[e138] link -> /watch?v=3URtTIdnXIk&pp=ygUKY2F0IHZpZGVvcw%3D%3D
```
**Why it's unclear:** No accessible name → no human-readable target; the agent
must reverse-engineer a video from a URL query string. It doesn't read as "a
clickable result". (These are the actual search results the task needs to click.)
**Proposed:** keep the line (it IS actionable — has a ref), add the click
affordance, and when the name is empty fall back to the link text/heading of the
sibling so it reads as a result:
```
[e138] link "These CATS are too FUNNY!" — click -> /watch?v=3URtTIdnXIk
```
(Name-recovery from a child `heading`/`text` is a secondary nicety; the
load-bearing fix is the `— click` affordance + trimming the tracking
`&pp=...` tail.)

### 5. Dialog controls — already clear (keep this as the template)

**Raw** (turn-002 lines 141-147):
```
- button "Reject the use of cookies and other data for the purposes described" [ref=e168]
- button "Accept the use of cookies and other data for the purposes described" [ref=e175]
- link "More options" [ref=e183]
```
**Current prose:**
```
[e168] button "Reject the use of cookies and other data for the purposes described"
[e175] button "Accept the use of cookies and other data for the purposes described"
[e183] link "More options" -> https://consent.youtube.com/...
```
**Why it's good:** named, ref'd, and the preamble already says *"clear it (click
its Accept/Reject/Dismiss control)"*. This is the model to extend: it names the
element, gives the ref, and states how to act. The only gap is the per-line
affordance verb, which §A adds uniformly.

### 6. `generic` paragraph promoted to a pseudo-interactable

**Raw** (turn-002 line 69): a long `generic "..."` with a ref.
**Current:** `[e88] generic "You can turn on watch and search history..."`
**Why it's unclear:** It carries a ref so it survives, prints with a `[eN]`
identifier identical in shape to a button — the agent may try to "click the
ref". It is not interactable.
**Proposed:** non-interactable roles (`generic`, `heading`, `text`, `img`)
should NOT carry a bracketed ref in the actionable column; render their ref (if
any) inertly or drop it, reserving the `[eN]` + affordance shape for true
controls. See scheme §B.

---

## Recommended labeling scheme (the #70 spec)

**Design goal:** make every interactable *unmistakable* — what it is, that it's
actionable, and *how* to act — while KEEPING the native `[ref]` (the issue-#64
decision: the ref is the tool's `target`, no second mapping table).

Grounded in WebArena's `[<id>] <role> "<name>" <props>` line format (already the
module's basis) + ARIA role semantics. Because Playwright omits `editable`/
`focusable` here, **affordance is derived from role**, with ARIA state used when
present.

### §A. Affordance suffix — every actionable line ends with "how to act"

Append a terse, tool-named action hint. Keep WebArena's
`[ref] role "name" [state]` prefix; add ` — <verb>` after it.

| ARIA role (raw) | Render as | Affordance suffix | Maps to tool |
|-----------------|-----------|-------------------|--------------|
| `textbox`, `searchbox` | `textfield` | `— type a value with fill` | `fill` / `type` |
| `combobox` *(no listbox options / editable)* | `textfield` | `— type a value with fill, then Enter` | `fill`+`press_key` |
| `combobox`/`listbox` *(true `<select>`)* | `dropdown` | `— pick an option with select_option` | `select_option` |
| `button` | `button` | `— click` | `click` |
| `link` | `link` | `— click` (+ `-> url`) | `click` |
| `checkbox`, `switch` | `checkbox` | `— toggle with check/uncheck` | `check`/`uncheck` |
| `radio` | `radio` | `— select with check` | `check` |
| `tab`, `menuitem`, `option` | (role) | `— click` | `click` |
| `slider`, `spinbutton` | (role) | `— adjust (click/scroll)` | `click`/`hover` |

**Disambiguating `combobox`:** the headline bug. A `combobox` that is editable
(text-entry; YouTube search) must render as `textfield` and route to `fill`. Only
a `combobox` backed by a real option list (a `<select>`) should keep the
`dropdown`/`select_option` label. Heuristics, in order: (1) if ARIA exposes
`editable`/`hasPopup=listbox`+`autocomplete`, treat as text; (2) if the node has
no child `option` nodes, treat as text (a `<select>` enumerates its options in
the tree); (3) default an unqualified `combobox` to **textfield** (the common
web case — search/address bars — and the safer failure: `fill` on a true select
is a clean error, whereas `select_option` on a text field silently misfires).
The current code does the opposite default and is wrong.

### §B. Reserve the `[ref]` + affordance shape for real interactables

- **Interactable roles** (the §A table) → `[eN] role "name" [state] — verb`.
- **Non-interactable roles** (`generic`, `heading`, `text`, `img`, `list`,
  `listitem`, structural) → keep for context but **without** a leading `[eN]`
  affordance token, so the agent never confuses prose/context with a control.
  Use a non-bracketed prefix (e.g. `· heading "Comments"`).
- This makes "lines that start with `[eN] <role> … — <verb>`" a clean, scannable
  set = exactly the actionable elements.

### §C. Refless interactables — say it explicitly

Replace the overloaded `[-]` with:
- `(no ref)` for an element that *would* be actionable but has no Playwright
  handle, plus `— not directly targetable`.
- Decorative/context nodes simply carry no identifier at all (per §B).

So `[-] slider "Volume"` → `(no ref) slider "Volume" — not directly targetable`.

### §D. State tokens — keep, but translate the load-bearing ones

Keep `[disabled]`, `[level=N]`. Translate interaction-relevant state to plain
words: `[expanded]` on a textfield → drop it (it's autocomplete chrome, and it
mis-cues "dropdown"); `[active]` on the dialog → already handled by the preamble.
`[checked]`/`[selected]` (when present) → render as `[checked]`/`[selected]` so
the agent sees current toggle state before acting.

### §E. Keep the blocking-dialog preamble (it works) and the WebArena pruning

No change to `_find_active_dialog` preamble or the noise-pruning / depth-collapse
/ text-dedup — those are sound. #70 is purely about the **per-line interactable
labeling** above.

---

## Before / after on the real YouTube home snapshot (ashley turn-002)

**Before (current transform output):**
```
[e9] button "Guide"
[e14] link "YouTube Home" -> /
[e34] dropdown "Search" [expanded]
[e35] button "Search"
[e49] link "Sign in" -> https://accounts.google.com/ServiceLogin?...
[e88] generic "You can turn on watch and search history at any time..."
[-] slider "Volume"
[e168] button "Reject the use of cookies and other data for the purposes described"
[e175] button "Accept the use of cookies and other data for the purposes described"
```

**After (proposed scheme):**
```
[e9]   button "Guide" — click
[e14]  link "YouTube Home" — click -> /
[e34]  textfield "Search" — type a value with fill, then submit (Enter or click [e35])
[e35]  button "Search" — click
[e49]  link "Sign in" — click -> accounts.google.com/ServiceLogin
·      "You can turn on watch and search history at any time..."   (context, not a control)
(no ref) slider "Volume" — not directly targetable
[e168] button "Reject the use of cookies and other data..." — click
[e175] button "Accept the use of cookies and other data..." — click
```

The agent now reads `[e34] textfield "Search" — type a value with fill` and
knows the element, that it's actionable, and the exact tool — instead of being
told it's a "dropdown".

---

## Implementation notes for #70 (non-binding pointers)

- All changes localize to `_ROLE_LABEL` + `_render_node_line` (+ a small
  `combobox` editable/option-count check in `_parse_line`/`_render_node_line`)
  in `vibeharness/snapshot_prose.py`. The tree build, pruning, and preamble are
  untouched.
- Affordance verbs should match the live tool names in `vibeharness/web.py`
  (`fill`, `click`, `select_option`, `check`/`uncheck`, `type`, `press_key`,
  `hover`) so the prose vocabulary and the tool surface agree.
- Add a fixture test that asserts `combobox "Search"` → `textfield … fill`
  (regression guard for the headline bug) using the ashley turn-002 raw snapshot.
