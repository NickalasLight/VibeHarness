# Snapshot Scoping & Max-Cap Analysis (issues #24 / #28 / #29 / #40 / #42)

Research output for the `research/snapshot-scoping` worktree. Goal: make the
auto-injected per-turn page snapshot (`vibeharness/web.py::capture_page_snapshot`,
issue #24) **smaller-but-usable** (scoping), and determine **how high we can safely
raise** `config.web_snapshot_char_limit` (currently 40000).

All measurements taken with `playwright-cli` v0.1.14 (`@playwright/cli`), Chrome,
on a live headless-capable session. Token counts use the `cl100k_base` tokenizer
as a proxy; the model is VibeThinker-3B (Qwen tokenizer), so token figures are
approximate — budgeting below is deliberately conservative to absorb that drift.

---

## TL;DR

- **The bug is real and reproduced.** On the YouTube watch page the consent dialog
  "Before you continue to YouTube" begins at char **36,432** but its **Accept all
  (e1275) / Reject all (e1268)** buttons land at char **40,300** — *past* the
  current 40,000 cap. The agent sees the dialog heading but the buttons it is told
  to click are truncated off. Same pattern on w3schools (Accept at ~40,677) and BBC
  News (consent "Yes, I agree" at ~53,888). Late-DOM overlay controls are emitted
  **last** in the ARIA tree, so a char cap is exactly the wrong tool to clip with.

- **Best scoping method: `snapshot --depth=N` for big size wins, but it is LOSSY on
  deeply-nested controls. The safe production change is a post-process interactive/
  visible PRUNE** (prototype in `scripts/snapshot_prune_prototype.py`) which is
  *lossless on every actionable element* (keeps 100% of buttons/links/inputs and
  their refs) at a modest 12–28% reduction. Best of both: prune + a higher cap.

- **The cap can be raised massively.** With num_ctx=131072 and a generous,
  conservative overhead model, ~**75,000 tokens (~260k chars)** remain free for the
  snapshot. **Recommended cap: `web_snapshot_char_limit = 120000`** (~32k tokens) —
  fits every real page measured *whole* (largest was 56.6k chars) with huge headroom,
  while still bounding any single pathological page so it can't crowd out history.

---

## PART A — Scoping options found

### Native `playwright-cli snapshot` flags (from `snapshot --help` + SKILL.md)

| Flag / form              | Effect                                                                 | Useful for scoping? |
|--------------------------|------------------------------------------------------------------------|---------------------|
| `snapshot <target>`      | **Partial snapshot** rooted at a CSS selector / ref (e.g. `snapshot "#main"`) | Yes — region scope, but requires knowing the selector up-front |
| `snapshot --depth=N`     | Limit ARIA tree depth; "unlimited by default"                          | Yes — biggest size win, but **lossy** (see below) |
| `--raw` (global)         | Strips the `### Page` / generated-code / snapshot wrapper sections      | Marginal — saves only ~215 chars on YT |
| `snapshot --boxes`       | Adds `[box=x,y,w,h]` per element                                        | No — *increases* size |
| `snapshot --filename=`   | Writes to a file instead of returning text                             | No (orthogonal) |

There is **no native "visible-only" or "interactive-only" flag.**

### Playwright ARIA-snapshot semantics (what it already prunes)

Playwright's accessibility snapshot is built from the accessibility tree, so it
*already* drops `display:none` / `visibility:hidden` / `aria-hidden="true"` nodes
and most `interestingOnly`-pruned presentational nodes. What remains and bloats the
output is **real, visible content**: long link lists (nav, video recommendations,
article rivers), and chains of bare `generic` layout wrappers. So "scoping" here
means pruning *layout filler* and (optionally) *non-interactive body text*, not
hidden nodes — those are already gone.

### `--depth` is a big lever but LOSSY on deep controls

Depth prunes by tree level. Consent/cookie overlays are **shallow** top-level nodes,
so they survive even aggressive depth limits — which is why depth "rescues" consent.
But genuinely useful controls that are deeply nested get clipped:

- **docs (MDN) at depth=6 loses the "Yes"/"No" feedback buttons** (a deep widget at
  page bottom) while a 53% size cut.
- **docs at depth=6 drops 83 of 143 links; news drops 26 of 124 links.**

So `--depth` cannot be the production default — it silently removes actionable
elements whose depth varies per site.

### Post-process PRUNE prototype (recommended scoping)

`scripts/snapshot_prune_prototype.py` (RESEARCH prototype, not wired into
production) post-processes the ARIA-yaml that `capture_page_snapshot` already
captures. It **keeps** any line that is interactive (button/link/textbox/combobox/
checkbox/radio/menuitem/tab/dialog/heading/…), carries an accessible name in quotes,
has `[cursor=pointer]`, or is a visible-text leaf; it **drops** bare `generic`/`img`
layout wrappers with no name and no text, while re-emitting the minimal ancestor
chain so indentation stays valid. Conservative by design: when unsure, keep.

It is **lossless on actionable elements** — every button/link/input ref present in
the full snapshot is present in the pruned one — at a modest, page-dependent size cut.

---

## Measurement table

Page snapshots captured live. "Full" = exactly what `cli.run("snapshot")` returns
(the `### Page` + ```yaml``` block the harness injects). depth=6 and the prune
prototype shown for comparison. "Consent + interactive survived?" checks the
consent/Accept-Reject controls and all buttons/inputs.

| Page (URL)                         | Full chars | depth=6 chars (Δ) | pruned chars (Δ) | Consent survived? | All buttons/inputs survived? |
|------------------------------------|-----------:|------------------:|-----------------:|-------------------|------------------------------|
| YouTube watch `?v=dQw4w9WgXcQ`     |    41,160  |   9,056 (−78%)    |  30,969 (−24%)   | depth6 ✅ / prune ✅ (Accept e1275 + Reject e1268 kept w/ refs) | prune ✅ (44/44) · depth6 keeps consent+search+signin |
| BBC News `bbc.com/news`            |    56,595  |  27,192 (−52%)    |  49,726 (−12%)   | depth6 ✅ / prune ✅ ("Yes, I agree" + "I agree" kept) | prune ✅ (16/16 btns, 124/124 links) · depth6 keeps all 16 btns but drops 26 links |
| MDN docs `Web/JavaScript`          |    39,449  |  18,646 (−53%)    |  29,461 (−25%)   | n/a (no consent) | prune ✅ (13/13 btns, 143/143 links) · **depth6 LOSES "Yes"/"No" + 83 links** |
| w3schools `html/default.asp`       |    43,721  |  40,026 (−8%)     |  31,591 (−28%)   | depth6 ✅ / prune ✅ (iframe "Accept" f9e30 kept) | prune ✅ (13/13 btns) · depth6 keeps all |

Token sizes of the full snapshots (cl100k proxy): YouTube ≈ 10.9k tok, BBC ≈ 13.8k
tok, MDN ≈ 10.2k tok, w3schools ≈ 12.5k tok. Empirical ratio ≈ **3.77 chars/token**
for ARIA-yaml (refs/brackets tokenize densely).

### What this proves

1. **40k truncates real consent controls** on the heaviest pages (YT 40,300; w3
   40,677; BBC 53,888). The current cap is the root cause of #24's symptom.
2. **Pruning alone is safe but insufficient for the worst page**: after pruning,
   BBC's consent is still at char 46,655 (> 40k) — only a higher cap rescues it.
   (w3's pruned Accept drops to 30,851, under 40k.)
3. **`--depth` shrinks the most but silently clips deep interactive controls**
   (MDN's Yes/No) — disqualifying it as a blind default.
4. **Therefore the fix is BOTH**: prune the layout filler *and* raise the cap so
   even the largest page's late overlay controls fit.

---

## PART B — Maximum safe cap (headroom math)

Context window and per-turn budgets (`vibeharness/config.py`):

| Quantity                                  | Value (tokens) |
|-------------------------------------------|---------------:|
| `num_ctx` (whole window)                  | 131,072 |
| `reason_tokens` (phase 1, discarded)      | 2,048 |
| `action_tokens` (phase 2, constrained)    | 16,384 |

The snapshot is **stale-dropped every turn** — the whole system prompt is rebuilt
each turn and only the *latest* snapshot is ever present (`prompt.py`, issue #24).
It never accumulates across turns. So the snapshot competes only with: the
generation reservation, the (small) system base, and the growing *narrative* history.

Conservative overhead model (rounding up everywhere to stay safe):

| Component                                            | Tokens |
|------------------------------------------------------|-------:|
| Generation reservation (the larger phase = `action_tokens`) | 16,384 |
| System-prompt base (template + ~7 tool docs + guidance; template alone ≈ 993 chars) | 4,000 |
| Task header + per-turn reminder + scaffolding        | 1,500 |
| Growing narrative history (per-obs capped: web 14,000 chars ≈ 4k tok, fs 12,000 ≈ 3.4k tok; budget ~8 substantial obs) | 30,000 |
| Tokenizer drift (cl100k vs Qwen) + misc safety       | 4,000 |
| **Fixed overhead**                                   | **55,884** |
| **Free for snapshot** = 131,072 − 55,884             | **≈ 75,188 tok** |

At 3.5 chars/token that is **≈ 263,000 chars**; at 3.77, **≈ 283,000 chars**. The
*theoretical* max is therefore ~260k chars — orders of magnitude above any real page.

### Recommended cap (not the theoretical max)

Setting the cap to the theoretical max would let one pathological page consume the
whole window and starve history on a long run. Instead, size the cap to **fit the
largest real pages whole, with comfortable headroom, while still bounding outliers**:

> **`web_snapshot_char_limit = 120000`**  (~32,000 tokens)

Rationale:
- The largest page measured is BBC News at **56,595 chars (~13.8k tok)** — 120k is
  **>2x** that, so every real page (and most overlay-heavy outliers) renders *whole*;
  the consent controls that 40k clips are comfortably included.
- 120k chars ≈ 32k tok leaves **~43k tok of additional free headroom** under the 75k
  budget even at the conservative model above — history can grow far longer without
  the snapshot ever pushing it out.
- It still caps any genuinely pathological page (e.g. an infinite-scroll feed) so a
  single snapshot can never monopolize the window.

If maximum safety against a long, history-heavy run is preferred over fitting every
outlier whole, **80000** (~21k tok, still 1.4x the biggest measured page) is a more
conservative alternative. Either way, **40000 is demonstrably too low** and should
be raised.

---

## Combined recommendation (feeds #29 / #40 / #42)

1. **Scope (safe): add a post-process interactive/visible prune** to
   `capture_page_snapshot` (port `scripts/snapshot_prune_prototype.py`). It is
   lossless on actionable elements and trims 12–28% of layout/text bloat. Apply the
   prune *before* the char-cap truncation so the cap acts on already-scoped text.
   Do **not** adopt `--depth` as a blind default (clips deep controls); reserve
   region-scoped `snapshot <selector>` for targeted follow-up reads.
2. **Raise the cap: `web_snapshot_char_limit = 40000 → 120000`** (~32k tok). Headroom
   math above shows ~75k tok is free; 120k chars fits every measured page whole with
   >40k tok to spare for history, while bounding outliers.
3. **Float-modal priority (#40): emit overlay/dialog nodes FIRST.** The deeper fix is
   to *reorder* the injected snapshot so `dialog`/`alertdialog`/consent overlays
   (which the ARIA tree emits last) lead the snapshot — guaranteeing the controls the
   agent is told to click survive *any* cap. The prune in (1) is the natural place to
   hoist `role=dialog` subtrees to the top. With prune + 120k cap + dialog-first, the
   "consent past the cap" failure (#24) cannot recur.

Net effect for the YouTube case: with the prune the snapshot drops to ~31k chars
(Accept/Reject kept with refs e1275/e1268), the 120k cap never truncates it, and a
dialog-first reorder puts the consent controls at the very top of what the model sees.

---

## Reproduction / artifacts

- Prototype pruner: `scripts/snapshot_prune_prototype.py`
  (`python scripts/snapshot_prune_prototype.py <full_snapshot.txt> <out.txt>`).
- Measurement commands: `playwright-cli -s=<sess> snapshot` (full),
  `... snapshot --depth=6`, then the pruner on the full capture; char counts via
  `wc -c`, token counts via `tiktoken` cl100k.
- Browser processes from this research were torn down with `playwright-cli close` /
  `kill-all`.
