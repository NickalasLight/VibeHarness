# Snapshot → Natural-Language: Analysis (issue #61)

**Branch:** `research/snapshot-to-nl` · **Status:** analysis only, no production code
**Question:** VibeThinker-3B struggles to reason over Playwright ARIA snapshots (roles / `ref`s / nesting). Is there a documented, straightforward way to turn the snapshot into a *human-readable* page description so the model can decide what to click — e.g. *"a white form in the foreground with two black buttons labeled **Reject all** and **Accept all**; next to them a white button **More options**…"*?

---

## 1. What our snapshot actually is (ground truth)

We capture the page with the Playwright **Agent CLI** `snapshot` command (`vibeharness/web.py`,
`capture_page_snapshot_raw`). It returns Playwright's **AI-optimized ARIA snapshot** — a YAML
accessibility tree with stable element refs — wrapped in a small header. It is injected verbatim
into the per-turn system prompt under `# Current page (live snapshot)` (`vibeharness/prompt.py:120`).

A **real** captured snapshot — the YouTube consent page that motivates the example — lives at
`C:\git\vh-ashley38\.vibe\20260624_074859-diagnostics\turn-003-snapshot-20260624_075650_524351.txt`
(44,742 chars). Its consent dialog is literally the example scenario:

```yaml
- dialog "Before you continue to YouTube" [active] [ref=e1206]:
    - heading "Before you continue to YouTube" [level=2] [ref=e1241]
    - generic [ref=e1242]:
        - generic [ref=e1245]:
            - text: We use
            - link "cookies" [ref=e1246] [cursor=pointer]: { /url: https://policies.google.com/... }
            - text: and data to
        - list [ref=e1247]: …
    - generic [ref=e1289]:
        - generic [ref=e1290]:
            - button "Reject the use of cookies and other data for the purposes described" [ref=e1293] [cursor=pointer]:
                - generic [ref=e1294]: Reject all
            - button "Accept the use of cookies and other data for the purposes described" [ref=e1300] [cursor=pointer]:
                - generic [ref=e1301]: Accept all
        - link "More options" [ref=e1308] [cursor=pointer]:
            - /url: https://consent.youtube.com/d?...
            - generic [ref=e1309]: More options
```

### What the ARIA snapshot DOES contain
- **Role** per node — `dialog`, `button`, `link`, `heading`, `list`, `listitem`, `table`, `row`, `cell`, `combobox`, `slider`, `img`, `text`, and the very common semantically-empty `generic`.
- **Accessible name** — the quoted string (`button "Accept the use of cookies…"`). Note this is often the verbose ARIA label, while the *visible* text is a nested `generic: Accept all`.
- **Structure / nesting** — indentation encodes parent→child containment (the dialog contains the buttons).
- **Stable `ref`** (`[ref=e1300]`) — the handle our `browse` actions act on (click/fill `target`).
- **A small fixed set of ARIA state attributes** — per the Playwright docs, only: `checked, disabled, expanded, invalid, level, pressed, selected`. Plus Playwright extras seen in our data: `[active]` (the active/foreground dialog), `[cursor=pointer]`, `[expanded]`, `[disabled]`, `[level=N]`.
- **`/url`** for links, and raw text nodes.

### What it does NOT contain (the crux of the example)
The Playwright ARIA snapshot is the accessibility tree — *semantic structure only*. It has **no**:
- colors / background / `color` / borders (no "white form", no "black buttons"),
- font, size, or any CSS,
- bounding boxes, x/y coordinates, on-screen vs off-screen,
- z-index / true foreground ordering (the **closest** signal is `[active]` on the modal `dialog`),
- viewport/scroll position.

> **Therefore the literal example — "white form / black buttons / in the foreground" — is NOT derivable from the ARIA tree alone.** Producing those specific phrases requires *more*: either computed styles + geometry via `page.evaluate`, or a screenshot + a vision model. Every option below is explicit about what it can and cannot produce. (Confirmed against the official Playwright docs, which list the supported attributes and mention no visual/geometry data.)

---

## 2. Research / prior art (cited)

- **Playwright ARIA snapshots** — YAML tree of roles + accessible names + a fixed ARIA-attribute set + AI `ref`s; no styling/geometry. [playwright.dev/docs/aria-snapshots](https://playwright.dev/docs/aria-snapshots), [microsoft/playwright docs source](https://github.com/microsoft/playwright/blob/main/docs/src/aria-snapshots.md)
- **WebArena** — the canonical text observation for web agents IS the **accessibility tree** parsed by Playwright, "with rule-based post-processing such as merging consecutive text elements to improve readability." It is chosen precisely because it is *more compact than HTML* and "retains the structured information." This validates option 1 (rule-based) as the field-standard text representation. [arxiv 2307.13854](https://arxiv.org/pdf/2307.13854), [webarena resources](https://github.com/web-arena-x/webarena/blob/main/resources/README.md)
- **Mind2Web** — does NOT narrate; ranks candidate DOM elements with a small classifier, then feeds a pruned element snippet to the LLM. Lesson: for small models, **filtering/pruning the tree** matters as much as prose. [arxiv 2306.06070](https://arxiv.org/html/2306.06070v3)
- **Set-of-Marks (SoM) / WebVoyager** — overlay numbered boxes on a *screenshot* so a multimodal model refers to "element #5" rather than a selector. Directly analogous to keeping our `ref`s, but for the *visual* channel; requires a vision model. [WebVoyager arxiv 2401.13919](https://arxiv.org/html/2401.13919v1), [SoM origin (Set-of-Mark)](https://arxiv.org/html/2410.05243v1)
- **Observation-reduction literature** — "Read More, Think More" and related work show compacting/reducing the a11y observation improves agent performance — relevant because our snapshots hit 44k+ chars. [arxiv 2604.01535](https://arxiv.org/pdf/2604.01535)
- **Accessibility-tree → speech** — screen readers ARE the canonical "a11y tree → linear narration" engines (role + name + state read aloud in DOM order). There is **no widely-used off-the-shelf library that renders an a11y tree to descriptive English prose** — narration is either screen-reader-style linearization (deterministic, what option 1 reproduces) or LLM-generated. [web.dev: the accessibility tree](https://web.dev/articles/the-accessibility-tree), [A List Apart: Semantics to Screen Readers](https://alistapart.com/article/semantics-to-screen-readers/)
- **Local vision model** — `qwen2.5vl:7b` is already installed in our Ollama (confirmed via `ollama list`), enabling screenshot narration with zero new infra.

**Bottom line from the literature:** the *documented, standard, straightforward* representation for LLM web agents is the **rule-based accessibility tree** (WebArena), optionally **pruned** (Mind2Web) and optionally paired with a **marked screenshot** (SoM/WebVoyager). There is no documented turnkey "ARIA-YAML → descriptive prose" tool; descriptive prose is a custom rule-based or LLM step.

---

## 3. Options

### Option 1 — Deterministic rule-based ARIA → prose
Walk the YAML tree, emit one clause per actionable/named node: role + visible name + state + ref. Drop `generic`/`img` noise, collapse wrappers, group siblings, render the `[active]` dialog as "a dialog in front."

- **Straightforwardness / documentation:** High. This is exactly WebArena's approach (well-documented, field-standard). Pure string/tree transform, no model call.
- **Determinism:** Fully deterministic.
- **Local cost / latency:** ~0 ms, no GPU.
- **Closeness to example:** Partial. Produces *"A dialog 'Before you continue to YouTube' is in the foreground. It contains buttons **Reject all** and **Accept all**, and a link **More options**."* — i.e. structure + labels + which-is-modal, but **never** colors ("white/black") or precise spatial layout. Can say "in the foreground" honestly only via `[active]`.
- **Refs:** Trivially kept inline (`… **Accept all** [ref=e1300]`). Actions keep working.

### Option 2 — ARIA enriched with computed styles + bounding boxes (`page.evaluate`)
For each ref'd node, run JS to read `getBoundingClientRect()`, `getComputedStyle()` (background-color, color, visibility, z-index), and visibility. Merge into the prose so it can say color, on-screen position, and a geometry-based "foreground."

- **Straightforwardness / documentation:** Medium. `page.evaluate` / `getComputedStyle` are well documented, but there is **no documented end-to-end recipe** — we'd build the mapping (rgb→"white/black", rect→"top-right", z-index/`elementFromPoint`→"foreground") ourselves, and re-resolve each `ref` to a DOM node. Real-world CSS (gradients, pseudo-elements, semi-transparent overlays, theming) makes "is it white?" genuinely fuzzy. Our `browse` tool already exposes `eval`, so the seam exists.
- **Determinism:** Deterministic given a fixed page, but brittle across sites/themes; color-naming heuristics are arbitrary.
- **Local cost / latency:** No GPU, but one extra JS round-trip per snapshot (10s–100s ms; bounded by `web_cli_timeout`). Modest engineering + maintenance cost.
- **Closeness to example:** Highest of the deterministic options — this is the **only** path that can literally output "white form… black buttons… in the foreground" *without* a vision model.
- **Refs:** Native — we are already keyed by `ref`, just attach attributes.

### Option 3 — Screenshot + local vision model (`qwen2.5vl:7b`)
Take a screenshot (we already support `screenshot`), send to local `qwen2.5vl:7b` via Ollama, ask for a description. Optionally SoM-overlay numbered marks tied to refs.

- **Straightforwardness / documentation:** Medium. Vision-narration and SoM are well-documented in the agent literature; Ollama vision API is simple. But aligning the vision model's "button 3" back to a Playwright `ref` requires SoM overlay work (map ref → bounding box → drawn number), which is real engineering.
- **Determinism:** Non-deterministic (sampled generation); may hallucinate, miss small controls, or misread text.
- **Local cost / latency:** Heaviest. `qwen2.5vl:7b` is ~6 GB and competes with VibeThinker-3B for the same RTX 3080 Laptop VRAM; expect seconds per turn and possible model-swap thrash. A 7B describer may also be weaker at exact label text than the deterministic tree.
- **Closeness to example:** Highest overall — colors, layout, and true visual foreground come "for free" because it literally sees pixels. This is the only option that can match the *style* of the example unprompted.
- **Refs:** **The hard problem.** A raw screenshot has no refs. You MUST add SoM marks (ref↔number map) or the model's description is un-actionable. Best used *alongside* the ref'd tree, not instead of it.

### Option 4 — LLM summarization of the ARIA tree
Feed the raw ARIA YAML to a local LLM and ask for natural-language prose.

- **Straightforwardness / documentation:** High to wire up (one prompt), but no special documentation.
- **Determinism:** Non-deterministic; can hallucinate elements/labels and — worse — **invent or drop refs**, silently breaking actions.
- **Local cost / latency:** A second local LLM pass per turn. If we use VibeThinker-3B itself, it's the same model that already "struggles" with the tree (circular); a stronger summarizer means more infra.
- **Closeness to example:** Can *imitate* the style ("there is a white form…") but would be **fabricating** the colors — they aren't in the input. That is actively harmful: confident, wrong visual claims.
- **Refs:** Fragile — relies on the LLM faithfully copying `[ref=eNNNN]` through the rewrite.

---

## 4. Tradeoffs table

| Option | Straightforward / documented | Deterministic | Local cost / latency | Can produce "white/black/foreground"? | Refs preserved (actionable)? |
|---|---|---|---|---|---|
| **1. Rule-based ARIA→prose** | **High** — = WebArena standard | **Yes** | **~0 / none** | Structure + "foreground" via `[active]` only; **no colors** | **Yes**, trivially inline |
| **2. ARIA + computed styles/boxes** | Medium — build it ourselves | Yes (brittle heuristics) | Low (1 JS round-trip) | **Yes** — colors, position, z-index foreground | **Yes**, native (keyed by ref) |
| **3. Screenshot + qwen2.5vl** | Medium — SoM linking is work | No | **High** (6 GB VRAM, sec/turn, VRAM contention) | **Yes**, best/most natural | **Only with SoM overlay**; raw screenshot = no refs |
| **4. LLM summarize tree** | High to wire, no docs | No | Medium (extra LLM pass) | Only by **fabricating** colors (harmful) | **Fragile** — may drop/invent refs |

---

## 5. Concrete sample output for our real snapshot

Source: the consent dialog in `turn-003-snapshot-...txt` above.

### Under Option 1 (rule-based — recommended baseline), the model would see:

```
# Current page (live snapshot) — described

You are on YouTube: "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)".

A dialog titled "Before you continue to YouTube" is OPEN IN FRONT of the page and is
blocking it. It explains that Google uses cookies and data, and offers these choices:
  • Button "Reject all"  → ref e1300 ... (rejects extra cookies)        [ref=e1293]
  • Button "Accept all"  → accepts all cookies                          [ref=e1300]
  • Link  "More options" → opens detailed privacy settings              [ref=e1308]
  • Button "Language: English"                                          [ref=e1219]
  • Link  "Sign in"                                                     [ref=e1231]

Behind the dialog (currently blocked): a search box [ref=e104], "Sign in" [ref=e118],
the video player with Pause [ref=e28] / Mute [ref=e31] / Settings [ref=e38] /
Full screen [ref=e41], a "Subscribe to Rick Astley" button [ref=e197], Like [ref=e214],
Share [ref=e236], Save [ref=e248], and a long list of recommended videos.

To proceed, clear the dialog first: click "Reject all" (e1293) or "Accept all" (e1300).
```
*Honest about everything in the tree (labels, modal-in-front, refs) — says nothing about
color because the tree doesn't know. Directly actionable.*

### Under Option 2 (ARIA + computed styles), the SAME dialog could read:

```
A form-like dialog (white background, ~560px wide, centered, z-index above the page,
[active] = in the foreground) titled "Before you continue to YouTube". Near its bottom,
two buttons sit side by side: a dark/black button "Reject all" (ref e1293) and a dark/black
button "Accept all" (ref e1300); to their right, a white-background link "More options"
(ref e1308). The page behind it is dimmed and non-interactive.
```
*This is the only deterministic way to reproduce the literal example phrasing. It requires
our own rgb→name + rect→position + z-index→foreground heuristics layered on option 1.*

---

## 6. Recommendation

**Primary: build Option 1 (deterministic rule-based ARIA→prose) as the default**, keeping `ref`s
inline, and make it the snapshot renderer that feeds `# Current page (live snapshot)`.

Rationale:
- It is the **documented, field-standard** representation (WebArena uses exactly this) and the
  *straightforward* answer to the question — no GPU, no new model, fully deterministic, and it
  removes the YAML/indentation/`generic` noise that VibeThinker-3B trips over while **preserving
  the refs** the model must act on.
- It honestly conveys the one piece of "foreground" the tree truly has (`[active]` dialog),
  which is the single most important fact for the consent-banner use case (#61's real pain point).
- Pair it with light **pruning** (drop `generic`/`img`/decorative nodes, collapse wrappers,
  prefer visible nested text over verbose ARIA labels) — cheap, and the literature (Mind2Web /
  observation-reduction) says this helps small models most.

**Do NOT chase the literal "white/black" wording first.** Those words require Option 2 or 3 and
add brittleness/cost for a cosmetic gain; the agent needs *which control to click*, not its color.

**Secondary / phase 2 (optional):** add **Option 2** as an *enrichment toggle* — a single
`page.evaluate` that attaches background/text color + bounding box + z-index to each ref, merged
into the option-1 prose. This is the only deterministic path to the exact example phrasing and
reuses our existing `eval` seam and ref keying. Gate it behind a config flag; it's strictly
additive over option 1.

**Keep Option 3 (qwen2.5vl screenshot) on the bench** as a fallback for pages the ARIA tree
represents poorly (canvas/WebGL apps, image-only buttons). If adopted, it must use **SoM overlay**
so the vision description stays tied to actionable refs, and run *alongside* (not replacing) the
ref'd tree. Defer due to VRAM contention with VibeThinker-3B on the single RTX 3080 Laptop.

**Reject Option 4 (LLM-summarize-the-tree)** as the primary path: it would fabricate the very
colors the example asks for (they're not in the input) and can silently corrupt refs — high risk,
no determinism, and it leans on the same small model that already struggles.

> One-line answer to #61: *There is no turnkey "Playwright-snapshot → descriptive prose" tool, but
> the documented standard (WebArena-style rule-based accessibility-tree narration) is the
> straightforward win — adopt that, keep the refs inline, and add an optional computed-style
> enrichment only if we genuinely need "white form / black buttons / foreground" wording.*

---

### Sources
- [Playwright ARIA snapshots](https://playwright.dev/docs/aria-snapshots) · [docs source](https://github.com/microsoft/playwright/blob/main/docs/src/aria-snapshots.md)
- [WebArena paper](https://arxiv.org/pdf/2307.13854) · [WebArena resources](https://github.com/web-arena-x/webarena/blob/main/resources/README.md)
- [Mind2Web](https://arxiv.org/html/2306.06070v3)
- [WebVoyager](https://arxiv.org/html/2401.13919v1) · [Set-of-Mark / visual grounding](https://arxiv.org/html/2410.05243v1)
- [Observation reduction for web agents](https://arxiv.org/pdf/2604.01535)
- [The accessibility tree (web.dev)](https://web.dev/articles/the-accessibility-tree) · [Semantics to Screen Readers (A List Apart)](https://alistapart.com/article/semantics-to-screen-readers/)
</content>
</invoke>
