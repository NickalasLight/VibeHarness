# beta_qwen3coder — live validation log (issue #125)

Durable state for the autonomous validation of `beta_qwen3coder` (model
`qwen2.5-coder:3b-instruct`, codec `hermes`) against the FlashTec careers mock site.
Append one entry per iteration. This file is my memory across context windows.

## Mission (autonomous; user away ~12h)
Run the web agent against the mock job site, **3-minute wall-clock cap per run** (force-stop),
analyze the `.vibe/` log, fix defects on `beta_qwen3coder`, and **iterate hard** until:
1. The Qwen model generates **on the NVIDIA RTX 3080** (not the AMD iGPU / CPU) — verified by
   `ollama ps` = 100% GPU and `nvidia-smi` VRAM occupancy during a run.
2. Tool calls **encode + parse cleanly** (valid `<tool_call>{"name","arguments"}</tool_call>`,
   no garbling, codec parses every turn).
3. The agent makes real progress through the job-application flow (navigate → fill → submit).

## Run protocol
- Command: `python -m vibeharness --agent web --task-file test_assets/application_instruction.md`
  (run from repo root on `beta_qwen3coder` so the branch's code + test_assets are used).
- Cap: external `timeout 180` (force-stop at 3 min). Harness still flushes per-turn `.vibe/` logs.
- After each run: capture `ollama ps` + `nvidia-smi`; read the newest `.vibe/` log; classify the
  failure; apply ONE focused fix; commit `area(#125): …`; record below.

## Environment baseline (captured at start)
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 8192 MiB, CUDA index 0. Also present: AMD Radeon
  iGPU (NOT CUDA-capable → Ollama CUDA targets the 3080; real risk is CPU fallback, cf. #119).
- Ollama 0.30.8 running. Mock site `http://localhost:3000/careers/...` → HTTP 200 (React SPA,
  client-rendered; form at `/apply`). `playwright-cli` installed.
- Config (branch defaults): `model=qwen2.5-coder:3b-instruct`, `codec=hermes`, `num_gpu=99`.
- Model pull: `qwen2.5-coder:3b-instruct` — IN PROGRESS at start (was not yet pulled).

## GPU health — CONFIRMED (✅ resolves the user's #1 concern)
Instrumented API gen of `qwen2.5-coder:3b-instruct`: **100% GPU** (`ollama ps`), **2236 MiB
VRAM** on the RTX 3080 (`llama-server.exe` compute app), **136 tok/s**, load 2.45s. The model
generates on the NVIDIA dGPU, NOT the AMD iGPU / CPU. (Aside: `ollama run <model> "..."` via a
non-TTY background shell HUNG with no runner — use the `/api/generate` API or the harness, not
bare `ollama run`, for scripted checks.)

## Monitor mechanism (the 3-min loop)
Each iteration = a BACKGROUND task `timeout 180 python -m vibeharness --agent web
--task-file test_assets/application_instruction.md --no-color`. `timeout` force-stops at 180s
(exit 124); the background-task **completion notification** is the trigger to analyze → fix →
relaunch (which re-arms the monitor). Per-turn `.vibe/` logs are flushed even on an abrupt cap.

## Iterations

### Iteration 1 — DONE (task b12ub0ql9, exit 2 = stopped after 15 turns, ~well under 3 min)
**Result: 0/15 turns executed — every tool call REJECTED by the codec.** Root cause is an
encoding/parse mismatch, NOT GPU or model: the model generated fast and produced the CORRECT
content — `{"name":"goto","arguments":{"url":".../apply"}}` — but wrapped it in a ```json
markdown fence with **no `<tool_call>` tags**. The strict `hermes` parser required the tags and
rejected all 15 turns ("no <tool_call>...</tool_call> blocks found"). `qwen2.5-coder:3b-instruct`
is an instruct model with no `<think>`/`<tool_call>` habit (unlike the mythos fine-tune), so it
emits fenced JSON. (Also seen: it put a candidate call in the reason channel too — the two-phase
reason/act split is awkward for a non-thinking model; noted as a future transport follow-up.)

**Fix applied (`hermes_codec.py`):** made `parse()` TOLERANT — when no `<tool_call>` block is
present, recover the `{"name","arguments"}` JSON from ```json fences / bare text / a top-level
array (string-aware brace scan). A valid call is no longer discarded over a missing wrapper tag.
Added regression tests (fenced / bare / array); `tests/test_codec_hermes.py` = 26 passed.
Commit: see git log `fix(#125): hermes parser tolerant of fenced/bare JSON`.

### Iteration 2 — DONE (task b2e0wsnnf) — BIG progress, two new issues found
**Tolerant parser worked: tool calls now PARSE + EXECUTE.** The agent navigated to `/apply`
and successfully filled First name, Email, Apt, Phone, City via `fill` on real refs. Encoding
is healthy. Two problems surfaced:
1. **Two-phase waste (transport):** the model emits a tool call in BOTH the phase-1 "thinking"
   channel (DISCARDED) and the phase-2 "action" channel (executed). qwen2.5-coder is a
   non-thinking instruct model, so phase 1 = a wasted real call. Worse, the model's combobox
   RECOVERY clicks (e.g. `click e65` to open the State dropdown) kept landing in the discarded
   thinking channel while the failing action ran — blocking recovery.
2. **Custom combobox loop (next iteration):** State `e82` is a `<div role="listbox">` custom
   combobox, NOT a native `<select>`; `select_option` fails ("Element is not a <select>"). The
   agent looped 7 turns retrying `select_option(e82,"TX")` and froze (last-name field also got
   muddled). `select_option` can't drive a custom combo — needs click-open + click-option.

**Fix applied (this iteration → issue #1, the higher-leverage one):** added SINGLE-phase
generation. `Config.two_phase=False` (branch default) → `decide()` does ONE native `/api/chat`
call and the codec parses the tool call from it; no discarded duplicate. `llm.py:_chat` added;
two-phase preserved for VibeThinker/mythos. Full suite **548 passed**. Commit: `fix(#125):
single-phase generation for non-thinking instruct model`.

### Iteration 3 — DONE (task bqas38w7b) — single-phase works; new issue = ref comprehension
**Single-phase confirmed:** exactly ONE action/turn now, no discarded phase-1 call (turns 1-2
nailed `open_browser`→`goto`). But the model **burned ~10 turns on wrong refs**: hallucinated
`e163` (×5 identical failing clicks), clicked headings `e34`/`e35`, then tried to `fill` the
last-name **label** `e42` (a `<div>`, not the input). It only landed First name="Jason" once
(turn 13). Root cause: from the RAW ARIA tree it can't tell input refs from label/wrapper refs.
(Two-phase iter 2 stumbled onto right refs via the prefill; single-phase exposes the underlying
snapshot-comprehension weakness — the transport change is still correct.)

**Fix applied:** `Config.web_snapshot_prose=True` (branch default). The auto-injected page
snapshot now uses the deterministic WebArena-style ARIA→prose transform (#64) — pruned, one
ref-keyed line per interactable, with fillable affordances (#70) — which small models map to
the correct ref far more reliably. Updated 2 prose tests to the branch default; suite **548
passed**. Commit: `fix(#125): default web_snapshot_prose=True for ref comprehension`.

### Iteration 4 — IN PROGRESS
Re-run with prose snapshot. Watching for: fewer wasted/looped turns; correct input refs chosen
(input not label); more fields filled per turn; progress toward Next/step 2. Still pending:
custom State combobox handling (select_option fails on the div-listbox).

## Current status
RUNNING iteration 4 (3-min cap). Awaiting completion notification.
