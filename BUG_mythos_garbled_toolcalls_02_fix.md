# BUG (mythos #2/3 — FIX): align the harness to mythos_fast's native tool-call format

- **Branch:** `beta_mythos_fast` (ISOLATED — never merges back to `beta`).
- **Type:** implementation (fix the garbled tool calls).
- **Status:** COMPLETE — implemented per the mythos #1 analysis's confirmed root cause and
  recommendations (see `## IMPLEMENTATION`). All edits are on `beta_mythos_fast` only.
- **Depends on:** `BUG_mythos_garbled_toolcalls_01_analysis.md` (the analysis text file).
- **Gates:** `BUG_mythos_garbled_toolcalls_03_integration_test.md`.

---

## Goal

Make `hf.co/Shadow0482/mythos_fast:Q6_K` emit clean, parseable tool calls by aligning the
harness to the model's **native trained dialect** (see the captured ground truth in the
analysis item), and by **removing/disabling the bolted-on secondary constraint/encoding
step** IF — and only if — the analysis confirmed it as a (the) cause.

## Scope of changes (final shape DRIVEN BY the analysis findings)

1. **System prompt** — update so the active (`hermes`) path matches the native template:
   lean system message + the auto-appended `# Tools` block + the exact
   `<tools></tools>` / `<tool_call>` instruction wording the template uses. Strip or
   minimize harness scaffolding that pushes the model off-distribution.
2. **Tool descriptions / definitions** — reconcile `registry.tools_block` to the **bare**
   `tool | tojson` schema shape `{"name","description","parameters"}` that the live
   template renders (the orchestrator confirmed the current code emits the OpenAI-nested
   `{"type":"function","function":{…}}` wrapper instead). Use the analysis's final call on
   which shape the GGUF actually expects; keep one source of truth with `_args_schema()`.
3. **Constraint / encoding step** — if the analysis confirms it: disable the secondary
   constraint/encoding pass for this model end-to-end (config default, CLI resolution,
   any stale saved settings, and `OllamaClient._act`). Ensure the `hermes` path is truly
   unconstrained and, if recommended, route generation through the model's **native chat
   template** (e.g. Ollama `/api/chat` with a real `tools=[]` array) instead of the
   hand-rolled `raw=True` ChatML two-phase prefill.
4. **Protected-file discipline** — several targets are in the `MYTHOS_DIVERGENCE.md`
   protected set (`hermes_codec.py`, the `<tools>` seam in `prompt.py`/`registry.py`,
   `config.py` defaults, the chat template in `llm.py`). Preserve the mythos format;
   integrate logic carefully; never blind-overwrite. Include the `DIVERGENCE-REVIEWED`
   token in any PR body per the governance workflow.

## Acceptance
- [x] Implements the analysis's confirmed root-cause fix (cite the analysis sections):
      Suspect 3 / Recommendation #2 (bare tool-definition shape), Recommendation #4 +
      Suspect 4 (slim/native system-prompt instruction wording), Recommendation #5 +
      Suspect 2 (keep the JSON-schema constraint OFF, fix the stale docstring).
- [x] Tool definitions and the `<tool_call>` output instructions match the captured
      native format byte-for-byte where it matters (keys, tags, schema shape): `<tools>`
      lines are now BARE `{"name","description","parameters"}`; `format_instructions`
      leads with the template's verbatim "For each function call, return a json object …
      within <tool_call></tool_call> XML tags:" wording.
- [x] Secondary constraint/encoding step: confirmed already OFF for hermes and kept OFF,
      end to end, with comments referencing this item + the analysis. (The analysis's
      decision: "there is nothing active to disable" — `json_schema=None` already inert.)
- [x] `tests/test_codec_hermes.py` updated for the bare shape (and any new tests) pass;
      full suite green — **544 passed**.
- [x] `vibe --print-system --agent web` shows the corrected, native-aligned system prompt
      (bare `<tools>` block + native `<tool_call>` instruction wording). Verified.
- [x] Changes committed directly to `beta_mythos_fast` (per task direction).

## Notes
- Do NOT touch `beta`-only behaviour or the `json`/other codecs' defaults.
- Keep the change minimal and reversible; the integration test (#3) is the empirical gate.

## IMPLEMENTATION

### Transport decision (the key call) — Path B: in-prompt native alignment, NOT a full `/api/chat tools=[]` rewrite

The mythos #1 analysis ranks the **two-phase hand-rolled `raw=True` transport** as the
PRIMARY cause and recommends (P0) replacing it with a single native Ollama `/api/chat`
pass that passes a real `tools=[]` array and lets the model's embedded template render
itself. The task briefing authorises P0 **only if it can be done without breaking the
codec seam and the suite**, otherwise the "safest partial alignment."

I chose **Path B (safest partial alignment)**, deliberately, because the full P0 rewrite
is not contained:

- `LLMClient.decide(system, user, constraint, …)` receives only rendered strings — it has
  no registry/tool-schema handle, so a real `tools=[]` array can't be threaded without
  changing the public `LLMClient` interface (used by every codec, the agent loop, tests).
- The model's native multi-turn shape feeds prior results back as
  `<tool_response>` (`tool`-role) turns and the model's own prior `<tool_call>`s as
  assistant `tool_calls`. The harness instead renders history as an English NL narrative
  inside ONE synthetic user turn (`NarrativeMemory`, `agent.py`). A genuine native
  transport would require rewriting memory into role-typed messages — a large, risky
  change spanning the agent loop, well beyond a minimal/reversible fix.
- A *half* native transport (single `/api/chat` with `tools=` but history still a single
  NL user turn) would still be off-distribution on history while adding real interface
  risk — poor risk/reward for this stage.

Path B instead makes the harness speak the native format **within the existing seam**, so
it is minimal, fully reversible, and breaks nothing: it fixes the two confirmed,
self-contained contributors (tool-definition shape + instruction wording) and ensures the
model sees its trained `<tools>`/`<tool_call>` cue even though the `raw=True` ChatML
transport means the embedded template never auto-fires.

**Tradeoff / what Path B does NOT do:** it does not collapse the phase-1/phase-2 split,
does not pass a real `tools=[]` array, and does not convert history to `<tool_response>`
turns. Those remain the highest-value follow-up (the P0 transport rewrite) and are the
right scope for a dedicated transport issue once #3 quantifies the residual gap. The #3
integration test is the empirical gate: if bare-shape + native wording alone clears the
garbling, the transport rewrite may be unnecessary; if not, #3's evidence justifies the
larger interface change.

### Files / functions changed

1. `vibeharness/registry.py` → `ToolRegistry.tools_block(style="hermes")` (≈ lines 47-82):
   emit each tool as **bare** `{"name","description","parameters"}` (was the OpenAI-nested
   `{"type":"function","function":{…}}`). Still built from `t._args_schema()` — the single
   source of truth shared with the JSON constraint, so docs/schema cannot drift. Docstring
   corrected (it previously documented the nested shape as native).

2. `vibeharness/codecs/hermes_codec.py`:
   - Module docstring (lines ~1-27): corrected the wrong "OpenAI-nested" claim to **bare**,
     with a BUG #2 note citing the analysis's live template capture.
   - `format_instructions` (≈ lines 49-72): now LEADS with the template's verbatim native
     instruction wording ("You may call one or more functions…", "For each function call,
     return a json object … within <tool_call></tool_call> XML tags:" + the
     `{"name": …, "arguments": …}` exemplar), then the harness batching guidance. Keeps the
     model on its trained single-pass distribution since the embedded template never fires
     under the `raw=True` transport.
   - `tool_definitions` docstring: updated to say BARE shape.
   - `constraint` (≈ lines 86-96): added a BUG #2 comment — keep UNCONSTRAINED; do NOT
     re-introduce a `format` schema (it would fight the native dialect).

3. `vibeharness/llm.py`:
   - Module docstring (lines ~1-22): rewritten — the old text claimed phase 2 is ALWAYS
     JSON-schema-constrained (the stale always-`json` design). Now states constraint is
     codec-driven and is OFF for hermes, citing BUG #2 + the analysis, and notes the
     native-transport upgrade as a tracked follow-up.
   - `Decision.action_json` comment: clarified "constrained only if the codec supplies a
     json_schema."

4. `vibeharness/cli.py` → `--print-system` branch (≈ lines 605-628): resolve the **active**
   codec via `resolve_config(args)` (Config default `hermes` on this branch) and pass it +
   the resolved Config into `SystemPromptBuilder`, so `--print-system` shows the
   codec-native prompt the model actually receives (bare `<tools>` + native instructions)
   instead of the `json`-codec Markdown fallback. Respects saved settings and `--codec`.

5. `tests/test_codec_hermes.py`: `test_tools_block_lines_are_bare_function_schemas`
   (renamed from `…_openai_function_schemas`) now asserts the bare shape and the ABSENCE of
   `type`/`function` keys; `test_tools_block_parameters_match_args_schema` reads
   `parameters` from the bare top level. Existing `format_instructions` assertions
   (`<tool_call>`, `"name"`, `"arguments"`, "at most 4 tool calls") still hold under the new
   wording.

### Constraint status (kept OFF, end to end)
- `config.py` default `codec = "hermes"` (unchanged) → `cli.py resolve_config` →
  `hermes_codec.constraint` returns `DecodeConstraint(json_schema=None)` →
  `llm._act` only sets `payload["format"]` when `json_schema is not None`. No `format` is
  ever sent for hermes. No code path was disabled (nothing was active); the only change is
  documentation/comments making this explicit per the analysis.

### Protected-file discipline (MYTHOS_DIVERGENCE.md)
- `hermes_codec.py`, the `<tools>` seam in `registry.py`/`prompt.py`, and `config.py`
  defaults are protected for their mythos FORMAT/VALUES. I changed the hermes `<tools>`
  format from nested→bare — this is the intended point of this item (the analysis confirmed
  the bare shape IS the model's true native format; nested was a documented-but-wrong
  assumption). `config.py` default values (`model`, `codec`) are UNCHANGED. The `json` and
  other codecs are untouched. No beta-only behaviour changed.

### Verification
- `python -m pytest -q` → **544 passed**.
- `python -m vibeharness --print-system --agent web` → prints the native-aligned prompt:
  bare `<tools>` lines and the native `<tool_call>` instruction wording (captured above).

### Follow-ups for #3 (integration test)
- #3 is the empirical gate. If live mythos_fast runs still garble, the next step is the P0
  transport rewrite (native `/api/chat` + real `tools=[]` array, collapse the two-phase
  split, history as `<tool_response>` turns) — out of scope here for risk/seam reasons.
- Watch specifically whether the model now emits clean `<tool_call>` blocks given the bare
  `<tools>` shape; the analysis predicted the nested shape "mis-frames every tool."
