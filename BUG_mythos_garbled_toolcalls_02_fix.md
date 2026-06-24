# BUG (mythos #2/3 — FIX): align the harness to mythos_fast's native tool-call format

- **Branch:** `beta_mythos_fast` (ISOLATED — never merges back to `beta`).
- **Type:** implementation (fix the garbled tool calls).
- **Status:** BLOCKED — do not start until `BUG_mythos_garbled_toolcalls_01_analysis.md`
  is marked **COMPLETE**. The implementation MUST follow that analysis's confirmed root
  cause and recommendations, not this file's prompts alone.
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
- [ ] Implements the analysis's confirmed root-cause fix (cite the analysis sections).
- [ ] Tool definitions and the `<tool_call>` output instructions match the captured
      native format byte-for-byte where it matters (keys, tags, schema shape).
- [ ] Secondary constraint/encoding step disabled where the analysis directed (with a
      clear comment explaining why, referencing this item + the analysis).
- [ ] `tests/test_codec_hermes.py` (and any new tests) pass; full suite green.
- [ ] `vibe --print-system --agent web` shows the corrected, native-aligned system prompt.
- [ ] Changes committed directly to `beta_mythos_fast` (per task direction).

## Notes
- Do NOT touch `beta`-only behaviour or the `json`/other codecs' defaults.
- Keep the change minimal and reversible; the integration test (#3) is the empirical gate.
