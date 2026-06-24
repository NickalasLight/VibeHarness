# MYTHOS_DIVERGENCE.md — beta ↔ beta_mythos_fast governance

`beta_mythos_fast` is a **long-term isolated branch** for the mythos_fast fine-tune line.
It diverges from `beta` in **intentional, load-bearing ways** (the fine-tune's value
depends on them). A blind `git merge beta` would clobber these and defeat the fine-tune.

**Rules (see also issues #107, #105, #36, and the project memory):**
- Sync is **one-way**: `beta → beta_mythos_fast` only. **Never** merge `beta_mythos_fast` into `beta`.
- Do **not** blind-merge. Apply each desired beta fix carefully, preserving the protected
  files/areas below. Prefer the `mythos-sync` agent (`.claude/agents/mythos-sync.md`).
- Any PR into `beta_mythos_fast` must confirm this manifest was reviewed (enforced by
  `.github/workflows/mythos-divergence-check.yml` — include the token `DIVERGENCE-REVIEWED`
  in the PR body once you've checked the protected files below).

## Protected files / areas (keep the mythos-specific version)

> **FINALIZED** (#107) after #105/#117 landed. The set below is verified against the
> actual code on `beta_mythos_fast`. The divergence is small and surgical: it is the
> hermes tool-call **format** (codec + `<tools>` definition seam) and the two `config.py`
> **default values** that point the harness at the fine-tune. Everything else — including
> the chat template — is shared with `beta` and takes beta changes normally.

| Path / area | Rule | Why | Concrete divergence (verified) |
|---|---|---|---|
| `vibeharness/codecs/hermes_codec.py` | **keep-mythos** (mythos-only file) | the Hermes/Qwen2.5 native tool-call codec (#105/#117/#35) | whole file is mythos-only; `beta` has no such file. Audited SOUND. |
| `vibeharness/registry.py` → `ToolRegistry.tools_block(style="hermes")` | **keep-mythos FORMAT**; reconcile other logic | renders the `<tools>` OpenAI-nested function-schema block the fine-tune was trained on, built from each tool's `_args_schema()` (same source as the JSON constraint → no drift) | the `tools_block()` method is mythos-only (`beta` has only `docs()`). Preserve it; merge any new `beta` registry logic around it. |
| `vibeharness/prompt.py` → `SystemPromptBuilder.build` `<tools>` substitution | **keep-mythos FORMAT**; reconcile other logic | substitutes the codec's `tool_definitions()` (`<tools>` block) for the Markdown `registry.docs()` when the codec supplies one; all other codecs return `None` and render unchanged | the `docs = self._codec.tool_definitions(...) or self._registry.docs()` lines are mythos-only. Preserve them; merge other prompt changes normally. |
| `vibeharness/codec.py` → `ToolCallCodec.tool_definitions(registry)` seam | **keep-mythos FORMAT**; reconcile other logic | base default returns `None` (Markdown docs); the hermes codec overrides it. This is the open/closed hook the `<tools>` rendering rides on | the `tool_definitions()` base method (+12 lines) is mythos-only. Preserve; merge other codec-base changes normally. |
| `vibeharness/config.py` defaults — `model`, `codec` | **keep-mythos VALUES**; merge new keys | branch defaults that point at the fine-tune | ONLY two lines differ from `beta`: `model = "hf.co/Shadow0482/mythos_fast:Q6_K"` (beta: `"vibethinker"`) and `codec = "hermes"` (beta: `"json"`). ALL other config keys (`web_session`, `num_ctx`, token budgets, `web_snapshot_prose`, snapshot-budget knobs…) are IDENTICAL to beta — merge new beta keys normally. |
| `vibeharness/llm.py` → `_render_chatml` / chat template | **NOT diverged — takes beta changes normally** | per #105 the ChatML template was left UNCHANGED from beta; the native alignment is achieved by the codec + `<tools>` format, not by a custom template | `llm.py` is byte-identical between `beta` and `beta_mythos_fast`. NOT protected. Listed here only to record that #105 did not touch it. |
| any hermes-aligned prompt / tool-description text | keep-mythos | training-format alignment | the hermes codec's `format_instructions` / `turn_action_hint` (inside `hermes_codec.py`, already covered above). No other prompt text diverges. |

## Merge procedure (per beta fix)
1. Identify the beta change(s) and the files they touch.
2. For each file NOT in the table → cherry-pick / apply directly.
3. For each file in the table → manually integrate the *logic* of the beta change while
   preserving the mythos-specific format/values; never overwrite wholesale.
4. Run the suite; document each protected-file decision in the commit/PR.

_Last updated: 2026-06-24 (#107 FINALIZED after #105/#117 landed; verified against code on `beta_mythos_fast`)._
