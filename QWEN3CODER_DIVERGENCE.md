# QWEN3CODER_DIVERGENCE.md — beta ↔ beta_qwen3coder governance

`beta_qwen3coder` is a **long-term isolated branch** for the ~3B Qwen-Coder line (issue
#123). It diverges from `beta` in **intentional, load-bearing ways** (the model swap's
value depends on them). A blind `git merge beta` would clobber these and defeat the
model alignment.

**Rules (see also CLAUDE.md §6 and issue #123):**
- Sync is **one-way**: `beta → beta_qwen3coder` only. **Never** merge `beta_qwen3coder`
  into `beta` (or into `main` / `beta_mythos_fast`).
- Do **not** blind-merge. Apply each desired beta fix carefully, preserving the protected
  files/areas below. The `mythos-sync` agent (`.claude/agents/mythos-sync.md`) is the
  pattern to mirror; a dedicated `qwen3coder-sync` agent is a noted follow-up.
- Generic harness fixes belong on `beta` FIRST, then sync outward — never fix a shared
  bug here and expect it to reach the other lines.
- A `qwen3coder-divergence-check` Action (mirroring
  `.github/workflows/mythos-divergence-check.yml`, requiring a `DIVERGENCE-REVIEWED`
  token in PR bodies into this branch) is a **noted follow-up** — not yet wired.

## ⚠️ 3B-PARITY (the load-bearing reason this branch exists)

There is **no dense ~3B Qwen3-Coder** — the Qwen3-Coder line is MoE only (smallest
30B-A3B = 30B total / 3B active, then 80B-A3B "Next", then 480B-A35B). Those break
apples-to-apples parity with the 3B-dense VibeThinker-3B and won't fit the 8 GB GPU. So
this branch uses **`qwen2.5-coder:3b-instruct`** (a true ~3B DENSE coder; VibeThinker
itself derives from Qwen2.5-(Coder-)3B) as the flagged, documented substitute. Keeping
the model in the 3B-dense class is the WHOLE POINT — do not "upgrade" the default to a
30B/80B/480B Qwen3-Coder MoE; that silently destroys the comparison. See
`QWEN3CODER_ANALYSIS.md` §A. Revisit only if a dense ~3B Qwen3-Coder ships.

## Protected files / areas (keep the qwen3coder-specific version)

| Path / area | Rule | Why |
|---|---|---|
| `vibeharness/config.py` defaults — `model = qwen2.5-coder:3b-instruct`, `codec = hermes` | keep-qwen3coder values; merge new keys | branch model + dialect defaults (#123) |
| `vibeharness/codecs/hermes_codec.py` | keep-qwen3coder (model-line file) | the Qwen2.5/Hermes native `<tool_call>{"name","arguments"}` + `<tools>` codec |
| `vibeharness/registry.py` `tools_block(style="hermes")` | keep-qwen3coder; reconcile other logic | the bare `<tools>` function-schema rendering the model's template expects |
| tool-definition seam in `vibeharness/prompt.py` (`SystemPromptBuilder` codec `tool_definitions()` substitution) | keep-qwen3coder format; reconcile other logic | routes tools to `<tools>` instead of Markdown for this model |
| `vibeharness/codec.py` `ToolCallCodec.tool_definitions()` hook | additive seam; keep | enables the per-codec tool-definition rendering |
| `vibeharness/llm.py` two-phase docstring + codec-driven `format` gating | keep-qwen3coder wording | phase-2 is UNCONSTRAINED for `hermes` (json_schema=None) |
| `Modelfile` (`FROM qwen2.5-coder:3b-instruct`) | keep-qwen3coder | branch model |
| `QWEN3CODER_ANALYSIS.md`, `QWEN3CODER_DIVERGENCE.md`, README qwen3coder sections | keep-qwen3coder | this line's docs |
| `tests/test_codec_hermes.py` | keep-qwen3coder | codec contract tests |

> NOTE: most of the seams above (`tool_definitions` hook, `tools_block`, the prompt
> substitution, the `hermes` codec) are GENERIC harness machinery that also exist on
> `beta_mythos_fast`. They are additive and conflict-free; the truly branch-SPECIFIC
> divergences are the **config defaults** (`model`/`codec`) and the **`Modelfile`**.

## Merge procedure (per beta fix)
1. Identify the beta change(s) and the files they touch.
2. For each file NOT in the table → cherry-pick / apply directly.
3. For each file in the table → manually integrate the *logic* of the beta change while
   preserving the qwen3coder-specific format/values; never overwrite wholesale. In
   particular, NEVER let a sync revert the `model`/`codec` defaults or swap the model out
   of the 3B-dense class.
4. Run the suite (`python -m pytest -q`); document each protected-file decision in the
   commit/PR.

_Last updated: 2026-06-24 (#123 — initial fork of `beta` to the Qwen2.5-Coder-3B line)._
