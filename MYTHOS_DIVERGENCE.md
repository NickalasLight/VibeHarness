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

> **STUB — finalized once #105 lands** (it creates the concrete divergences). Expected set:

| Path / area | Rule | Why |
|---|---|---|
| `vibeharness/codecs/hermes_codec.py` | keep-mythos (mythos-only file) | the Hermes/Qwen native tool-call codec (#105/#35) |
| tool-definition rendering seam (the `<tools>` function-schemas vs Markdown `Tool.doc()`) in `prompt.py` / `registry.py` | keep-mythos format; reconcile other logic | the format the fine-tune was trained on |
| `config.py` defaults — `model = Shadow0482/mythos_fast`, `codec = hermes` | keep-mythos values; merge new keys | branch defaults |
| `OllamaClient._render_chatml` / chat template (if diverged by #105) | keep-mythos | native template alignment |
| any hermes-aligned prompt / tool-description text | keep-mythos | training-format alignment |

## Merge procedure (per beta fix)
1. Identify the beta change(s) and the files they touch.
2. For each file NOT in the table → cherry-pick / apply directly.
3. For each file in the table → manually integrate the *logic* of the beta change while
   preserving the mythos-specific format/values; never overwrite wholesale.
4. Run the suite; document each protected-file decision in the commit/PR.

_Last updated: 2026-06-24 (stub; #107 finalizes after #105)._
