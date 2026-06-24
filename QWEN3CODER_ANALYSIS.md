# Qwen3-Coder line Analysis — model selection + ground-truthed tool-call format (#123)

Branch: **`beta_qwen3coder`** (ISOLATED model line; forked from `beta`; one-way
`beta → beta_qwen3coder`, never merges back — CLAUDE.md §6).

Goal (#123): fork `beta` and FULLY replace VibeThinker-3B with a **~3B Qwen3-Coder**
model, end to end, keeping ~3B parity for an apples-to-apples comparison with
VibeThinker-3B (and the `beta_mythos_fast` 3B fine-tune line).

---

## Section A — Model selection (⚠️ Qwen3-Coder ~3B does NOT exist)

### A.1 Finding: the Qwen3-Coder line is MoE-only; there is no dense ~3B

Research via WebSearch / authoritative sources (Qwen blog, HuggingFace, Ollama library,
Unsloth docs — June 2026):

| Qwen3-Coder model | total params | active params | local footprint | ~3B dense? |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct | 30B (MoE) | **3B active** | ~18 GB+ even quantized | **NO** (30B total) |
| Qwen3-Coder-Next (80B-A3B) | 80B (MoE) | 3B active | tens of GB | **NO** |
| Qwen3-Coder-480B-A35B-Instruct | 480B (MoE) | 35B active | hundreds of GB | **NO** |

The "3B" advertised for Qwen3-Coder is **active** parameters of a Mixture-of-Experts
model — the *total* weight footprint (what must load into VRAM / what defines the model
class) is **30B at the very smallest**. None of these:

1. is a **dense ~3B** model, so none is an apples-to-apples peer of the 3B-dense
   VibeThinker-3B (or the 3B mythos_fast fine-tune); and
2. fits the project's **8 GB GPU** under the pinned `num_ctx=32768` (issue #77) — a
   30B-A3B GGUF is ~18 GB+ even at aggressive quants.

So **`Qwen3-Coder` cannot satisfy the "~3B, apples-to-apples" requirement.** This is the
discrepancy the issue told us to flag rather than silently substitute a larger model.

### A.2 Decision: closest true ~3B dense coder — `qwen2.5-coder:3b-instruct`

Per the #123 fallback instruction ("If no official Qwen3-Coder ~3B exists … fall back to
`Qwen2.5-Coder-3B`, and EXPLICITLY FLAG the discrepancy"), this branch uses:

- **Ollama tag (default in `config.py`):** `qwen2.5-coder:3b-instruct`
  - `ollama pull qwen2.5-coder:3b-instruct` → ~1.9 GB (Q4_K_M default), 32K context.
  - Quant variants on the same library: `…-q2_K` (1.3 GB), `…-q3_K_S` (1.5 GB),
    `…-q3_K_M` (1.6 GB), default Q4_K_M (1.9 GB) — all comfortably fit 8 GB.
- **HF source of truth:** `Qwen/Qwen2.5-Coder-3B-Instruct`.

Why this is the right 3B-for-3B substitute:

- It is a **dense 3B** model (Qwen2.5-Coder ships 0.5B / 1.5B / **3B** / 7B / 14B / 32B),
  so it IS apples-to-apples with VibeThinker-3B on size.
- **VibeThinker-3B itself derives from Qwen2.5-3B / Qwen2.5-Coder-3B** (per the
  `beta_mythos_fast` analysis), so this is the most direct "swap the model, keep the
  class" comparison available.
- It is a purpose-built **coder** model (the spirit of "Qwen3-Coder"), available as an
  official Ollama library tag with a published, authoritative chat template.

**This is a documented, flagged substitute — NOT a silent swap.** If/when Alibaba ships a
dense ~3B *Qwen3-Coder*, change the one `model =` line in `config.py` (the codec/format
below is the same Qwen lineage and would very likely carry over unchanged).

---

## Section B — Ground-truthed tool-call FORMAT (captured, NOT assumed)

The Qwen card warns tool-use performance "depends heavily on the format and structure of
tool definitions provided at inference time", so the EXACT trained format matters. Captured
from the **authoritative** source: the model's own chat template in
`Qwen/Qwen2.5-Coder-3B-Instruct/tokenizer_config.json` (`chat_template`), fetched live.

### B.1 Special tokens / envelope (ChatML)

Roles are delimited with **`<|im_start|>` … `<|im_end|>`** (standard Qwen2.5 ChatML).
The harness's hand-rolled `OllamaClient._render_chatml` already emits exactly this
envelope, so no transport change is required for the ChatML wrapping.

Default system message (when none supplied):
> "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

### B.2 Tool DEFINITIONS — bare function schemas in `<tools>` (one per line)

The template renders each tool with a **bare** `tool | tojson` — i.e. the BARE object
`{"name", "description", "parameters"}` with **NO** `{"type":"function","function":{…}}`
envelope — wrapped in `<tools>…</tools>` inside the **system** turn:

```
You may call one or more functions to assist with the user query. You are
provided with function signatures within <tools></tools> XML tags
<tools>
{"name": ..., "description": ..., "parameters": {...}}   ← one bare schema per line
</tools>
For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

### B.3 Assistant tool-call OUTPUT — `<tool_call>{"name","arguments"}</tool_call>`

The model emits one or more consecutive blocks per turn:

```
<tool_call>
{"name": "<tool-name>", "arguments": {<args object>}}
</tool_call>
```

Keys are **`name`** and **`arguments`** (NOT `tool`/`args`).

### B.4 Tool RESULT feedback — `<tool_response>` in a user turn

Results are fed back in a **user** turn wrapped in `<tool_response>…</tool_response>`
(consecutive results coalesce under one `<|im_start|>user`).

### B.5 Reasoning

Qwen2.5-Coder-3B-Instruct is **NOT** a long-chain reasoning model — it does not emit
verbose `<think>` chains by default (unlike VibeThinker). The harness's two-phase
reason→act loop still works: phase-1 simply returns quickly (the `</think>` stop is
harmless if never reached, capped by `reason_tokens`), phase-2 emits the `<tool_call>`
block parsed by the codec. `reason_tokens=4096` is kept as a harmless ceiling (see the
config note), preserving the beta input-budget math and all snapshot/budget tests.

---

## Section C — Codec decision: REUSE the `hermes` codec (no new codec needed)

### C.1 The captured format == the Qwen2.5 / Hermes convention

The format in Section B is **exactly** the standard Qwen2.5 / Hermes tool-calling
convention (bare `<tools>` schemas + `<tool_call>{"name","arguments"}` + `<tool_response>`).
That is the convention the `hermes` codec implements.

### C.2 Does an existing codec match? — YES (`hermes`)

| codec | wire format | match to Qwen2.5-Coder native? |
|---|---|---|
| `json` (beta default) | JSON array `[{"tool","args"}]`, schema-constrained | No — wrong keys, no `<tool_call>` tag |
| `tagged_json` | `<local_toolcall>{"tool","args"}</local_toolcall>` | No — wrong tag + wrong keys |
| `xml` | `<tool_call name=…><arg…>` nested XML body | No — right outer tag, but XML body not a JSON object |
| `codeact` / `gbnf` | Python / grammar variant | No |
| **`hermes`** | **`<tool_call>{"name","arguments"}</tool_call>` + bare `<tools>` schemas** | **YES — exact match** |

**Decision:** REUSE `hermes` (per #123: "reuse an existing codec if it matches"). It is
authored on this branch as `vibeharness/codecs/hermes_codec.py` (it originated on
`beta_mythos_fast` for the same Qwen lineage; it is not present on `beta`). It is the
default codec here. No NEW codec is needed because the format is identical.

### C.3 Implementation seams (mirrors the mythos pattern)

- `vibeharness/codec.py` — `ToolCallCodec.tool_definitions()` hook (default `None`).
- `vibeharness/registry.py` — `tools_block(style="hermes")` builds the BARE `<tools>`
  block from each tool's `_args_schema()` (single source of truth; never drifts from the
  JSON constraint).
- `vibeharness/prompt.py` — `SystemPromptBuilder` substitutes the codec's
  `tool_definitions()` for the Markdown `docs()` when present (every codec but `hermes`
  returns `None`, so other formats are byte-for-byte unchanged).
- `vibeharness/codecs/hermes_codec.py` — `format_instructions` (native wording),
  `tool_definitions` (`<tools>` block), `constraint` (UNCONSTRAINED — the `<tool_call>`
  tag isn't expressible as a JSON-schema `format`; parsing does the work), `parse`
  (regex `<tool_call>` blocks → `(name, arguments)`, tolerant of missing close tag /
  absent or null arguments / surrounding prose).
- `vibeharness/llm.py` — phase-2 sets Ollama `format` ONLY when `json_schema is not
  None`; `hermes` returns `json_schema=None`, so it is unconstrained end to end. No
  transport change required (the ChatML envelope already matches).
- `vibeharness/config.py` — defaults realigned: `model = qwen2.5-coder:3b-instruct`,
  `codec = hermes` (commented as branch-local divergences).

### C.4 Tests

`tests/test_codec_hermes.py` — codec discovery, native `format_instructions`, the bare
`<tools>` block shape (no `type`/`function` envelope; parameters == `_args_schema()`),
unconstrained `constraint`, and `parse` over single / multiple-in-order / prose+`<think>`
/ missing-close-tag / absent-args / null-args / malformed inputs.

---

## Section D — Status & follow-ups

- ✅ Implemented + unit-tested + offline-rendered (`--print-system --agent web`).
- ⏳ **LIVE end-to-end run is a documented follow-up** (covered by validation issue
  **#125**). NOT run here to avoid a multi-GB download / live dependency; repro:
  ```
  ollama pull qwen2.5-coder:3b-instruct
  python -m vibeharness --agent web "…task…"        # uses the hermes codec by default
  ```
  No live result is fabricated.
- 📌 Follow-up: if a dense ~3B *Qwen3-Coder* ever ships, swap the `model =` tag in
  `config.py` and re-validate (format should carry over — same Qwen lineage).
- 📌 Follow-up (optional, mirrors mythos): a `qwen3coder-sync` agent + a
  divergence-check GitHub Action enforcing `QWEN3CODER_DIVERGENCE.md` on PRs into this
  branch.
