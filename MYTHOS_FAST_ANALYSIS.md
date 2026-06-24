# mythos_fast Analysis (GitHub #35 — gates #36)

Model under study: **Shadow0482/mythos_fast** — <https://huggingface.co/Shadow0482/mythos_fast>
A VibeThinker-3B fine-tune for tool-use / agentic execution (~2,000,000 tool-call /
function-calling / agent-trace samples; trained with Unsloth). GGUF-only repo; Ollama
drop-in.

Branch: **beta_mythos_fast** (ISOLATED mythos_fast line — like the HF-API branch, this is
NOT merged into `beta` until validated; #36's build continues on this same branch).

> **LICENSE — UNSPECIFIED.** The model card states **no license**. Evaluation/analysis is
> fine, but **adoption is blocked until a license is confirmed** with the author. The base
> model is `WeiboAI/VibeThinker-3B` (itself derived from Qwen2.5-3B / Qwen2.5-Coder-3B);
> the base's license terms do not automatically transfer to a redistributed fine-tune
> whose card is silent. Flag explicitly before any production use.

---

## Section A — Stand-up + evaluation

### A.1 Stand-up (Ollama, 8 GB card)

The HF repo is GGUF-only (no `tokenizer_config.json`; the chat template is **embedded in
the GGUF** — the card's only usage hint is `llama-cli -hf Shadow0482/mythos_fast --jinja`,
i.e. "use the embedded jinja template"). GGUF quants available in the repo:

| quant | size | fits 8 GB? |
|---|---|---|
| BF16 / F16 | 6.18 GB | yes but wasteful |
| Q8_0 | 3.29 GB | yes |
| **Q6_K** | **2.54 GB** | **yes — chosen** |
| Q5_K_M | 2.22 GB | yes |
| Q4_K_M | 1.93 GB | yes |
| Q3_K_M / Q2_K_L | 1.59 / 1.35 GB | yes (lower quality) |

Chose **Q6_K** — near-Q8 quality, smaller resident footprint, ample VRAM headroom on the
RTX 3080 Laptop (8 GB) for the harness's pinned `num_ctx=32768`.

```
ollama pull hf.co/Shadow0482/mythos_fast:Q6_K
```

- **Load placement:** `ollama ps` →
  `hf.co/Shadow0482/mythos_fast:Q6_K   2.8 GB   100% GPU   4096`. Full-GPU offload
  confirmed (no CPU/partial split).
- **Smoke generation:** prompt "What is 2+2? Answer in one word." → model emits a
  `<think>…</think>` reasoning chain (VibeThinker's signature long reasoning is intact),
  then answers **"Four"**. Generation healthy.

### A.2 File-op benchmark — base VibeThinker vs mythos_fast

`benchmarks/runner.py`, **10 file-operation tasks, NO browser**, `json` codec (the harness
baseline), `max_steps=15`. Same harness, same codec, same tasks; only `--model` differs.

> The harness drives the model through its OWN hand-rolled ChatML + JSON-schema-constrained
> action phase (see Section B/C) — it does **NOT** use the model's native `<tool_call>`
> template. So these numbers measure mythos_fast UNDER the current harness contract, which
> is exactly the like-for-like base-vs-mythos comparison #35 asked for. (Whether routing
> through the model's native trained format helps is the #36 question — see Section C.)

<!-- BENCH_RESULTS -->
> **RUN STATUS — INCOMPLETE (benchmark crashed/interrupted).** The file-op sweep did not
> finish in this session. The base run and a parallel sweep attempt both died early — the
> run logs contain only the `=== codec: json ===` header and, for the base run, a single
> completed task before the process was lost. **No full base-vs-mythos scorecard was
> captured.** Recorded honestly rather than fabricated; re-running the sweep is the first
> follow-up for #36 (see C.4).

**What WAS captured (partial):**

| model | task | result | turns | time |
|---|---|---|---|---|
| base `vibethinker` (json codec) | 1. create_file | **PASS** (`greeting.txt` exact text) | 1 | 43.29s |
| base `vibethinker` | 2–10 | not reached (run lost) | — | — |
| `mythos_fast` Q6_K | 1–10 | **not run** (process died before this model's sweep) | — | — |

The only datapoint: base VibeThinker solves the simplest file-op task in a single turn
(~43s) under the `json` codec — its normal baseline. **No comparative conclusion** about
mythos_fast vs base on file-ops can be drawn from this session; deferred to the #36 re-run.

**Re-run command (for #36 — run SEQUENTIALLY; the 8 GB card holds one model at a time and
`OLLAMA_MAX_LOADED_MODELS=1` evicts the previous):**
```
python -m benchmarks.runner --codec json   --model vibethinker                         --json-out base.json   --transcript-dir base_tx
python -m benchmarks.runner --codec hermes  --model hf.co/Shadow0482/mythos_fast:Q6_K   --json-out mythos.json --transcript-dir mythos_tx
```
The `hermes` codec referenced here was IMPLEMENTED on this branch (commit
`mythos_fast #105: hermes codec + <tools> seam`) per the Section C spec — it now exists and
is the default codec, so the re-run can A/B `json` vs `hermes` directly.

**Metrics to capture:** pass rate (deterministic `check`), total turns (turn efficiency),
wall-clock seconds, and from saved transcripts — tool-call success and format adherence (did
the model emit parseable actions in the codec's wire format).

### A.3 Deferred: ashley WEB task

DEFERRED per task instructions. The browser daemon is currently buggy (#101 / #75 in
flight; see worktrees `fix/browser-daemon-dies`, `research/daemon-crash-resume`), so web
eval is unreliable right now. The file-op benchmark is the primary comparison.
**Follow-up:** re-run the ashley web benchmark on this branch (via `vibe`) once the daemon
fix lands.

---

## Section B — Training tool-call FORMAT (the critical part; gates #36)

The card warns: *"Performance on tool-use tasks depends on the format and structure of tool
definitions provided at inference time."* So the exact trained format matters. Three
independent sources triangulate to the **standard Qwen2.5 / Hermes** tool-calling format.

### B.1 Evidence

1. **Base model jinja** (`WeiboAI/VibeThinker-3B` `tokenizer_config.json` `chat_template`)
   is the stock Qwen2.5 tool template (tools as JSON in `<tools></tools>`; calls as
   `<tool_call>{json}</tool_call>`; results as `<tool_response>`).
2. **mythos_fast's embedded GGUF template** (`ollama show --template
   hf.co/Shadow0482/mythos_fast:Q6_K`) is the **same Hermes structure**, and the card tells
   you to run it with `--jinja` (i.e. honor that embedded template).
3. The fine-tune is ~2M **tool-call** samples on this base — i.e. it *reinforces* the base's
   native format rather than replacing it.

### B.2 The EXACT format

**(a) How TOOLS are DEFINED in the prompt** — JSON function schemas, one per line, wrapped
in `<tools>…</tools>` inside the **system** turn. mythos_fast's embedded template uses the
**OpenAI-nested** wrapper:

```
<|im_start|>system
{system text}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {<json-schema of tool 1>}}
{"type": "function", "function": {<json-schema of tool 2>}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

Note: the base **jinja** emits each tool as a bare `tool | tojson` (no `{"type":"function"}`
wrapper); mythos_fast's **embedded GGUF template** adds the `{"type": "function", "function":
{…}}` wrapper. Either is "Hermes/Qwen", but the inference-time tool *definition* the model
most expects is the OpenAI-nested object the GGUF template renders.

**(b) SYSTEM-prompt structure** — the user's system text, then the auto-appended
`# Tools` block above. The `<tool_call>` output convention is restated to the model *inside
the system turn* (the template literally instructs the format).

**(c) ASSISTANT tool-call OUTPUT format** — the model emits one or more:

```
<tool_call>
{"name": "<tool-name>", "arguments": {<args object>}}
</tool_call>
```

Multiple calls = multiple consecutive `<tool_call>` blocks. Keys are **`name`** and
**`arguments`** (NOT `tool`/`args`).

**(d) TOOL-RESULT format** — results are fed back in a **user** turn wrapped in
`<tool_response>…</tool_response>` (one per result; consecutive results coalesce into a
single user turn).

VibeThinker's `<think>…</think>` reasoning precedes the `<tool_call>` (confirmed in the
smoke test); the fine-tune keeps the reason-then-act shape.

### B.3 The chat template (mythos_fast, from the GGUF, Ollama Go-template form)

```
{{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>
{{- else if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}
```

Special tokens: ChatML (`<|im_start|>` / `<|im_end|>`); FIM tokens present
(`<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`). `config.json`:
`Qwen2ForCausalLM`, `eos_token_id = bos_token_id = 151643` (`<|endoftext|>`), vocab 151936,
36 layers, hidden 2048 — i.e. Qwen2.5-3B geometry, unchanged by the fine-tune.

### B.4 How it DIFFERS from the base ChatML the harness hand-rolls

The harness hand-rolls ChatML in `vibeharness/llm.py` `OllamaClient._render_chatml`:

```python
f"<|im_start|>system\n{system}<|im_end|>\n"
f"<|im_start|>user\n{user}<|im_end|>\n"
f"<|im_start|>assistant\n"
```

Differences that matter for mythos_fast:

| dimension | harness hand-rolled ChatML | mythos_fast native template |
|---|---|---|
| **ChatML envelope** | `<|im_start|>`/`<|im_end|>` | **same** ✓ |
| **tool DEFINITION** | tools rendered as **Markdown docs** inside the system prompt (`SystemPromptBuilder` + each `Tool.doc()`); no `<tools>` block | tools as **JSON function schemas** in `<tools></tools>` |
| **tool-call OUTPUT** | model is told (by the `json` codec) to emit a **JSON array** `[{"tool","args"}]`, decode-constrained by Ollama `format` (JSON schema) | model emits **`<tool_call>{"name","arguments"}</tool_call>`** blocks (free-form text, no schema constraint) |
| **call object keys** | **`tool`** / **`args`** | **`name`** / **`arguments`** |
| **tool RESULT feedback** | woven into a natural-language narrative in the next user turn | `<tool_response>…</tool_response>` in a user turn |
| **reasoning** | two-phase: free `<think>` then a raw continuation prefilled past `</think>`, constrained to JSON | single pass; `<think>` then `<tool_call>` — model decides when to act |

Net: the **envelope** matches, but the **tool-definition block, the call wire-format, the
key names, and the result feedback** are all different from what mythos_fast was trained on.
The harness currently asks mythos_fast to speak a dialect (`{"tool","args"}` JSON array,
schema-constrained) that is NOT its native trained dialect (`<tool_call>{"name","arguments"}`).

---

## Section C — Conversion recommendation (the #36 spec)

> **STATUS:** this spec was **IMPLEMENTED on this branch** in commit
> `mythos_fast #105: hermes codec + <tools> seam` — `vibeharness/codecs/hermes_codec.py`
> (the new `hermes` codec emitting `<tool_call>{"name","arguments"}</tool_call>`),
> `registry.tools_block(style="hermes")` (the OpenAI-nested `<tools>` block), a
> `ToolCallCodec.tool_definitions()` seam in `SystemPromptBuilder`, and `config.py` defaults
> realigned (model → `hf.co/Shadow0482/mythos_fast:Q6_K`, codec → `hermes`). The spec below
> is the design of record; the remaining open item is the empirical A/B (C.4), which is
> blocked on the crashed benchmark re-run.

### C.1 Does an existing codec match? — NO

| codec | wire format | match to mythos_fast native? |
|---|---|---|
| `json` (default) | JSON array `[{"tool","args"}]`, JSON-schema constrained | **No** — wrong keys (`tool`/`args`), wrong shape (bare array, no `<tool_call>` tag) |
| `tagged_json` | `<local_toolcall>{"tool","args"}</local_toolcall>` | **Closest in spirit** (JSON-in-a-tag, unconstrained) but **wrong tag** (`local_toolcall` ≠ `tool_call`) and **wrong keys** (`tool`/`args` ≠ `name`/`arguments`) |
| `xml` | `<tool_call name="..."><arg name="...">val</arg></tool_call>` | **No** — right OUTER tag name (`tool_call`) but the BODY is nested `<arg>` XML, whereas mythos_fast emits a **JSON object** inside `<tool_call>` |
| `codeact` | model writes Python | No |
| `gbnf` | grammar-constrained variant | No |

**Conclusion: a NEW codec is required.** None of the five emit
`<tool_call>{"name","arguments"}</tool_call>` with JSON function-schema tool definitions.

### C.2 The #36 spec — a `hermes` (Qwen/Hermes native) codec

Add `vibeharness/codecs/hermes_codec.py` exposing `CODEC = HermesCodec()` (the codec system
is open/closed — a new file, no edits to the registry/agent/transport; see
`vibeharness/codec.py` `get_codec`). It implements the three `ToolCallCodec` methods:

1. **`format_instructions(max_actions)`** — restate the native convention so the model that
   was trained on it sees the familiar instruction:
   > Each turn, emit one or more tool calls, each as
   > `<tool_call>\n{"name": <tool>, "arguments": {…}}\n</tool_call>`.
   And — critically per the card — the **tool DEFINITIONS** must be presented as JSON
   function schemas in a `<tools></tools>` block, NOT the current Markdown `Tool.doc()`
   text. This is the highest-leverage change: it's the "format and structure of tool
   definitions" the card says performance "depends heavily" on. Build the `<tools>` block
   from the registry, reusing `Tool._args_schema()` but emitting the **Hermes/OpenAI**
   object shape `{"type":"function","function":{"name":…,"description":…,"parameters":{…}}}`
   rather than the current `{"tool":<const>,"args":{…}}` `call_schema()` shape.
   - This likely needs a small seam in `SystemPromptBuilder`/`registry` so a codec can
     supply its own tool-definition rendering (today the prompt builder always emits Markdown
     docs). Concretely: let the codec contribute the tool-definition block, or add a
     `registry.tools_block(style="hermes")` helper. Keep it codec-local to preserve O/C.

2. **`constraint(registry, max_actions)`** — return `DecodeConstraint(json_schema=None, …)`,
   i.e. **unconstrained** (like `xml`/`tagged_json`). The `<tool_call>` wrapper is not
   expressible as a JSON-schema `format`, so parsing does the structural work.
   - Optional upgrade: a **GBNF** grammar (`DecodeConstraint.gbnf`) that locks output to
     `<tool_call>\n{json}\n</tool_call>` repeated — but GBNF is honored only by the
     `llamacpp` backend, not Ollama's `format`, so this is a backend-gated enhancement, not
     the baseline. Start unconstrained; the model is fine-tuned to emit this shape natively.

3. **`parse(raw)`** — regex out each `<tool_call>…</tool_call>` block (tolerant of a missing
   final close tag, like `tagged_json`), `json.loads` the inner object, read **`name`** and
   **`arguments`** (map to the harness's internal `(tool, args)` tuple). Reject blocks
   without a `name`; coerce `arguments` to `{}` if absent.

4. **`turn_action_hint()`** — "Respond with one or more `<tool_call>{json}</tool_call>`
   blocks."

### C.3 Two-phase generation note (llm.py)

`OllamaClient._act` currently prefills a raw ChatML continuation and constrains phase 2 to a
JSON schema. For the `hermes` codec (unconstrained), `_act` already supports
`json_schema=None` (it just doesn't set `format`), so **no transport change is strictly
required** to ship the codec. The reason-then-act split still works: phase 1 `<think>`,
phase 2 emits `<tool_call>` blocks parsed by the new codec. The stop token stays
`<|im_end|>`. (A later refinement could let the harness call Ollama's native `/api/chat`
with a real `tools=[…]` array so Ollama renders the embedded template itself — but that is a
bigger transport change and optional; the codec above gets the native format end-to-end
within the existing two-phase machinery.)

### C.4 #36 acceptance test

Add `hermes` to the benchmark codec sweep and run **mythos_fast** under `json` (baseline)
vs `tagged_json` vs **`hermes`** on the 10 file-op tasks. Expectation: `hermes` (native
trained format + JSON-schema tool definitions in `<tools>`) should show the best
format-adherence and pass rate for mythos_fast, validating the conversion. Gate #36 merge
(into beta_mythos_fast) on that.
