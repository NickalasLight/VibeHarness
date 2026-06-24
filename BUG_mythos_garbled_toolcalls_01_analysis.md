# BUG (mythos #1/3 — ANALYSIS): mythos_fast emits badly garbled tool calls under the harness

- **Branch:** `beta_mythos_fast` (ISOLATED mythos line — never merges back to `beta`; see `MYTHOS_DIVERGENCE.md`)
- **Type:** root-cause analysis + recommendations (gates the fix)
- **Severity:** high — the whole point of this branch is to run a tool-use fine-tune; garbled tool calls make the agent non-functional.
- **Status:** OPEN — analysis to be completed by the analysis subagent (this file IS the "analysis text file").
- **Depends on:** nothing.
- **Gates:** `BUG_mythos_garbled_toolcalls_02_fix.md` (the fix MUST NOT start until this analysis is marked COMPLETE).

---

## Symptom

The default model on this branch is the tool-use fine-tune
`hf.co/Shadow0482/mythos_fast:Q6_K` (~2M tool-call samples, Qwen2.5-3B geometry),
paired with the `hermes` codec. Despite the fine-tune being trained specifically to
emit tool calls, **live runs produce badly garbled / malformed tool-call output.**

Working hypothesis from the user: a **"bolted-on" encoding / constraining step** in the
harness is fighting the model's native trained format and corrupting the output. The fix
will **likely disable that secondary constraint/encoding step** if this analysis confirms
it as the cause.

## Ground truth — the model's REAL trained format (captured live)

Captured **2026-06-24** by rendering the model's chat template in a headed Chrome browser
via Playwright at:
`https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Shadow0482%2Fmythos_fast`
(the real app is the cross-origin iframe
`https://huggingfacejs-chat-template-playground.hf.space/?modelId=Shadow0482%2Fmythos_fast`).
Evidence screenshots: `mythos-playground-initial.png`, `mythos-playground-toolusage.png`.
The model is **GGUF-only** (no `tokenizer_config.json` on the HF repo — that is why a raw
fetch 404s); the canonical template is the one the playground renders, reproduced here.

### The Jinja chat template (verbatim, from the playground)

```jinja
{%- if tools -%}
	{{- "<|im_start|>system\n" -}}
	{%- if messages[0]["role"] == "system" -%}
		{{- messages[0]["content"] -}}
	{%- else -%}
		{{- "You are a helpful assistant." -}}
	{%- endif -%}
	{{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" -}}
	{%- for tool in tools -%}
		{{- "\n" -}}
		{{- tool | tojson -}}
	{%- endfor -%}
	{{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" -}}
{%- elif messages[0]["role"] == "system" -%}
	{{- "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n" -}}
{%- else -%}
	{{- "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
	{%- if message.role == "user" or message.role == "system" and not loop.first or message.role == "assistant" and not message.tool_calls -%}
		{{- "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>" + "\n" -}}
	{%- elif message.role == "assistant" -%}
		{{- "<|im_start|>" + message.role -}}
		{%- if message.content -%}
			{{- "\n" + message.content -}}
		{%- endif -%}
		{%- for tool_call in message.tool_calls -%}
			{%- if tool_call.function is defined -%}{%- set tool_call = tool_call.function -%}{%- endif -%}
			{{- "\n<tool_call>\n{\"name\": \"" -}}{{- tool_call.name -}}{{- "\", \"arguments\": " -}}{{- tool_call.arguments | tojson -}}{{- "}\n</tool_call>" -}}
		{%- endfor -%}
		{{- "<|im_end|>\n" -}}
	{%- elif message.role == "tool" -%}
		{%- if loop.index0 == 0 or messages[loop.index0 - 1].role != "tool" -%}{{- "<|im_start|>user" -}}{%- endif -%}
		{{- "\n<tool_response>\n" -}}{{- message.content -}}{{- "\n</tool_response>" -}}
		{%- if loop.last or messages[loop.index0 + 1].role != "tool" -%}{{- "<|im_end|>\n" -}}{%- endif -%}
	{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
	{{- "<|im_start|>assistant\n" -}}
```

### Rendered tool-use example (verbatim Output panel)

System turn (tools present) →
```
<|im_start|>system
You are a helpful assistant that can use tools to get information for the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "Get current weather information for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "..."}, "unit": {"type": "string", "enum": ["celsius","fahrenheit"], "description": "..."}}, "required": ["location"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```
Assistant turn (reason + prose + call, ALL in one turn) →
```
<|im_start|>assistant
<think>
The user is asking about the weather in New York. I should use the weather tool...
</think>
I'll check the current weather in New York for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}
</tool_call><|im_end|>
```
Tool result →
```
<|im_start|>user
<tool_response>
{"temperature": 22, "condition": "Sunny", ...}
</tool_response><|im_end|>
```

### The four invariants the trained format requires
1. **Tool definitions** = **bare** JSON schema per line inside `<tools>…</tools>`:
   `{"name":…, "description":…, "parameters":<json-schema>}` (template uses `tool | tojson`).
2. **Tool call** = `<tool_call>\n{"name":…, "arguments":{…}}\n</tool_call>` (keys `name`/`arguments`).
3. **Reasoning + content + tool_call** are emitted together in ONE assistant turn
   (`<think>…</think>` then optional prose then `<tool_call>`).
4. **Tool results** come back as `<tool_response>…</tool_response>` in a user turn.

---

## Suspect areas in the harness (for the analysis subagent to confirm/refute)

> The analysis subagent MUST read the actual branch code and confirm each point with a
> file:line citation; do not take these prompts as conclusions.

1. **`vibeharness/llm.py` — the two-phase `raw=True` generation (`OllamaClient._act`).**
   Phase 2 hand-rolls ChatML (`_render_chatml`) and prefills past `</think>` with
   `raw=True`, **bypassing the model's embedded chat template entirely**. The fine-tune's
   native single-pass `<think>…</think> … <tool_call>` shape is split/forced by the
   harness. Is the prefill corrupting the `<tool_call>` emission? Is the model ever
   given its native template (via Ollama `/api/chat` with a real `tools=[]` array)?

2. **The constraining step.** Confirm whether the `hermes` codec truly runs UNCONSTRAINED
   (`hermes_codec.HermesCodec.constraint` returns `DecodeConstraint(json_schema=None)`),
   and whether ANY residual JSON-schema `format` (from `json` codec defaults, settings,
   or a stale saved `~/.vibeharness/settings.json`) is still being applied. The user
   suspects an encoding/constraining step is "bolted on" — verify it is actually OFF for
   this model, end to end (config default, CLI resolution, saved settings, `_act`).

3. **Tool-definition shape mismatch (CONFIRMED by orchestrator).**
   `vibeharness/registry.py::tools_block(style="hermes")` emits the **OpenAI-nested**
   `{"type":"function","function":{…}}` wrapper, but the **live template uses bare**
   `tool | tojson` → `{"name","description","parameters"}`. The card warns tool-use
   performance "depends heavily on the format of tool definitions." Quantify the impact.

4. **System-prompt structure drift.** The harness `SystemPromptBuilder` wraps the
   `<tools>` block in its own scaffolding (`# How the loop works`, `# Guidance`,
   `# Working with your tools`, task/workspace/page headers). The native template expects
   a lean system message + the auto-appended `# Tools` block. Does the extra scaffolding
   (and the harness's NL-narrative history instead of `<tool_response>` turns) push the
   model off-distribution? Note the harness deliberately drops reasoning from context and
   uses a prose narrative — both diverge from the trained multi-turn shape.

5. **Stop tokens / `</think>` handling.** `_reason` stops at `</think>`; `_act` stops at
   `<|im_end|>`. Confirm these line up with the template and do not truncate a
   `<tool_call>` block.

## Deliverable (acceptance for this item)
- [ ] Each suspect area above confirmed or refuted with file:line evidence from the branch.
- [ ] A definitive **root cause** (or ranked set of contributing causes) for the garbling.
- [ ] An explicit decision on **whether to disable the secondary constraint/encoding step**,
      and if so exactly which code path (with the safest mechanism).
- [ ] A prioritized, concrete **recommendations** list that the fix item can implement
      directly (system-prompt text, tool-definition shape, codec/transport changes).
- [ ] This file updated in place with a `## FINDINGS` and `## RECOMMENDATIONS` section, then
      status flipped to **COMPLETE**.
- [ ] NO production code changes in this item — analysis only.

## FINDINGS
_(to be completed by the analysis subagent)_

## RECOMMENDATIONS
_(to be completed by the analysis subagent)_
