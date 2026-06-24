# BUG (mythos #1/3 — ANALYSIS): mythos_fast emits badly garbled tool calls under the harness

- **Branch:** `beta_mythos_fast` (ISOLATED mythos line — never merges back to `beta`; see `MYTHOS_DIVERGENCE.md`)
- **Type:** root-cause analysis + recommendations (gates the fix)
- **Severity:** high — the whole point of this branch is to run a tool-use fine-tune; garbled tool calls make the agent non-functional.
- **Status:** COMPLETE — root-cause analysis + recommendations filled in (see `## FINDINGS` / `## RECOMMENDATIONS`).
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

All citations are from the actual `beta_mythos_fast` working tree.

### Summary verdict
The garbling is **NOT** caused by the JSON-schema constraint (it is genuinely OFF for
`hermes`). It is caused by the harness **never speaking the model's native template** and
instead **hand-rolling its own ChatML** in a two-phase split, while feeding tool
definitions in the **wrong shape** (OpenAI-nested instead of bare) and history in a
**foreign shape** (English narrative instead of `<tool_response>` turns). The model is
driven far off the single-pass distribution it was fine-tuned on, so it emits malformed
tool calls.

---

### Suspect 1 — two-phase `raw=True` prefill — CONFIRMED, PRIMARY CAUSE

- Phase 1 `_reason` calls `/api/chat` with only `messages=[system,user]` and **no
  `tools=[]` array** (`vibeharness/llm.py:90-99`). Because Ollama's chat template only
  emits the `# Tools` block / `<tool_call>` instructions under `{%- if tools -%}`
  (template lines reproduced above, the `{%- if tools -%}` branch), passing no tools means
  the model's **native tool scaffolding never fires at all** — not in phase 1.
- Phase 2 `_act` sets `raw=True` and hand-rolls ChatML via `_render_chatml`
  (`vibeharness/llm.py:102-118`, `185-191`). `raw=True` **bypasses the embedded chat
  template entirely** (confirmed by the module docstring `vibeharness/llm.py:1-11` and the
  `format` comment at `llm.py:113-114`). So even the bare ChatML `<|im_start|>` framing is
  reconstructed by hand and is the ONLY thing the model sees.
- `_render_chatml` (`vibeharness/llm.py:185-191`) emits
  `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`.
  The native template (lines 38-50 above) instead wraps the system content, then appends a
  fixed `# Tools` preamble, the bare `<tools>` list, and the verbatim
  `For each function call, return a json object ... <tool_call>...</tool_call>` instruction
  block. **None of that fixed native preamble/instruction text is reproduced** — only
  whatever the SystemPromptBuilder happened to put in `{system}`.
- `_continue_after_reasoning` (`vibeharness/llm.py:193-201`) splices the phase-1 text back
  in after the reopened assistant turn and appends `</think>\n`. The model is therefore
  forced to resume *after* a `</think>` it did not itself decide to close, on a prompt that
  never carried the native tool-call instruction block. This is exactly the "split/forced
  single-pass shape" the bug doc suspected, and it is the leading driver of malformed
  output: the model is mid-assistant-turn with no native cue that a `<tool_call>` is what
  comes next.
- Net: the model is **never** given its native template with a real `tools=[]` array in
  either phase. Confirmed by grep — the only producer of a tools array is the string
  `tools_block` embedded in the system prompt; no `/api/chat` call passes `tools=`
  (`vibeharness/llm.py:91-99`; no `tools=` key anywhere in `vibeharness/`).

### Suspect 2 — the JSON-schema constraint / "bolted-on encoding step" — REFUTED (already OFF for hermes)

Traced end-to-end:
- Default codec is `hermes` (`vibeharness/config.py:41`).
- CLI resolves it: `cfg = Settings.apply(Config())` then `get_codec(config.codec)`
  (`vibeharness/cli.py:212`, `320`).
- No saved settings exist to override it (`~/.vibeharness/settings.json` absent; verified
  live). `Settings.apply` only overrides known Config fields (`vibeharness/settings.py:102-107`),
  and the only constraint-relevant field is `codec` itself.
- `HermesCodec.constraint(...)` returns `DecodeConstraint(json_schema=None)`
  (`vibeharness/codecs/hermes_codec.py:76-80`).
- The agent threads that straight through: `constraint = self._codec.constraint(...)`
  (`vibeharness/agent.py:142`) → `self._client.decide(..., constraint)`
  (`vibeharness/agent.py:272`, `282`).
- `_act` only sets `payload["format"]` when `constraint.json_schema is not None`
  (`vibeharness/llm.py:115-116`) — so for hermes **no `format` is ever sent**. GBNF is
  also ignored on the Ollama backend (`vibeharness/llm.py:113-114`).

Conclusion: the schema/`format` constraint does **not** fire for this model. The only way
it could is if a user/stale settings file set `codec` back to `json` (which would then run
`json` codec's schema constraint). It is not the cause of the garbling here. **There is no
constraint to disable.** (The misleading legacy docstring at `vibeharness/llm.py:1-11`
still describes the old always-constrained two-phase design; it is stale, not active.)

### Suspect 3 — tool-definition shape mismatch (nested vs bare) — CONFIRMED, STRONG CONTRIBUTOR

- `tools_block(style="hermes")` emits the **OpenAI-nested** wrapper
  `{"type":"function","function":{"name","description","parameters"}}`
  (`vibeharness/registry.py:63-78`), and `HermesCodec.tool_definitions` returns exactly
  that (`vibeharness/codecs/hermes_codec.py:71-74`).
- The native template renders each tool as **bare** `tool | tojson`
  (template line 48 above): `{"name":..., "description":..., "parameters":{...}}` — no
  `type`/`function` envelope.
- Verified by rendering the real web toolset (command run): every line is the nested form,
  e.g. `{"type": "function", "function": {"name": "goto", ...}}`.
- Impact: the model was fine-tuned on ~2M samples whose `<tools>` lines are the bare shape;
  the card explicitly warns tool-use quality "depends heavily on the format of tool
  definitions." Feeding the nested shape mis-frames every tool the model must call. Note
  also the codec's own docstring (`vibeharness/codecs/hermes_codec.py:6-8`) *wrongly*
  claims the native format is OpenAI-nested — that documented assumption is the root of the
  mismatch and must be corrected. (Encoding artifact: the rendered output also shows `—`
  em-dashes mojibaked to `?`/`�` in this console, but that is a terminal codepage issue,
  not a model-input bug; `ensure_ascii=False` is correct.)

### Suspect 4 — system-prompt scaffolding + narrative-history drift — CONFIRMED, CONTRIBUTOR

- The SystemPromptBuilder wraps the `<tools>` block in heavy harness scaffolding:
  `# How the loop works`, the codec `format_instructions`, `# Tools`, optional
  `# Working with your tools`, `# Guidance`, plus a `# YOUR ASSIGNED TASK` /
  `# Workspace` / `# Current page (live snapshot…)` header
  (`vibeharness/prompt.py:32-55`, `88-149`). The native system message is lean: just the
  user system line + the auto-appended `# Tools` preamble (template lines 38-50). The
  harness's `# Tools\n{docs}` heading (`prompt.py:46-47`) is ALSO redundant/conflicting
  with the native `# Tools` the template would add — but since the template never fires
  (Suspect 1), the model only sees the harness's hand-written version.
- The codec's `format_instructions` (`vibeharness/codecs/hermes_codec.py:49-65`) re-teaches
  the `<tool_call>` shape in prose. This is well-intentioned but redundant with — and
  textually different from — the model's trained instruction string, adding noise.
- History is rendered as English narrative ("First, you… Then, you…")
  (`vibeharness/memory.py:17-24`) and reasoning is deliberately discarded
  (`vibeharness/agent.py` keeps `decision.reasoning` only for logs). The native multi-turn
  shape feeds prior tool results back as `<tool_response>...</tool_response>` user turns
  (template lines 69-72). So the model never sees its trained multi-turn conversation
  structure — only a prose summary in a single synthetic user turn. This is off-distribution
  but secondary to Suspects 1 and 3.

### Suspect 5 — stop tokens / `</think>` handling — CONFIRMED CORRECT (not a cause)

- Phase 1 stops at `</think>` (`vibeharness/llm.py:98`); phase 2 stops at `<|im_end|>` plus
  any codec stops (`vibeharness/llm.py:111`). These match the template's turn terminator
  `<|im_end|>` (template lines 50, 68) and do not truncate a `<tool_call>` block, since the
  block closes before `<|im_end|>`. The stop config itself is fine; the problem is the
  *prompt* the stops are applied to, not the stops.

---

### Ranked root cause
1. **(Primary) The two-phase hand-rolled `raw=True` prefill bypasses the model's native
   chat template and never passes a `tools=[]` array**, so the model never receives its
   trained tool-call scaffolding and is forced to resume an assistant turn after an
   injected `</think>`. (`vibeharness/llm.py:90-118`, `185-201`)
2. **(Strong contributor) Tool definitions are sent in the OpenAI-nested shape, not the
   bare `tool|tojson` shape the fine-tune expects.** (`vibeharness/registry.py:63-78`)
3. **(Contributor) System-prompt scaffolding + English-narrative history** push the prompt
   off the lean native single-pass distribution. (`vibeharness/prompt.py`,
   `vibeharness/memory.py`)
4. The JSON-schema constraint is **not** a cause — it is already off for hermes.

## RECOMMENDATIONS

Prioritized; directly implementable by the fix subagent. **All edits stay on
`beta_mythos_fast` only.**

1. **(P0) Stop bypassing the native template — let Ollama apply it with a real `tools`
   array.** The single highest-leverage fix. Add a native-tool-call generation path for the
   hermes codec that calls Ollama `/api/chat` ONCE with:
   - `messages=[{role:"system", content:<lean system>}, {role:"user", content:<turn>}, …]`
   - `tools=[<bare function schemas>]` (Ollama accepts the OpenAI-style array and renders it
     through the model's own template, which emits the bare `tool|tojson` lines).
   Drop the phase-1/phase-2 split for this model: emit `<think>…</think>` + `<tool_call>` in
   one assistant turn, exactly as trained. Concretely: in `vibeharness/llm.py`, gate `decide`
   so that when the codec signals "native tool transport" it runs a single `/api/chat` pass
   instead of `_reason`+`_act`. Do **not** use `raw=True` on this path. (Touches
   `vibeharness/llm.py:66-118`.)

2. **(P0) Fix the tool-definition shape to bare.** Either (a) when using the native
   `/api/chat tools=` path, hand Ollama the parameter schemas and let the template render
   the bare form; or (b) if keeping a string-embedded `<tools>` block, add a
   `style="hermes_bare"` (or change the hermes branch) in
   `vibeharness/registry.py:63-78` that emits bare
   `{"name","description","parameters"}` per line (drop the `{"type":"function","function":…}`
   envelope) and update `HermesCodec.tool_definitions`
   (`vibeharness/codecs/hermes_codec.py:71-74`). Also correct the now-wrong docstrings at
   `vibeharness/codecs/hermes_codec.py:6-8` and `vibeharness/registry.py:54-58` that claim
   the native shape is OpenAI-nested. Prefer (a); fall back to (b) only if a native
   `tools=` transport is out of scope for the fix.

3. **(P1) Feed history back as native `<tool_response>` turns when on the native path.**
   Map the narrative memory's per-action observations to `{role:"tool", content:…}` messages
   (rendered by the template as `<tool_response>…</tool_response>` user turns, template lines
   69-72) and assistant `tool_calls` for the model's prior calls, so the multi-turn shape
   matches training. If a full rewrite of `NarrativeMemory` is too large for the fix, at
   minimum wrap each observation in a `tool` role message. (Touches
   `vibeharness/agent.py` history rendering + `vibeharness/memory.py`.)

4. **(P1) Slim the system prompt for the hermes path.** Let the native template own the
   `# Tools` block and the `<tool_call>` instruction; remove the harness's duplicate
   `# Tools` heading and the codec `format_instructions` prose when the native transport is
   active, keeping only the lean task/workspace/page context. (Touches
   `vibeharness/prompt.py:32-55`, `88-149`; consider making `format_instructions` / the
   `# Tools` heading conditional on transport.)

5. **(P2) Do NOT add or re-enable any JSON-schema `format` constraint for hermes.** It is
   correctly off (`vibeharness/codecs/hermes_codec.py:76-80`, `vibeharness/llm.py:115-116`);
   keep it off. Also delete/rewrite the stale module docstring at
   `vibeharness/llm.py:1-11` so future readers don't think the constraint is active. No code
   path needs disabling — the suspected "bolted-on constraint" is already inert for this
   model.

### Decision on the constraint/encoding step
**Do not disable anything — there is nothing active to disable.** The JSON-schema
constraint already does not fire for the `hermes` codec (traced: `config.py:41` →
`cli.py:212,320` → `hermes_codec.py:80` → `llm.py:115-116`). The real "bolted-on encoding"
that corrupts output is the **hand-rolled two-phase `raw=True` ChatML transport**
(`vibeharness/llm.py:102-118,185-201`) combined with the **nested tool-definition shape**
(`vibeharness/registry.py:63-78`). Replace the transport with a single native
`/api/chat` + `tools=` pass and fix the tool shape; that is the fix, not toggling a
constraint flag.
