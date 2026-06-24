# Qwen2.5-Coder Tool-Call Format Analysis

Issue: **#129** — ground-truth analysis of the `hermes` codec's tool-call behaviour with
`qwen2.5-coder:3b-instruct` on branch `beta_qwen3coder`.

**Status: research-only.** No `.py` harness file was modified. Every claim below is tied to
one of three evidence sources:

- **MODELFILE** — `ollama show qwen2.5-coder:3b-instruct --modelfile` (the actual GGUF chat
  template Ollama ships and applies; captured live for this report).
- **RUNLOG** — `.vibe/20260624_200826.json` (the last real run; 20 turns; cited by `turn N`).
- **CODE** — source on `origin/beta_qwen3coder`
  (`vibeharness/codecs/hermes_codec.py`, `vibeharness/llm.py`, `vibeharness/config.py`,
  `vibeharness/agent.py`, `vibeharness/registry.py`, `vibeharness/prompt.py`).

Where evidence is absent I say so and flag the inference's uncertainty explicitly.

---

## 0. Executive summary (read this first)

1. **The native chat template is NEVER engaged for tools.** The harness sends tool
   definitions and the `<tool_call>` instruction as plain text *inside the system string*
   and never populates Ollama's `tools:` field. So Ollama's template renders with
   `.Tools` empty — the model's trained `# Tools` framing, its verbatim instruction line,
   and its `<tool_response>` result framing all sit dormant. The harness hand-rebuilds most
   of this, but **drops the one line that prevents fences.** (CODE, MODELFILE.)

2. **The markdown-fence problem is near-total, not occasional.** In RUNLOG, **20 of 20**
   `raw_action` payloads are ```` ```json ... ``` ```` fences; **zero** `<tool_call>` tags
   appear in the entire run. This is not a tail behaviour — it is the model's default output
   under the current prompt. (RUNLOG turns 1-20.)

3. **PRIMARY root cause of the fences:** the harness's `format_instructions()` reproduces
   the native instruction *except* the negative constraint the template carries verbatim —
   `"with NO other text. Do not include any backticks or ```json."` Its absence, combined
   with a coder model whose pre-training distribution is saturated with ```` ```json ````
   code fences, lets the model fall back to its strongest prior. This is fixable in the
   *prompt text* with zero transport change. (MODELFILE vs CODE `format_instructions`.)

4. **The single-call-per-turn behaviour is mostly correct model behaviour, not a bug.** The
   task is a fill-one-field-then-resnapshot loop where the result of each action changes the
   page; batching is genuinely unsafe for most turns. Turn 18 *proves the model CAN batch*
   (two bare JSON objects, parsed correctly). So batching is a capability the model has but
   rarely needs here. (RUNLOG turn 18.)

5. **The biggest lever is not the codec format at all.** RUNLOG shows the run died on a
   `<select>`/combobox the model could not operate and on ref hallucination (`State`,
   `zip_code_element`), looping under a weak anti-loop guard. Format alignment will clean up
   the wire protocol, but the run-ending failures are tool-affordance and grounding
   problems. (RUNLOG turns 9-20.) Flagged so the format fix is not oversold.

---

## 1. Native Format (verified from modelfile)

Captured verbatim from `ollama show qwen2.5-coder:3b-instruct --modelfile`. The
`TEMPLATE """..."""` block is the authoritative ground truth.

### 1.1 ChatML envelope

Roles are delimited `<|im_start|>{role}\n ... <|im_end|>`. A system turn is emitted when
**either** `.System` **or** `.Tools` is set:

```
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}
... tools block ...
{{- end }}<|im_end|>
```

The harness's `OllamaClient._render_chatml` (used only by the two-phase `_act`, not by the
single-phase path) reproduces this envelope exactly:
`<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`.
So the *envelope* is aligned. The divergence is entirely in the **tools** and **tool-result**
sections, below.

### 1.2 Native tool DEFINITIONS block (fires only when `.Tools` is set)

```
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools>:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within
<tool_call></tool_call> with NO other text. Do not include any backticks or ```json.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

Two facts that matter for the rest of the report:

- **The native per-tool wrapper is `{"type": "function", "function": {...}}`** — an
  ENVELOPED object, NOT bare. (See §2.1 — this contradicts the codec's "bare" claim for
  the Ollama-rendered path.)
- **The native instruction ends with `with NO other text. Do not include any backticks or
  ```json.`** — an explicit anti-fence guard. The harness's reproduction omits it (§3).

### 1.3 Native tool-CALL output

```
{{ else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
```

Keys are `name` / `arguments`. Note the native template wraps **all** calls of a turn in a
**single** `<tool_call>...</tool_call>` pair with each call on its own line — it does NOT
emit one `<tool_call>` pair per call. (See §5 — this differs from the harness instruction,
which tells the model to emit *consecutive* `<tool_call>` blocks.)

### 1.4 Native tool-RESULT feedback

```
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
```

A `role:"tool"` message is rendered as a **user** turn wrapping the content in
`<tool_response>...</tool_response>`. The harness sends results as English narrative inside
the next user turn instead (see §2.3 and `memory.py`).

### 1.5 Stop tokens

The only structural terminator in the template is **`<|im_end|>`** (every turn ends with
it). There is no model-level `<tool_call>`-specific stop. (MODELFILE; confirmed by the
absence of any other terminal token in the template.)

---

## 2. Harness vs Native: Alignment Gaps

### Q1 — Does the harness's manual injection match the native template? **No, in three ways.**

| Aspect | Native template (MODELFILE) | Harness (`hermes` codec + transport) | Aligned? |
|---|---|---|---|
| ChatML envelope | `<|im_start|>…<|im_end|>` | identical (`_render_chatml`) | ✅ |
| Where tools live | `# Tools` section, rendered by template from `.Tools` | plain text baked into the `system` string; `.Tools` is **empty** | ❌ template never fires |
| Per-tool wrapper | `{"type":"function","function":{…}}` (ENVELOPED) | bare `{"name","description","parameters"}` (`registry.tools_block`) | ❌ envelope dropped |
| `<tools>` preamble line | `…within <tools></tools>:` | `…within <tools></tools> XML tags (see the # Tools section below).` | ~ paraphrased |
| Anti-fence clause | `with NO other text. Do not include any backticks or ```json.` | **absent** | ❌ (the decisive gap) |
| Call output shape | ONE `<tool_call>` pair, calls newline-separated inside | "emit consecutive `<tool_call>` blocks" (one pair per call) | ❌ contradicts native |
| Tool result framing | `role:"tool"` → `<tool_response>` user turn | English narrative in the user turn (`NarrativeMemory`) | ❌ |

#### 2.1 The "bare schema" claim is wrong for the Ollama path (important correction)

`hermes_codec.py` and `registry.tools_block` assert the native template renders each tool
with a **bare** `tool | tojson` and therefore emit bare `{"name","description","parameters"}`.
The captured **Ollama** modelfile template renders
`{"type": "function", "function": {{ .Function }}}` — i.e. **enveloped**. (MODELFILE §1.2.)

Reconciliation / uncertainty flag: the codec cites `Qwen/Qwen2.5-Coder-3B-Instruct`'s
**HuggingFace `tokenizer_config.json`**, which historically did render a bare
`{{ tool | tojson }}`. The two sources genuinely differ — Ollama re-authored the template
with the `{"type":"function","function":…}` envelope (this matches the broader Qwen2.5
template revision). **Since the harness runs through Ollama, the Ollama template is the
operative ground truth.** So the harness is currently injecting tools in a shape (bare) that
does NOT match what *this backend's* native template would produce (enveloped). I am
confident on what each source says; I am moderately confident the envelope vs bare
difference has only a *secondary* effect on output (the model tolerates both; the fence
problem is driven by the instruction text, §3), but this should be A/B-tested, not assumed.

#### 2.2 The native template never fires for tools

The single-phase path `OllamaClient._chat` (CODE `llm.py`) sends:

```python
"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

with **no `tools:` key**. Ollama therefore evaluates the template with `.Tools` empty: the
entire `# Tools` block — including the verbatim anti-fence instruction and the
`<tool_response>` result framing — is skipped. The model only ever sees whatever
`SystemPromptBuilder` baked into the `system` string via `codec.format_instructions()` +
`codec.tool_definitions()` (CODE `prompt.py` lines 114-120). **This is the central
structural fact of the whole analysis.**

#### 2.3 Tool results are narrative, not `<tool_response>`

`NarrativeMemory.render()` (CODE `memory.py`) produces `"First, you …\nThen, you …"` prose
that is concatenated into the next *user* message. The model never sees `<tool_response>`
framing it was trained to associate with "a tool ran; here is its output; now continue."
This weakens the model's turn-structure prior and contributes to the confusion in RUNLOG
(e.g. re-issuing `open_browser` after it already ran, turns 2/8/10).

---

## 3. Root Cause: Markdown Fence Issue

### Q3 — Why does the model emit ```` ```json ``` ```` instead of `<tool_call>`?

**Evidence (RUNLOG):** every single `raw_action` is a fence. Representative:

- turn 1: `` ```json\n{\n  "name": "open_browser",\n  "arguments": {}\n}\n``` ``
- turn 5: `` ```json\n{\n  "name": "fill", "arguments": {"target": "e41", "text": "Jason"}}\n``` ``
- turn 18: `` ```json\n{"name":"evaluate",…}\n{"name":"fill",…}\n``` `` (two objects, one fence)

`<tool_call>` appears **0 times** in 20 turns. The fence is the model's deterministic
default here, so the cause must be systematic, not stochastic.

Evaluating the candidate causes from the issue:

- **(a) Format mismatch between harness prompt and native template — CONTRIBUTING, strong.**
  Because `.Tools` is empty (§2.2), the model is in a *plain instruct* posture, not its
  trained tool-calling posture. In that posture its instinct for "emit a structured object"
  is the ```` ```json ```` code block — the single most common JSON-emission pattern in a
  coder model's instruct/SFT data.

- **(b) Training-data distribution — PRIMARY, in combination with (d).**
  `qwen2.5-coder` is a **code** model: its instruct data is dominated by answers that fence
  JSON/code. Absent an explicit "no backticks" instruction it reverts to that prior. The
  fact that the JSON *content* is always perfectly shaped (`{"name","arguments"}`) but always
  fenced shows the model learned the *payload* from the tool dialect but defaults the
  *wrapper* to its code-fence prior.

- **(d) Unconstrained generation with no anti-fence guard — PRIMARY (the missing line).**
  The codec deliberately sets `DecodeConstraint(json_schema=None)` (CODE `hermes_codec.py`
  `constraint()`), so nothing at decode time forbids the fence. Crucially the *prompt* does
  not forbid it either: `format_instructions()` omits the native line **"with NO other text.
  Do not include any backticks or ```json."** The native template authors added that exact
  sentence precisely because this model fences by default. The harness reproduced everything
  around it but dropped the one clause that matters.

- **(c) Temperature — NOT a cause.** The action path runs at `action_temperature = 0.0`
  (greedy) (CODE `config.py`; `_chat` overrides temperature to `action_temperature`). Greedy
  decoding cannot be blamed for the fence; it means the fence is literally the
  highest-probability continuation — reinforcing (b)/(d).

- **(e) "Something else" — minor: the contradictory call-shape instruction (§5) and the
  missing `<tool_response>` framing (§2.3)** further push the model off its trained
  tool-calling manifold, but these are secondary to the missing anti-fence clause.

**Verdict — PRIMARY cause:** the prompt neither engages the native tool template (so the
trained anti-fence instruction never appears) **nor** reproduces that anti-fence instruction
manually, while the model's coder pre-training prior strongly favours ```` ```json ````.
It is (d)+(b) acting together; (a) is the enabling condition. Temperature (c) is ruled out.

**The codec's `_FENCE_RE` fallback is treating the symptom.** It successfully *recovers*
fenced calls (which is why the run progressed at all), but recovery masks the misalignment
and guarantees the model is never nudged back onto its native distribution.

---

## 4. Ollama Native Tool Support: Recommendation

### Q2 — Should the harness use Ollama's `tools:` field instead of manual injection?

**What `tools:` does.** When `/api/chat` is called with a `tools:` JSON array, Ollama sets
`.Tools` in the template, so the native `# Tools` block (§1.2) renders — including the
verbatim anti-fence instruction — and Ollama parses `<tool_call>` blocks out of the response
into a structured `message.tool_calls` array.

**Sub-questions:**

- *Does Ollama's tool injection match the native template exactly?* **Yes — by construction.**
  Using `tools:` IS the native template; the per-tool wrapper becomes the enveloped
  `{"type":"function","function":{…}}` form (§1.2), and the anti-fence instruction appears
  for free. This is strictly more aligned than the harness's hand-rebuild.

- *Does Ollama expose `role:"tool"` results with Qwen's template?* **Yes.** A
  `{"role":"tool","content":…}` message renders as the `<tool_response>` user turn (§1.4).
  That is the exact result framing the model was trained on — the harness currently forgoes
  it (§2.3).

- *What do tool-result messages look like natively vs what the harness sends?* Native:
  `<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>`. Harness: English
  narrative ("Then, you filled e41…") in a normal user turn (CODE `memory.py`).

**Trade-offs for THIS harness (the honest accounting):**

- **For:** instant, guaranteed template alignment; the anti-fence instruction returns; native
  `<tool_response>` results; Ollama does the `<tool_call>` parsing; eliminates the entire
  class of "hand-rebuilt prompt drifts from the template" bugs (which is exactly #129).

- **Against / friction:**
  1. **Architectural collision with the two-phase design.** The whole `llm.py` is built
     around prefill-continuation (`_render_chatml` + `raw:True`). `tools:` only works on
     `/api/chat`, not on the raw `/api/generate` prefill path. On `beta_qwen3coder`
     `two_phase=False` already, so the single-phase `_chat` path *is* `/api/chat` and could
     adopt `tools:` cleanly — but the codec seam (`format_instructions`, `tool_definitions`,
     `parse`) would be partly bypassed, which is an abstraction change, not a tweak.
  2. **Loses the NarrativeMemory design.** The harness's deliberate choice (memory.py
     docstring) is English narrative over a ChatML message log. Native `tools:`/`role:"tool"`
     pulls toward a structured message history — a philosophical reversal for this codebase.
  3. **Streaming + parse ownership move into Ollama.** The codec's tolerant `parse()` (which
     also recovers fences and arrays) would be redundant for the happy path; you'd depend on
     Ollama's parser version behaviour.
  4. **Per-turn snapshot budgeting (#43) assumes a single rendered system string.** Moving
     tools into `tools:` changes the token accounting the snapshot budget relies on.

**Recommendation (nuanced):** Do **not** rip out the codec and switch wholesale. Instead, in
priority order:

1. **First, close the cheap gap (HIGH/easy):** add the native anti-fence clause to
   `format_instructions()` and align the call-shape instruction (§5). This is a *report-only*
   recommendation here; it likely removes most fences with zero architectural risk and lets
   you measure how much of #129 is just the missing sentence.
2. **Then, if fences persist, pilot `tools:` on the single-phase path only** (it is already
   `/api/chat`), behind a config flag, and compare RUNLOGs. The single-phase Qwen branch is
   the *one* place in this multi-model repo where native `tools:` fits without fighting the
   two-phase VibeThinker design — so it is a reasonable experiment, but it is a bigger change
   than the prompt fix and should be gated on the prompt fix not being sufficient.

---

## 5. Multi-Call Batching: Evidence and Recommendation

### Q4 — Can Qwen2.5-Coder-3B reliably batch multiple calls per turn?

**Direct evidence it CAN (RUNLOG turn 18):**

```
```json
{"name": "evaluate", "arguments": {"expression": "el => el.getAttribute('value')", "target": "zip_code_element"}}
{"name": "fill", "arguments": {"target": "zip_code_element", "text": "75201"}}
```
```

Two distinct JSON objects in one turn, and the codec's `_iter_json_values` parsed **both**
into two actions (RUNLOG turn 18 `actions` array has two entries). So **the 3B model is
capable of multi-call output**, and the harness parser handles it. This refutes "the model
can only emit one call." The *capacity* exists.

**Why it almost never does it here:**

1. **The task structure punishes batching.** This is a web form where each action mutates the
   page and the next correct ref is only known after re-snapshotting. The model (correctly)
   emits one action, waits for the new snapshot, then acts again. Single-call-per-turn is the
   *right* behaviour for most of this task — it is not a defect. (RUNLOG turns 1-17 are all
   single sequential dependent steps.)

2. **Training-data shape (inference, MODERATE confidence).** Hermes/Qwen function-calling SFT
   does include multi-call turns, but they are a minority of examples and skew toward
   *independent* parallel calls (e.g. two weather lookups). A 3B model's prior for "one call,
   then look" is strong and, for this task, accurate. I have no token-level training-data
   audit, so this is reasoned inference, flagged.

3. **The harness instruction is itself ambivalent** — `format_instructions()` says "Batch
   independent or predictable calls… emit a single call when you must see its result." For a
   form, almost every step "must see its result," so the model's single-call default agrees
   with the harness's own guidance.

4. **The instruction contradicts the native call shape (a real bug worth noting).** The
   harness tells the model to emit *consecutive `<tool_call>` blocks* (one pair per call),
   but the native template emits **one** `<tool_call>` pair with calls newline-separated
   inside (§1.3). When the model batched in turn 18 it used neither — it emitted two bare
   newline-separated objects in a single fence, which is closest to the *native* multi-call
   shape, not the harness instruction. This is more evidence the model follows its trained
   distribution over the harness text.

**Recommendation:** Treat single-call-per-turn as **expected and acceptable** for form-fill
tasks; do not chase batching as a primary goal. If batching is wanted for throughput on
*independent* steps, align the instruction to the native shape ("put each call on its own
line inside ONE `<tool_call>` block") rather than "consecutive blocks," because that is what
the model actually produces (turn 18). LOW priority relative to the fence and grounding fixes.

---

## 6. Constraining Pass: To Use or Not

### Q5 — Should a grammar/schema constraint be (re)introduced?

The `json` codec uses `format: <json-schema>` (Ollama grammar). The `hermes` codec
deliberately uses none (`json_schema=None`).

- **A JSON-schema `format` constraint would HURT here.** Ollama's `format` constrains output
  to *valid JSON for the schema* — but the native `hermes` output is `<tool_call>{json}
  </tool_call>` (or, when batched, multiple objects). A whole-output JSON-schema cannot
  express the `<tool_call>` tag wrapper or multiple top-level objects, so turning it on would
  force the model OFF its native distribution into a single bare JSON object — the opposite of
  alignment. The codec's own comment (CODE `hermes_codec.py` `constraint()`) makes this point
  and it is correct. **Do not re-introduce `format` for hermes.**

- **A GBNF grammar (`<tool_call>\n{json}\n</tool_call>`) WOULD help — but Ollama ignores it.**
  GBNF can encode the tag wrapper exactly and would *structurally eliminate the fence*. But
  CODE (`llm.py` `_act`/`_chat`) only forwards `format` (JSON-schema) to Ollama; `gbnf` is
  honoured solely by a llama.cpp backend (`DecodeConstraint.gbnf`, `config.backend`). So on
  the Ollama backend GBNF is a non-option today. It is the cleanest *eventual* fix if/when
  the llama.cpp backend is the target. (Flagged: this is backend-gated, not free.)

- **Ollama options that help WITHOUT fighting the distribution:**
  1. **Stop sequences** (see Q6) — `DecodeConstraint.stop` is already plumbed end-to-end
     (`_chat` passes `list(constraint.stop)`); hermes currently sets none.
  2. **The `tools:` field** (§4) — not a grammar, but it makes Ollama parse `<tool_call>`
     natively and re-introduces the anti-fence instruction, which is the practical equivalent
     of a soft constraint.
  3. **Prompt-level anti-fence instruction** (§3) — the cheapest "constraint" of all.

- **Stop tokens to terminate tool-call generation / prevent runaway:** see §7. The key one is
  **`</tool_call>`** (terminate after the call) — but only once the model actually emits the
  tag; while it still fences, `</tool_call>` never appears, so the *prompt* fix must land
  first or the stop is inert. `<|im_end|>` is the backstop terminator.

**Verdict:** keep hermes unconstrained at decode time; fix compliance through the prompt
(anti-fence clause) and stop sequences, with native `tools:` as the next step and GBNF as a
llama.cpp-only future upgrade.

---

## 7. Optimal Settings

All values are reasoned from MODELFILE + RUNLOG + CODE. Where I lack live A/B data I label
the value a *starting point to verify*, not a proven optimum.

| Setting | Current (CODE `config.py`) | Recommended | Rationale / evidence |
|---|---|---|---|
| `temperature` (phase-1) | `0.3` | `0.3` (keep) — or irrelevant: with `two_phase=False`, phase-1 never runs (CODE `decide`). | Only the action path executes; phase-1 temp is dead config on this branch. |
| `action_temperature` | `0.0` (greedy) | **`0.0` keep** | Verbatim string fidelity for refs/URLs/field values is essential for web actions; RUNLOG shows arg payloads are always well-formed — greedy is right. Greedy also confirms the fence is the top token, so the fix must be prompt/stop, not sampling. |
| `top_p` | `0.95` | `0.95` (inert at temp 0) | Ignored under greedy; harmless. |
| `top_k` | `0` | `0` (inert at temp 0) | Same. |
| `num_predict` (action) | `reason_tokens + action_tokens` = `4096 + 4096 = 8192` (CODE `_chat`) | **~1024-2048** for the action path | A `<tool_call>` with ≤4 calls is a few hundred tokens (CODE config comment agrees). 8192 invites runaway prose after a recovered fence. Smaller cap + a real stop sequence prevents the model from continuing to narrate. *Verify against the largest legitimate batch.* |
| **Stop sequences** | none for hermes (`constraint.stop = ()`) | **`["</tool_call>", "<|im_end|>"]`** (and consider `"\n```\n"` to cut a fence early while the prompt fix is rolling out) | `</tool_call>` ends generation the instant a native call closes; `<|im_end|>` is the ChatML backstop. NOTE both are inert until the model stops fencing — so add them *with* the prompt fix. A fence-closing stop is a stopgap, not a real fix. |
| `num_ctx` | `32768` | keep | Justified at length in CODE (#77/#92); fits 8 GB; single-runner invariant. Out of scope for #129. |
| Native `tools:` field vs manual injection | manual injection | **manual + add anti-fence clause first; pilot `tools:` second** (§4) | Cheapest high-impact fix is the prompt clause; `tools:` is the principled but larger follow-up. |
| `two_phase` | `False` | keep `False` | Correct for a non-thinking instruct model (CODE config rationale, confirmed by RUNLOG: all `reasoning` fields are empty). |

---

## 8. Prioritised Recommendations

Each: **impact** (HIGH/MED/LOW) · **effort** (easy/medium/hard) · notes. All are
*recommendations*; no code was changed (research-only per #129).

1. **Restore the native anti-fence instruction in `format_instructions()`.**
   **HIGH · easy.** Append the template's verbatim clause: *"... within `<tool_call>`
   `</tool_call>` XML tags **with NO other text. Do not include any backticks or ```json.**"*
   Evidence this is the primary lever: RUNLOG = 20/20 fences; MODELFILE shows the template
   carries this exact sentence and the harness dropped it (§3). Lowest-risk change with the
   highest expected payoff. Measure fence rate before/after.

2. **Add stop sequences to the hermes `DecodeConstraint` and shrink the action
   `num_predict`.** **HIGH · easy.** `stop = ("</tool_call>", "<|im_end|>")` and cap the
   action budget to ~1-2k. Prevents runaway continuation once the model emits tags. Plumbing
   already exists (`_chat` forwards `constraint.stop`). Caveat: inert until rec. #1 lands, so
   ship them together (§6, §7).

3. **Fix the call-shape instruction to match the native template.** **MED · easy.** Change
   "emit consecutive `<tool_call>` blocks" to "put each call on its own line inside ONE
   `<tool_call>` block" — this is what the model actually produces when it batches (RUNLOG
   turn 18) and what the template renders (§1.3). Reduces instruction/behaviour conflict.

4. **Pilot Ollama's native `tools:` field on the single-phase path, behind a flag.**
   **HIGH · medium.** Only if #1-#2 don't fully clear the fences. Gives guaranteed template
   alignment, native `<tool_response>` results, and Ollama-side `<tool_call>` parsing (§4).
   Bigger change (touches the codec seam, snapshot budgeting, and the NarrativeMemory
   philosophy) — gate it on #1 being insufficient and A/B it against RUNLOGs.

5. **Reconcile the bare-vs-enveloped tool-definition shape.** **MED · medium.** The codec
   emits **bare** schemas; the *Ollama* template renders **enveloped**
   `{"type":"function","function":{…}}` (§2.1). Either move to `tools:` (which fixes this
   automatically) or update `registry.tools_block` to the enveloped shape so manual injection
   matches the operative backend. Flag: confirm with an A/B before changing — the model
   tolerates both and this is likely secondary to the fence fix.

6. **Feed tool results as `<tool_response>` framing (only meaningful with `tools:`/structured
   history).** **MED · hard.** The model was trained on `<tool_response>` result turns (§1.4);
   the harness uses English narrative (§2.3). Aligning this likely reduces the
   "re-open browser / lost-context" confusion seen in RUNLOG turns 2/8/10. Hard because it
   reverses the NarrativeMemory design — treat as a research spike, not a quick fix.

7. **Strengthen grounding / dropdown handling and the anti-loop guard (NOT a format issue,
   but it is what actually killed the run).** **HIGH · medium.** RUNLOG shows the run failed
   on a combobox the model could not operate (`State` e65 is a `<button>`/combobox, turns
   11-17), ref hallucination (`State`, `zip_code_element`, turns 12/18), and tight loops the
   guard only weakly broke (turns 13-20). Format alignment will not fix these. Called out so
   the #129 fixes are not oversold: the wire-format work is necessary but not sufficient for a
   successful run.

---

### Appendix — evidence index

- **MODELFILE:** `ollama show qwen2.5-coder:3b-instruct --modelfile` — template captured in
  full; key fragments quoted in §1.
- **RUNLOG:** `.vibe/20260624_200826.json` — 20 turns; fence count 20/20; multi-call proof at
  turn 18; run-ending grounding/loop failures turns 9-20; all `reasoning` fields empty
  (confirms single-phase).
- **CODE (`origin/beta_qwen3coder`):** `vibeharness/codecs/hermes_codec.py`
  (`format_instructions`, `constraint`, `parse`, `_FENCE_RE`); `vibeharness/llm.py`
  (`_chat` single-phase path, `_render_chatml`, no `tools:` field, `constraint.stop`
  plumbing); `vibeharness/config.py` (`action_temperature=0.0`, `two_phase=False`,
  `codec="hermes"`, token budgets); `vibeharness/registry.py` (`tools_block` bare schema);
  `vibeharness/prompt.py` (system-prompt assembly, lines 114-120); `vibeharness/agent.py`
  (anti-loop guard, `max_actions_per_turn`); `vibeharness/memory.py` (narrative results).

**Uncertainty register (explicit):** (1) bare-vs-enveloped tool-definition impact on output
is reasoned, not measured — flagged in §2.1/#5. (2) Training-data batching distribution is
inference from the Hermes/Qwen lineage, not a token-level audit — flagged in §5. (3) The
exact fence-reduction from rec. #1 is an expectation grounded in the template's own design
intent, to be confirmed by a before/after RUNLOG — flagged in §3/#1.
