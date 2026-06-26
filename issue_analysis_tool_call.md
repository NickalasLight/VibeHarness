## Analysis: Qwen2.5-Coder tool-call format, multi-call batching, and Ollama integration — ground-truth audit

### Motivation

Recent iterative runs on `beta_qwen3coder` revealed that `qwen2.5-coder:3b-instruct`:
- Emits ` ```json ... ``` ` markdown fences instead of native `<tool_call>...</tool_call>` tags
- Emits exactly 1 tool call per turn despite instructions to batch multiple
- The harness currently bypasses Ollama's native tool-call support entirely, injecting `<tools>` manually into the system prompt

Before implementing any further tooling fixes, we need a rigorous, evidence-based answer to the following questions. This is a **research and analysis issue only** — no code changes should be made here. Implementation of recommendations goes in a dependent issue.

### Questions to answer (with evidence, not assumption)

#### 1. Verified native format
What does the Qwen2.5-Coder-3B-Instruct model's chat template actually emit for tool calls?
- Obtain the actual chat template (`ollama show qwen2.5-coder:3b-instruct --modelfile` or model card on HuggingFace)
- What role/tag format does the template emit for: (a) tool definitions, (b) a single tool call, (c) a tool result returned to the model?
- Is `<tool_call>` the canonical format or does the template use something else?

#### 2. Ollama native tool-call support
Ollama's `/api/chat` accepts a `tools:` field and injects tool definitions via the model's own chat template.
- Does the current harness use this? (It appears not — it injects `<tools>` manually in the system prompt.)
- Would using Ollama's native `tools:` field produce better-aligned prompts for Qwen2.5-Coder?
- What are the trade-offs: (a) does Ollama's tool injection match what the model was trained on? (b) does it support all tool schemas we need? (c) does it expose `role: "tool"` for tool results?
- Is the constraining phase (`format: json_schema`) needed at all if we use Ollama native tools?

#### 3. Why is the model generating ` ```json ``` ` fences?
Analyse the root cause: model training, prompt format mismatch, temperature, or the single-phase `two_phase=False` approach?
- Does the `_chat()` single-pass approach give the model the same signals as the native template would?
- Is the `<tools>` block format the harness emits identical to what the Ollama template would generate?
- What does the raw prompt look like when it hits the model (inspect via diagnostics logs or reconstruct)?

#### 4. Multi-call batching in Qwen's training
Does Qwen2.5-Coder's training data include examples of consecutive `<tool_call>` blocks in a single assistant turn?
- Check the model's training methodology (Hermes dataset, function-calling datasets) for multi-call patterns.
- Is "emit multiple tool_call blocks" a behaviour the 3B size can reliably reproduce, or is this a capacity issue?
- Is the new concrete two-call example in `turn_action_hint` the right signal, or would something else help more?

#### 5. Constraining pass necessity
The `json` codec enforces `format: json_schema` (Ollama grammar constraint); the `hermes` codec does not.
- For Qwen with Ollama: does a grammar/schema constraint actually improve tool-call reliability, or does it fight the model's native training distribution?
- Are there Ollama settings (grammar, format, stop tokens) that improve Qwen tool-call compliance without fighting the model's weights?
- What stop tokens should terminate Qwen's tool-call generation (currently: none beyond `action_tokens` cap)?

#### 6. Recommended settings
Based on the analysis, what are the optimal `num_ctx`, `temperature`, `action_temperature`, `top_p`, `top_k`, `num_predict`, and stop-token settings for Qwen2.5-Coder-3B tool calling under Ollama?

### Deliverable

A file `qwen_tool_call_analysis_report.md` in the repo root containing:
1. Verified answer to each question above with cited evidence (model card, Ollama docs, training data, or direct model inspection).
2. Prioritised list of recommended changes, each labelled: HIGH/MED/LOW impact, easy/hard to implement.
3. Recommended approach for the implementation sprint (dependent issue).

### Notes
- **No code changes in this issue.** Analysis only.
- Dependent issue: "Implementation: Apply Qwen tool-call best-practice recommendations (Dependent on #NN)" — deferred to next sprint.
- Branch: `beta_qwen3coder`
