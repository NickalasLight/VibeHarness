## Feature: Stateful chat history + Qwen-optimised context management

### Background

The current harness uses a **stateless per-turn transport**: every call to `OllamaClient.decide()` sends only `system` + one `user` message (containing the `NarrativeMemory` prose summary of past turns). The model never sees its own previous `<tool_call>` outputs or tool results as actual chat messages — only a paraphrased summary written by the harness.

This creates several problems for the Qwen2.5-Coder-3B base agent:

1. **Loop / failure learning**: The model can't learn from its own previous mistakes because tool errors are fed as prose ("Then you called fill on e42 and got: ERROR: not interactable") rather than as proper conversation turns in the format the model was trained on.
2. **Format confusion**: Qwen2.5-Coder is trained with multi-turn ChatML history (system / user / assistant / tool). When it only receives a single `user` message, it can't apply its trained multi-turn behaviour.
3. **No per-turn assistant anchoring**: We never send the model's own previous output as an `assistant` role message — the prose summary is a lossy representation.
4. **Context overflow management is absent**: When chat history grows beyond `num_ctx` tokens, there is no trimming strategy.

### Goal

Replace the single-`user`-message + NarrativeMemory prose transport with **proper stateful multi-turn chat messages** (system / user / assistant / tool roles), trimmed oldest-first when approaching the context window.

### Analysis required (to be done by the implementing agent BEFORE writing code)

**Grounding in Qwen2.5-Coder native format:**
- Read the model's tokenizer_config.json to confirm the exact ChatML chat-template and how it renders `role: "tool"` messages.
- Confirm whether Ollama's `/api/chat` correctly injects tool-role messages into the Qwen2.5 chat template. If `role: "tool"` is not natively supported, determine the correct fallback (e.g. wrapping tool results as a `user` message).

**Best practice decision points** (decide with evidence, not assumption):
1. **Tool result role**: `role: "tool"` (Qwen native, Ollama-supported?) vs `role: "user"` with XML wrapper vs plain `user` prose. Ground truth: what does the Qwen2.5-Coder training data look like for tool responses?
2. **Error / failure results**: Should failed tool calls be formatted differently from successes (e.g. XML `<tool_error>` wrapper)? Evidence from Qwen docs and observed model behaviour.
3. **Advisor / validator results**: Best role: `user` with a wrapper tag (current approach) or a dedicated system-level injection each turn?
4. **Context trimming strategy**: FIFO message eviction (drop oldest user+assistant+tool triplet first) vs summarisation. Ground truth: least-information-loss strategy for a 32k window with a 3B model.
5. **Live snapshot handling**: With stateful history, do we still inject the snapshot per-turn into the system prompt, or into the most-recent user message?

### Acceptance criteria

- [ ] `OllamaClient.decide()` (or a new overload) accepts and sends a growing list of chat messages instead of a single `user` string.
- [ ] `RalphAgent` maintains a `chat_history: list[dict]` (role+content) across turns instead of NarrativeMemory prose — or hybridises (chat history + minimal prose for novel observations).
- [ ] Tool results (success and failure) are inserted as the appropriate role determined by the analysis.
- [ ] When `system + history` tokens approach `num_ctx - output_reservation`, the harness trims OLDEST messages first (preserving system and recent messages) until it fits.
- [ ] Advisor / validator injections follow the best-practice role determined by the analysis.
- [ ] All existing tests pass; new tests cover context trimming and message-list construction.
- [ ] A brief QWEN3CODER_DIVERGENCE.md entry is added noting this as a Qwen-specific chat transport divergence from `beta`.

### Files likely to change

- `vibeharness/llm.py` — `OllamaClient.decide()` / `_chat()` / `_reason()` / `_act()`
- `vibeharness/agent.py` — `RalphAgent.run()` turn loop, memory / history management
- `vibeharness/memory.py` — `NarrativeMemory` (may be deprecated or reduced)
- `vibeharness/snapshot_budget.py` — ensure snapshot sizing accounts for history tokens
- `vibeharness/config.py` — add `chat_history_max_turns` or similar config knob

### Branch

`beta_qwen3coder` — this is a Qwen-specific divergence; must NOT be synced to `beta` or `beta_mythos_fast`.

### Priority

**Critical** — the 3B model's failure to learn from loops and errors is directly caused by this architecture gap.
