"""Runtime configuration. One immutable value object passed where needed (DIP-friendly)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # model / sampling
    # ISSUE #123 / branch beta_qwen3coder (ISOLATED): the default model on this line is
    # a ~3B Qwen2.5-Coder, paired with the `hermes` codec below so the harness speaks
    # the model's native trained dialect. (On `beta` the default is "vibethinker" +
    # "json"; this branch never merges back, so these defaults stay local to it.)
    #
    # ⚠️ 3B-PARITY NOTE: there is NO dense ~3B *Qwen3-Coder* model — the entire
    # Qwen3-Coder line is MoE (smallest is 30B-A3B: 30B total / 3B ACTIVE; then
    # 80B-A3B "Next"; then 480B-A35B). Using any of those breaks apples-to-apples
    # parity with VibeThinker-3B (a 3B DENSE model) and won't fit the 8 GB card. So
    # this branch uses the closest true ~3B DENSE coder: Qwen2.5-Coder-3B-Instruct
    # (1.9 GB at Q4_K_M; same Qwen2.5/Hermes tool dialect; VibeThinker itself derives
    # from Qwen2.5-(Coder-)3B, so this is the cleanest 3B-for-3B substitute). The
    # Qwen3-Coder discrepancy is flagged in QWEN3CODER_ANALYSIS.md and
    # QWEN3CODER_DIVERGENCE.md. Swap the tag here if/when a dense ~3B Qwen3-Coder ships.
    model: str = "qwen2.5-coder:3b-instruct"
    temperature: float = 0.3          # phase-1 reasoning temperature (some diversity helps)
    action_temperature: float = 0.0   # phase-2 action: greedy, for verbatim string fidelity
    top_p: float = 0.95
    top_k: int = 0
    num_gpu: int = 99                 # force full GPU offload (NVIDIA via CUDA)

    # loop
    max_steps: int = 15               # <= 0 means unlimited
    max_actions_per_turn: int = 4     # cap on tool calls the model may emit per turn

    # Per-turn wall-clock budget (seconds). 0 (default) disables the guard,
    # preserving the original behaviour exactly: decide() is called inline with no
    # threading. When > 0, each turn's blocking decide() is run in a daemon worker
    # thread and joined with this timeout; if it overruns, the turn is recorded as a
    # failure and the run ends gracefully (RunResult.finished stays False). Caps a
    # turn's wall-clock TIME, complementing reason_tokens/action_tokens which only
    # cap token COUNT. The stuck generation thread is a daemon so it cannot keep the
    # process alive after the run returns (tradeoff: that thread may still be running
    # detached until the process exits or its blocking I/O unwinds).
    turn_timeout_seconds: int = 0

    # tool-call wire format (see vibeharness.codec.get_codec). "json" is the
    # decode-constrained baseline; other codecs add alternative formats.
    # ISSUE #123 (beta_qwen3coder): default to "hermes" — the native Qwen2.5 / Hermes
    # <tool_call>{"name","arguments"} format + bare <tools> function-schema definitions
    # that Qwen2.5-Coder reads (ground-truthed from the model's tokenizer_config.json
    # chat template; see QWEN3CODER_ANALYSIS.md), so the harness aligns to the model's
    # trained dialect. On `beta` this default is "json".
    codec: str = "hermes"

    # ISSUE #125 (beta_qwen3coder): SINGLE-phase generation. The two-phase
    # reason-then-act transport (free <think> pass, then a raw continuation constrained
    # to the action) exists for VibeThinker, which emits long <think> chains. But
    # qwen2.5-coder:3b-instruct is a NON-thinking instruct model: it emits a tool call
    # immediately, so phase 1 produces a real tool call that is then DISCARDED and phase 2
    # emits another — halving throughput and dropping the model's primary intent (observed
    # in #125 iter 2: recovery clicks landed in the thrown-away phase-1 channel). With
    # two_phase=False, decide() does ONE native /api/chat generation and the codec parses
    # the call from it. On `beta`/`beta_mythos_fast` (VibeThinker) this stays True.
    two_phase: bool = False

    # context + per-turn token budgets.
    # num_ctx is the whole window (system prompt + history + generation share it).
    #
    # ISSUE #77 (single-runner fix): num_ctx is PINNED to one value that fits the
    # 8 GB GPU. Background (diagnosed in #76 / PR #87): the old 131072 request never
    # actually fit — Ollama's auto-fit shrank each request to a VARYING size
    # (4096/16384/32768) depending on free VRAM at the moment. Ollama keys a
    # llama-server runner by (model, context-size), so every distinct auto-fit size
    # spawned a NEW runner; with MAX_LOADED_MODELS/KEEP_ALIVE unset they were never
    # evicted, stacked up, and exhausted the GPU -> the OOM/CUDA crash. Sending one
    # fixed num_ctx on EVERY request (see OllamaClient._options) means only ONE
    # runner shape is ever requested, so at most one runner exists.
    #
    # 32768 is chosen as the largest of the observed auto-fit sizes — it has actually
    # loaded on this card, so it is known-fittable, and it is large enough that a
    # heavy real page still fits: the worst-case YouTube watch snapshot (~45k chars
    # ≈ ~11k tokens at 4 chars/token) plus the prompt and output reservation stays
    # well under 32768. If 32768 proves unstable in live runs, drop to 16384 (also
    # an observed-fitting size) — both keep the single-runner invariant.
    num_ctx: int = 32768
    # ISSUE #92 (token rebalance): the output reservation
    # (reason_tokens + action_tokens) is subtracted from num_ctx before anything else
    # can be fed in (see vibeharness.snapshot_budget.input_budget_tokens). With the
    # pinned 32768 window, the OLD reservation (2048 + 16384 = 18432, 56% of the
    # window) left only ~13312 input tokens — so after a heavy ~11k-token page
    # snapshot, barely ~2k tokens remained for the system prompt + history, and the
    # snapshot got trimmed in realistic multi-step turns. We rebalance:
    #
    #   action_tokens 16384 -> 4096. Phase 2 emits the CONSTRAINED JSON tool call
    #   (see llm.py::_act). It is capped at max_actions_per_turn (=4) tool calls; even
    #   four web actions with verbose string args is a few hundred tokens — well under
    #   1k. 16384 was wildly oversized. 4096 keeps ~10x headroom over the realistic
    #   worst case while reclaiming 12288 tokens of input budget.
    #
    #   reason_tokens 2048 -> 4096. Phase 1 is the model's free <think> chain (llm.py
    #   ::_reason, stops at </think>). The VibeThinker 3B emits LONG reasoning, so 2048
    #   risked truncating its thinking mid-chain. We DOUBLE it to 4096 so reasoning is
    #   not starved, while staying small enough to keep a large positive input budget.
    #
    # New reservation = 4096 + 4096 = 8192 (was 18432). With the 1024 safety margin,
    # input_budget = 32768 - 8192 - 1024 = 23552 tokens (was 13312) — a ~11k-token
    # snapshot + a realistic ~2-3k-token system prompt + several k of history all fit
    # without dropping the snapshot. See tests/test_snapshot_budget.py.
    #
    # ISSUE #123 (beta_qwen3coder) NOTE: unlike VibeThinker, Qwen2.5-Coder-3B-Instruct
    # is NOT a long-chain reasoning model — it does not emit verbose <think> chains by
    # default, so phase-1 typically returns quickly (well under 4096). We KEEP 4096 as a
    # harmless ceiling (the input-budget math is identical to beta, so all snapshot/budget
    # tests carry over unchanged); it is a cap, not a target. If a future run shows the
    # model never reasons at length, this can be lowered to reclaim input budget without
    # affecting correctness. The action reservation is unchanged — phase-2 emits the
    # compact <tool_call> blocks (a few hundred tokens), so 4096 has ~10x headroom.
    reason_tokens: int = 4096         # phase 1 (free reasoning, discarded) — see #92/#123
    action_tokens: int = 4096         # phase 2 (codec action; unconstrained for hermes) — #92

    # observation rendering
    observation_char_limit: int = 12000  # truncate big tool outputs in the narrative

    # backend
    backend: str = "ollama"           # "ollama" or "llamacpp"
    ollama_url: str = "http://127.0.0.1:11434"
    # ISSUE #77: a CONSTANT keep_alive sent on every Ollama request. Keeping the
    # value identical across requests (paired with the pinned num_ctx and
    # OLLAMA_MAX_LOADED_MODELS=1) means a fresh request reuses / re-pins the single
    # existing runner instead of leaving idle runners around to be re-loaded. "30m"
    # comfortably spans a long agent run without holding the GPU forever after it.
    ollama_keep_alive: str = "30m"
    llamacpp_url: str = "http://127.0.0.1:8080"
    request_timeout: int = 600

    # web toolset (Playwright Agent CLI)
    web_session: str = "vibe"
    web_cli_timeout: int = 90
    web_observation_char_limit: int = 14000
    # Cap on the auto-injected live page snapshot rendered into the per-turn system
    # prompt (issue #24). ARIA snapshots can be 800+ lines; truncate so the current
    # page state is shown without crowding the task out of context.
    # Raised 6000 -> 40000 (~10k tokens) per the issue #28 snapshot-size analysis
    # (SNAPSHOT_SIZE_ANALYSIS.md): 8/9 real pages exceed 6000 chars, and a 6k cap
    # truncates BEFORE the consent/Accept buttons the agent is told to click (those
    # late-DOM overlays land at the END of the ARIA tree, e.g. YouTube ~45k). 40000
    # is the knee of the coverage/budget curve. NOTE: even 40k can still clip the
    # very largest pages (e.g. w3schools ~63k) because overlay controls are emitted
    # last; the longer-term fix is to PRIORITIZE interactive/overlay controls in the
    # injected snapshot rather than rely on cap size (future enhancement, see #28).
    #
    # As of issue #43 this fixed cap is no longer the primary control: the snapshot
    # is sized DYNAMICALLY each turn so it may be as large as the remaining context
    # window allows (a 200k snapshot beside a 50k message is fine; a 500k snapshot
    # beside the same message is truncated only because together they would overflow
    # num_ctx). ``web_snapshot_char_limit`` is kept as an ABSOLUTE CEILING / safety
    # fallback: the dynamic budget is min(dynamic_budget, web_snapshot_char_limit).
    # Set it very high (or rely on the dynamic budget alone) to let the window be the
    # only limit. The dynamic computation lives in vibeharness.snapshot_budget.
    web_snapshot_char_limit: int = 2_000_000

    # --- dynamic snapshot budget (issue #43) ---
    # The live page snapshot injected into the per-turn system prompt is truncated
    # ONLY when including it whole would push the full model message (system prompt +
    # per-turn user/history) past the usable input window. The usable input window is
    #   input_budget = num_ctx - (reason_tokens + action_tokens) - snapshot_safety_margin_tokens
    # all in TOKENS. We estimate tokens from characters with a fixed, conservative
    # ratio (chars per token). 4.0 is the long-standing English rule of thumb; we keep
    # it configurable and deliberately treat it as a *floor* (real ARIA snapshots —
    # refs, URLs, punctuation — often pack FEWER chars per token, i.e. more tokens per
    # char, so a too-high ratio would UNDER-count tokens and risk overflow). Combined
    # with the explicit token safety margin this keeps us safely under num_ctx.
    snapshot_chars_per_token: float = 4.0
    # Extra tokens held back on top of the output reservation, absorbing chat-template
    # wrapping, role tokens, and tokenizer estimate error so we never reach num_ctx.
    snapshot_safety_margin_tokens: int = 1024
    web_headless: bool = False        # headed by default so a human can watch
    web_browser: str = "chrome"

    # --- snapshot prose rendering (issue #64) ---
    # When True, the auto-injected live page snapshot is run through the deterministic
    # WebArena-style ARIA->prose transform (vibeharness.snapshot_prose) before injection,
    # instead of the raw Playwright ARIA-YAML. The prose prunes generic/img noise and
    # emits one ref-keyed line per interesting control, which small models reason over
    # far more reliably. Refs are preserved inline so the discrete web subtools (click/
    # fill/…) keep working unchanged. This is an A/B SEAM, not a replacement: set False
    # (default) to inject the raw ARIA snapshot exactly as before. Budgeting/diagnostics
    # are unchanged — only the text fed into the page section differs.
    web_snapshot_prose: bool = False
