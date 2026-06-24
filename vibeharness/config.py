"""Runtime configuration. One immutable value object passed where needed (DIP-friendly)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # model / sampling
    model: str = "vibethinker"
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
    codec: str = "json"

    # context + per-turn token budgets.
    # num_ctx is the whole window (system prompt + history + generation share it).
    # 131072 is the model's max; on an 8 GB card the KV overflow spills to system
    # RAM (Windows shared GPU memory), so context fills slow down but don't OOM.
    # Use OLLAMA_NUM_PARALLEL=1 so a single instance gets the whole window.
    num_ctx: int = 131072
    reason_tokens: int = 2048         # phase 1 (free reasoning, discarded)
    action_tokens: int = 16384        # phase 2 (constrained JSON action) — can be large

    # observation rendering
    observation_char_limit: int = 12000  # truncate big tool outputs in the narrative

    # backend
    backend: str = "ollama"           # "ollama" or "llamacpp"
    ollama_url: str = "http://127.0.0.1:11434"
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
