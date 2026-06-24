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
    ollama_url: str = "http://127.0.0.1:11434"
    request_timeout: int = 600

    # web toolset (Playwright Agent CLI)
    web_session: str = "vibe"
    web_cli_timeout: int = 90
    web_observation_char_limit: int = 14000
    web_headless: bool = False        # headed by default so a human can watch
    web_browser: str = "chrome"
