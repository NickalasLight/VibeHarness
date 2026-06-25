"""Runtime configuration. One immutable value object passed where needed (DIP-friendly)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # model / sampling
    model: str = "vibethinker"
    temperature: float = 0.3          # 0.3 gave clean content in our experiments
    top_p: float = 0.95
    top_k: int = 0
    num_gpu: int = 99                 # force full GPU offload (NVIDIA via CUDA)

    # loop
    max_steps: int = 15               # <= 0 means unlimited

    # context + per-turn token budgets.
    # num_ctx is the whole window (system prompt + history + generation share it).
    # 131072 is the model's max; on an 8 GB card the KV overflow spills to system
    # RAM (Windows shared GPU memory), so context fills slow down but don't OOM.
    # Use OLLAMA_NUM_PARALLEL=1 so a single instance gets the whole window.
    num_ctx: int = 131072
    reason_tokens: int = 2048         # phase 1 (free reasoning, discarded)
    action_tokens: int = 16384        # phase 2 (constrained JSON action) — can be large

    # observation rendering
    observation_char_limit: int = 1500  # truncate big tool outputs in the narrative

    # backend
    ollama_url: str = "http://127.0.0.1:11434"
    request_timeout: int = 600

    # web toolset (Playwright Agent CLI)
    web_session: str = "vibe"
    web_cli_timeout: int = 90
    web_observation_char_limit: int = 4000
    web_headless: bool = False        # headed by default so a human can watch
    web_browser: str = "chrome"

    # --- Escalation / API provider ---
    # When the local model gets stuck, the agent swaps its LLM client to an
    # external API model (same browser/session) for a stronger second attempt.
    # base_url / api_key_env live in providers.PROVIDERS, NOT here — no secrets in Config.
    escalation_enabled: bool = True
    escalation_provider: str = "zhipuai"
    escalation_model: str = "glm-5.2"
    escalation_stuck_threshold: int = 3       # consecutive identical tool calls -> stuck
    escalation_on_premature_validate: bool = True  # escalate on first premature validate

    # --- Validation provider ---
    # The validator uses this API model by default for an independent verdict. Leave
    # validation_provider blank ("") to fall back to the main (local) client.
    validation_provider: str = "zhipuai"
    validation_model: str = "glm-5.2"
