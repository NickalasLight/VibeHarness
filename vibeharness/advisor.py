"""VibeThinker Advisor — periodic free-text hint generator for the Qwen base agent.

Every N Qwen turns the advisor is called with the task + the last N action turns and
returns concise plain-English advice: what the agent is doing wrong, what to try next, etc.

The advice is injected into the Qwen agent's next-turn user message as:
  <user_advice>The human user gives you the following hint: '...'</user_advice>

VibeThinker is called with temperature=1.0 (Config.advisor_temperature) and NO constrained
output schema — just free reasoning + plain-text continuation.  This is intentional: the
advisor is a hint channel, not a decision maker.  Two-phase generation is used because
VibeThinker always emits a <think> chain first; the continuation (phase 2, no JSON schema)
is the actual advice text.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .config import Config
from .llm import OllamaClient
from .codec import DecodeConstraint

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SYSTEM = (
    "You are an expert web-form automation observer. "
    "You watch an AI agent filling a job-application form and give it concise, direct advice. "
    "State specifically: (1) what is going wrong or what progress has been made, "
    "(2) what the agent should do NEXT to advance the form. "
    "Be concrete — name field labels, actions, or problems. 2-5 sentences only. "
    "Plain English. No JSON, no code, no lists."
)


def _build_user(task: str, history: str) -> str:
    return (
        f"TASK:\n{task}\n\n"
        f"RECENT AGENT ACTIONS (last turns):\n{history}\n\n"
        "What is going wrong (if anything) and what should the agent do next?"
    )


def format_turns_for_advisor(turns: list, n: int) -> str:
    """Render the last ``n`` turns as readable action history for the advisor prompt."""
    import json
    recent = turns[-n:] if len(turns) >= n else turns
    lines = []
    for turn in recent:
        lines.append(f"Turn {turn.index}:")
        for a in turn.actions:
            status = "OK" if a.ok else "FAIL"
            tool = a.tool or "(none)"
            args_s = json.dumps(a.args, ensure_ascii=False)[:120] if a.args else ""
            obs = (a.observation or "")[:200]
            lines.append(f"  [{status}] {tool} {args_s}")
            lines.append(f"         -> {obs}")
    return "\n".join(lines)


class VibeThinkerAdvisor:
    """Calls VibeThinker as a free-text advisor alongside the Qwen base agent.

    Uses a separate OllamaClient scoped to the advisor model + temperature, derived
    from the run Config via ``dataclasses.replace``.  OLLAMA_MAX_LOADED_MODELS must
    be 2 (set by the caller before constructing OllamaClient) so both models stay hot.
    """

    def __init__(self, config: Config) -> None:
        advisor_cfg = replace(
            config,
            model=config.advisor_model,
            temperature=config.advisor_temperature,
            action_temperature=config.advisor_temperature,
            reason_tokens=2048,   # advisor reasoning cap — shorter than the main agent
            action_tokens=1024,   # advice text cap
            two_phase=True,       # VibeThinker must reason before answering
        )
        self._client = OllamaClient(advisor_cfg)
        self._interval = config.advisor_interval
        print(f"[advisor] VibeThinkerAdvisor active — model={config.advisor_model} "
              f"temp={config.advisor_temperature} every={config.advisor_interval} turns",
              flush=True)

    def advise(self, task: str, turns: list) -> str:
        """Generate free-text advice from the last ``self._interval`` turns."""
        history = format_turns_for_advisor(turns, self._interval)
        system = _SYSTEM
        user = _build_user(task, history)
        print("\n[advisor] calling VibeThinker for advice...", flush=True)
        # Two-phase: phase 1 = free <think> reasoning (discarded), phase 2 = advice text.
        reasoning = self._client._reason(system, user, on_token=None)
        advice = self._client._act(system, user, reasoning,
                                   DecodeConstraint(json_schema=None), on_token=None)
        # Strip any leaked <think> tags from phase-2 text (defensive).
        advice = _THINK_RE.sub("", advice).strip()
        print(f"[advisor] hint: {advice[:120]}{'...' if len(advice) > 120 else ''}",
              flush=True)
        return advice
