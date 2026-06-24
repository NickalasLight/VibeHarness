"""Natural-language narrative memory.

The agent's past kept as a plain English account ("First, you ... Then, you ...").

FATE under native stateful chat (issue #129/#130/#131): on the native path the model's
transport-level memory is now the REAL ``chat_history`` (system/user/assistant/tool
messages) maintained by :class:`~vibeharness.agent.RalphAgent`, NOT this prose. So when
``native_tools`` is active the narrative is no longer injected into the model's user
message. It is KEPT (not removed) as a SUPPLEMENT for two consumers that read prose, not
transport messages:

  * the VibeThinker advisor's history rendering, and
  * the human-readable run transcript / logs (``RunResult.transcript``).

On the LEGACY (non-native) path it remains the agent's only memory and is injected each
turn exactly as before. Thinking is never stored here.
"""
from __future__ import annotations


class NarrativeMemory:
    def __init__(self) -> None:
        self._steps: list[str] = []

    def record(self, observation: str) -> None:
        self._steps.append(observation)

    def render(self) -> str:
        if not self._steps:
            return "You have not taken any actions yet."
        lines = []
        for i, obs in enumerate(self._steps):
            connector = "First" if i == 0 else "Then"
            lines.append(f"{connector}, {obs}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._steps)
