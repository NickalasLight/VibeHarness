"""Natural-language narrative memory.

Instead of a JSON/ChatML message history, the agent's past is kept as a plain
English account ("First, you ... Then, you ...") which is regenerated into the
prompt each turn. Thinking is never stored here.
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
