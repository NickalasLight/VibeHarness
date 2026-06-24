"""Advisor multi-turn message construction (issue #129/#130/#131).

The advisor now replays recent turns as a real role-tagged ``messages`` list instead of
a flattened prose blob, and is deliberately NOT given the native tools: field (VibeThinker
is confused by enveloped schemas — verified live). These cover the pure message-building.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from vibeharness.advisor import build_advisor_messages


@dataclass
class _Action:
    tool: str
    args: dict
    observation: str
    ok: bool = True


@dataclass
class _Turn:
    index: int
    raw_action: str = ""
    actions: list = field(default_factory=list)


class BuildAdvisorMessagesTest(unittest.TestCase):
    def _turns(self):
        return [
            _Turn(1, raw_action='I will fill the name.\n<tool_call>{"name":"fill"}</tool_call>',
                  actions=[_Action("fill", {"target": "e1", "text": "Al"}, "you filled e1")]),
            _Turn(2, raw_action='<tool_call>{"name":"click"}</tool_call>',
                  actions=[_Action("click", {"target": "e2"}, "click FAILED", ok=False)]),
        ]

    def test_starts_with_system_and_task(self):
        msgs = build_advisor_messages("TASK X", self._turns(), 5)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("TASK X", msgs[1]["content"])

    def test_each_turn_yields_assistant_then_tool(self):
        msgs = build_advisor_messages("T", self._turns(), 5)
        roles = [m["role"] for m in msgs]
        # system, user(task), [assistant, tool] x2, user(ask)
        self.assertEqual(roles, ["system", "user", "assistant", "tool",
                                 "assistant", "tool", "user"])

    def test_preamble_becomes_assistant_content(self):
        msgs = build_advisor_messages("T", self._turns(), 5)
        assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
        self.assertIn("fill the name", assistants[0])

    def test_tool_message_has_status_and_observation(self):
        msgs = build_advisor_messages("T", self._turns(), 5)
        tools = [m for m in msgs if m["role"] == "tool"]
        self.assertIn("OK", tools[0]["content"])
        self.assertIn("you filled e1", tools[0]["content"])
        self.assertIn("FAIL", tools[1]["content"])

    def test_ends_with_user_ask(self):
        msgs = build_advisor_messages("T", self._turns(), 5)
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("should the agent do next", msgs[-1]["content"])

    def test_only_last_n_turns_kept(self):
        turns = [_Turn(i, actions=[_Action("x", {}, "o")]) for i in range(1, 6)]
        msgs = build_advisor_messages("T", turns, 2)
        assistants = [m for m in msgs if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 2)  # only last 2 turns


if __name__ == "__main__":
    unittest.main()
