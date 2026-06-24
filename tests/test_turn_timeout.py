"""Per-turn wall-clock budget (issue #9).

Each turn's blocking decide() is bounded by Config.turn_timeout_seconds. When the
budget is exceeded the run aborts gracefully (not finished, with a clear reason)
instead of hanging forever. These tests use fakes only — no live model — and must
themselves finish fast: a hanging test means the guard is broken.
"""
from __future__ import annotations

import threading
import time
import unittest

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry

from tests._fakes import FakeLLMClient, FakeValidator


VALIDATE = {"tool": "validate", "args": {"summary": "done"}}


class SleepingLLMClient(LLMClient):
    """A fake whose decide() blocks in time.sleep longer than any tiny budget.

    Backed by a daemon-friendly Event so the test process never waits on it: the
    guard must cut the turn off well before the sleep elapses.
    """

    def __init__(self, sleep_seconds: float = 30.0):
        self._sleep = sleep_seconds
        self.started = threading.Event()

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        self.started.set()
        time.sleep(self._sleep)
        return Decision(reasoning="", action_json='{"tool": "validate", "args": {}}')


class TurnTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def _agent(self, client, *, turn_timeout_seconds=0, max_steps=10, validator=None):
        return RalphAgent(
            client, self.registry, "SYSTEM",
            Config(max_steps=max_steps, turn_timeout_seconds=turn_timeout_seconds),
            validator or FakeValidator(passed=True),
        )

    def test_blocking_turn_is_aborted_under_budget(self):
        # decide() sleeps 30s but the budget is a fraction of a second; the run must
        # return almost immediately, marked not-finished, with the timeout reason.
        client = SleepingLLMClient(sleep_seconds=30.0)
        agent = self._agent(client, turn_timeout_seconds=1, max_steps=5)

        start = time.monotonic()
        result = agent.run("slow task")
        elapsed = time.monotonic() - start

        # Cut off well before the 30s sleep (generous ceiling for slow CI).
        self.assertLess(elapsed, 10.0, "guard did not cut off the blocking turn")
        self.assertTrue(client.started.is_set(), "decide() never actually ran")
        self.assertFalse(result.finished)
        self.assertIn("generation budget", result.stop_reason)
        self.assertIn("aborting", result.stop_reason)
        # The abort is observable as a failed action on the (single) recorded turn.
        self.assertEqual(len(result.turns), 1)
        last = result.turns[0].actions[-1]
        self.assertFalse(last.ok)
        self.assertIn("generation budget", last.observation)

    def test_fast_turn_under_budget_completes_normally(self):
        # A normal scripted fake well within the budget runs to a clean finish even
        # with the guard enabled.
        client = FakeLLMClient([VALIDATE])
        agent = self._agent(client, turn_timeout_seconds=5, max_steps=5,
                            validator=FakeValidator(passed=True, reason="ok"))
        result = agent.run("fast task")
        self.assertTrue(result.finished)
        self.assertEqual(result.final_summary, "ok")
        self.assertEqual(result.stop_reason, "")

    def test_disabled_guard_is_behaviour_preserving(self):
        # turn_timeout_seconds=0 (default): no threading, identical to before.
        client = FakeLLMClient([VALIDATE])
        agent = self._agent(client, turn_timeout_seconds=0, max_steps=5,
                            validator=FakeValidator(passed=True, reason="done"))
        result = agent.run("normal task")
        self.assertTrue(result.finished)
        self.assertEqual(result.final_summary, "done")
        self.assertEqual(result.stop_reason, "")


if __name__ == "__main__":
    unittest.main()
