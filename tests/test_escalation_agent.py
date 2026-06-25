"""Agent-level wiring of stuck-detection escalation: the live LLM client is
swapped in-place (same session) when the model gets stuck or validates prematurely."""
import json
import os
import tempfile
import unittest
from unittest import mock

from vibeharness import providers
from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry
from vibeharness.validation import Validator, Verdict


class FakeLLMClient(LLMClient):
    def __init__(self, actions):
        self._actions = actions
        self._i = 0

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        payload = action if isinstance(action, str) else json.dumps(action)
        return Decision(reasoning="", action_json=payload)


class FakeValidator(Validator):
    def __init__(self, passed=True, reason="ok"):
        self._passed, self._reason = passed, reason

    def validate(self, task, history, on_reason=None, on_action=None):
        return Verdict(self._passed, self._reason)


class EscalationAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def tearDown(self):
        self.tmp.cleanup()

    def _agent(self, actions, cfg, validator=None):
        client = FakeLLMClient(actions)
        return client, RalphAgent(client, self.registry, "SYS", cfg,
                                  validator or FakeValidator(passed=True))

    def test_stuck_triggers_client_swap(self):
        # Same list_directory call repeated -> 3 consecutive -> escalate once.
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=5, escalation_enabled=True,
                     escalation_stuck_threshold=3, validation_provider="")
        original, agent = self._agent([action], cfg)
        swapped = FakeLLMClient([action])
        with mock.patch.object(providers, "make_api_client",
                               return_value=swapped) as mk:
            agent.run("loop")
        mk.assert_called_once_with("zhipuai", "glm-5.2")
        self.assertIs(agent._client, swapped)        # client replaced in-place

    def test_escalation_disabled_does_not_swap(self):
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=5, escalation_enabled=False, validation_provider="")
        original, agent = self._agent([action], cfg)
        with mock.patch.object(providers, "make_api_client") as mk:
            agent.run("loop")
        mk.assert_not_called()
        self.assertIs(agent._client, original)

    def test_premature_validate_triggers_swap(self):
        VALIDATE = {"tool": "validate", "args": {"summary": "done"}}
        cfg = Config(max_steps=2, escalation_enabled=True,
                     escalation_on_premature_validate=True, validation_provider="")
        original, agent = self._agent([VALIDATE], cfg,
                                      validator=FakeValidator(passed=False, reason="nope"))
        swapped = FakeLLMClient([VALIDATE])
        with mock.patch.object(providers, "make_api_client",
                               return_value=swapped) as mk:
            agent.run("t")
        mk.assert_called_once()
        self.assertIs(agent._client, swapped)

    def test_swap_happens_at_most_once(self):
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=10, escalation_enabled=True,
                     escalation_stuck_threshold=3, validation_provider="")
        _, agent = self._agent([action], cfg)
        with mock.patch.object(providers, "make_api_client",
                               return_value=FakeLLMClient([action])) as mk:
            agent.run("loop")
        self.assertEqual(mk.call_count, 1)            # never re-escalates

    def test_missing_api_key_is_safe_noop(self):
        # No env var -> make_api_client raises RuntimeError -> agent logs and continues.
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=4, escalation_enabled=True,
                     escalation_stuck_threshold=3, validation_provider="")
        original, agent = self._agent([action], cfg)
        with mock.patch.dict(os.environ, {}, clear=True):
            result = agent.run("loop")
        self.assertIs(agent._client, original)        # fell back, no crash
        self.assertEqual(len(result.turns), 4)


if __name__ == "__main__":
    unittest.main()
