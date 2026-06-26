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
        # ISSUE #197: default escalator is now deepseek / deepseek-v4-flash (zhipuai key
        # is account rate-limited → 429, so the old default silently no-oped).
        # ISSUE #198: the escalation client also carries the run's browser User-Agent.
        mk.assert_called_once_with("deepseek", "deepseek-v4-flash",
                                   user_agent=cfg.request_user_agent)
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

    # ---- issue #191: per-model codec/path switch + recorded events ----
    def test_takeover_switches_codec_to_json_single_shot(self):
        # Escalating to a GLM model (json policy) MUST flip the agent to the json codec
        # + single-shot path + the escalator's per-turn cap (glm-5.2 -> 10 since #206), not leave it
        # on the local native/hermes path (the #179 failure).
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=5, escalation_enabled=True, escalation_stuck_threshold=3,
                     escalation_provider="zhipuai", escalation_model="glm-5.2",
                     max_actions_per_turn=3,   # the local/base cap, BEFORE take-over
                     validation_provider="")
        client, agent = self._agent([action], cfg)
        self.assertEqual(agent._max_actions, 3)         # base cap before escalation
        swapped = FakeLLMClient([action])   # non-native single-shot fake (API stand-in)
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            result = agent.run("loop")
        self.assertIs(agent._client, swapped)
        self.assertEqual(agent._codec.name, "json")     # codec = json for the API path
        self.assertFalse(agent._native)                 # single-shot path
        self.assertEqual(agent._max_actions, 10)        # flipped to glm-5.2 policy cap (10 since #206)
        # a VISIBLE success event is recorded in the run log
        evs = [e for e in result.escalation_events if e["success"]]
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["codec"], "json")
        self.assertFalse(evs[0]["native"])
        self.assertEqual(evs[0]["model"], "glm-5.2")

    def test_failed_escalation_records_distinct_event(self):
        # A missing provider key must produce a DISTINCT, observable failure event in the
        # run log (not a silent terminal-only degrade) — the #191 acceptance criterion.
        action = {"tool": "list_directory", "args": {"path": self.tmp.name}}
        cfg = Config(max_steps=4, escalation_enabled=True,
                     escalation_stuck_threshold=3, validation_provider="")
        original, agent = self._agent([action], cfg)
        with mock.patch.dict(os.environ, {}, clear=True):
            result = agent.run("loop")
        fails = [e for e in result.escalation_events if not e["success"]]
        self.assertEqual(len(fails), 1)
        self.assertIsNotNone(fails[0]["error"])
        self.assertIn("[ESCALATION] FAILED", fails[0]["message"])
        self.assertEqual(len(result.escalation_events), 1)   # recorded once, no spam


if __name__ == "__main__":
    unittest.main()
