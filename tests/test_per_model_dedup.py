"""Per-model consecutive-duplicate tool-call collapse (issue #201).

The same-turn CONSECUTIVE-DUPLICATE filter in ``RalphAgent.run`` (the ``_dedup_actions``
loop) used to collapse identical back-to-back tool calls into ONE for EVERY model. That is
a crutch for the small local ``qwen3:4b`` (which tends to repeat itself); for capable API
models (GLM / DeepSeek) a repeated call can be legitimate and must NOT be silently dropped.

This is now per-model, built on the existing policy registry:
  * ``ModelToolPolicy.collapse_consecutive_dup_tool_calls`` (True only for qwen3:4b),
  * resolved by ``config.resolve_model_collapse_dups`` (spec → registry → True default),
  * read by the agent via ``self._collapse_dup_tool_calls`` (re-resolved on ``_escalate``).

The gate covers BOTH the execution path AND the replayed chat-state history symmetrically:
  * Collapse ON  → the duplicate never executes AND leaves no trace in history (1 call +
    1 tool_response).
  * Collapse OFF → the duplicate executes AND both the call and its tool_response are
    retained in history (2 calls + 2 tool_responses).
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from vibeharness import providers
from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import (Config, ModelSpec, ModelToolPolicy,
                                resolve_model_collapse_dups)
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry
from vibeharness.tools import Tool, ToolResult

from tests._fakes import FakeLLMClient, FakeValidator


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class CountingClick(Tool):
    """A ``click`` tool that counts how many times it actually executed. ``click`` is in
    the agent's SOFT-REPEAT set, so a same-turn repeat that survives the collapse re-runs
    (rather than being no-op-blocked by the #125 anti-loop guard) — exactly the signal we
    need to tell "collapsed" (1 run) from "preserved" (2 runs) apart."""

    name = "click"
    description = "counts clicks"

    def __init__(self):
        self.calls = 0

    @property
    def parameters(self):
        return []

    def run(self, args) -> ToolResult:
        self.calls += 1
        return ToolResult(True, f"you clicked target #{self.calls}")


class _ScriptedNativeClient(LLMClient):
    """Returns scripted Decisions and records the ``messages`` it was handed via
    ``decide_chat`` — so a test can inspect the committed stateful chat history (mirrors the
    helper in test_native_ollama_chat.py). Overriding ``decide_chat`` reports native-capable."""

    def __init__(self, decisions):
        self._decisions = decisions
        self._i = 0
        self.seen_messages = []

    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    def decide_chat(self, messages, tools, constraint, on_reason=None, on_action=None):
        self.seen_messages.append([dict(m) for m in messages])
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


# Two identical click calls in ONE turn, encoded per codec.
_HERMES_DUP = (
    '<tool_call>{"name": "click", "arguments": {"target": "e1"}}</tool_call>\n'
    '<tool_call>{"name": "click", "arguments": {"target": "e1"}}</tool_call>'
)
_HERMES_VALIDATE = '<tool_call>{"name": "validate", "arguments": {}}</tool_call>'
_JSON_DUP = [{"tool": "click", "args": {"target": "e1"}},
             {"tool": "click", "args": {"target": "e1"}}]
_JSON_VALIDATE = {"tool": "validate", "args": {}}


def _registry() -> ToolRegistry:
    return ToolRegistry([CountingClick()])


# --------------------------------------------------------------------------- #
# 1) Resolver — the per-model policy value
# --------------------------------------------------------------------------- #
class ResolverTest(unittest.TestCase):
    def _spec(self, model):
        return ModelSpec(provider="x", model=model)

    def test_qwen_collapses(self):
        self.assertTrue(resolve_model_collapse_dups(Config(), self._spec("qwen3:4b")))

    def test_api_models_do_not_collapse(self):
        for m in ("glm-4.7", "glm-4.7-flash", "glm-5.2",
                  "deepseek-chat", "deepseek-reasoner"):
            self.assertFalse(
                resolve_model_collapse_dups(Config(), self._spec(m)),
                f"{m} must NOT collapse duplicates")

    def test_family_fallbacks_do_not_collapse(self):
        # An unrecognised GLM / DeepSeek variant resolves to its family policy (collapse off).
        self.assertFalse(resolve_model_collapse_dups(Config(), self._spec("glm-9-future")))
        self.assertFalse(resolve_model_collapse_dups(Config(), self._spec("deepseek-v9")))

    def test_unknown_model_defaults_to_collapse(self):
        # Conservative default: an unconfigured (likely small/local) model keeps the crutch.
        self.assertTrue(resolve_model_collapse_dups(Config(), self._spec("mystery-1b")))

    def test_spec_override_wins(self):
        # An explicit spec field beats the registry both ways.
        off = ModelSpec(provider="x", model="qwen3:4b",
                        collapse_consecutive_dup_tool_calls=False)
        on = ModelSpec(provider="x", model="deepseek-chat",
                       collapse_consecutive_dup_tool_calls=True)
        self.assertFalse(resolve_model_collapse_dups(Config(), off))
        self.assertTrue(resolve_model_collapse_dups(Config(), on))

    def test_policy_field_defaults_true(self):
        # The dataclass default is the conservative True (only API entries set it False).
        self.assertTrue(ModelToolPolicy(codec="json", max_actions_per_turn=1)
                        .collapse_consecutive_dup_tool_calls)


# --------------------------------------------------------------------------- #
# 2) The agent resolves the flag from the ACTIVE (base) model
# --------------------------------------------------------------------------- #
class AgentFlagTest(unittest.TestCase):
    def _agent(self, model):
        cfg = Config(model=model, max_steps=3, escalation_enabled=False)
        return RalphAgent(FakeLLMClient([]), _registry(), "SYS", cfg,
                          FakeValidator(passed=True))

    def test_base_qwen_flag_on(self):
        self.assertTrue(self._agent("qwen3:4b")._collapse_dup_tool_calls)

    def test_base_deepseek_flag_off(self):
        self.assertFalse(self._agent("deepseek-chat")._collapse_dup_tool_calls)


# --------------------------------------------------------------------------- #
# 3) Execution through each model's real codec/decode path (legacy single-shot)
# --------------------------------------------------------------------------- #
class CodecExecutionTest(unittest.TestCase):
    """Feed a 2x-duplicate turn through the model's actual codec and assert how many calls
    actually executed. ``FakeLLMClient`` only implements ``decide`` → the single-shot legacy
    path (native off), so ``self._codec`` is the codec we pass."""

    def _run(self, *, model, codec_name, dup, validate):
        tool = CountingClick()
        client = FakeLLMClient([dup, validate])
        cfg = Config(model=model, max_steps=4, escalation_enabled=False)
        agent = RalphAgent(client, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec(codec_name))
        agent.run("go")
        return tool.calls

    def test_qwen_hermes_collapses_to_one(self):
        self.assertEqual(
            self._run(model="qwen3:4b", codec_name="hermes",
                      dup=_HERMES_DUP, validate=_HERMES_VALIDATE), 1)

    def test_deepseek_json_preserves_both(self):
        self.assertEqual(
            self._run(model="deepseek-chat", codec_name="json",
                      dup=_JSON_DUP, validate=_JSON_VALIDATE), 2)

    def test_glm_json_preserves_both(self):
        self.assertEqual(
            self._run(model="glm-4.7", codec_name="json",
                      dup=_JSON_DUP, validate=_JSON_VALIDATE), 2)


# --------------------------------------------------------------------------- #
# 4) HISTORY symmetry (addendum) — the committed chat-state, per model.
# --------------------------------------------------------------------------- #
class HistorySymmetryTest(unittest.TestCase):
    """The collapse also EDITS replayed chat history. Using the native (hermes) transport so
    ``_commit_turn_to_history`` runs and the committed messages are inspectable, we hold the
    codec constant and vary ONLY the model id — isolating the per-model gate's effect on the
    history. The turn-1 assistant block + tool_responses are inspected in turn-2's request."""

    def _history(self, model):
        tool = CountingClick()
        client = _ScriptedNativeClient([
            Decision("", _HERMES_DUP),        # turn 1: two identical clicks
            Decision("", _HERMES_VALIDATE),   # turn 2: finish, so turn-1 is replayed
        ])
        cfg = Config(model=model, max_steps=4, escalation_enabled=False)
        agent = RalphAgent(client, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("hermes"))
        self.assertTrue(agent._native, "native path required to inspect committed history")
        agent.run("go")
        msgs = client.seen_messages[1]      # turn-2 request replays turn-1
        assistant = [m for m in msgs if m["role"] == "assistant"][0]
        n_calls = assistant.get("content", "").count("<tool_call>")
        tool_resp_msgs = [m for m in msgs if m["role"] == "user"
                          and "<tool_response>" in (m.get("content") or "")]
        n_resp = sum(m["content"].count("<tool_response>") for m in tool_resp_msgs)
        return tool.calls, n_calls, n_resp

    def test_qwen_history_keeps_single_call_and_response(self):
        executed, calls, responses = self._history("qwen3:4b")
        self.assertEqual(executed, 1)    # only one click ran
        self.assertEqual(calls, 1)       # history shows ONE tool call
        self.assertEqual(responses, 1)   # history shows ONE tool_response

    def test_deepseek_history_keeps_both_calls_and_responses(self):
        executed, calls, responses = self._history("deepseek-chat")
        self.assertEqual(executed, 2)    # both clicks ran
        self.assertEqual(calls, 2)       # history retains BOTH tool calls
        self.assertEqual(responses, 2)   # history retains BOTH tool_responses


# --------------------------------------------------------------------------- #
# 5) Escalation take-over lifts the collapse mid-run (qwen3:4b -> deepseek)
# --------------------------------------------------------------------------- #
class EscalationTest(unittest.TestCase):
    def test_takeover_to_deepseek_preserves_duplicates(self):
        tool = CountingClick()
        # Base qwen3:4b (collapse ON) gets STUCK on a repeated no-op, escalates to DeepSeek
        # (collapse OFF). The swapped client then emits a 2x-duplicate click turn that, with
        # the collapse lifted, executes BOTH calls.
        base = FakeLLMClient([
            {"tool": "noop_missing", "args": {}},   # unknown tool -> stuck loop, no clicks
        ])
        swapped = FakeLLMClient([_JSON_DUP, _JSON_VALIDATE])
        cfg = Config(
            model="qwen3:4b", max_steps=8, escalation_enabled=True,
            escalation_stuck_threshold=3,
            escalation_provider="deepseek", escalation_model="deepseek-chat",
            validation_provider="")
        agent = RalphAgent(base, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("json"))
        self.assertTrue(agent._collapse_dup_tool_calls)   # qwen base: collapse ON
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            agent.run("go")
        self.assertIs(agent._client, swapped)             # take-over happened
        self.assertFalse(agent._collapse_dup_tool_calls)  # #201: collapse lifted on take-over
        self.assertEqual(tool.calls, 2)                   # both duplicates ran post-take-over


if __name__ == "__main__":
    unittest.main()
