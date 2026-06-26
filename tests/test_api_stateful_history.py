"""Structured stateful chat history for the API/json (non-native) path (issue #207).

The native path keeps a real ``chat_history``; before #207 the API/json path (GLM,
DeepSeek) got only a flattened third-person prose narrative (``memory.render()``). These
cover, WITHOUT a live model, the new structured path behind ``Config.
api_stateful_chat_history`` (default True):

  * a multi-turn run replays a REAL ``system/user/assistant/user(<tool_response>)`` array
    (assert the message shape, not prose) across >= 3 turns;
  * the flag OFF restores the legacy prose path (regression guard) — ``decide`` is called
    with the ``memory.render()`` narrative in the user turn, ``decide_messages`` never;
  * the shared #193 budgeting evicts oldest history on the API path too;
  * a native -> API escalation keeps the history coherent (sanitized for the provider);
  * #204 dedup OFF records BOTH consecutive duplicates into the structured history.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry

from tests._fakes import FakeValidator


def _registry() -> ToolRegistry:
    return ToolRegistry(build_default_tools(FileSystem(), 1000))


class _ScriptedApiClient(LLMClient):
    """A single-shot API stand-in: NON-native but structured-history capable (issue #207).

    Records the FULL ``messages`` array handed to ``decide_messages`` (proves the agent
    replays a structured multi-turn history) and, separately, the ``user`` handed to the
    legacy ``decide`` (proves the prose path when the flag is off)."""

    def __init__(self, decisions):
        self._decisions = decisions
        self._i = 0
        self.seen_messages: list[list[dict]] = []   # one snapshot per decide_messages call
        self.decide_users: list[str] = []           # user per legacy decide call

    def supports_native_tools(self) -> bool:
        return False

    def supports_structured_history(self) -> bool:
        return True

    def _next(self) -> Decision:
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        self.decide_users.append(user)
        return self._next()

    def decide_messages(self, messages, constraint, on_reason=None, on_action=None):
        self.seen_messages.append([dict(m) for m in messages])
        return self._next()


def _act(tool, **args) -> str:
    return json.dumps([{"tool": tool, "args": args}])


VALIDATE = _act("validate")


class ApiStructuredHistoryTest(unittest.TestCase):
    def setUp(self):
        self.reg = _registry()
        # native_tools defaults True, but the json codec speaks no native tools, so the
        # path is non-native; api_stateful_chat_history defaults True.
        self.cfg = Config(max_steps=6, escalation_enabled=False, validation_provider="")

    def _agent(self, client, cfg=None):
        return RalphAgent(client, self.reg, "SYS", cfg or self.cfg,
                          FakeValidator(passed=True), codec=get_codec("json"))

    def test_path_is_api_structured_by_default(self):
        agent = self._agent(_ScriptedApiClient([Decision("", VALIDATE)]))
        self.assertFalse(agent._native)
        self.assertTrue(agent._api_structured_history)
        self.assertTrue(agent._structured_history)

    def test_three_turn_structured_history_replayed(self):
        client = _ScriptedApiClient([
            Decision("", _act("create_file", path="a.txt", content="hi")),
            Decision("", _act("list_directory", path=".")),
            Decision("", VALIDATE),
        ])
        result = self._agent(client).run("do work")
        self.assertTrue(result.finished)
        # decide_messages was used every turn; the legacy prose decide() never.
        self.assertEqual(len(client.seen_messages), 3)
        self.assertEqual(client.decide_users, [])
        # By turn 3 the model sees a REAL multi-turn array, NOT prose. Two committed turns
        # (each: user, assistant, batched <tool_response> user) + the live user turn.
        msgs = client.seen_messages[2]
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")             # the live turn
        # assistant turns carry the model's EMITTED JSON actions (not prose).
        assistants = [m for m in msgs if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 2)
        self.assertIn("create_file", assistants[0]["content"])
        self.assertIn("list_directory", assistants[1]["content"])
        # observations ride role:user <tool_response> batches.
        tool_resps = [m for m in msgs if m["role"] == "user"
                      and "<tool_response>" in (m.get("content") or "")]
        self.assertEqual(len(tool_resps), 2)
        self.assertIn("a.txt", tool_resps[0]["content"])           # create_file observation
        self.assertIn("listed the directory", tool_resps[1]["content"])
        # the stored history is NOT a third-person prose narrative.
        joined = " ".join(m.get("content") or "" for m in msgs)
        self.assertNotIn("First, you", joined)
        self.assertNotIn("Then, you", joined)

    def test_flag_off_uses_legacy_prose_path(self):
        cfg = replace(self.cfg, api_stateful_chat_history=False)
        client = _ScriptedApiClient([
            Decision("", _act("write_file", path="a.txt", content="hi")),
            Decision("", VALIDATE),
        ])
        agent = self._agent(client, cfg)
        self.assertFalse(agent._api_structured_history)
        self.assertFalse(agent._structured_history)
        result = agent.run("do work")
        self.assertTrue(result.finished)
        # legacy single-shot decide() used; decide_messages never.
        self.assertEqual(client.seen_messages, [])
        self.assertEqual(len(client.decide_users), 2)
        # turn 1 user carries the "no actions yet" narrative; turn 2 carries the prose of
        # turn 1's observation — the flattened narrative the flag preserves.
        self.assertIn("You have not taken any actions yet", client.decide_users[0])
        self.assertIn("First, you", client.decide_users[1])
        # and NOT a structured array shape.
        self.assertNotIn("<tool_response>", client.decide_users[1])

    def test_budget_evicts_oldest_history_on_api_path(self):
        reg = _registry()
        # Tiny window so a couple of turns overflow and force eviction on the API history.
        cfg = Config(max_steps=8, num_ctx=400, reason_tokens=50, action_tokens=50,
                     snapshot_safety_margin_tokens=0, snapshot_chars_per_token=4.0,
                     escalation_enabled=False, validation_provider="")
        big = "x" * 600
        client = _ScriptedApiClient(
            [Decision("", _act("write_file", path="a.txt", content=big))] * 6
            + [Decision("", VALIDATE)])
        agent = RalphAgent(client, reg, "SYS", cfg, FakeValidator(passed=True),
                           codec=get_codec("json"))
        agent.run("loop")
        lengths = [len(m) for m in client.seen_messages]
        # history stayed bounded (eviction happened), never growing by a full triple/turn.
        self.assertTrue(max(lengths) < 3 * len(client.seen_messages),
                        f"history grew unbounded: {lengths}")
        self.assertTrue(all(n >= 1 for n in lengths))

    def test_dedup_off_records_both_consecutive_duplicates(self):
        # #204: collapse OFF for GLM/DeepSeek -> both consecutive duplicate calls survive in
        # the committed structured history (the assistant action JSON is not trimmed).
        client = _ScriptedApiClient([
            # one turn emitting the SAME write_file twice in a row.
            Decision("", json.dumps([
                {"tool": "write_file", "args": {"path": "a.txt", "content": "hi"}},
                {"tool": "write_file", "args": {"path": "a.txt", "content": "hi"}},
            ])),
            Decision("", VALIDATE),
        ])
        agent = self._agent(client)
        agent._collapse_dup_tool_calls = False        # GLM/DeepSeek policy
        result = agent.run("dup")
        self.assertTrue(result.finished)
        msgs = client.seen_messages[1]                # turn-2 request sees turn-1 committed
        assistant = [m for m in msgs if m["role"] == "assistant"][0]
        # BOTH duplicate calls are present in the committed assistant action (not collapsed).
        self.assertEqual(assistant["content"].count("write_file"), 2)


class ApiHistorySanitizeTest(unittest.TestCase):
    """The native -> API escalation sanitizer keeps a carried-over history valid for an
    OpenAI-compatible provider (issue #207)."""

    def test_flattens_native_assistant_tool_calls(self):
        native_hist = [
            {"role": "user", "content": "t1"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "click", "arguments": {"target": "e1"}}}]},
            {"role": "user", "content": "<tool_response>\nclicked\n</tool_response>"},
        ]
        out = RalphAgent._sanitize_history_for_api(native_hist)
        asst = out[1]
        # tool_calls dropped; the call serialised into plain content (OpenAI-valid).
        self.assertNotIn("tool_calls", asst)
        self.assertIn("click", asst["content"])
        # other messages preserved; original not mutated.
        self.assertEqual(out[0], {"role": "user", "content": "t1"})
        self.assertIn("tool_calls", native_hist[1])

    def test_api_shaped_history_is_unchanged(self):
        api_hist = [
            {"role": "user", "content": "t1"},
            {"role": "assistant", "content": '[{"tool":"click","args":{"target":"e1"}}]'},
            {"role": "user", "content": "<tool_response>\nclicked\n</tool_response>"},
        ]
        out = RalphAgent._sanitize_history_for_api(api_hist)
        self.assertEqual(out, api_hist)


class _ScriptedNativeClient(LLMClient):
    """Native base client emitting STRUCTURED tool_calls (so the committed assistant
    messages carry ``tool_calls`` + empty content — the shape the escalation sanitizer
    must flatten)."""

    def __init__(self, decisions):
        self._decisions = decisions
        self._i = 0

    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    def decide_chat(self, messages, tools, constraint, on_reason=None, on_action=None):
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


class NativeToApiEscalationContinuityTest(unittest.TestCase):
    def test_escalation_switches_to_api_structured_and_keeps_history_coherent(self):
        from vibeharness import providers
        from unittest import mock
        reg = _registry()
        # qwen native base: repeat the SAME structured list_directory call until stuck (3x),
        # then escalate to a GLM (json) API model that finishes.
        call = {"function": {"name": "list_directory", "arguments": {"path": "."}}}
        base = _ScriptedNativeClient([Decision("", "", tool_calls=(call,))] * 4)
        cfg = Config(max_steps=8, escalation_enabled=True, escalation_stuck_threshold=3,
                     escalation_provider="zhipuai", escalation_model="glm-5.2",
                     validation_provider="")
        agent = RalphAgent(base, reg, "SYS", cfg, FakeValidator(passed=True),
                           codec=get_codec("hermes"))
        self.assertTrue(agent._native)
        self.assertFalse(agent._api_structured_history)
        swapped = _ScriptedApiClient([Decision("", VALIDATE)])
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            result = agent.run("loop")
        # took over onto the API structured path.
        self.assertIs(agent._client, swapped)
        self.assertFalse(agent._native)
        self.assertTrue(agent._api_structured_history)
        self.assertTrue(agent._structured_history)
        self.assertTrue(result.finished)
        # the post-escalation request carried the carried-over history, SANITIZED: no
        # message may have empty content AND structured tool_calls (provider-invalid).
        self.assertTrue(swapped.seen_messages)
        for msgs in swapped.seen_messages:
            for m in msgs:
                self.assertFalse(m.get("tool_calls") and not (m.get("content") or "").strip(),
                                 f"un-sanitized native message leaked: {m}")
            # the prior native turns are present (history was preserved across the swap).
            self.assertTrue(any("list_directory" in (m.get("content") or "") for m in msgs))


if __name__ == "__main__":
    unittest.main()
