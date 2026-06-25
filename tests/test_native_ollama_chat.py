"""Native Ollama tool calling + stateful chat history (issue #129/#130/#131).

These cover the transport-layer rewrite WITHOUT a live model:
  * enveloped tool-schema output for the ``tools:`` field,
  * structured ``tool_calls`` parsing + the schema-leak / nested-arg repairs,
  * the agent's stateful ``chat_history`` construction (user/assistant/tool order),
  * FIFO eviction when the history exceeds the token budget,
  * the native system prompt omitting the harness-injected `# Tools` block.

Ground truth that motivated these (live /api/chat, qwen2.5-coder:3b, Ollama 0.30.8):
the model returns its call as TEXT in ``message.content`` (Ollama leaves ``tool_calls``
null) and frequently malforms the arguments — so the codec's text parse + repairs stay
load-bearing even on the native path.
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
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry

from tests._fakes import FakeValidator


def _registry() -> ToolRegistry:
    return ToolRegistry(build_default_tools(FileSystem(), 1000))


class EnvelopedToolSchemaTest(unittest.TestCase):
    def setUp(self):
        self.reg = _registry()
        self.codec = get_codec("hermes")

    def test_tools_are_enveloped(self):
        tools = self.codec.tools(self.reg)
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), len(self.reg.all()))
        for t in tools:
            self.assertEqual(t["type"], "function")
            fn = t["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_envelope_parameters_match_args_schema(self):
        # Same source as the JSON constraint and the bare <tools> block -> no drift.
        tools = {t["function"]["name"]: t["function"] for t in self.codec.tools(self.reg)}
        for tool in self.reg.all():
            self.assertEqual(tools[tool.name]["parameters"], tool._args_schema())

    def test_non_native_codecs_return_none(self):
        # Only hermes speaks native tools; json/xml/etc opt out so they are unaffected.
        for name in ("json", "xml", "tagged_json"):
            self.assertIsNone(get_codec(name).tools(self.reg))


class StructuredToolCallParseTest(unittest.TestCase):
    def setUp(self):
        self.codec = get_codec("hermes")

    def test_parse_structured_ollama_shape(self):
        calls = [{"function": {"name": "fill", "arguments": {"target": "e1", "text": "x"}}}]
        self.assertEqual(self.codec.parse_tool_calls(calls),
                         [("fill", {"target": "e1", "text": "x"})])

    def test_parse_structured_string_arguments(self):
        calls = [{"function": {"name": "click", "arguments": '{"target": "e5"}'}}]
        self.assertEqual(self.codec.parse_tool_calls(calls), [("click", {"target": "e5"})])

    def test_parse_structured_repairs_schema_leak(self):
        calls = [{"function": {"name": "fill", "arguments": {
            "text": {"type": "string", "description": "the text", "value": "Alice"}}}}]
        self.assertEqual(self.codec.parse_tool_calls(calls), [("fill", {"text": "Alice"})])

    def test_parse_structured_skips_garbage(self):
        self.assertEqual(self.codec.parse_tool_calls([{}, {"function": {}}, None]), [])

    def test_text_parse_repairs_nested_arg(self):
        # NESTED-ARG bug: {"target": {"target": "e9"}} -> {"target": "e9"}
        actions, err = self.codec.parse(
            '{"name": "click", "arguments": {"target": {"target": "e9"}}}')
        self.assertIsNone(err)
        self.assertEqual(actions, [("click", {"target": "e9"})])

    def test_text_parse_accepts_parameters_key(self):
        actions, err = self.codec.parse(
            '{"name": "click", "parameters": {"target": "e5"}}')
        self.assertIsNone(err)
        self.assertEqual(actions, [("click", {"target": "e5"})])


class _ScriptedNativeClient(LLMClient):
    """An LLMClient that returns scripted Decisions and records the messages/tools it
    was handed via decide_chat (proves the agent builds a faithful stateful history)."""

    def __init__(self, decisions):
        self._decisions = decisions
        self._i = 0
        self.seen_messages = []   # one snapshot of `messages` per decide_chat call
        self.seen_tools = []

    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    def decide_chat(self, messages, tools, constraint, on_reason=None, on_action=None):
        # Deep-copy the messages so later mutation/eviction can't rewrite the snapshot.
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_tools.append(tools)
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


class StatefulHistoryTest(unittest.TestCase):
    def setUp(self):
        self.reg = _registry()
        self.cfg = Config(max_steps=5)  # native_tools defaults True, two_phase False

    def _agent(self, client):
        return RalphAgent(client, self.reg, "SYS", self.cfg,
                          FakeValidator(passed=True), codec=get_codec("hermes"))

    def test_native_path_is_active_by_default(self):
        agent = self._agent(_ScriptedNativeClient([Decision("", "{}")]))
        self.assertTrue(agent._native)
        self.assertIsNotNone(agent._tools)

    def test_native_off_when_two_phase(self):
        cfg = replace(self.cfg, two_phase=True)
        agent = RalphAgent(_ScriptedNativeClient([Decision("", "{}")]), self.reg, "SYS",
                           cfg, FakeValidator(), codec=get_codec("hermes"))
        self.assertFalse(agent._native)

    def test_native_off_for_non_native_codec(self):
        agent = RalphAgent(_ScriptedNativeClient([Decision("", "{}")]), self.reg, "SYS",
                           self.cfg, FakeValidator(), codec=get_codec("json"))
        self.assertFalse(agent._native)

    def test_history_has_user_assistant_tool_sequence(self):
        # Turn 1: one write_file call (text content). Turn 2: validate -> finish.
        call = '<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}}</tool_call>'
        client = _ScriptedNativeClient([
            Decision("", call),
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>'),
        ])
        agent = self._agent(client)
        result = agent.run("write a.txt")
        self.assertTrue(result.finished)
        # The SECOND decide_chat call must have seen turn-1 committed as
        # system, user(t1), assistant(t1), tool(t1), user(t2).
        msgs = client.seen_messages[1]
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        self.assertEqual(roles[2], "assistant")
        self.assertEqual(roles[3], "tool")
        self.assertEqual(roles[-1], "user")  # the current turn

    def test_tool_message_carries_observation_and_name(self):
        call = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        client = _ScriptedNativeClient([
            Decision("", call),
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>'),
        ])
        result = self._agent(client).run("list")
        tool_msgs = [m for m in client.seen_messages[1] if m["role"] == "tool"]
        self.assertTrue(tool_msgs)
        self.assertEqual(tool_msgs[0]["tool_name"], "list_directory")
        self.assertTrue(tool_msgs[0]["content"])  # the observation text

    def test_structured_tool_calls_preferred_over_text(self):
        # Turn 1 returns STRUCTURED tool_calls (with junk text content); the agent must
        # act on the structured call AND store it as the assistant message's tool_calls,
        # not the junk text. Turn 2 finishes so we can inspect turn-1 in the history.
        t1 = Decision("", "JUNK TEXT NOT A CALL", tool_calls=(
            {"function": {"name": "list_directory", "arguments": {"path": "."}}},))
        t2 = Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')
        client = _ScriptedNativeClient([t1, t2])
        result = self._agent(client).run("done")
        self.assertTrue(result.finished)
        # turn-1 assistant message in turn-2's request carries the structured call.
        assistant = [m for m in client.seen_messages[1] if m["role"] == "assistant"][0]
        self.assertIn("tool_calls", assistant)
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "list_directory")
        # and a tool observation for list_directory was produced (the call ran).
        tool_msgs = [m for m in client.seen_messages[1] if m["role"] == "tool"]
        self.assertEqual(tool_msgs[0]["tool_name"], "list_directory")

    def test_tools_field_passed_each_turn(self):
        client = _ScriptedNativeClient([
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        self._agent(client).run("x")
        self.assertTrue(client.seen_tools)
        self.assertEqual(client.seen_tools[0][0]["type"], "function")


class FifoEvictionTest(unittest.TestCase):
    def test_evicts_oldest_when_over_budget(self):
        reg = _registry()
        # Tiny window so a couple of turns overflow and force eviction.
        cfg = Config(max_steps=8, num_ctx=400, reason_tokens=50, action_tokens=50,
                     snapshot_safety_margin_tokens=0, snapshot_chars_per_token=4.0)
        # Each turn writes a file with a big content so the tool observation is large.
        big = "x" * 600
        call = ('<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", '
                f'"content": "{big}"}}}}</tool_call>')
        client = _ScriptedNativeClient([Decision("", call)] * 6 + [
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        agent = RalphAgent(client, reg, "SYS", cfg, FakeValidator(passed=True),
                           codec=get_codec("hermes"))
        agent.run("loop")
        # Across turns the recorded history length must stay bounded (eviction happened),
        # never growing unbounded with every turn's user+assistant+tool triple.
        lengths = [len(m) for m in client.seen_messages]
        self.assertTrue(max(lengths) < 3 * len(client.seen_messages),
                        f"history grew unbounded: {lengths}")
        # And at least one message was always retained.
        self.assertTrue(all(n >= 1 for n in lengths))

    def test_fixed_turn_cap_applied(self):
        reg = _registry()
        cfg = Config(max_steps=8, chat_history_max_turns=4, num_ctx=32768)
        call = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        client = _ScriptedNativeClient([Decision("", call)] * 5 + [
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        agent = RalphAgent(client, reg, "SYS", cfg, FakeValidator(passed=True),
                           codec=get_codec("hermes"))
        agent.run("loop")
        # The stored history (excluding the always-fresh system) is capped at 4 + the
        # current user turn appended for the request.
        for msgs in client.seen_messages:
            non_system = [m for m in msgs if m["role"] != "system"]
            self.assertLessEqual(len(non_system), 4 + 1)


class NativeSystemPromptTest(unittest.TestCase):
    def setUp(self):
        self.reg = _registry()
        self.builder = SystemPromptBuilder(self.reg, 5, get_codec("hermes"))

    def test_native_prompt_omits_tools_block_and_format_instructions(self):
        prompt = self.builder.build("do a thing", native_tools=True)
        self.assertNotIn("# Tools", prompt)
        self.assertNotIn("<tools>", prompt)       # no hand-injected schema block
        self.assertNotIn("<tool_call>", prompt)   # no hand-injected call-format block
        self.assertIn("# How the loop works", prompt)
        self.assertIn("do a thing", prompt)

    def test_legacy_prompt_still_injects_tools(self):
        prompt = self.builder.build("do a thing", native_tools=False)
        self.assertIn("# Tools", prompt)
        self.assertIn("<tools>", prompt)


class SnapshotOnUserTurnTest(unittest.TestCase):
    """Issue #146: the live page snapshot rides the END of the USER turn (recency slot),
    NOT the system prompt, and stale snapshots are pruned from chat history so only the
    latest one is visible to the model."""

    def setUp(self):
        self.reg = _registry()
        self.cfg = Config(max_steps=5)

    def _agent(self, client, snapshots):
        # snapshots: a list yielding one snapshot string per turn (the cli wires this
        # to the budgeted live capture). We ignore the user arg and pop in order.
        snaps = iter(snapshots)
        def provider(user=""):
            try:
                return next(snaps)
            except StopIteration:
                return ""
        return RalphAgent(client, self.reg, "SYS", self.cfg,
                          FakeValidator(passed=True), codec=get_codec("hermes"),
                          snapshot_provider=provider)

    def test_snapshot_appended_to_current_user_turn_not_system(self):
        from vibeharness.prompt import SNAPSHOT_USER_MARKER
        call = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        client = _ScriptedNativeClient([
            Decision("", call),
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        agent = self._agent(client, ["### Page\nSNAP-TURN-1 ref e9"])
        agent.run("do it")
        # Turn-1 request: system carries NO snapshot; the user turn ends with it.
        msgs = client.seen_messages[0]
        system = [m for m in msgs if m["role"] == "system"][0]["content"]
        self.assertNotIn(SNAPSHOT_USER_MARKER, system)
        self.assertNotIn("SNAP-TURN-1", system)
        user = [m for m in msgs if m["role"] == "user"][-1]["content"]
        self.assertIn(SNAPSHOT_USER_MARKER, user)
        self.assertIn("SNAP-TURN-1 ref e9", user)
        self.assertTrue(user.rstrip().endswith("SNAP-TURN-1 ref e9"))

    def test_old_snapshots_pruned_from_history(self):
        from vibeharness.prompt import (SNAPSHOT_USER_MARKER,
                                         SNAPSHOT_PRUNED_PLACEHOLDER)
        call = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        client = _ScriptedNativeClient([
            Decision("", call),                        # turn 1
            Decision("", call),                        # turn 2
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        agent = self._agent(client, [
            "### Page\nSNAP-ONE", "### Page\nSNAP-TWO", "### Page\nSNAP-THREE"])
        agent.run("loop")
        # On turn 3's request, the history holds turns 1 & 2's user messages. Only the
        # MOST RECENT user turn with a snapshot keeps it; older ones are placeholdered.
        msgs = client.seen_messages[2]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        with_marker = [m for m in user_msgs if SNAPSHOT_USER_MARKER in m["content"]]
        # Exactly one user message still carries a live snapshot block (the current turn).
        self.assertEqual(len(with_marker), 1)
        self.assertIn("SNAP-THREE", with_marker[0]["content"])
        # The earlier turns were pruned to the placeholder, and their old snapshot text
        # is gone, but the user message body (task reminder) survives.
        joined = "\n".join(m["content"] for m in user_msgs)
        self.assertIn(SNAPSHOT_PRUNED_PLACEHOLDER, joined)
        self.assertNotIn("SNAP-ONE", joined)
        self.assertNotIn("SNAP-TWO", joined)

    def test_tool_observations_survive_pruning(self):
        # Pruning strips only the snapshot block; action observations (tool messages)
        # are untouched so the model keeps the record of what it did.
        call = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        client = _ScriptedNativeClient([
            Decision("", call),
            Decision("", call),
            Decision("", '<tool_call>{"name": "validate", "arguments": {}}</tool_call>')])
        agent = self._agent(client, ["### Page\nA", "### Page\nB", "### Page\nC"])
        agent.run("loop")
        msgs = client.seen_messages[2]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        self.assertGreaterEqual(len(tool_msgs), 2)  # one per executed turn-1/turn-2 call
        self.assertTrue(all(m.get("content") for m in tool_msgs))


if __name__ == "__main__":
    unittest.main()
