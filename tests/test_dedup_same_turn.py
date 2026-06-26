"""Tests for same-turn duplicate tool-call suppression (issue #162).

The feature drops EXACT duplicate calls within a single assistant turn for web tools
that opt in via ``config.web_dedup_same_turn_tools``: the duplicate is not executed, is
stripped from the assistant tool-call block, and never produces a tool_response — so the
replayed history shows no sign the model requested a duplicate. It is OPT-IN per tool,
WEB-FLOW only (a live snapshot provider is present), and a complete no-op by default.

Most coverage targets the pure helper ``_suppress_same_turn_duplicates`` (and its block
trimmer) so every branch — structured tool_calls, ``<tool_call>`` text, legacy mode, and
the parity-mismatch consistency bail — is exercised deterministically. One end-to-end loop
test proves the duplicate truly never executes.
"""
from __future__ import annotations

import re
import os
import tempfile
import unittest
from unittest import mock

from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import Config
from vibeharness.llm import Decision
from vibeharness.registry import ToolRegistry
from vibeharness.settings import Settings
from vibeharness.tools import Tool, ToolResult

from tests._fakes import FakeLLMClient, FakeValidator


def _counting_tool(tool_name: str) -> Tool:
    class _T(Tool):
        name = tool_name
        description = "counts its runs"

        def __init__(self):
            self.calls = 0

        @property
        def parameters(self):
            return []

        def run(self, args) -> ToolResult:
            self.calls += 1
            return ToolResult(True, f"{tool_name} #{self.calls}")

    return _T()


# A click + fill registry mirrors the real web subtools used in the issue's examples.
def _web_registry() -> ToolRegistry:
    return ToolRegistry([_counting_tool("click"), _counting_tool("fill")])


def _make_agent(*, dedup_tools=(), web=True, native=False) -> RalphAgent:
    cfg = Config(
        max_steps=10,
        web_dedup_same_turn_tools=frozenset(dedup_tools),
        native_tools=native,
        two_phase=False,
    )
    codec = get_codec("hermes") if native else None
    snapshot = (lambda: "") if web else None
    agent = RalphAgent(
        FakeLLMClient([]), _web_registry(), "SYS", cfg, FakeValidator(passed=True),
        codec=codec, raw_snapshot_provider=snapshot,
    )
    if native:
        # Sanity: the native path must actually be active for the native trims to apply.
        assert agent._native, "expected native path active with hermes codec"
    return agent


# A non-adjacent duplicate: the consecutive filter upstream keeps all three (no two equal
# calls are adjacent); only the same-turn pass removes the repeat.
_ACTIONS = [
    ("click", {"target": "a"}),
    ("fill", {"target": "b", "text": "x"}),
    ("click", {"target": "a"}),
]


class SuppressUnitTest(unittest.TestCase):
    # ---- positive suppression across the three transports ----
    def test_legacy_drops_action_but_leaves_decision(self):
        # Legacy (non-native) mode: no committed assistant block exists, so dropping the
        # duplicate action is always safe and the decision is returned untouched.
        agent = _make_agent(dedup_tools={"click"}, native=False)
        dec = Decision(reasoning="", action_json="[irrelevant]")
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS[:2])
        self.assertIs(new_dec, dec)

    def test_native_structured_tool_calls_trimmed(self):
        agent = _make_agent(dedup_tools={"click"}, native=True)
        tcs = (
            {"function": {"name": "click", "arguments": {"target": "a"}}},
            {"function": {"name": "fill", "arguments": {"target": "b", "text": "x"}}},
            {"function": {"name": "click", "arguments": {"target": "a"}}},
        )
        dec = Decision(reasoning="", action_json="", tool_calls=tcs)
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS[:2])
        self.assertEqual(list(new_dec.tool_calls), list(tcs[:2]))

    def test_native_tool_call_text_blocks_trimmed(self):
        agent = _make_agent(dedup_tools={"click"}, native=True)
        aj = (
            '<tool_call>{"name":"click","arguments":{"target":"a"}}</tool_call>\n'
            '<tool_call>{"name":"fill","arguments":{"target":"b","text":"x"}}</tool_call>\n'
            '<tool_call>{"name":"click","arguments":{"target":"a"}}</tool_call>'
        )
        dec = Decision(reasoning="", action_json=aj)
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS[:2])
        blocks = re.findall(r"<tool_call>[\s\S]*?</tool_call>", new_dec.action_json)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(new_dec.action_json.count('"click"'), 1)
        self.assertIn('"fill"', new_dec.action_json)

    # ---- exact-match only: similar-but-different calls survive ----
    def test_near_but_not_equal_args_kept(self):
        agent = _make_agent(dedup_tools={"click"}, native=True)
        actions = [("click", {"target": "a"}), ("click", {"target": "a", "force": True})]
        dec = Decision(reasoning="", action_json="", tool_calls=(
            {"function": {"name": "click", "arguments": {"target": "a"}}},
            {"function": {"name": "click", "arguments": {"target": "a", "force": True}}},
        ))
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(actions), dec)
        self.assertEqual(new_actions, actions)            # nothing dropped
        self.assertEqual(len(new_dec.tool_calls), 2)

    # ---- scoping: opt-in, web-flow, per-tool ----
    def test_flag_off_is_noop(self):
        agent = _make_agent(dedup_tools=(), native=True)
        dec = Decision(reasoning="", action_json="", tool_calls=(
            {"function": {"name": "click", "arguments": {"target": "a"}}},
        ))
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS)           # unchanged
        self.assertIs(new_dec, dec)

    def test_non_web_flow_is_noop(self):
        # No snapshot provider => not the web-agent flow => never suppress, even opted in.
        agent = _make_agent(dedup_tools={"click"}, web=False, native=True)
        dec = Decision(reasoning="", action_json="", tool_calls=(
            {"function": {"name": "click", "arguments": {"target": "a"}}},
        ))
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS)
        self.assertIs(new_dec, dec)

    def test_only_opted_in_tools_are_deduped(self):
        # 'fill' duplicates but is NOT opted in => kept; 'click' is not duplicated here.
        agent = _make_agent(dedup_tools={"click"}, native=True)
        actions = [("fill", {"target": "b"}), ("click", {"target": "a"}),
                   ("fill", {"target": "b"})]
        dec = Decision(reasoning="", action_json="", tool_calls=(
            {"function": {"name": "fill", "arguments": {"target": "b"}}},
            {"function": {"name": "click", "arguments": {"target": "a"}}},
            {"function": {"name": "fill", "arguments": {"target": "b"}}},
        ))
        new_actions, _ = agent._suppress_same_turn_duplicates(list(actions), dec)
        self.assertEqual(new_actions, actions)            # fill not opted in -> unchanged

    # ---- consistency bail: never split a call from its result ----
    def test_parity_mismatch_bails_entirely(self):
        # Native mode but the assistant block has FEWER tool_calls than parsed actions
        # (e.g. a malformed entry was skipped during parse). Index-based trimming would
        # misalign calls and results, so suppression must abandon and change NOTHING.
        agent = _make_agent(dedup_tools={"click"}, native=True)
        dec = Decision(reasoning="", action_json="", tool_calls=(
            {"function": {"name": "click", "arguments": {"target": "a"}}},
            {"function": {"name": "fill", "arguments": {"target": "b", "text": "x"}}},
        ))  # 2 calls vs 3 actions
        new_actions, new_dec = agent._suppress_same_turn_duplicates(list(_ACTIONS), dec)
        self.assertEqual(new_actions, _ACTIONS)           # duplicate NOT dropped
        self.assertIs(new_dec, dec)


class SuppressLoopTest(unittest.TestCase):
    """End-to-end: the duplicate genuinely never executes (and the contrast run shows it
    would, with the flag off)."""

    def _run(self, dedup_tools):
        registry = _web_registry()
        click = registry.get("click")
        client = FakeLLMClient([
            [  # one turn, batched: click a, fill b, click a (non-adjacent duplicate)
                {"tool": "click", "args": {"target": "a"}},
                {"tool": "fill", "args": {"target": "b", "text": "x"}},
                {"tool": "click", "args": {"target": "a"}},
            ],
            {"tool": "validate", "args": {}},
        ])
        cfg = Config(max_steps=5, web_dedup_same_turn_tools=frozenset(dedup_tools))
        agent = RalphAgent(client, registry, "SYS", cfg, FakeValidator(passed=True),
                           raw_snapshot_provider=lambda: "")
        with mock.patch("vibeharness.agent.time.sleep", lambda *_a, **_k: None):
            agent.run("t")
        return click.calls

    def test_duplicate_not_executed_when_opted_in(self):
        self.assertEqual(self._run({"click"}), 1)         # second click a suppressed

    def test_duplicate_executes_when_flag_off(self):
        # Click is a soft-repeat tool, so without suppression the repeat DOES run.
        self.assertEqual(self._run(set()), 2)


class SettingsWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("VIBEHARNESS_HOME")
        os.environ["VIBEHARNESS_HOME"] = self.tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VIBEHARNESS_HOME", None)
        else:
            os.environ["VIBEHARNESS_HOME"] = self._prev
        self.tmp.cleanup()

    def test_set_persists_csv_and_applies(self):
        field, value = Settings.set("web-dedup-same-turn-tools", "click, fill , click")
        self.assertEqual(field, "web_dedup_same_turn_tools")
        self.assertEqual(value, ["click", "fill"])        # trimmed + de-duplicated
        cfg = Settings.apply(Config())
        self.assertEqual(set(cfg.web_dedup_same_turn_tools), {"click", "fill"})

    def test_default_is_empty_and_off(self):
        self.assertEqual(Config().web_dedup_same_turn_tools, frozenset())


if __name__ == "__main__":
    unittest.main()
