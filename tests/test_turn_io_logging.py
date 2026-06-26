"""Tests for per-turn IO logging (issues #170 / #171).

The full-turn IO dump (`RunLogger.dump_full_turn_io`) was silently dead: its keyword-only
signature mismatched the positional call site, so every call threw a swallowed TypeError
and NO input/output file was ever written. This suite covers the fix:
  - the request payload is dumped BEFORE generation (so a mid-generation crash still leaves
    the crashing turn's exact input on disk), with an estimated request token size;
  - the model output is dumped after;
  - RalphAgent invokes the input logger before the model call and the output logger after;
  - the input dump survives a generation crash (the key #170 observability guarantee).
"""
from __future__ import annotations

import datetime
import re
import tempfile
import unittest
from pathlib import Path

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry
from vibeharness.runlog import RunLogger
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools

from tests._fakes import FakeLLMClient, FakeValidator

VALIDATE = {"tool": "validate", "args": {}}


class RunLoggerDumpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lg = RunLogger(self.tmp.name, datetime.datetime(2026, 6, 26, 20, 0, 0))
        self.diag = Path(self.tmp.name) / ".vibe" / "20260626_200000-diagnostics"

    def tearDown(self):
        self.tmp.cleanup()

    def test_input_dump_written_with_token_estimate(self):
        msgs = [{"role": "assistant", "content": "x" * 40},
                {"role": "user", "content": "y" * 40}]
        self.lg.dump_turn_input(3, system="S" * 80, messages=msgs, user="U" * 40,
                                chars_per_token=4.0)
        files = list(self.diag.glob("turn-003-input-*.txt"))
        self.assertEqual(len(files), 1)
        body = files[0].read_text(encoding="utf-8")
        self.assertIn("captured BEFORE generation", body)
        self.assertIn("SYSTEM PROMPT", body)
        self.assertIn("MESSAGE HISTORY", body)
        # The estimated request token size must be present and non-zero.
        m = re.search(r"estimated request size: ~(\d+) tokens", body)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.group(1)), 0)

    def test_output_dump_written(self):
        self.lg.dump_turn_output(3, reasoning="thinking", action="<tool_call>...")
        files = list(self.diag.glob("turn-003-output-*.txt"))
        self.assertEqual(len(files), 1)
        body = files[0].read_text(encoding="utf-8")
        self.assertIn("REASONING", body)
        self.assertIn("ACTION / TOOL CALL", body)
        self.assertIn("<tool_call>", body)

    def test_input_dump_never_throws_on_unserializable_message(self):
        # A non-JSON value must degrade (default=str), not throw — the dump is the very
        # thing we rely on during a crash, so it must be robust.
        self.lg.dump_turn_input(1, system="S", messages=[{"x": object()}], user="U")
        self.assertEqual(len(list(self.diag.glob("turn-001-input-*.txt"))), 1)


class AgentWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def tearDown(self):
        self.tmp.cleanup()

    def _agent(self, client, **kw):
        return RalphAgent(client, self.registry, "SYSTEM", Config(max_steps=3),
                          FakeValidator(passed=True), **kw)

    def test_input_logged_before_output_each_turn(self):
        order: list = []
        ins: list = []
        outs: list = []

        def in_log(turn, system, messages, user, cpt):
            order.append(("in", turn))
            ins.append((turn, system, messages, user, cpt))

        def out_log(turn, reasoning, action):
            order.append(("out", turn))
            outs.append((turn, reasoning, action))

        client = FakeLLMClient([
            {"tool": "read_file", "args": {"path": "nope.txt"}},
            VALIDATE,
        ])
        agent = self._agent(client, turn_input_logger=in_log, turn_output_logger=out_log)
        agent.run("t")
        # Both fired, and for each turn the INPUT is logged strictly before the OUTPUT.
        self.assertTrue(ins and outs)
        self.assertEqual(order[0], ("in", 1))
        self.assertEqual(order[1], ("out", 1))
        # chars_per_token threaded through from config.
        self.assertEqual(ins[0][4], Config().snapshot_chars_per_token)

    def test_input_dump_survives_a_generation_crash(self):
        # The whole point of #170/#171: if generation crashes, the crashing turn's input
        # must already be on disk. Here the model call raises; assert the input logger
        # captured the turn while the output logger never fired.
        captured_in: list = []
        captured_out: list = []

        class RaisingClient(LLMClient):
            def decide(self, system, user, constraint, on_reason=None, on_action=None):
                raise RuntimeError("boom mid-generation")

        agent = self._agent(
            RaisingClient(),
            turn_input_logger=lambda *a: captured_in.append(a[0]),
            turn_output_logger=lambda *a: captured_out.append(a[0]),
        )
        with self.assertRaises(RuntimeError):
            agent.run("t")
        self.assertEqual(captured_in, [1])   # input persisted before the crash
        self.assertEqual(captured_out, [])   # output never reached


if __name__ == "__main__":
    unittest.main()
