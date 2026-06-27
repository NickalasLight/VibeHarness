"""Real test of the live-streaming-to-file behaviour.

This runs the actual RalphAgent with the real RunLogger writing to a real file on
disk, and proves the log is written *after every turn* (not just at the end) by
reading the file back from disk at each checkpoint and checking it grows. The
file is created in a temp workspace and removed when the test finishes.

Uses fake LLM/validator (no model needed) so it stays fast and deterministic,
but the file I/O and agent<->logger wiring are 100% real.
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vibeharness.agent import RalphAgent, RunResult
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.registry import ToolRegistry
from vibeharness.runlog import RunLogger
from vibeharness.tools import Tool, ToolResult

from tests._fakes import FakeLLMClient as _ScriptedClient
from tests._fakes import FakeValidator as _PassValidator


class _BoomTool(Tool):
    """Raises an unexpected error mid-turn (models the #16 interruption)."""

    name = "boom"
    description = "raises mid-turn"

    @property
    def parameters(self):
        return []

    def run(self, args) -> ToolResult:
        raise RuntimeError("kaboom from a tool mid-turn")


class StreamingLogTest(unittest.TestCase):
    def test_log_streams_to_a_real_file_growing_each_turn(self):
        ls = lambda d: {"tool": "list_directory", "args": {"path": d}}
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            registry = ToolRegistry(build_default_tools(FileSystem(), 1000))
            # 3 working turns, then validate (passes) -> 4 turns total
            actions = [ls(d), ls(d), ls(d), {"tool": "validate", "args": {}}]
            # The 3 identical list_directory turns would trip the stuck-detector and
            # escalate to the API model; disable escalation so this log-streaming test
            # stays hermetic (escalation is covered in test_escalation_agent.py).
            agent = RalphAgent(_ScriptedClient(actions), registry, "SYS",
                               Config(max_steps=10, escalation_enabled=False),
                               _PassValidator())
            logger = RunLogger(workspace, datetime(2026, 1, 1, 0, 0, 0))

            turns_on_disk = []

            def checkpoint(result):
                logger.write("demo task", Config(), result)
                # read the file straight back from disk at this moment in the run
                self.assertTrue(logger.json_path.exists(), "log file missing mid-run")
                data = json.loads(logger.json_path.read_text(encoding="utf-8"))
                turns_on_disk.append(len(data["turns"]))

            result = agent.run("demo task", on_turn=checkpoint)

            # streamed once per turn, growing 1 -> 2 -> 3 -> 4 (NOT written only at the end)
            self.assertEqual(turns_on_disk, [1, 2, 3, 4])

            # the real file on disk reflects the finished, validated run
            self.assertTrue(logger.json_path.exists())
            final = json.loads(logger.json_path.read_text(encoding="utf-8"))
            self.assertTrue(final["finished"])
            self.assertEqual(len(final["turns"]), 4)
            self.assertEqual(final["validations"][-1]["passed"], True)
            self.assertTrue(logger.json_path.with_suffix(".md").exists())

            saved_path = logger.json_path

        # the temp workspace (and its .vibe log) is removed after the test
        self.assertFalse(saved_path.exists(), "log file should be cleaned up after the test")

    def test_partial_run_leaves_a_valid_log_on_disk(self):
        # Simulate a run that is interrupted: only the first turn's checkpoint fires.
        # The on-disk file must still be valid JSON reflecting partial progress.
        ls = lambda d: {"tool": "list_directory", "args": {"path": d}}
        with tempfile.TemporaryDirectory() as d:
            registry = ToolRegistry(build_default_tools(FileSystem(), 1000))
            agent = RalphAgent(_ScriptedClient([ls(d)]), registry, "SYS",
                               Config(max_steps=5), _PassValidator())
            logger = RunLogger(Path(d), datetime(2026, 1, 1, 0, 0, 0))

            stop = {"n": 0}

            def checkpoint(result):
                logger.write("demo", Config(), result)
                stop["n"] += 1
                if stop["n"] == 1:
                    raise KeyboardInterrupt  # mimic the user killing the run

            with self.assertRaises(KeyboardInterrupt):
                agent.run("demo", on_turn=checkpoint)

            # even though the run was interrupted, a valid log exists with turn 1
            data = json.loads(logger.json_path.read_text(encoding="utf-8"))
            self.assertFalse(data["finished"])
            self.assertEqual(len(data["turns"]), 1)

    def test_unexpected_mid_turn_crash_still_flushes_completed_actions(self):
        # #16: a turn does one real action, then an UNEXPECTED tool error aborts the
        # turn before its normal end-of-turn on_turn fires. Mirroring the CLI run
        # path (seed log at start + on_turn checkpoint each turn), the real .vibe log
        # on disk must still contain that turn's completed action — not just the seed.
        # Pre-fix, agent.run raised before on_turn fired and the action was dropped.
        ls = lambda d: {"tool": "list_directory", "args": {"path": d}}
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            registry = ToolRegistry(build_default_tools(FileSystem(), 1000) + [_BoomTool()])
            actions = [[ls(d), {"tool": "boom", "args": {}}]]  # success then crash, one turn
            agent = RalphAgent(_ScriptedClient(actions), registry, "SYS",
                               Config(max_steps=5), _PassValidator())
            logger = RunLogger(workspace, datetime(2026, 1, 1, 0, 0, 0))

            # seed log at run start, exactly like the CLI does
            logger.write("demo", Config(), RunResult(task="demo"))
            checkpoint = lambda res: logger.write("demo", Config(), res)

            with self.assertRaises(RuntimeError):
                agent.run("demo", on_turn=checkpoint)

            # the real on-disk log holds the interrupted turn's completed action
            data = json.loads(logger.json_path.read_text(encoding="utf-8"))
            self.assertFalse(data["finished"])
            self.assertEqual(len(data["turns"]), 1)
            tools_on_disk = [a["tool"] for a in data["turns"][0]["actions"]]
            self.assertIn("list_directory", tools_on_disk)   # the completed action survived
            # the markdown transcript reflects it too
            md = logger.json_path.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("list_directory", md)


if __name__ == "__main__":
    unittest.main()
