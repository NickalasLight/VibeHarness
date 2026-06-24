"""CLI run-logging behaviour (issue #2): the `vibe` CLI must reliably leave a
`.vibe/` chat log — write a seed log at run START, and never silently swallow a
log-write failure.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path

from vibeharness import cli
from vibeharness.agent import RunResult
from vibeharness.config import Config
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.runlog import RunLogger
from vibeharness.toolset import default_catalog

STAMP = datetime(2026, 1, 2, 3, 4, 5)


class BoomLogger:
    """A logger whose write always fails — to prove _safe_log surfaces, not hides."""
    json_path = Path("nowhere") / ".vibe" / "boom.json"

    def write(self, *a, **k):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "simulated snapshot failure")


class SafeLogTest(unittest.TestCase):
    def test_safe_log_surfaces_failure_and_does_not_raise(self):
        err = io.StringIO()
        with redirect_stderr(err):
            cli._safe_log(BoomLogger(), "task", Config(), RunResult(task="task"))
        msg = err.getvalue()
        self.assertIn("warning", msg.lower())
        self.assertIn("could not write run log", msg)
        self.assertIn("UnicodeEncodeError", msg)   # the real cause is named, not hidden

    def test_safe_log_writes_when_it_can(self):
        with tempfile.TemporaryDirectory() as td:
            logger = RunLogger(td, STAMP)
            err = io.StringIO()
            with redirect_stderr(err):
                cli._safe_log(logger, "task", Config(), RunResult(task="task"))
            self.assertEqual(err.getvalue(), "")      # nothing surfaced on success
            self.assertTrue(logger.json_path.exists())


class StartLogContractTest(unittest.TestCase):
    def test_seed_log_exists_before_any_turn_completes(self):
        # The CLI writes this seed log right after building the logger, so a run that
        # raises/hangs during turn 1 (no on_turn ever fires) still leaves a .vibe log.
        with tempfile.TemporaryDirectory() as td:
            logger = RunLogger(td, STAMP)
            cli._safe_log(logger, "do a thing", Config(), RunResult(task="do a thing"))
            self.assertTrue(logger.json_path.exists(), "seed log not written at start")
            self.assertEqual(logger.json_path.parent.name, ".vibe")
            data = json.loads(logger.json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["task"], "do a thing")
            self.assertEqual(data["turns"], [])   # zero turns, but a log nonetheless


def _build_system_prompt(names) -> str:
    """Assemble the system prompt for the given toolset(s) exactly as cli.py does:
    a registry from those toolsets + their assembled per-toolset guidance."""
    catalog = default_catalog()
    toolsets = catalog.select(names)
    registry = catalog.build_registry(toolsets, Config())
    guidance = SystemPromptBuilder.assemble_guidance(toolsets)
    return SystemPromptBuilder(registry, guidance=guidance).build("do a thing")


class PerAgentTypePromptDumpTest(unittest.TestCase):
    """Issue #37 (#3): the system prompt DUMPED for a given --agent type must
    describe THAT agent's tools + guidance — verification that the per-agent-type
    prompt framework (#19/#31) is reflected in what is logged. We assemble the prompt
    the way cli.py does (agent -> default toolset -> registry+guidance), dump it via
    the real RunLogger diagnostics path, and assert on the file contents."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logger = RunLogger(self.tmp.name, STAMP)

    def _dumped_prompt(self, names) -> str:
        prompt = _build_system_prompt(names)
        self.logger.dump_turn_diagnostics(1, snapshot=None, system_prompt=prompt)
        files = list(self.logger.diagnostics_dir.glob("turn-001-system-prompt-*.txt"))
        self.assertEqual(len(files), 1)
        return files[0].read_text(encoding="utf-8")

    def test_web_agent_prompt_describes_web_tools_and_guidance(self):
        dumped = self._dumped_prompt(["web"])
        self.assertIn("# Working with your tools", dumped)
        self.assertIn("browse", dumped)                       # the web tool
        self.assertIn("# Current page (live snapshot)", dumped)  # web guidance hook
        self.assertNotIn("create_file", dumped)               # NOT the fs tools

    def test_fs_agent_prompt_describes_fs_tools_and_guidance(self):
        dumped = self._dumped_prompt(["fs"])
        self.assertIn("# Working with your tools", dumped)
        self.assertIn("create_file", dumped)                  # an fs tool
        self.assertIn("write_file", dumped)                   # fs guidance mentions it
        self.assertNotIn("browse", dumped)                    # NOT the web tool


if __name__ == "__main__":
    unittest.main()
