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
from vibeharness.runlog import RunLogger

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


if __name__ == "__main__":
    unittest.main()
