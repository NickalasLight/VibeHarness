import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vibeharness.agent import Action, RunResult, Turn
from vibeharness.config import Config
from vibeharness.runlog import RunLogger

STAMP = datetime(2026, 1, 2, 3, 4, 5)


class RunLoggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, finished=True) -> RunResult:
        turn = Turn(index=1, reasoning="<think>checking</think>",
                    raw_action='[{"tool":"validate","args":{"summary":"done"}}]')
        turn.actions.append(Action("validate", {"summary": "done"},
                                   "validation PASSED — looks good", ok=True, final=True))
        return RunResult(task="demo", turns=[turn], finished=finished, final_summary="looks good",
                         validations=[{"turn": 1, "passed": True, "reason": "looks good",
                                       "reasoning": "<think>ok</think>"}])

    def test_writes_json_and_md_into_hidden_dir(self):
        logger = RunLogger(self.workspace, STAMP)
        path = logger.write("demo", Config(), self._result())
        self.assertTrue(path.exists())
        self.assertEqual(path.parent.name, ".vibe")
        self.assertTrue(path.with_suffix(".md").exists())

    def test_json_contains_reasoning_and_validations(self):
        path = RunLogger(self.workspace, STAMP).write("demo", Config(), self._result())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["task"], "demo")
        self.assertTrue(data["finished"])
        self.assertEqual(data["turns"][0]["reasoning"], "<think>checking</think>")
        self.assertEqual(data["validations"][0]["passed"], True)

    def test_streaming_overwrites_same_file(self):
        # writing twice (e.g. per turn) reuses the same timestamped file
        logger = RunLogger(self.workspace, STAMP)
        p1 = logger.write("demo", Config(), self._result(finished=False))
        p2 = logger.write("demo", Config(), self._result(finished=True))
        self.assertEqual(p1, p2)
        data = json.loads(p2.read_text(encoding="utf-8"))
        self.assertTrue(data["finished"])   # reflects the latest state

    def test_defensive_unicode_does_not_crash_the_write(self):
        # A transcript/observation carrying lone surrogates + astral chars (the kind
        # a browser snapshot can produce) must NOT raise UnicodeEncodeError; the log
        # is still written. This is the real failure the old silent _safe_log hid.
        nasty = "snapshot \udce9 \udfff \U0001f600 café"   # lone surrogates + emoji + accent
        turn = Turn(index=1, reasoning=f"<think>{nasty}</think>",
                    raw_action='[{"tool":"read_file","args":{"path":"x"}}]')
        turn.actions.append(Action("read_file", {"path": "x"},
                                   f"you read the page: {nasty}", ok=True))
        result = RunResult(task=nasty, turns=[turn], finished=False)
        logger = RunLogger(self.workspace, STAMP)
        path = logger.write(nasty, Config(), result)   # must not raise
        self.assertTrue(path.exists())
        self.assertTrue(path.with_suffix(".md").exists())
        self.assertIn("snapshot", path.with_suffix(".md").read_text(encoding="utf-8"))

    def test_empty_result_writes_a_valid_start_log(self):
        # The CLI writes a seed log at run START with a zero-turn RunResult, so a run
        # that hangs/raises during turn 1 still leaves a trace. That seed must be a
        # valid, parseable log.
        path = RunLogger(self.workspace, STAMP).write("seed task", Config(),
                                                      RunResult(task="seed task"))
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["task"], "seed task")
        self.assertEqual(data["turns"], [])
        self.assertFalse(data["finished"])


class TurnDiagnosticsTest(unittest.TestCase):
    """Per-turn diagnostic dumps (issue #37): the raw untruncated snapshot and the
    exact injected system prompt are written into <stamp>-diagnostics/."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_dump_has_turn_number_full_text_and_length(self):
        logger = RunLogger(self.workspace, STAMP)
        snap = "live page text " * 100   # well over any inline cap
        logger.dump_turn_diagnostics(3, snapshot=snap, system_prompt=None)
        files = list(logger.diagnostics_dir.glob("turn-003-snapshot-*.txt"))
        self.assertEqual(len(files), 1, "exactly one turn-3 snapshot dump expected")
        text = files[0].read_text(encoding="utf-8")
        self.assertIn(f"char length: {len(snap)}", text)   # records true size
        self.assertIn(snap, text)                            # full, untruncated text

    def test_system_prompt_dump_contains_the_injected_prompt(self):
        logger = RunLogger(self.workspace, STAMP)
        prompt = "# YOUR ASSIGNED TASK\ndo a thing\n\n# Working with your tools\n..."
        logger.dump_turn_diagnostics(7, snapshot=None, system_prompt=prompt)
        files = list(logger.diagnostics_dir.glob("turn-007-system-prompt-*.txt"))
        self.assertEqual(len(files), 1)
        self.assertIn(prompt, files[0].read_text(encoding="utf-8"))

    def test_omitted_dumps_are_not_written(self):
        # When web is inactive, snapshot is None and only the prompt dump appears.
        logger = RunLogger(self.workspace, STAMP)
        logger.dump_turn_diagnostics(1, snapshot=None, system_prompt="prompt text")
        self.assertEqual(list(logger.diagnostics_dir.glob("*-snapshot-*.txt")), [])
        self.assertEqual(len(list(logger.diagnostics_dir.glob("*-system-prompt-*.txt"))), 1)

    def test_dump_failure_is_swallowed(self):
        # A throwing dump (here: an unwritable diagnostics dir) must NOT raise — the
        # agent loop calls this per turn and a logging failure can't be allowed to break it.
        logger = RunLogger(self.workspace, STAMP)
        # Force mkdir to explode by pointing diagnostics_dir at something unusable.
        broken = Path(self.tmp.name) / "afile"
        broken.write_text("x", encoding="utf-8")   # a FILE where a dir is expected
        logger.dir = broken                         # mkdir(parents=True) under a file fails
        logger.dump_turn_diagnostics(1, snapshot="s", system_prompt="p")  # must not raise

    def test_unicode_snapshot_does_not_crash_the_dump(self):
        logger = RunLogger(self.workspace, STAMP)
        nasty = "snap \udce9 \U0001f600 café"   # lone surrogate + astral + accent
        logger.dump_turn_diagnostics(2, snapshot=nasty, system_prompt=nasty)
        self.assertTrue(list(logger.diagnostics_dir.glob("turn-002-snapshot-*.txt")))
        self.assertTrue(list(logger.diagnostics_dir.glob("turn-002-system-prompt-*.txt")))


if __name__ == "__main__":
    unittest.main()
