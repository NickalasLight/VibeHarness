import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from vibeharness.reporting import ConsoleReporter


class ConsoleReporterTest(unittest.TestCase):
    """Exercises the real console rendering (color off so we assert on plain text)."""

    def _render(self, fn):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(ConsoleReporter(color=False))
        return buf.getvalue()

    def test_validator_subagent_block_is_labeled(self):
        out = self._render(lambda r: r.validator_start())
        self.assertIn("validator subagent", out)

    def test_validator_reasoning_and_verdict_are_streamed(self):
        def go(r):
            r.validator_start()
            r.validator_reasoning_token("judging")
            r.validator_verdict_token('{"verdict":"pass"}')
        out = self._render(go)
        self.assertIn("thinking:", out)
        self.assertIn("judging", out)
        self.assertIn("verdict:", out)
        self.assertIn('{"verdict":"pass"}', out)

    def test_action_result_truncates_long_preview(self):
        # The console preview cap is 2000 chars; exceed it so truncation kicks in.
        action = SimpleNamespace(ok=True, observation="z" * 2500, tool="read_file")
        out = self._render(lambda r: r.action_result(action))
        self.assertIn("more chars", out)
        self.assertIn("✓", out)  # success mark

    def test_action_result_marks_failure(self):
        action = SimpleNamespace(ok=False, observation="boom", tool="write_file")
        out = self._render(lambda r: r.action_result(action))
        self.assertIn("✗", out)  # failure mark
        self.assertIn("boom", out)

    def test_run_end_reports_finished(self):
        result = SimpleNamespace(turns=[1, 2], finished=True, final_summary="all done")
        out = self._render(lambda r: r.run_end(result))
        self.assertIn("done in 2 turns", out)
        self.assertIn("all done", out)

    def test_run_end_reports_unfinished(self):
        result = SimpleNamespace(turns=[1, 2, 3], finished=False, final_summary="")
        out = self._render(lambda r: r.run_end(result))
        self.assertIn("stopped after 3 turns", out)


if __name__ == "__main__":
    unittest.main()
