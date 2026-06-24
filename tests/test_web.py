"""Unit tests for BrowseTool's pure error/guard surface.

The arg-mapping happy paths (goto/snapshot/click/eval -> CLI args) are deliberately
NOT re-asserted here against a recorder: they merely restate BrowseTool's
implementation and are now proven for real, end to end, in
tests/integration/test_web_live.py. What remains here are the pure-unit paths a
live browser can't cheaply force on demand: the fill-missing-text guard,
unknown-action, output truncation, and the schema branch.
"""
import subprocess
import sys
import time
import unittest

from vibeharness.web import BrowseTool, PlaywrightCli

from tests._fakes import FakeCli


class BrowseToolTest(unittest.TestCase):
    def _tool(self, ok=True, output="### Page URL: https://example.com"):
        self.cli = FakeCli(ok=ok, output=output)
        return BrowseTool(self.cli, observation_limit=1000)

    def test_fill_requires_target_and_text(self):
        res = self._tool().run({"action": "fill", "target": "e3"})
        self.assertFalse(res.ok)
        self.assertIn("text", res.observation)
        self.assertEqual(self.cli.calls, [])  # never reached the CLI

    def test_unknown_action_is_error(self):
        res = self._tool().run({"action": "teleport"})
        self.assertFalse(res.ok)
        self.assertIn("unknown browser action", res.observation)

    def test_output_is_truncated(self):
        tool = BrowseTool(FakeCli(output="x" * 5000), observation_limit=100)
        res = tool.run({"action": "snapshot"})
        self.assertIn("truncated", res.observation)

    def test_failed_eval_surfaces_as_error_observation(self):
        # A null/throwing eval (CLI exits non-zero) must come back as a normal
        # ok=False observation the agent can adapt to — not a hang (issue #4).
        tool = self._tool(ok=False, output="Error: Cannot read properties of null")
        res = tool.run({"action": "eval",
                        "expression": "() => document.querySelector('video').play()"})
        self.assertFalse(res.ok)
        self.assertIn("failed", res.observation)
        self.assertIn("null", res.observation)

    def test_browse_schema_branch_present(self):
        tool = self._tool()
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "browse")
        self.assertIn("action", schema["properties"]["args"]["properties"])


class _SleepCli(PlaywrightCli):
    """A REAL PlaywrightCli whose command is a Python child that sleeps far
    longer than the configured timeout. Drives the actual bounded-execution path
    (Popen + communicate(timeout) + kill-tree) in ``run`` — only the program
    being launched stands in for the live browser CLI (issue #4)."""

    def __init__(self, timeout, sleep_seconds=30):
        super().__init__(session="test", timeout=timeout)
        self._binary = sys.executable          # python is a real, on-PATH binary
        self._sleep = sleep_seconds

    def _command(self, *args):
        # Ignore the playwright args; just sleep so the command outlives timeout.
        return [self._binary, "-c", f"import time; time.sleep({self._sleep})"]


class TimeoutTest(unittest.TestCase):
    """Real-behavior tests for issue #4: a web action must be HARD-BOUNDED by the
    configured timeout and must never hang the agent turn."""

    def test_slow_command_returns_timeout_error_without_hanging(self):
        # A real child that sleeps 30s, bounded by a 1s timeout. If the bug
        # regressed (the timeout did not bound the call, or the post-kill drain
        # blocked on a child holding the captured pipe), this would hang ~30s;
        # instead it must return promptly with a clear timeout error.
        cli = _SleepCli(timeout=1, sleep_seconds=30)
        start = time.monotonic()
        ok, out = cli.run("eval", "() => null")
        elapsed = time.monotonic() - start

        self.assertFalse(ok)
        self.assertIn("timed out after 1s", out)
        # Returns well inside the sleep window (1s timeout + a few s kill grace),
        # proving the call did NOT hang past the bound.
        self.assertLess(elapsed, 15, "web action hung past its timeout (issue #4)")

    def test_timeout_error_surfaces_as_browse_tool_observation(self):
        # The agent only ever sees a ToolResult; assert the timeout becomes a
        # clear ok=False observation naming the action, so the agent can adapt.
        cli = _SleepCli(timeout=1, sleep_seconds=30)
        tool = BrowseTool(cli, observation_limit=1000)
        start = time.monotonic()
        res = tool.run({"action": "eval",
                        "expression": "() => document.querySelector('video').play()"})
        elapsed = time.monotonic() - start

        self.assertFalse(res.ok)
        self.assertIn("evaluated JavaScript on", res.observation)  # the 'eval' verb
        self.assertIn("timed out after 1s", res.observation)
        self.assertLess(elapsed, 15, "browse tool hung past its timeout (issue #4)")

    def test_timeout_expired_is_caught_not_propagated(self):
        # If communicate() raises TimeoutExpired (the documented timeout signal),
        # run() must catch it and return an error tuple, never let it propagate.
        import vibeharness.web as web_mod

        class FakeProc:
            pid = 999999
            returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 1)

            def kill(self):
                pass

        cli = PlaywrightCli(session="test", timeout=1)
        cli._binary = "fake-binary"
        orig = web_mod.subprocess.Popen
        web_mod.subprocess.Popen = lambda *a, **k: FakeProc()
        try:
            ok, out = cli.run("eval", "() => null")
        finally:
            web_mod.subprocess.Popen = orig
        self.assertFalse(ok)
        self.assertIn("timed out after 1s", out)


if __name__ == "__main__":
    unittest.main()
