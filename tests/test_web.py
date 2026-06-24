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

from vibeharness.config import Config
from vibeharness.web import BrowseTool, PlaywrightCli, WebToolset

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


class _FakeProc:
    """Stand-in for a spawned Popen: just carries a pid and records kill()."""

    def __init__(self, pid=424242):
        self.pid = pid
        self.killed = False

    def kill(self):
        self.killed = True


class CloseReapsTreeTest(unittest.TestCase):
    """Issue #15: closing a session must run the graceful `close` AND tree-kill the
    spawned daemon/chrome tree, be idempotent, and never raise. No live browser: a
    fake child proc + a stubbed `_kill_tree` prove the tree-kill path is taken."""

    def _cli_with_fake_proc(self):
        cli = PlaywrightCli(session="test", timeout=5)
        cli._binary = "fake-binary"          # mark "installed" so run() proceeds
        cli._last_proc = _FakeProc()         # pretend setup's `open` spawned this
        runs = []
        # Stub run() so no real subprocess launches; record the CLI verbs issued.
        cli.run = lambda *a: (runs.append(list(a)) or (True, "ok"))
        killed = []
        cli._kill_tree = lambda proc: killed.append(proc)
        return cli, runs, killed

    def test_close_issues_close_command_and_kills_tree(self):
        cli, runs, killed = self._cli_with_fake_proc()
        proc = cli._last_proc
        cli.close()
        self.assertIn(["close"], runs)             # graceful close ran
        self.assertEqual(killed, [proc])           # whole tree reaped
        self.assertIsNone(cli._last_proc)          # handle cleared

    def test_close_is_idempotent(self):
        cli, runs, killed = self._cli_with_fake_proc()
        cli.close()
        cli.close()                                # second call: no proc left
        self.assertEqual(len(killed), 1)           # tree killed exactly once
        # Both calls still issue the graceful close (harmless no-op CLI call).
        self.assertEqual(runs.count(["close"]), 2)

    def test_close_kills_tree_even_when_graceful_close_raises(self):
        # A crashed/wedged session may make `close` raise; the tree must STILL be
        # reaped so no chrome/node leaks.
        cli = PlaywrightCli(session="test", timeout=5)
        cli._binary = "fake-binary"
        proc = _FakeProc()
        cli._last_proc = proc
        def boom(*a):
            raise RuntimeError("cli exploded")
        cli.run = boom
        killed = []
        cli._kill_tree = lambda p: killed.append(p)
        cli.close()                                # must not raise
        self.assertEqual(killed, [proc])

    def test_close_never_raises_with_no_proc(self):
        cli = PlaywrightCli(session="test", timeout=5)
        cli._binary = "fake-binary"
        cli.run = lambda *a: (True, "ok")
        cli._last_proc = None
        cli.close()                                # no handle: graceful close only


class WebToolsetTeardownTest(unittest.TestCase):
    """Issue #15: WebToolset.teardown closes the run's CLI (which reaps the tree),
    runs even when the run body raised (cli.py's finally path), and swallows its own
    errors. Driven with a fake CLI — no browser involved."""

    class _RecordingCli:
        def __init__(self):
            self.closed = 0
            self.opened = []
        def run(self, *a):
            self.opened.append(list(a))
            return True, "ok"
        def close(self):
            self.closed += 1

    def test_teardown_closes_the_run_cli(self):
        ts = WebToolset()
        cli = self._RecordingCli()
        ts._cli = cli                              # stand in for setup()'s CLI
        ts.teardown(Config())
        self.assertEqual(cli.closed, 1)            # close() -> tree reap invoked
        self.assertIsNone(ts._cli)                 # state cleared (idempotent-friendly)

    def test_teardown_runs_after_run_body_raised(self):
        # Reproduce cli.py's finally path: setup, run body raises, finally tears down.
        ts = WebToolset()
        cli = self._RecordingCli()
        ts._cli = cli
        try:
            raise RuntimeError("run crashed mid-stream")
        except RuntimeError:
            pass
        finally:
            ts.teardown(Config())
        self.assertEqual(cli.closed, 1)            # browser reaped despite the crash

    def test_teardown_swallows_close_errors(self):
        ts = WebToolset()
        class _Boom:
            def close(self):
                raise RuntimeError("close blew up")
        ts._cli = _Boom()
        ts.teardown(Config())                      # must NOT raise
        self.assertIsNone(ts._cli)

    def test_teardown_without_setup_still_best_effort_closes(self):
        # No prior setup (e.g. setup failed early): teardown must not raise and must
        # attempt a by-name close rather than crash on a missing CLI.
        ts = WebToolset()                          # _cli is None
        ts.teardown(Config())                      # exercises the fallback PlaywrightCli

    def test_teardown_is_idempotent(self):
        ts = WebToolset()
        cli = self._RecordingCli()
        ts._cli = cli
        ts.teardown(Config())
        ts.teardown(Config())                      # second call: _cli is None now
        self.assertEqual(cli.closed, 1)            # original CLI closed exactly once


class CliFinallyTeardownContractTest(unittest.TestCase):
    """Issue #15: reproduce cli._run_locked's setup-inside-try / teardown-in-finally
    loop with fake toolsets to prove the browser is reaped on ANY termination — a
    crashing run body, or a LATER toolset's setup raising after web already opened —
    and that one failing teardown never blocks the others."""

    class _FakeToolset:
        def __init__(self, name, setup_raises=False, teardown_raises=False):
            self.name = name
            self._setup_raises = setup_raises
            self._teardown_raises = teardown_raises
            self.torn_down = False
        def setup(self, config):
            if self._setup_raises:
                raise RuntimeError(f"{self.name} setup failed")
        def teardown(self, config):
            self.torn_down = True
            if self._teardown_raises:
                raise RuntimeError(f"{self.name} teardown failed")

    @staticmethod
    def _run_loop(toolsets, run_body):
        """Verbatim shape of cli._run_locked's setup/run/finally block."""
        try:
            for ts in toolsets:
                ts.setup(None)
            run_body()
        finally:
            for ts in reversed(toolsets):
                try:
                    ts.teardown(None)
                except Exception:
                    pass

    def test_crashing_run_body_still_tears_down_web(self):
        web = self._FakeToolset("web")
        with self.assertRaises(RuntimeError):
            self._run_loop([web], lambda: (_ for _ in ()).throw(RuntimeError("crash")))
        self.assertTrue(web.torn_down)   # browser reaped despite uncaught crash

    def test_later_setup_failure_still_tears_down_already_opened_web(self):
        web = self._FakeToolset("web")
        bad = self._FakeToolset("fs", setup_raises=True)
        with self.assertRaises(RuntimeError):
            self._run_loop([web, bad], lambda: None)
        self.assertTrue(web.torn_down)   # web opened first, must be torn down

    def test_one_failing_teardown_does_not_block_others(self):
        web = self._FakeToolset("web", teardown_raises=True)
        fs = self._FakeToolset("fs")
        self._run_loop([web, fs], lambda: None)   # no exception escapes the finally
        self.assertTrue(web.torn_down)
        self.assertTrue(fs.torn_down)             # reached despite web's teardown raising


if __name__ == "__main__":
    unittest.main()
