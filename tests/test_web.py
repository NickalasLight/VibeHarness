"""Unit tests for the discrete web subtools' pure error/guard surface (issue #51).

The monolithic ``browse(action=...)`` tool was split into one first-class Tool per
playwright-cli operation (goto, click, fill, type, press_key, select_option, …) and
the agent-facing ``snapshot`` tool was removed entirely. The ``evaluate`` (run-JS)
tool was also removed (issue #67) so the limited agent can never execute arbitrary
JavaScript. The arg-mapping happy paths
are proven end to end in tests/integration/test_web_live.py; what remains here are the
pure-unit paths a live browser can't cheaply force: the missing-required-param guard,
the CLI->argv mapping (against a recorder, since there is no single dispatcher to
prove anymore), output truncation, error-observation phrasing, and the per-tool
call schema.
"""
import subprocess
import sys
import time
import unittest

from vibeharness.config import Config
from vibeharness.web import (
    PlaywrightCli, WebToolset,
    GotoTool, ClickTool, FillTool, TypeTool, PressKeyTool, SelectOptionTool,
    CheckTool, UncheckTool, HoverTool, DragTool, UploadTool,
    ScreenshotTool, NavigateBackTool, NavigateForwardTool, ReloadTool,
    _WEB_TOOL_CLASSES,
)

from tests._fakes import FakeCli


class _SnapshotThenResultCli:
    """A PlaywrightCli stand-in that answers `snapshot` with one canned page and
    every other (action) command with a separate canned result. Lets a test set up
    the live page the issue-#73 target guard validates against independently of what
    the action itself returns. Records every call for argv assertions."""

    def __init__(self, snapshot="", result_ok=True, result_output="### Page\nok"):
        self._snapshot = snapshot
        self._result_ok = result_ok
        self._result_output = result_output
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        if args and args[0] == "snapshot":
            return (bool(self._snapshot), self._snapshot)
        return (self._result_ok, self._result_output)


class SubtoolMappingTest(unittest.TestCase):
    """Each discrete subtool maps its validated args to the right playwright-cli argv,
    phrases a clear past-tense observation, and surfaces CLI failures as ok=False."""

    # A snapshot listing every ref the mapping tests target, so the issue-#73
    # target guard (which captures a fresh snapshot and checks the ref is present)
    # lets these happy-path calls proceed to the CLI argv mapping.
    _SNAPSHOT_WITH_REFS = (
        "Page: \"Example\"\n"
        "[e1] checkbox\n[e2] button\n[e3] textbox\n"
        "[e5] link\n[e6] button\n[e9] dropdown\n"
    )

    def _make(self, cls, ok=True, output=None):
        if output is None:
            output = self._SNAPSHOT_WITH_REFS
        cli = FakeCli(ok=ok, output=output)
        return cls(cli, observation_limit=1000), cli

    def test_goto_maps_to_goto_url(self):
        tool, cli = self._make(GotoTool)
        res = tool.run({"url": "https://example.com"})
        self.assertTrue(res.ok)
        self.assertEqual(cli.calls, [["goto", "https://example.com"]])
        self.assertIn("navigated to", res.observation)

    def test_click_maps_to_click_target(self):
        tool, cli = self._make(ClickTool)
        tool.run({"target": "e6"})
        # Targeted tools snapshot first to validate the ref (issue #73), then act.
        self.assertEqual(cli.calls[-1], ["click", "e6"])
        self.assertIn(["snapshot"], cli.calls)

    def test_fill_maps_to_fill_target_text(self):
        tool, cli = self._make(FillTool)
        tool.run({"target": "e3", "text": "John"})
        self.assertEqual(cli.calls[-1], ["fill", "e3", "John"])

    def test_type_maps_to_type_text(self):
        tool, cli = self._make(TypeTool)
        tool.run({"text": "hello"})
        self.assertEqual(cli.calls, [["type", "hello"]])

    def test_press_key_maps_to_press_key(self):
        tool, cli = self._make(PressKeyTool)
        tool.run({"key": "Enter"})
        self.assertEqual(cli.calls, [["press", "Enter"]])

    def test_select_option_maps_to_select(self):
        tool, cli = self._make(SelectOptionTool)
        tool.run({"target": "e9", "value": "TX"})
        self.assertEqual(cli.calls[-1], ["select", "e9", "TX"])

    def test_check_and_uncheck_map_through(self):
        t1, c1 = self._make(CheckTool)
        t1.run({"target": "e1"})
        self.assertEqual(c1.calls[-1], ["check", "e1"])
        t2, c2 = self._make(UncheckTool)
        t2.run({"target": "e1"})
        self.assertEqual(c2.calls[-1], ["uncheck", "e1"])

    def test_hover_maps_to_hover(self):
        tool, cli = self._make(HoverTool)
        tool.run({"target": "e5"})
        self.assertEqual(cli.calls[-1], ["hover", "e5"])

    def test_drag_maps_to_drag_start_end(self):
        tool, cli = self._make(DragTool)
        tool.run({"target": "e1", "end": "e2"})
        self.assertEqual(cli.calls[-1], ["drag", "e1", "e2"])

    def test_upload_maps_to_upload_file(self):
        tool, cli = self._make(UploadTool)
        tool.run({"file": "/tmp/cv.pdf"})
        self.assertEqual(cli.calls, [["upload", "/tmp/cv.pdf"]])

    def test_navigate_back_forward_reload_map_through(self):
        for cls, expect in ((NavigateBackTool, ["go-back"]),
                            (NavigateForwardTool, ["go-forward"]),
                            (ReloadTool, ["reload"])):
            tool, cli = self._make(cls)
            tool.run({})
            self.assertEqual(cli.calls, [expect])

    def test_screenshot_optional_target(self):
        t1, c1 = self._make(ScreenshotTool)
        t1.run({})
        self.assertEqual(c1.calls, [["screenshot"]])
        t2, c2 = self._make(ScreenshotTool)
        t2.run({"target": "e7"})
        self.assertEqual(c2.calls, [["screenshot", "e7"]])

    def test_fill_requires_target_and_text(self):
        tool, cli = self._make(FillTool)
        res = tool.run({"target": "e3"})         # missing text
        self.assertFalse(res.ok)
        self.assertIn("text", res.observation)
        self.assertEqual(cli.calls, [])          # never reached the CLI

    def test_goto_requires_url(self):
        tool, cli = self._make(GotoTool)
        res = tool.run({})
        self.assertFalse(res.ok)
        self.assertIn("url", res.observation)
        self.assertEqual(cli.calls, [])

    def test_output_is_truncated(self):
        tool = GotoTool(FakeCli(output="x" * 5000), observation_limit=100)
        res = tool.run({"url": "https://example.com"})
        self.assertIn("truncated", res.observation)

    def test_failed_action_surfaces_as_error_observation(self):
        # A failing CLI action (exits non-zero) must come back as a normal ok=False
        # observation the agent can adapt to — not a hang (issue #4). The ref IS on
        # the page (so the issue-#73 guard lets it through) but the CLI run fails.
        cli = _SnapshotThenResultCli(
            snapshot="Page: \"X\"\n[e404] button \"Boom\"\n",
            result_ok=False, result_output="Error: element not found")
        tool = ClickTool(cli, observation_limit=1000)
        res = tool.run({"target": "e404"})
        self.assertFalse(res.ok)
        self.assertIn("failed", res.observation)
        self.assertIn("not found", res.observation)


class SnapshotRefEnforcementTest(unittest.TestCase):
    """Issue #73: a targeted web tool must validate its `target` against the CURRENT
    live snapshot's refs (rejecting guessed CSS selectors before any browser call),
    and must report a no-match as ok=False (the ok-on-no-match status bug)."""

    # A realistic YouTube-consent snapshot (the exact shape from the ground-truth
    # run): refs appear as [eN] tokens.
    _SNAPSHOT = (
        "Page: \"YouTube\"\n"
        "URL: https://www.youtube.com/watch?v=2K9V8y4gZ3c\n"
        "[e18] button \"Guide\"\n"
        "[e87] dialog \"Before you continue to YouTube\" [active]\n"
        "  [e156] button \"Reject the use of cookies\"\n"
        "  [e163] button \"Accept the use of cookies\"\n"
    )

    def test_guessed_css_selector_rejected_and_lists_available_refs(self):
        # The exact bug trigger from the ground-truth run: a guessed class selector.
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT)
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": ".ytd-play-button"})
        self.assertFalse(res.ok)
        # The browser was never asked to click — only the validating snapshot ran.
        self.assertNotIn(["click", ".ytd-play-button"], cli.calls)
        self.assertEqual([c for c in cli.calls if c[0] == "click"], [])
        # The message names the offending target and lists the real refs.
        self.assertIn(".ytd-play-button", res.observation)
        for ref in ("e18", "e87", "e156", "e163"):
            self.assertIn(ref, res.observation)

    def test_unknown_ref_rejected(self):
        # A ref-shaped target that isn't on the page is still rejected.
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT)
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": "e999"})
        self.assertFalse(res.ok)
        self.assertEqual([c for c in cli.calls if c[0] == "click"], [])
        self.assertIn("e163", res.observation)  # available refs listed

    def test_valid_ref_proceeds_to_cli(self):
        # A ref that IS in the snapshot is allowed through to the playwright call.
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT,
                                     result_output="### Page\nclicked")
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": "e163"})
        self.assertTrue(res.ok)
        self.assertEqual(cli.calls[-1], ["click", "e163"])

    def test_bracketed_ref_form_is_accepted(self):
        # The snapshot prints [e163]; accept that bracketed form too.
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT,
                                     result_output="### Page\nclicked")
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": "[e163]"})
        self.assertTrue(res.ok)
        self.assertEqual(cli.calls[-1], ["click", "[e163]"])

    def test_no_match_result_is_ok_false_regression(self):
        # THE bug: playwright-cli exits 0 but its output says the target matched
        # nothing. That must surface as ok=False, not ok=true-with-an-error.
        # (Ref is present so it passes the guard and reaches the CLI.)
        no_match = ('### Error\nError: ".ytd-play-button" does not match any elements.')
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT,
                                     result_ok=True, result_output=no_match)
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": "e163"})
        self.assertFalse(res.ok, "a no-match must be ok=False (issue #73)")
        self.assertIn("does not match any elements", res.observation)

    def test_no_match_detection_helper(self):
        from vibeharness.web import output_signals_no_match
        self.assertTrue(output_signals_no_match(
            '### Error\nError: ".x" does not match any elements.'))
        self.assertTrue(output_signals_no_match("Error: ref not found"))
        self.assertFalse(output_signals_no_match("### Page\n- Page Title: YouTube"))
        self.assertFalse(output_signals_no_match(""))

    def test_parse_snapshot_refs(self):
        from vibeharness.web import parse_snapshot_refs
        self.assertEqual(parse_snapshot_refs(self._SNAPSHOT),
                         {"e18", "e87", "e156", "e163"})

    def test_normalize_ref(self):
        from vibeharness.web import normalize_ref
        self.assertEqual(normalize_ref("e163"), "e163")
        self.assertEqual(normalize_ref("[e163]"), "e163")
        self.assertEqual(normalize_ref(" e163 "), "e163")
        self.assertIsNone(normalize_ref(".ytd-play-button"))
        self.assertIsNone(normalize_ref("#play"))
        self.assertIsNone(normalize_ref("button"))

    def test_guard_fails_open_when_snapshot_unavailable(self):
        # If the snapshot can't be captured, don't block a legitimate action.
        cli = _SnapshotThenResultCli(snapshot="", result_output="### Page\nok")
        tool = ClickTool(cli, observation_limit=2000)
        res = tool.run({"target": "e5"})
        self.assertTrue(res.ok)
        self.assertEqual(cli.calls[-1], ["click", "e5"])

    def test_drag_validates_both_endpoints(self):
        cli = _SnapshotThenResultCli(snapshot=self._SNAPSHOT)
        tool = DragTool(cli, observation_limit=2000)
        # e163 is valid, e999 is not -> reject, never drag.
        res = tool.run({"target": "e163", "end": "e999"})
        self.assertFalse(res.ok)
        self.assertEqual([c for c in cli.calls if c[0] == "drag"], [])
        self.assertIn("e999", res.observation)


class SubtoolGuidanceTest(unittest.TestCase):
    """Issue #73: descriptions and system guidance must require a snapshot ref and
    forbid guessing a CSS selector."""

    def test_click_description_requires_snapshot_ref(self):
        tool = ClickTool(FakeCli(), 1000)
        desc = tool.description.lower()
        self.assertIn("ref", desc)
        self.assertIn("never guess", desc)
        # and the param doc spells out "never a guessed CSS selector".
        target_doc = tool.parameters[0].description.lower()
        self.assertIn("never", target_doc)
        self.assertIn("css selector", target_doc)

    def test_system_guidance_forbids_guessing_selectors(self):
        guidance = WebToolset().system_guidance().lower()
        self.assertIn("ref", guidance)
        self.assertIn("never guess", guidance)
        self.assertIn("css selector", guidance)


class SubtoolSchemaTest(unittest.TestCase):
    def test_each_subtool_call_schema_names_itself(self):
        for cls in _WEB_TOOL_CLASSES:
            tool = cls(FakeCli(), observation_limit=1000)
            schema = tool.call_schema()
            self.assertEqual(schema["properties"]["tool"]["const"], tool.name)
            # args is an object schema derived from the tool's params.
            self.assertEqual(schema["properties"]["args"]["type"], "object")

    def test_no_subtool_is_named_browse_or_snapshot(self):
        names = {cls.name for cls in _WEB_TOOL_CLASSES}
        self.assertNotIn("browse", names)
        self.assertNotIn("snapshot", names)

    def test_required_params_marked_in_schema(self):
        schema = FillTool(FakeCli(), 1000).call_schema()
        required = schema["properties"]["args"].get("required", [])
        self.assertIn("target", required)
        self.assertIn("text", required)


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
        ok, out = cli.run("goto", "https://example.com")
        elapsed = time.monotonic() - start

        self.assertFalse(ok)
        self.assertIn("timed out after 1s", out)
        # Returns well inside the sleep window (1s timeout + a few s kill grace),
        # proving the call did NOT hang past the bound.
        self.assertLess(elapsed, 15, "web action hung past its timeout (issue #4)")

    def test_timeout_error_surfaces_as_web_tool_observation(self):
        # The agent only ever sees a ToolResult; assert the timeout becomes a
        # clear ok=False observation naming the action, so the agent can adapt.
        cli = _SleepCli(timeout=1, sleep_seconds=30)
        tool = GotoTool(cli, observation_limit=1000)
        start = time.monotonic()
        res = tool.run({"url": "https://example.com"})
        elapsed = time.monotonic() - start

        self.assertFalse(res.ok)
        self.assertIn("navigated to", res.observation)  # the 'goto' verb
        self.assertIn("timed out after 1s", res.observation)
        self.assertLess(elapsed, 15, "web tool hung past its timeout (issue #4)")

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
            ok, out = cli.run("goto", "https://example.com")
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


class _ReopenableCli(PlaywrightCli):
    """A PlaywrightCli stand-in modelling a session whose persistent daemon has
    DIED: until ``open`` is called, every command (snapshot included) fails with the
    real "browser is not open" text; after ``open`` it answers normally.

    It is a REAL ``PlaywrightCli`` subclass overriding only the low-level
    ``_run_once`` seam, so the genuine self-healing ``run`` (detect close -> reopen
    -> re-navigate -> retry) is exercised end to end (issues #101/#75/#102). Records
    every low-level command for sequence assertions."""

    NOT_OPEN = "The browser 'vibe' is not open, please run open first playwright-cli -s=vibe open"

    def __init__(self, snapshot_after_open="Page: \"X\"\n[e1] textbox\n",
                 result_after_open="### Page\nok", *, open_succeeds=True,
                 open_flags=("--headed",)):
        super().__init__(session="vibe", timeout=5, open_flags=list(open_flags))
        self._binary = "fake-binary"          # mark installed so _run_once is reached
        self._is_open = False
        self._open_succeeds = open_succeeds
        self._snapshot = snapshot_after_open
        self._result = result_after_open
        self.calls = []

    def _run_once(self, *args):
        self.calls.append(list(args))
        cmd = args[0] if args else ""
        if cmd == "open":
            if not self._open_succeeds:
                return (False, "open failed: could not launch browser")
            self._is_open = True
            return (True, "### Browser `vibe` opened with pid 1234.")
        if cmd == "close":
            self._is_open = False
            return (True, "Browser 'vibe' closed")
        if not self._is_open:
            return (False, self.NOT_OPEN)
        if cmd == "snapshot":
            return (True, self._snapshot)
        return (True, self._result)


class SessionClosedDetectionTest(unittest.TestCase):
    """The session-death signal (#101) is distinct from a per-element no-match (#73)."""

    def test_output_signals_session_closed(self):
        from vibeharness.web import output_signals_session_closed
        self.assertTrue(output_signals_session_closed(
            "The browser 'vibe' is not open, please run open first playwright-cli -s=vibe open"))
        self.assertTrue(output_signals_session_closed("Error: session closed"))
        self.assertFalse(output_signals_session_closed("### Page\n- Page Title: YouTube"))
        self.assertFalse(output_signals_session_closed(""))

    def test_no_match_is_not_a_session_close(self):
        # A per-element miss (#73) must NOT be mistaken for the daemon dying, or we
        # would needlessly reopen on every bad ref.
        from vibeharness.web import output_signals_session_closed
        self.assertFalse(output_signals_session_closed(
            '### Error\nError: ".x" does not match any elements.'))


class RootCauseDaemonSurvivesTimeoutTest(unittest.TestCase):
    """Issue #101 root cause: a SINGLE per-command timeout/kill must NOT tear down
    the shared persistent daemon. The harness's per-command kill path
    (``_kill_tree``) acts only on that command's own spawned process; it must never
    issue a session `close`/`stop`, which is what actually kills the daemon."""

    def test_timed_out_command_does_not_close_the_session(self):
        # Drive the REAL bounded-execution path with a child that outlives the
        # timeout, then assert the wrapper never asked playwright to `close`/`stop`
        # the session — i.e. one slow command cannot take down the daemon.
        closed = {"count": 0}

        class _SleepCliNoClose(_SleepCli):
            def close(self):
                closed["count"] += 1
                super().close()

        cli = _SleepCliNoClose(timeout=1, sleep_seconds=30)
        ok, out = cli.run("snapshot")
        self.assertFalse(ok)
        self.assertIn("timed out", out)
        # The timeout path tree-kills only THIS command's own child; it must not have
        # invoked the session-closing close() at all.
        self.assertEqual(closed["count"], 0,
                         "a per-command timeout must not close/stop the shared daemon (#101)")

    def test_kill_tree_targets_only_the_commands_own_proc(self):
        # The kill on timeout is scoped to the timed-out command's Popen handle, NOT
        # a broadcast daemon kill: assert _kill_tree is handed exactly that proc.
        import vibeharness.web as web_mod
        killed = []

        class FakeProc:
            pid = 555
            returncode = None
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 1)
            def kill(self):
                pass

        the_proc = FakeProc()
        cli = PlaywrightCli(session="vibe", timeout=1)
        cli._binary = "fake"
        orig_popen = web_mod.subprocess.Popen
        # Restore the staticmethod via the class __dict__ so we don't accidentally
        # rebind it as an instance method (which would feed `self` and break later
        # tests that exercise the real kill path).
        orig_kill = PlaywrightCli.__dict__["_kill_tree"]
        web_mod.subprocess.Popen = lambda *a, **k: the_proc
        cli._kill_tree = lambda p: killed.append(p)   # per-instance override only
        try:
            cli.run("snapshot")
        finally:
            web_mod.subprocess.Popen = orig_popen
        # Exactly the command's own proc was killed — nothing broader.
        self.assertEqual(killed, [the_proc])
        self.assertIs(PlaywrightCli.__dict__["_kill_tree"], orig_kill)  # class untouched


class OpenBrowserToolTest(unittest.TestCase):
    """Issue #75: an agent-callable open_browser tool that (re)opens the session."""

    def test_open_browser_is_registered(self):
        from vibeharness.web import OpenBrowserTool
        names = [c.name for c in _WEB_TOOL_CLASSES]
        self.assertIn("open_browser", names)
        # And it is the FIRST tool, so it's prominent / always available.
        self.assertIs(_WEB_TOOL_CLASSES[0], OpenBrowserTool)

    def test_open_browser_opens_the_session(self):
        from vibeharness.web import OpenBrowserTool
        cli = _ReopenableCli()
        tool = OpenBrowserTool(cli, observation_limit=1000)
        res = tool.run({})
        self.assertTrue(res.ok)
        self.assertIn("open", res.observation.lower())
        # It used cli.open() (carrying the run's flags), not a bare run("open").
        self.assertEqual(cli.calls[0][0], "open")

    def test_open_browser_surfaces_failure_clearly(self):
        from vibeharness.web import OpenBrowserTool
        cli = _ReopenableCli(open_succeeds=False)
        tool = OpenBrowserTool(cli, observation_limit=1000)
        res = tool.run({})
        self.assertFalse(res.ok)
        self.assertIn("failed", res.observation.lower())

    def test_open_uses_captured_flags_on_real_cli(self):
        # PlaywrightCli.open() defaults to the session's captured open flags so a
        # reopen restores the same headed/channel browser (#101/#75). It goes through
        # the low-level _run_once seam (never the self-healing run, to avoid recursion).
        recorded = []
        cli = PlaywrightCli(session="vibe", timeout=5, open_flags=["--headed", "--browser", "chrome"])
        cli._run_once = lambda *a: (recorded.append(list(a)) or (True, "ok"))
        cli.open()
        self.assertEqual(recorded, [["open", "--headed", "--browser", "chrome"]])


class AutoRecoveryTest(unittest.TestCase):
    """Issue #101/#75/#102: a web action against a CLOSED session transparently
    reopens (and re-navigates to the last URL) and retries once, at the CLI seam —
    the agent sees success, not a dead-end "not open" loop."""

    def test_fill_reopens_and_retries_on_dead_session(self):
        cli = _ReopenableCli(snapshot_after_open="Page: \"X\"\n[e1] textbox\n",
                             result_after_open="### Page\nfilled")
        tool = FillTool(cli, observation_limit=1000)
        res = tool.run({"target": "e1", "text": "hi"})
        self.assertTrue(res.ok, "auto-recovery should make the retried action succeed")
        # Proof of the sequence: the daemon was reopened, then the action replayed.
        verbs = [c[0] for c in cli.calls]
        self.assertIn("open", verbs)
        self.assertEqual(verbs[-1], "fill")        # last call is the successful replay

    def test_goto_reopens_and_retries_on_dead_session(self):
        # goto has no target guard, so it's the cleanest proof of the action-level path.
        cli = _ReopenableCli(result_after_open="### Page\nnavigated")
        tool = GotoTool(cli, observation_limit=1000)
        res = tool.run({"url": "https://example.com"})
        self.assertTrue(res.ok)
        opens = [c for c in cli.calls if c[0] == "open"]
        self.assertEqual(len(opens), 1)                  # reopened exactly once
        self.assertEqual(opens[0], ["open", "--headed"])  # with the run's captured flags
        self.assertEqual(cli.calls[-1], ["goto", "https://example.com"])

    def test_resume_renavigates_to_last_url(self):
        # After a successful goto, last_url is tracked; a later dead-session command
        # re-navigates there as part of resume (issue #102) so the page is restored.
        cli = _ReopenableCli(snapshot_after_open="Page: \"X\"\nPage URL: https://example.com/app\n",
                             result_after_open="### Page\nfilled")
        # First, a live goto records last_url.
        cli._is_open = True
        GotoTool(cli, 1000).run({"url": "https://example.com/app"})
        self.assertEqual(cli.state.last_url, "https://example.com/app")
        # Now the session dies; a fill triggers resume which must re-goto last_url.
        cli._is_open = False
        FillTool(cli, 1000).run({"target": "e1", "text": "hi"})
        self.assertIn(["goto", "https://example.com/app"], cli.calls[-3:])

    def test_resume_is_bounded(self):
        # If reopen never sticks (open "succeeds" but the session stays dead), resume
        # must be capped across the run so a broken environment terminates instead of
        # re-opening on every single command forever. Each run() does at most one
        # resume; once the cap is hit, further dead-session commands stop resuming.
        cli = _ReopenableCli()
        # open() claims success but the session stays "closed", forcing repeated death.
        def _broken(*args):
            cli.calls.append(list(args))
            cmd = args[0] if args else ""
            if cmd in ("open", "close"):
                return (True, f"{cmd} ok")
            return (False, _ReopenableCli.NOT_OPEN)
        cli._run_once = _broken
        cli._state.max_resumes = 3
        for _ in range(6):                              # many commands, all dead
            cli.run("goto", "https://example.com")
        self.assertEqual(cli._state.resumes, 3)         # capped, did not spin forever
        # Once capped, later dead-session commands no longer trigger an open.
        before = sum(1 for c in cli.calls if c[0] == "open")
        cli.run("goto", "https://example.com")
        after = sum(1 for c in cli.calls if c[0] == "open")
        self.assertEqual(before, after)                 # no further reopen attempts

    def test_reopen_failure_returns_clear_recovery_path_not_bare_error(self):
        # If the session is dead AND reopening fails, the agent must get an
        # actionable "call open_browser" message, never a silent dead-end.
        cli = _ReopenableCli(open_succeeds=False)
        tool = GotoTool(cli, observation_limit=1000)
        res = tool.run({"url": "https://example.com"})
        self.assertFalse(res.ok)
        self.assertIn("open_browser", res.observation)

    def test_recovery_only_triggers_on_session_close_not_no_match(self):
        # A per-element no-match (#73) must NOT trigger a reopen — only a genuine
        # session death does. Use a live (open) session whose action no-matches.
        cli = _SnapshotThenResultCli(
            snapshot="Page: \"X\"\n[e1] button\n",
            result_ok=True,
            result_output='### Error\nError: "e1" does not match any elements.')
        tool = ClickTool(cli, observation_limit=1000)
        res = tool.run({"target": "e1"})
        self.assertFalse(res.ok)
        self.assertNotIn(["open"], cli.calls)      # never tried to reopen
        self.assertIn("does not match", res.observation)


class WebToolsetGuidanceRecoveryTest(unittest.TestCase):
    """Issue #75: system guidance must tell the agent to open the browser when there
    is no current page / the session was closed."""

    def test_guidance_mentions_open_browser_when_no_page(self):
        guidance = WebToolset().system_guidance().lower()
        self.assertIn("open_browser", guidance)
        self.assertIn("not open", guidance)


class FindOptionRefTest(unittest.TestCase):
    """#125: match an OPEN custom-combobox option by its visible text (combobox fallback)."""

    SNAP = (
        '- combobox "State" [ref=e82]\n'
        '- listbox [ref=e90]:\n'
        '  - option "Alabama" [ref=e91]\n'
        '  - option "Texas" [ref=e92]\n'
        '  - option "TX" [ref=e93]\n'
    )

    def test_exact_match_wins_over_substring(self):
        from vibeharness.web import find_option_ref_by_text
        self.assertEqual(find_option_ref_by_text(self.SNAP, "TX"), "e93")

    def test_case_insensitive_and_startswith(self):
        from vibeharness.web import find_option_ref_by_text
        self.assertEqual(find_option_ref_by_text(self.SNAP, "texas"), "e92")
        self.assertEqual(find_option_ref_by_text(self.SNAP, "Tex"), "e92")

    def test_no_match_returns_none(self):
        from vibeharness.web import find_option_ref_by_text
        self.assertIsNone(find_option_ref_by_text(self.SNAP, "Wyoming"))
        self.assertIsNone(find_option_ref_by_text(self.SNAP, ""))


if __name__ == "__main__":
    unittest.main()
