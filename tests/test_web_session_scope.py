"""Run-scoped browser session: unique GUID name + shared SessionState (#111/#112/#113).

The persistent playwright-cli daemon is keyed by SESSION NAME, so every wrapper bound
to the same name talks to — and tears down — the SAME browser. When ``web_session`` was
the constant ``"vibe"``, every concurrent run shared one daemon and one run's teardown
killed another's mid-run (the "browser is not open" deaths, #101/#111). These tests pin
the two halves of the fix:

* #112 (closes #111): each RUN gets a unique ``vibe-<guid>`` session name, and ALL web
  tools + both snapshot providers within one run share that one name.
* #113: ONE ``SessionState`` per run is shared by every web tool and both snapshot
  providers, so recovery bookkeeping written by one tool is visible to another.
"""
import unittest
from dataclasses import replace

from vibeharness.config import Config
from vibeharness.web import (
    DEFAULT_WEB_SESSION,
    PlaywrightCli,
    WebToolset,
    drop_session_state,
    make_raw_snapshot_provider,
    make_snapshot_provider,
    resolve_web_session,
    shared_session_state,
)


def _session_of(cli_or_tool) -> str:
    """The session name a web tool (or PlaywrightCli) is bound to."""
    cli = getattr(cli_or_tool, "_cli", cli_or_tool)
    return cli.session


def _state_of(cli_or_tool):
    """The SessionState a web tool (or PlaywrightCli) is bound to."""
    cli = getattr(cli_or_tool, "_cli", cli_or_tool)
    return cli.state


class UniqueSessionNameTest(unittest.TestCase):
    """#112/#111: each run resolves a fresh, unique session name by default."""

    def test_default_config_yields_a_generated_unique_name(self):
        name = resolve_web_session(Config())
        self.assertNotEqual(name, DEFAULT_WEB_SESSION)
        self.assertTrue(name.startswith(DEFAULT_WEB_SESSION + "-"))

    def test_two_runs_get_different_session_names(self):
        # Two independent resolutions (two runs) never collide on the daemon.
        a = resolve_web_session(Config())
        b = resolve_web_session(Config())
        self.assertNotEqual(a, b)

    def test_explicit_session_name_is_honoured_as_override(self):
        cfg = replace(Config(), web_session="my-explicit-session")
        self.assertEqual(resolve_web_session(cfg), "my-explicit-session")


class OneRunSharesNameAndStateTest(unittest.TestCase):
    """#112 + #113: within ONE run, every web tool and both snapshot providers share
    the SAME session name AND the SAME SessionState instance."""

    def setUp(self):
        # Simulate cli.resolve_config: mint the run's unique name once, then every
        # consumer reads it off the same frozen config.
        self.config = replace(Config(), web_session=resolve_web_session(Config()))

    def tearDown(self):
        drop_session_state(self.config.web_session)

    def test_all_web_tools_share_one_name_and_one_state(self):
        tools = WebToolset().create_tools(self.config)
        names = {_session_of(t) for t in tools}
        states = {id(_state_of(t)) for t in tools}
        self.assertEqual(names, {self.config.web_session})
        self.assertEqual(len(states), 1, "all web tools must share ONE SessionState")

    def test_tools_and_both_snapshot_providers_share_one_state(self):
        tools = WebToolset().create_tools(self.config)
        # Building the providers binds them to the same keyed shared state.
        make_raw_snapshot_provider(self.config)
        make_snapshot_provider(self.config)
        shared = shared_session_state(self.config.web_session)
        for t in tools:
            self.assertIs(_state_of(t), shared)
            self.assertEqual(_session_of(t), self.config.web_session)

    def test_two_runs_get_distinct_states(self):
        run_a = replace(Config(), web_session=resolve_web_session(Config()))
        run_b = replace(Config(), web_session=resolve_web_session(Config()))
        tools_a = WebToolset().create_tools(run_a)
        tools_b = WebToolset().create_tools(run_b)
        try:
            state_a = _state_of(tools_a[0])
            state_b = _state_of(tools_b[0])
            self.assertIsNot(state_a, state_b)
            self.assertNotEqual(_session_of(tools_a[0]), _session_of(tools_b[0]))
        finally:
            drop_session_state(run_a.web_session)
            drop_session_state(run_b.web_session)


class SharedRecoveryBookkeepingTest(unittest.TestCase):
    """#113: recovery bookkeeping written by one tool's wrapper is visible to another
    tool in the same run, because they share one SessionState."""

    class _NavThenDeadCli:
        """playwright-cli stand-in: a `goto` succeeds (and prints a Page URL the run
        layer records as last_url); the next command reports the daemon dead so the
        self-healing ``run`` path resumes, mutating the SHARED state."""

        def __init__(self):
            self.calls = []

        def __call__(self, *args):  # not used; PlaywrightCli drives _run_once
            raise AssertionError

    def setUp(self):
        self.config = replace(Config(), web_session=resolve_web_session(Config()))

    def tearDown(self):
        drop_session_state(self.config.web_session)

    def _cli_pair(self):
        """Two PlaywrightCli wrappers bound to the run's name + shared state, exactly
        as create_tools / setup / the snapshot providers build them. We stub the
        process-level seam (_run_once) so no real browser is touched."""
        state = shared_session_state(self.config.web_session)
        tool_cli = PlaywrightCli(self.config.web_session, self.config.web_cli_timeout, state=state)
        snap_cli = PlaywrightCli(self.config.web_session, self.config.web_cli_timeout, state=state)
        return tool_cli, snap_cli, state

    def test_last_url_written_by_one_tool_visible_to_another(self):
        tool_cli, other_cli, state = self._cli_pair()
        # Drive a successful goto through the first wrapper; it records last_url onto
        # the SHARED state.
        tool_cli._run_once = lambda *a: (True, "Navigated\nPage URL: https://example.com/job")
        ok, _ = tool_cli.run("goto", "https://example.com/job")
        self.assertTrue(ok)
        self.assertEqual(state.last_url, "https://example.com/job")
        # A DIFFERENT wrapper in the same run sees that last_url (one shared state).
        self.assertEqual(other_cli.state.last_url, "https://example.com/job")

    def test_resume_triggered_by_one_tool_is_counted_on_the_shared_state(self):
        tool_cli, other_cli, state = self._cli_pair()
        # First, a successful goto so the resume has a URL to re-navigate to.
        tool_cli._run_once = lambda *a: (True, "Page URL: https://example.com/start")
        tool_cli.run("goto", "https://example.com/start")
        self.assertEqual(state.resumes, 0)
        # Now a command that reports the daemon dead, then succeeds on retry. The
        # self-healing run path bumps ``resumes`` on the SHARED state.
        seq = iter([
            (False, "Error: The browser is not open, please run open first"),  # first attempt
            (True, "ok"),    # close (in _resume)
            (True, "ok"),    # open  (in _resume)
            (True, "ok"),    # goto last_url (in _resume)
            (True, "clicked"),  # retry of the original command
        ])
        tool_cli._run_once = lambda *a: next(seq)
        ok, out = tool_cli.run("click", "e5")
        self.assertTrue(ok)
        self.assertEqual(out, "clicked")
        # The resume counter on the shared state is now visible to the OTHER wrapper.
        self.assertEqual(state.resumes, 1)
        self.assertEqual(other_cli.state.resumes, 1)


class SessionStateRegistryHygieneTest(unittest.TestCase):
    """The keyed shared-state registry returns one instance per name and is dropped on
    teardown so it never accumulates across runs."""

    def test_same_name_returns_same_instance(self):
        name = resolve_web_session(Config())
        try:
            self.assertIs(shared_session_state(name), shared_session_state(name))
        finally:
            drop_session_state(name)

    def test_open_flags_seed_only_when_first_created(self):
        name = resolve_web_session(Config())
        try:
            first = shared_session_state(name, ["--headed"])
            self.assertEqual(first.open_flags, ["--headed"])
            # A later caller with different flags does not overwrite the seeded ones.
            again = shared_session_state(name, ["--browser", "firefox"])
            self.assertIs(again, first)
            self.assertEqual(again.open_flags, ["--headed"])
        finally:
            drop_session_state(name)

    def test_teardown_drops_the_runs_state(self):
        config = replace(Config(), web_session=resolve_web_session(Config()))
        ts = WebToolset()

        class _RecordingCli:
            def close(self):
                pass

        ts._cli = _RecordingCli()
        # Materialise the shared state, then ensure teardown forgets it.
        before = shared_session_state(config.web_session)
        ts.teardown(config)
        after = shared_session_state(config.web_session)
        self.assertIsNot(before, after)  # a fresh instance => the old entry was dropped
        drop_session_state(config.web_session)


if __name__ == "__main__":
    unittest.main()
