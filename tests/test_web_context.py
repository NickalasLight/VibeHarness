"""Issue #24: the LATEST live page snapshot is auto-injected into the per-turn
system prompt for the web worker, and stale snapshots are dropped by prompt
regeneration (never accumulated in narrative memory).

No real browser is used: the snapshot source is INJECTABLE. ``capture_page_snapshot``
takes any object with a ``run(*args) -> (ok, output)`` method, so a fake returning
canned snapshot text drives the whole path. The per-turn injection is reproduced the
same way cli.py wires it: a zero-arg ``page`` provider passed into
``SystemPromptBuilder.build(..., page=...)`` each turn.
"""
import unittest

from vibeharness.config import Config
from vibeharness.memory import NarrativeMemory
from vibeharness.prompt import SystemPromptBuilder, build_turn_prompt
from vibeharness.toolset import default_catalog
from vibeharness.web import capture_page_snapshot, make_snapshot_provider


class _FakeSnapshotCli:
    """Injectable stand-in for PlaywrightCli: returns scripted snapshot text. A list
    of outputs is consumed one per ``run`` call so successive turns see DIFFERENT
    snapshots (proving stale-dropping)."""

    def __init__(self, outputs, ok=True):
        self._outputs = list(outputs)
        self._i = 0
        self.ok = ok
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return self.ok, out


def _registry(names):
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(names), Config())


class CapturePageSnapshotTest(unittest.TestCase):
    def test_captures_snapshot_text_from_session(self):
        cli = _FakeSnapshotCli(["### Page\nCONSENT BANNER"])
        text = capture_page_snapshot(cli, char_limit=1000)
        self.assertIn("CONSENT BANNER", text)
        # It captures via the SAME session by issuing a `snapshot` command.
        self.assertEqual(cli.calls, [["snapshot"]])

    def test_truncated_to_char_limit(self):
        cli = _FakeSnapshotCli(["y" * 5000])
        text = capture_page_snapshot(cli, char_limit=100)
        self.assertIn("truncated", text)
        self.assertLess(len(text), 5000)

    def test_truncates_at_configured_cap(self):
        # capture_page_snapshot truncates at exactly the cap it is given: the kept
        # body is char_limit chars, plus the appended "+N chars truncated" marker.
        cli = _FakeSnapshotCli(["q" * 50000])
        text = capture_page_snapshot(cli, char_limit=40000)
        self.assertTrue(text.startswith("q" * 40000))
        self.assertIn("truncated", text)
        self.assertIn("10000 chars truncated", text)

    def test_under_new_default_cap_not_truncated(self):
        # A page at/under the new 40000-char default is returned whole.
        cli = _FakeSnapshotCli(["a" * 40000])
        text = capture_page_snapshot(cli, char_limit=Config().web_snapshot_char_limit)
        self.assertEqual(text, "a" * 40000)
        self.assertNotIn("truncated", text)

    def test_failed_snapshot_returns_empty(self):
        cli = _FakeSnapshotCli(["boom"], ok=False)
        self.assertEqual(capture_page_snapshot(cli, char_limit=1000), "")

    def test_exception_returns_empty(self):
        class _Raises:
            def run(self, *a):
                raise RuntimeError("no session")
        self.assertEqual(capture_page_snapshot(_Raises(), char_limit=1000), "")


class PerTurnSnapshotInjectionTest(unittest.TestCase):
    """Reproduce cli.py's wiring: when web is active, each turn's regenerated system
    prompt carries the CURRENT snapshot under the page section."""

    def _provider(self, cli, limit=6000):
        return lambda: capture_page_snapshot(cli, limit)

    def test_web_active_prompt_contains_current_snapshot(self):
        cli = _FakeSnapshotCli(["### Page\nFIRST-SNAP consent dialog"])
        builder = SystemPromptBuilder(_registry(["web"]))
        page = self._provider(cli)
        sp = builder.build("DO THE THING", page=page())
        self.assertIn("# Current page (live snapshot)", sp)
        self.assertIn("FIRST-SNAP consent dialog", sp)

    def test_second_turn_drops_stale_snapshot(self):
        # Two turns, two DIFFERENT snapshots. Each turn the prompt is rebuilt from
        # scratch via the provider, so the new snapshot replaces the old one.
        cli = _FakeSnapshotCli(["### Page\nOLD-SNAP", "### Page\nNEW-SNAP"])
        builder = SystemPromptBuilder(_registry(["web"]))
        page = self._provider(cli)

        first = builder.build("DO THE THING", page=page())
        self.assertIn("OLD-SNAP", first)

        second = builder.build("DO THE THING", page=page())
        self.assertIn("NEW-SNAP", second)
        self.assertNotIn("OLD-SNAP", second)  # stale dropped by regeneration

    def test_snapshot_not_in_narrative_memory(self):
        # The snapshot lives ONLY in the regenerated system prompt; it must never be
        # recorded into narrative memory (which would accumulate stale snapshots).
        cli = _FakeSnapshotCli(["### Page\nSNAP-TEXT-XYZ"])
        page = self._provider(cli)
        _ = SystemPromptBuilder(_registry(["web"])).build("T", page=page())

        memory = NarrativeMemory()
        memory.record("you navigated to the page")
        turn_prompt = build_turn_prompt("T", memory.render())
        self.assertNotIn("SNAP-TEXT-XYZ", memory.render())
        self.assertNotIn("SNAP-TEXT-XYZ", turn_prompt)

    def test_web_inactive_has_no_page_section(self):
        # fs-only: cli.py passes no page provider, so build() gets page="" and emits
        # no page section.
        sp = SystemPromptBuilder(_registry(["fs"])).build("DO THE THING", page="")
        self.assertNotIn("# Current page (live snapshot)", sp)

    def test_page_section_truncated_to_cap(self):
        cli = _FakeSnapshotCli(["z" * 9000])
        sp = SystemPromptBuilder(_registry(["web"])).build(
            "T", page=capture_page_snapshot(cli, char_limit=200))
        self.assertIn("# Current page (live snapshot)", sp)
        self.assertIn("truncated", sp)


class SnapshotCapDefaultTest(unittest.TestCase):
    def test_default_cap_is_absolute_ceiling(self):
        # Issue #43: web_snapshot_char_limit is no longer the PRIMARY cap (the dynamic
        # budget is — see test_snapshot_budget.py). It is now an absolute ceiling /
        # safety fallback, defaulted very high so the context window is the real limit.
        self.assertGreaterEqual(Config().web_snapshot_char_limit, 1_000_000)

    def test_provider_uses_config_ceiling(self):
        # make_snapshot_provider binds the config ceiling, so a snapshot up to that
        # ceiling passes through untruncated (the dynamic budget does the real work in
        # the live run; this fixed-cap provider remains for back-compat).
        cli = _FakeSnapshotCli(["b" * 40000])
        provider = lambda: capture_page_snapshot(cli, Config().web_snapshot_char_limit)
        self.assertEqual(provider(), "b" * 40000)


class SnapshotProviderFactoryTest(unittest.TestCase):
    def test_provider_is_zero_arg_callable_using_config_cap(self):
        # make_snapshot_provider returns a zero-arg seam (like render_workspace). It
        # binds the run's session/timeout from config; with no live browser the
        # snapshot call fails and it returns "" — proving it never raises.
        provider = make_snapshot_provider(Config())
        self.assertEqual(provider(), "")


class CliSnapshotProviderGatingTest(unittest.TestCase):
    """cli.py wires the page-snapshot provider ONLY when 'web' is among the selected
    toolsets, and renders no page section otherwise. These tests reproduce cli.py's
    exact gating expression and feed the resulting provider through build(page=...),
    so the wiring is exercised with no live browser and no agent run.
    """

    @staticmethod
    def _wire(names):
        # Mirrors cli._run_locked verbatim:
        #   snapshot_provider = (make_snapshot_provider(config) if "web" in names else None)
        #   render_page = lambda: snapshot_provider() if snapshot_provider else ""
        snapshot_provider = (make_snapshot_provider(Config()) if "web" in names else None)
        return snapshot_provider, (lambda: snapshot_provider() if snapshot_provider else "")

    def test_provider_wired_only_when_web_selected(self):
        web_provider, _ = self._wire(["web", "fs"])
        fs_provider, _ = self._wire(["fs"])
        self.assertIsNotNone(web_provider)   # web active -> a snapshot provider exists
        self.assertIsNone(fs_provider)       # fs-only   -> no provider at all

    def test_fs_only_render_page_is_empty_so_no_page_section(self):
        # With no provider, render_page() is "" and build() emits no page heading —
        # exactly what an fs-only run does.
        _, render_page = self._wire(["fs"])
        self.assertEqual(render_page(), "")
        sp = SystemPromptBuilder(_registry(["fs"])).build("DO THE THING", page=render_page())
        self.assertNotIn("# Current page (live snapshot)", sp)

    def test_web_render_page_feeds_snapshot_into_page_section(self):
        # web active: swap in a fake CLI (no browser) so the provider yields canned
        # snapshot text, and confirm it lands under the page section.
        _, render_page = self._wire(["web"])
        # render_page closes over a real provider whose CLI has no session, so on its
        # own it returns ""; substitute a fake-backed provider to prove the data flow.
        cli = _FakeSnapshotCli(["### Page\nWIRED-SNAP consent banner"])
        render_page = lambda: capture_page_snapshot(cli, 6000)
        sp = SystemPromptBuilder(_registry(["web"])).build("T", page=render_page())
        self.assertIn("# Current page (live snapshot)", sp)
        self.assertIn("WIRED-SNAP consent banner", sp)


if __name__ == "__main__":
    unittest.main()
