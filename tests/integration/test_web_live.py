"""Live integration tests for the web toolset (issue #51).

These drive the REAL discrete web subtools (goto/click/fill/…) through the REAL
`playwright-cli` against the demo app at http://localhost:3000. They exercise the
agent's actual tool-calling surface end to end, so integration bugs (e.g. the page
capture coming back empty because of a Windows codec crash on emoji) are caught here
rather than in production.

The page is OBSERVED the way the live run observes it — via the internal
`capture_page_snapshot_raw` helper that auto-injects the page each turn — NOT via an
agent tool, because the agent-facing `snapshot` tool was removed in #51.

They auto-skip when `playwright-cli` or the demo server isn't available, so the
fast unit suite and CI stay green.

Run the demo app first (it lives in the job_app_benchmark project), then:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import re
import shutil
import unittest

import pytest

from vibeharness.web import (PlaywrightCli, WebToolset, capture_page_snapshot_raw,
                             GotoTool, ClickTool, FillTool,
                             compute_hidden_refs, filter_hidden_snapshot)
from vibeharness.config import Config

BASE = "http://localhost:3000"
CAREERS = BASE + "/careers/senior-net-engineer-dallas-tx-FT-2024-8842"


@pytest.mark.needs_web
class WebLiveTest(unittest.TestCase):
    SESSION = "vibe-itest"

    @classmethod
    def setUpClass(cls):
        cls.cli = PlaywrightCli(cls.SESSION, timeout=90)
        cls.cli.run("close")
        ok, out = cls.cli.run("open", "--browser", "chrome")  # headless for tests
        if not ok:
            raise unittest.SkipTest(f"could not open browser: {out[:200]}")
        # Full content for the agent; tests assert on the real, untruncated output.
        cls.goto = GotoTool(cls.cli, observation_limit=20000)
        cls.click = ClickTool(cls.cli, observation_limit=20000)
        cls.fill = FillTool(cls.cli, observation_limit=20000)

    @classmethod
    def tearDownClass(cls):
        cls.cli.run("close")

    def _goto_careers(self):
        res = self.goto.run({"url": CAREERS})
        self.assertTrue(res.ok, res.observation)
        return res

    def _snapshot(self):
        # The page is observed exactly as the live run observes it: the internal raw
        # capture (auto-injected each turn), NOT an agent tool (#51 removed snapshot).
        snap = capture_page_snapshot_raw(self.cli)
        self.assertTrue(snap, "page capture came back empty")
        return snap

    # ---- the regression that the unit tests (fake CLI) could not catch ----
    def test_snapshot_is_not_empty(self):
        self._goto_careers()
        snap = self._snapshot()
        self.assertGreater(len(snap), 500,
                           "page capture came back empty/short — likely a decode crash")

    def test_snapshot_survives_unicode_emoji(self):
        # The careers page contains emoji (📍 🕑 📈 💰); a cp1252 decode would
        # crash the reader thread and return empty. Assert the content is intact.
        self._goto_careers()
        snap = self._snapshot()
        self.assertIn("Dallas", snap)
        self.assertIn("Senior .NET Engineer", snap)

    def test_snapshot_exposes_element_refs(self):
        self._goto_careers()
        snap = self._snapshot()
        self.assertRegex(snap, r"ref=e\d+")
        self.assertIn("Apply for this role", snap)

    # ---- navigation / interaction surface ----
    def test_goto_reports_title(self):
        res = self._goto_careers()
        self.assertIn("FlashTec Careers", res.observation)

    def test_click_apply_reaches_the_form(self):
        self._goto_careers()
        snap = self._snapshot()
        m = re.search(r'Apply for this role"\s*\[ref=(e\d+)\]', snap)
        self.assertIsNotNone(m, "could not find the Apply link ref in the snapshot")
        res = self.click.run({"target": m.group(1)})
        self.assertTrue(res.ok, res.observation)
        form = self._snapshot()
        self.assertIn("/apply", form)              # navigated to the form URL
        self.assertIn("Personal Information", form)  # step 1 of the application

    def test_fill_a_text_field(self):
        # Note: the accessibility snapshot shows placeholders for empty fields but
        # not the typed value, so we assert the fill *executed* (Playwright ran
        # .fill on the real element) rather than re-reading it from the snapshot.
        self._goto_careers()
        m = re.search(r'Apply for this role"\s*\[ref=(e\d+)\]', self._snapshot())
        self.click.run({"target": m.group(1)})
        form = self._snapshot()
        field = re.search(r'textbox\s+"[^"]*"\s*\[ref=(e\d+)\]', form)  # a labeled input
        if not field:
            self.skipTest("no labeled text field on the first form step")
        res = self.fill.run({"target": field.group(1), "text": "John"})
        self.assertTrue(res.ok, res.observation)
        self.assertIn("fill", res.observation.lower())  # Playwright executed a fill

    # ---- tool error surface (no browser needed, but real tool) ----
    def test_missing_required_param_is_reported(self):
        res = self.goto.run({})  # no url
        self.assertFalse(res.ok)
        self.assertIn("url", res.observation)

    # ---- issue #223: live visibility filter drops the honeypots ----
    def _goto_apply_form(self):
        res = self.goto.run({"url": CAREERS + "/apply"})
        self.assertTrue(res.ok, res.observation)
        return self._snapshot()

    def test_visibility_filter_drops_honeypots_keeps_real_fields(self):
        # The FlashTec apply form hides "Company website" / "Home fax" honeypot inputs in
        # an aria-hidden, 1px-clipped, off-screen wrapper. compute_hidden_refs must detect
        # them (ONE run-code pass over the SAME capture), and the filter must remove them
        # while keeping every visible real field. Ground-truthed live in #223.
        raw = self._goto_apply_form()
        # Pre-filter: the trap labels ARE present (proving the pipeline would leak them).
        self.assertIn("Company website", raw)
        self.assertIn("Home fax", raw)
        hidden = compute_hidden_refs(self.cli, raw)
        self.assertTrue(hidden, "no hidden refs detected — honeypot not found")
        # The trap inputs' refs are flagged hidden.
        m77 = re.search(r"Company website[\s\S]{0,60}?\[ref=(e\d+)\]", raw)
        m79 = re.search(r"Home fax[\s\S]{0,60}?\[ref=(e\d+)\]", raw)
        if m77:
            self.assertIn(m77.group(1), hidden)
        if m79:
            self.assertIn(m79.group(1), hidden)
        filtered = filter_hidden_snapshot(raw, self.cli)
        self.assertNotIn("Company website", filtered)
        self.assertNotIn("Home fax", filtered)
        # Real visible fields survive (no over-filtering).
        for label in ("First name", "ZIP code", "Continue"):
            self.assertIn(label, filtered)
        self.assertRegex(filtered, r"ref=e\d+")


@unittest.skipUnless(shutil.which("playwright-cli") is not None,
                     "needs playwright-cli installed")
class WebToolsetPrereqTest(unittest.TestCase):
    def test_prerequisites_satisfied_when_cli_present(self):
        self.assertEqual(WebToolset().check_prerequisites(), [])

    def test_toolset_creates_the_discrete_subtools(self):
        names = [t.name for t in WebToolset().create_tools(Config())]
        # The monolithic browse + snapshot tool are gone; navigate_back/forward and
        # screenshot are not in the run-loaded set here (#206 owns nav). ISSUE #203:
        # the evaluate/JS tool is now LOADED so capable API models can use it — qwen3:4b's
        # PER-MODEL toolset omits it (see test_remove_evaluate_67), so it is absent from the
        # small model's VIEW, not from the toolset.
        for absent in ("browse", "snapshot",
                       "navigate_back", "navigate_forward", "screenshot"):
            self.assertNotIn(absent, names)
        self.assertIn("evaluate", names)   # #203: loaded; gated per-model for qwen
        for expected in ("goto", "click", "fill", "type", "press_key",
                         "select_option", "hover", "reload"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
