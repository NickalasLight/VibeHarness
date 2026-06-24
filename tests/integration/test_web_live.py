"""Live integration tests for the web toolset.

These drive the REAL `browse` tool through the REAL `playwright-cli` against the
demo app at http://localhost:3000. They exercise the agent's actual tool-calling
surface end to end, so integration bugs (e.g. the snapshot coming back empty
because of a Windows codec crash on emoji) are caught here rather than in
production.

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

from vibeharness.web import BrowseTool, PlaywrightCli, WebToolset
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
        cls.tool = BrowseTool(cls.cli, observation_limit=20000)

    @classmethod
    def tearDownClass(cls):
        cls.cli.run("close")

    def _goto_careers(self):
        res = self.tool.run({"action": "goto", "url": CAREERS})
        self.assertTrue(res.ok, res.observation)
        return res

    def _snapshot(self):
        res = self.tool.run({"action": "snapshot"})
        self.assertTrue(res.ok, res.observation)
        return res.observation

    # ---- the regression that the unit tests (fake CLI) could not catch ----
    def test_snapshot_is_not_empty(self):
        self._goto_careers()
        snap = self._snapshot()
        self.assertGreater(len(snap), 500,
                           "snapshot came back empty/short — likely a decode crash")

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
        res = self.tool.run({"action": "click", "target": m.group(1)})
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
        self.tool.run({"action": "click", "target": m.group(1)})
        form = self._snapshot()
        field = re.search(r'textbox\s+"[^"]*"\s*\[ref=(e\d+)\]', form)  # a labeled input
        if not field:
            self.skipTest("no labeled text field on the first form step")
        res = self.tool.run({"action": "fill", "target": field.group(1), "text": "John"})
        self.assertTrue(res.ok, res.observation)
        self.assertIn("fill", res.observation.lower())  # Playwright executed a fill

    # ---- tool error surface (no browser needed, but real tool) ----
    def test_unknown_action_is_reported(self):
        res = self.tool.run({"action": "levitate"})
        self.assertFalse(res.ok)
        self.assertIn("unknown browser action", res.observation)

    def test_missing_required_param_is_reported(self):
        res = self.tool.run({"action": "goto"})  # no url
        self.assertFalse(res.ok)
        self.assertIn("url", res.observation)


@unittest.skipUnless(shutil.which("playwright-cli") is not None,
                     "needs playwright-cli installed")
class WebToolsetPrereqTest(unittest.TestCase):
    def test_prerequisites_satisfied_when_cli_present(self):
        self.assertEqual(WebToolset().check_prerequisites(), [])

    def test_toolset_creates_browse_tool(self):
        tools = WebToolset().create_tools(Config())
        self.assertEqual([t.name for t in tools], ["browse"])


if __name__ == "__main__":
    unittest.main()
