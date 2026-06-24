"""Unit tests for BrowseTool's pure error/guard surface.

The arg-mapping happy paths (goto/snapshot/click/eval -> CLI args) are deliberately
NOT re-asserted here against a recorder: they merely restate BrowseTool's
implementation and are now proven for real, end to end, in
tests/integration/test_web_live.py. What remains here are the pure-unit paths a
live browser can't cheaply force on demand: the fill-missing-text guard,
unknown-action, output truncation, and the schema branch.
"""
import unittest

from vibeharness.web import BrowseTool

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

    def test_browse_schema_branch_present(self):
        tool = self._tool()
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "browse")
        self.assertIn("action", schema["properties"]["args"]["properties"])


if __name__ == "__main__":
    unittest.main()
