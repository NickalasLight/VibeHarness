"""Unit tests for the pre-compaction snapshot VISIBILITY FILTER (issue #223).

Covers the PURE text transform (`filter_hidden_snapshot_refs`) end to end with a real
raw-ARIA fixture (the FlashTec honeypot block), the flag-off / passthrough / malformed
guards, the raw-snapshot ref extractor and `### Result` parser, and a MOCKED
`compute_hidden_refs` (no browser). The browser-needing live path is exercised in
tests/integration/test_web_live.py and marked there.
"""
import unittest

from vibeharness.config import Config
from vibeharness.web import (
    filter_hidden_snapshot_refs,
    filter_hidden_snapshot,
    compute_hidden_refs,
    _raw_snapshot_refs,
    _parse_run_code_result,
)


# A real raw ARIA snapshot slice mirroring the live FlashTec apply form (verified live):
# two visible real fields (ZIP e69, First name e41), the Country combobox (e73), then the
# aria-hidden honeypot wrapper e75 holding the trap inputs "Company website" [e77] and
# "Home fax" [e79], then the visible "Continue" button (e81).
RAW = """\
### Page
- Page URL: http://localhost:3000/apply
- Page Title: FlashTec Careers
### Snapshot
```yaml
- generic [ref=e33]:
  - generic [ref=e40]: First name*
  - textbox "First name" [ref=e41]: Jane
  - generic [ref=e68]: ZIP code*
  - textbox "ZIP code" [ref=e69]: "75201"
  - combobox "Country" [ref=e73] [cursor=pointer]:
    - generic [ref=e74]: Select…
  - generic [ref=e75]:
    - generic [ref=e76]:
      - text: Company website
      - textbox [ref=e77]
    - generic [ref=e78]:
      - text: Home fax
      - textbox [ref=e79]
  - button "Continue →" [ref=e81] [cursor=pointer]
```"""


class PureFilterTest(unittest.TestCase):
    def test_full_aria_hidden_subtree_removed_visible_siblings_kept(self):
        # compute_hidden_refs would return the WHOLE aria-hidden subtree.
        hidden = {"e75", "e76", "e77", "e78", "e79"}
        out = filter_hidden_snapshot_refs(RAW, hidden)
        for ref in hidden:
            self.assertNotIn(f"[ref={ref}]", out)
        self.assertNotIn("Company website", out)
        self.assertNotIn("Home fax", out)
        # Every visible real field + the container + header survive.
        for keep in ("e33", "e40", "e41", "e68", "e69", "e73", "e74", "e81"):
            self.assertIn(f"[ref={keep}]", out)
        self.assertIn("Continue", out)
        self.assertIn("Page URL", out)
        self.assertIn("```yaml", out)
        self.assertIn("```", out)

    def test_leaf_refs_only_prunes_emptied_wrappers_and_orphan_labels(self):
        # Issue's stated case: given ONLY the textbox refs, the now-empty label wrappers
        # (e76/e78) and their orphan "Company website"/"Home fax" labels are pruned too,
        # and the empty grand-wrapper e75 collapses — but visible siblings stay.
        out = filter_hidden_snapshot_refs(RAW, {"e77", "e79"})
        self.assertNotIn("[ref=e77]", out)
        self.assertNotIn("[ref=e79]", out)
        self.assertNotIn("Company website", out)   # orphan label gone
        self.assertNotIn("Home fax", out)
        self.assertNotIn("[ref=e76]", out)          # emptied wrapper pruned
        self.assertNotIn("[ref=e78]", out)
        self.assertNotIn("[ref=e75]", out)          # grand-wrapper collapsed
        # Visible sibling + the real fields above remain.
        self.assertIn("[ref=e81]", out)
        self.assertIn("[ref=e69]", out)
        self.assertIn("[ref=e73]", out)

    def test_partial_keeps_wrapper_with_remaining_ref_descendant(self):
        # Removing only e77 leaves e79 under e75 -> e75/e78 keep a ref descendant, so the
        # wrapper is NOT pruned; the visible Home-fax-side ref survives.
        out = filter_hidden_snapshot_refs(RAW, {"e76", "e77"})
        self.assertNotIn("[ref=e77]", out)
        self.assertNotIn("[ref=e76]", out)
        self.assertNotIn("Company website", out)
        self.assertIn("[ref=e75]", out)             # still holds e78/e79
        self.assertIn("[ref=e79]", out)

    def test_flag_off_passthrough_is_byte_identical(self):
        self.assertEqual(filter_hidden_snapshot_refs(RAW, set()), RAW)
        self.assertEqual(filter_hidden_snapshot_refs(RAW, None), RAW)

    def test_absent_refs_are_a_noop(self):
        # Hidden refs that aren't in the snapshot change nothing (byte-identical).
        self.assertEqual(filter_hidden_snapshot_refs(RAW, {"e999", "e1000"}), RAW)

    def test_untouched_text_block_is_preserved(self):
        # A pure-text disclaimer block with no actionable ref is NOT pruned when nothing
        # beneath it was removed (pruning is scoped to ancestors of a removed element).
        raw = (
            "- generic [ref=e1]:\n"
            "  - generic [ref=e2]:\n"
            "    - text: By submitting you agree to the terms.\n"
            "  - textbox [ref=e3]\n"
        )
        out = filter_hidden_snapshot_refs(raw, {"e3"})
        self.assertNotIn("[ref=e3]", out)
        self.assertIn("[ref=e2]", out)              # untouched text block survives
        self.assertIn("agree to the terms", out)

    def test_malformed_input_falls_back_to_raw(self):
        self.assertEqual(filter_hidden_snapshot_refs("", {"e1"}), "")
        self.assertIsNone(filter_hidden_snapshot_refs(None, {"e1"}))
        weird = "not a snapshot at all\njust prose\n"
        self.assertEqual(filter_hidden_snapshot_refs(weird, {"e5"}), weird)


class HelperTest(unittest.TestCase):
    def test_raw_snapshot_refs_ordered_unique(self):
        self.assertEqual(
            _raw_snapshot_refs(RAW)[:5], ["e33", "e40", "e41", "e68", "e69"])
        # de-dup
        dup = "- a [ref=e7]\n- b [ref=e7]\n- c [ref=e8]\n"
        self.assertEqual(_raw_snapshot_refs(dup), ["e7", "e8"])

    def test_parse_run_code_result(self):
        out = ('### Result\n["e75","e77","e79"]\n### Ran Playwright code\n```js\n...\n```')
        self.assertEqual(_parse_run_code_result(out), ["e75", "e77", "e79"])

    def test_parse_run_code_result_missing_or_bad(self):
        self.assertEqual(_parse_run_code_result(""), [])
        self.assertEqual(_parse_run_code_result("no result heading here"), [])
        self.assertEqual(_parse_run_code_result("### Result\nnot json\n### x"), [])


class _StubCli:
    """PlaywrightCli stand-in: records the run-code call and returns a canned result."""
    def __init__(self, ok=True, output=""):
        self._ok = ok
        self._output = output
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        return (self._ok, self._output)


class ComputeHiddenRefsTest(unittest.TestCase):
    def test_one_pass_resolves_hidden_subset(self):
        cli = _StubCli(ok=True,
                       output='### Result\n["e75","e76","e77","e78","e79"]\n### Ran')
        hidden = compute_hidden_refs(cli, RAW)
        self.assertEqual(hidden, {"e75", "e76", "e77", "e78", "e79"})
        # EXACTLY ONE evaluate pass per turn — a single run-code invocation.
        self.assertEqual(len(cli.calls), 1)
        self.assertEqual(cli.calls[0][0], "run-code")
        # The probe embeds the refs from the per-turn raw capture (no second snapshot).
        self.assertIn('"e77"', cli.calls[0][1])
        self.assertIn("aria-ref", cli.calls[0][1])

    def test_result_refs_not_in_snapshot_are_dropped(self):
        cli = _StubCli(ok=True, output='### Result\n["e77","e9999"]\n### Ran')
        self.assertEqual(compute_hidden_refs(cli, RAW), {"e77"})

    def test_no_refs_skips_the_pass(self):
        cli = _StubCli(ok=True, output="### Result\n[]\n### Ran")
        self.assertEqual(compute_hidden_refs(cli, "### Page\njust text\n"), set())
        self.assertEqual(cli.calls, [])             # nothing to resolve -> no DOM pass

    def test_cli_failure_yields_empty_set(self):
        cli = _StubCli(ok=False, output="boom")
        self.assertEqual(compute_hidden_refs(cli, RAW), set())

    def test_cli_exception_is_swallowed(self):
        class _Boom:
            def run(self, *a):
                raise RuntimeError("dead session")
        self.assertEqual(compute_hidden_refs(_Boom(), RAW), set())


class FilterHiddenSnapshotTest(unittest.TestCase):
    def test_end_to_end_drops_honeypots(self):
        cli = _StubCli(ok=True,
                       output='### Result\n["e75","e76","e77","e78","e79"]\n### Ran')
        out = filter_hidden_snapshot(RAW, cli)
        self.assertNotIn("Company website", out)
        self.assertNotIn("Home fax", out)
        self.assertIn("[ref=e81]", out)

    def test_nothing_hidden_returns_raw_unchanged(self):
        cli = _StubCli(ok=True, output="### Result\n[]\n### Ran")
        self.assertEqual(filter_hidden_snapshot(RAW, cli), RAW)

    def test_error_falls_back_to_unfiltered_raw(self):
        class _Boom:
            def run(self, *a):
                raise RuntimeError("dead")
        self.assertEqual(filter_hidden_snapshot(RAW, _Boom()), RAW)

    def test_blank_raw_passthrough(self):
        cli = _StubCli()
        self.assertEqual(filter_hidden_snapshot("", cli), "")


class ConfigFlagTest(unittest.TestCase):
    def test_flag_default_on(self):
        self.assertTrue(Config().web_snapshot_visibility_filter)


if __name__ == "__main__":
    unittest.main()
