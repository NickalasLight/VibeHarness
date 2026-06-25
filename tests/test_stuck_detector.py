import unittest

from vibeharness.escalation import StuckDetector, _make_sig


class StuckDetectorTest(unittest.TestCase):
    def test_two_identical_not_stuck_three_is(self):
        d = StuckDetector(threshold=3)
        self.assertFalse(d.record("click", {"target": "e329"}))   # 1
        self.assertFalse(d.record("click", {"target": "e329"}))   # 2
        self.assertTrue(d.record("click", {"target": "e329"}))    # 3 -> stuck

    def test_non_consecutive_resets_the_run(self):
        d = StuckDetector(threshold=3)
        d.record("click", {"target": "e329"})                     # 1
        d.record("click", {"target": "e329"})                     # 2
        self.assertFalse(d.record("type", {"text": "x"}))         # different -> reset to 1
        self.assertFalse(d.record("click", {"target": "e329"}))   # back to 1
        self.assertFalse(d.record("click", {"target": "e329"}))   # 2
        self.assertTrue(d.record("click", {"target": "e329"}))    # 3 -> stuck

    def test_arg_order_does_not_matter(self):
        d = StuckDetector(threshold=2)
        self.assertFalse(d.record("t", {"a": 1, "b": 2}))
        self.assertTrue(d.record("t", {"b": 2, "a": 1}))          # same signature

    def test_different_args_are_distinct(self):
        d = StuckDetector(threshold=2)
        self.assertFalse(d.record("click", {"target": "e1"}))
        self.assertFalse(d.record("click", {"target": "e2"}))     # different -> not stuck

    def test_threshold_of_one_is_immediate(self):
        d = StuckDetector(threshold=1)
        self.assertTrue(d.record("click", {"target": "e1"}))

    def test_threshold_floor_is_one(self):
        d = StuckDetector(threshold=0)
        self.assertTrue(d.record("x", {}))

    def test_premature_validate_always_triggers(self):
        d = StuckDetector(threshold=3)
        self.assertTrue(d.record_premature_validate())

    def test_premature_validate_resets_consecutive_run(self):
        d = StuckDetector(threshold=3)
        d.record("click", {"target": "e1"})
        d.record("click", {"target": "e1"})
        d.record_premature_validate()                              # resets
        self.assertFalse(d.record("click", {"target": "e1"}))     # back to 1

    def test_reset_clears_everything(self):
        d = StuckDetector(threshold=2)
        d.record("x", {})
        d.escalated = True
        d.reset()
        self.assertFalse(d.escalated)
        self.assertFalse(d.record("x", {}))                       # counter cleared

    def test_make_sig_handles_unhashable_args(self):
        sig = _make_sig("t", {"nested": {"a": [1, 2]}})
        self.assertIn("t::", sig)


if __name__ == "__main__":
    unittest.main()
