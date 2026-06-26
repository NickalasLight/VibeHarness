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

    # ---- no-progress CYCLE detection (issue #191) ----
    def test_two_step_no_progress_cycle_escalates(self):
        # The exact #191 failure mode: a 2-step A,B,A,B,... loop that a consecutive-only
        # counter (each B resets A's run) would NEVER flag. threshold=3 -> trips after
        # three full (A,B) repeats = 6 recorded calls.
        d = StuckDetector(threshold=3)
        A = ("goto", {"url": "https://form"})
        B = ("page_snapshot", {})
        self.assertFalse(d.record(*A))   # A           (1)
        self.assertFalse(d.record(*B))   # A,B         (cycle x1)
        self.assertFalse(d.record(*A))   # A,B,A       (1.5)
        self.assertFalse(d.record(*B))   # A,B,A,B     (cycle x2)
        self.assertFalse(d.record(*A))   # A,B,A,B,A
        self.assertTrue(d.record(*B))    # A,B,A,B,A,B (cycle x3) -> STUCK

    def test_three_step_cycle_detected(self):
        d = StuckDetector(threshold=2)
        seq = [("a", {}), ("b", {}), ("c", {})]
        # one full cycle: not yet (need 2 repeats at threshold=2)
        for s in seq:
            self.assertFalse(d.record(*s))
        self.assertFalse(d.record(*seq[0]))   # a
        self.assertFalse(d.record(*seq[1]))   # b
        self.assertTrue(d.record(*seq[2]))    # c -> (a,b,c) repeated 2x -> STUCK

    def test_progressing_distinct_calls_never_stuck(self):
        # Distinct (tool,args) every call -> no repeating cycle -> never flagged.
        d = StuckDetector(threshold=3)
        for n in range(12):
            self.assertFalse(d.record("fill", {"target": f"e{n}", "text": str(n)}))


if __name__ == "__main__":
    unittest.main()
