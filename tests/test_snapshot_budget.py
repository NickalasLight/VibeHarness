"""Issue #43: DYNAMIC snapshot budget.

The live page snapshot is no longer truncated to a FIXED character cap. Instead it
may be as large as possible and is truncated ONLY when including it whole would push
the FULL model message (system prompt + per-turn user/history) past the usable input
window. These tests pin every branch of the budget maths and the end-to-end wiring,
all without a browser, a model, or an agent run.

Worked spec example (named tests below):
  * a 200k-char snapshot beside a 50k-char message -> injected whole;
  * a 500k-char snapshot beside the same message -> truncated, because together they
    would overflow the context window.
"""
import unittest
from dataclasses import replace

from vibeharness.config import Config
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.snapshot_budget import (
    SnapshotBudget,
    compute_snapshot_budget,
    estimate_tokens,
    input_budget_tokens,
    render_budgeted_snapshot,
    truncate_snapshot,
)
from vibeharness.toolset import default_catalog


def _registry(names):
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(names), Config())


# A tiny, easy-to-reason-about config: 4 chars/token, no fractional rounding noise.
def _cfg(**over):
    base = Config(
        num_ctx=10_000,
        reason_tokens=500,
        action_tokens=1_500,
        snapshot_safety_margin_tokens=0,
        snapshot_chars_per_token=4.0,
        web_snapshot_char_limit=10_000_000,  # ceiling out of the way unless tested
    )
    return replace(base, **over) if over else base


class TokenEstimateTest(unittest.TestCase):
    def test_estimate_rounds_up_never_undercounts(self):
        # 5 chars at 4 chars/token = 1.25 tokens -> must round UP to 2, never 1.
        self.assertEqual(estimate_tokens("x" * 5, 4.0), 2)
        self.assertEqual(estimate_tokens("x" * 4, 4.0), 1)
        self.assertEqual(estimate_tokens("x" * 8, 4.0), 2)
        self.assertEqual(estimate_tokens("", 4.0), 0)

    def test_nonpositive_ratio_is_worst_case_one_token_per_char(self):
        self.assertEqual(estimate_tokens("abcde", 0), 5)
        self.assertEqual(estimate_tokens("abcde", -1), 5)

    def test_input_budget_subtracts_reservation_and_margin(self):
        cfg = _cfg(num_ctx=10_000, reason_tokens=500, action_tokens=1_500,
                   snapshot_safety_margin_tokens=200)
        # 10000 - (500 + 1500) - 200 = 7800
        self.assertEqual(input_budget_tokens(cfg), 7_800)

    def test_input_budget_never_negative(self):
        cfg = _cfg(num_ctx=100, reason_tokens=500, action_tokens=1_500,
                   snapshot_safety_margin_tokens=0)
        self.assertEqual(input_budget_tokens(cfg), 0)


class TruncateTest(unittest.TestCase):
    def test_fits_returns_whole_byte_for_byte(self):
        raw = "abc123" * 100
        body, dropped = truncate_snapshot(raw, len(raw))
        self.assertEqual(body, raw)
        self.assertEqual(dropped, 0)

    def test_truncates_to_exact_boundary(self):
        raw = "y" * 5000
        body, dropped = truncate_snapshot(raw, 1234)
        self.assertEqual(len(body), 1234)          # EXACT boundary
        self.assertEqual(body, "y" * 1234)
        self.assertEqual(dropped, 5000 - 1234)

    def test_zero_budget_drops_everything(self):
        body, dropped = truncate_snapshot("hello", 0)
        self.assertEqual(body, "")
        self.assertEqual(dropped, 5)

    def test_render_appends_marker_only_when_truncated(self):
        self.assertEqual(render_budgeted_snapshot("abc", 10), "abc")
        out = render_budgeted_snapshot("y" * 100, 30)
        self.assertTrue(out.startswith("y" * 30))
        self.assertIn("70 chars truncated", out)
        self.assertEqual(render_budgeted_snapshot("y" * 100, 0), "")


class ComputeBudgetTest(unittest.TestCase):
    def test_small_rest_leaves_large_snapshot_budget(self):
        cfg = _cfg()  # input_budget = 10000 - 2000 - 0 = 8000 tokens
        rest = "r" * 400  # 100 tokens
        b = compute_snapshot_budget(cfg, rest)
        self.assertFalse(b.overflow)
        self.assertEqual(b.input_budget_tokens, 8_000)
        self.assertEqual(b.rest_tokens, 100)
        self.assertEqual(b.snapshot_budget_tokens, 7_900)
        # 7900 tokens * 4 chars/token = 31600 chars.
        self.assertEqual(b.budget_chars, 31_600)

    def test_budget_shrinks_as_rest_grows(self):
        cfg = _cfg()
        small = compute_snapshot_budget(cfg, "r" * 400).budget_chars
        large = compute_snapshot_budget(cfg, "r" * 4000).budget_chars
        self.assertLess(large, small)

    def test_rest_at_or_over_budget_overflows_to_zero(self):
        cfg = _cfg()  # 8000-token input budget
        rest = "r" * (8_000 * 4)  # exactly 8000 tokens
        b = compute_snapshot_budget(cfg, rest)
        self.assertTrue(b.overflow)
        self.assertEqual(b.budget_chars, 0)

    def test_rest_over_budget_overflows(self):
        cfg = _cfg()
        rest = "r" * (9_000 * 4)  # 9000 tokens > 8000 input budget
        b = compute_snapshot_budget(cfg, rest)
        self.assertTrue(b.overflow)
        self.assertEqual(b.budget_chars, 0)

    def test_absolute_ceiling_clamps_budget(self):
        # A huge window but a low ceiling: the ceiling wins.
        cfg = _cfg(num_ctx=10_000_000, web_snapshot_char_limit=12_345)
        b = compute_snapshot_budget(cfg, "r" * 40)
        self.assertEqual(b.budget_chars, 12_345)

    def test_config_plumbing_feeds_computation(self):
        # Changing each knob measurably moves the budget — proving every field is wired.
        base = _cfg()
        b0 = compute_snapshot_budget(base, "r" * 40).budget_chars
        # More num_ctx -> bigger budget.
        self.assertGreater(
            compute_snapshot_budget(replace(base, num_ctx=20_000), "r" * 40).budget_chars, b0)
        # More reserved output tokens -> smaller budget.
        self.assertLess(
            compute_snapshot_budget(replace(base, action_tokens=5_000), "r" * 40).budget_chars, b0)
        self.assertLess(
            compute_snapshot_budget(replace(base, reason_tokens=2_000), "r" * 40).budget_chars, b0)
        # Bigger safety margin -> smaller budget.
        self.assertLess(
            compute_snapshot_budget(replace(base, snapshot_safety_margin_tokens=1_000),
                                    "r" * 40).budget_chars, b0)
        # More chars/token -> the same token budget buys MORE chars.
        self.assertGreater(
            compute_snapshot_budget(replace(base, snapshot_chars_per_token=8.0),
                                    "r" * 40).budget_chars, b0)


class NeverExceedsNumCtxTest(unittest.TestCase):
    """The headline safety property: the estimated total (rest + budgeted snapshot +
    the reserved output) must never exceed num_ctx, across a sweep of message sizes."""

    def test_total_estimate_stays_within_num_ctx(self):
        # When the snapshot is injected (non-overflow), rest + snapshot + reserved
        # output must never exceed num_ctx. (When rest ALONE already overflows the
        # window the budget is 0 and nothing is injected — covered separately; the
        # snapshot can't be blamed for a too-big rest.)
        cfg = _cfg(snapshot_chars_per_token=4.0, snapshot_safety_margin_tokens=64)
        checked_non_overflow = False
        for rest_chars in (0, 100, 1_000, 7_000, 30_000, 31_999, 32_000, 50_000, 200_000):
            rest = "r" * rest_chars
            b = compute_snapshot_budget(cfg, rest)
            if b.overflow:
                self.assertEqual(b.budget_chars, 0)
                continue
            checked_non_overflow = True
            snapshot = "s" * b.budget_chars  # inject the full budget
            total_tokens = (estimate_tokens(rest, cfg.snapshot_chars_per_token)
                            + estimate_tokens(snapshot, cfg.snapshot_chars_per_token)
                            + cfg.reason_tokens + cfg.action_tokens)
            self.assertLessEqual(
                total_tokens, cfg.num_ctx,
                msg=f"rest_chars={rest_chars} total_tokens={total_tokens} > {cfg.num_ctx}")
        self.assertTrue(checked_non_overflow)  # the sweep actually exercised the live path

    def test_overflow_path_injects_nothing(self):
        cfg = _cfg()
        rest = "r" * (50_000)  # way over the 8000-token window
        b = compute_snapshot_budget(cfg, rest)
        self.assertTrue(b.overflow)
        self.assertEqual(render_budgeted_snapshot("s" * 1000, b.budget_chars), "")


class WorkedExampleTest(unittest.TestCase):
    """The exact spec example, with a window large enough that a 200k snapshot fits
    beside a 50k message but a 500k one does not.

    Using 4 chars/token: 50k message = 12.5k tokens; output reservation default
    (2048+16384) = 18432 tokens. Pick num_ctx so input_budget comfortably holds the
    200k (50k tokens) case but not the 500k (125k tokens) case.
    """

    def setUp(self):
        # input_budget = num_ctx - 18432 - margin. We want it ~ 75k tokens so:
        #   200k snapshot = 50k tokens, + 12.5k rest = 62.5k <= 75k  -> fits
        #   500k snapshot = 125k tokens, + 12.5k rest = 137.5k > 75k -> truncates
        self.cfg = Config(
            num_ctx=93_432,                 # 75000 input budget after reservation
            reason_tokens=2048,
            action_tokens=16384,
            snapshot_safety_margin_tokens=0,
            snapshot_chars_per_token=4.0,
            web_snapshot_char_limit=10_000_000,
        )
        self.message_50k = "m" * 50_000     # the "50k message"

    def test_200k_snapshot_with_50k_message_fits_whole(self):
        b = compute_snapshot_budget(self.cfg, self.message_50k)
        self.assertFalse(b.overflow)
        raw = "s" * 200_000
        self.assertLessEqual(len(raw), b.budget_chars)        # whole snapshot fits
        rendered = render_budgeted_snapshot(raw, b.budget_chars)
        self.assertEqual(rendered, raw)                       # byte-for-byte, no marker
        self.assertNotIn("truncated", rendered)

    def test_500k_snapshot_with_50k_message_is_truncated(self):
        b = compute_snapshot_budget(self.cfg, self.message_50k)
        self.assertFalse(b.overflow)
        raw = "s" * 500_000
        self.assertGreater(len(raw), b.budget_chars)          # cannot fit whole
        rendered = render_budgeted_snapshot(raw, b.budget_chars)
        self.assertIn("truncated", rendered)
        # The kept body is EXACTLY the computed budget.
        kept = rendered.split("\n…[")[0]
        self.assertEqual(len(kept), b.budget_chars)
        # And the whole thing still respects num_ctx.
        total = (estimate_tokens(self.message_50k, 4.0)
                 + estimate_tokens(kept, 4.0)
                 + self.cfg.reason_tokens + self.cfg.action_tokens)
        self.assertLessEqual(total, self.cfg.num_ctx)


class LongRunShrinkingBudgetTest(unittest.TestCase):
    """Over a long run the narrative history grows, so 'rest' grows turn over turn and
    the snapshot budget must shrink monotonically (until it hits overflow)."""

    def test_growing_history_shrinks_snapshot_budget(self):
        cfg = _cfg()
        budgets = []
        for turn in range(1, 8):
            # Simulate the user/history message growing each turn.
            rest = "r" * (200 + turn * 3_000)
            b = compute_snapshot_budget(cfg, rest)
            budgets.append(b.budget_chars)
        # Strictly non-increasing, and strictly decreasing while not yet overflowed.
        for prev, cur in zip(budgets, budgets[1:]):
            self.assertLessEqual(cur, prev)
        self.assertLess(budgets[-1], budgets[0])


class EndToEndProviderTest(unittest.TestCase):
    """Drive the actual cli.make_system_prompt_provider with a fake raw-snapshot source
    (no browser, no model). Proves the FULL message is what gets budgeted and that the
    truncation now happens at prompt-build time inside the provider."""

    def _provider(self, cfg, raw_text, task="DO THE THING", workspace="ws-tree"):
        from vibeharness.cli import make_system_prompt_provider
        builder = SystemPromptBuilder(_registry(["web"]))
        return make_system_prompt_provider(
            builder, cfg, task, lambda: workspace, (lambda: raw_text))

    def test_small_snapshot_injected_whole(self):
        cfg = _cfg()
        raw = "### Page\nCONSENT BANNER e6"
        sp = self._provider(cfg, raw)(user="short user message")
        self.assertIn("# Current page (live snapshot", sp)
        self.assertIn("CONSENT BANNER e6", sp)
        self.assertNotIn("truncated", sp)

    def test_large_snapshot_truncated_against_full_message(self):
        # Small window so even a modest snapshot must truncate; the user message is part
        # of 'rest', so a bigger user message yields a smaller injected snapshot. The
        # web system prompt now documents 16 discrete subtools (#51), so this window is
        # sized to leave room for that larger 'rest' PLUS a truncatable snapshot.
        cfg = _cfg(num_ctx=8_000, reason_tokens=500, action_tokens=1_500,
                   snapshot_safety_margin_tokens=0)
        raw = "z" * 100_000
        small_user = self._provider(cfg, raw)(user="u" * 100)
        big_user = self._provider(cfg, raw)(user="u" * 4_000)
        self.assertIn("truncated", small_user)
        self.assertIn("truncated", big_user)
        # Bigger user message -> 'rest' bigger -> fewer snapshot chars survive.
        self.assertGreater(len(small_user), len(big_user))

    def test_overflow_injects_no_page_and_does_not_raise(self):
        # 'rest' alone already exceeds the window: provider must inject NO page section.
        cfg = _cfg(num_ctx=600, reason_tokens=200, action_tokens=300,
                   snapshot_safety_margin_tokens=0)
        sp = self._provider(cfg, "z" * 100_000)(user="u" * 8_000)
        self.assertNotIn("# Current page (live snapshot", sp)

    def test_no_snapshot_renders_no_page_section(self):
        cfg = _cfg()
        sp = self._provider(cfg, "")(user="hi")
        self.assertNotIn("# Current page (live snapshot", sp)

    def test_provider_accepts_user_message(self):
        # The provider must accept the per-turn user message (arity contract for #43).
        import inspect
        from vibeharness.cli import make_system_prompt_provider
        builder = SystemPromptBuilder(_registry(["web"]))
        p = make_system_prompt_provider(builder, _cfg(), "t", lambda: "ws", (lambda: "snap"))
        sig = inspect.signature(p)
        self.assertEqual(len(sig.parameters), 1)


class PinnedNumCtxFitsHeavyPageTest(unittest.TestCase):
    """ISSUE #77 interaction with #43: the default num_ctx (32768, confirmed safe for the
    #140 qwen3:4b upgrade with OLLAMA_FLASH_ATTENTION=1) sizes the dynamic snapshot
    budget. ISSUE #92 rebalances the output reservation (reason/action both 4096), giving
    input_budget = 32768 - 8192 - 1024 = 23552 input tokens. Confirm the DEFAULT config
    leaves real room and a heavy page fits whole (or, when the budget is smaller, the #43
    dynamic snapshot budget TRIMS it gracefully)."""

    def test_default_num_ctx_is_the_pinned_value(self):
        # ISSUE #140: kept at 32768 for the qwen3:4b upgrade (safe with
        # OLLAMA_FLASH_ATTENTION=1: ~4360-5086 MiB on the RTX 3080 8GB). See config.py / #139.
        self.assertEqual(Config().num_ctx, 32768)

    def test_default_output_reservation_is_rebalanced(self):
        # ISSUE #92: reason and action tokens are both 4096 now (were 2048 / 16384).
        self.assertEqual(Config().reason_tokens, 4096)
        self.assertEqual(Config().action_tokens, 4096)

    def test_default_input_budget_is_positive_and_sane(self):
        # ISSUE #140: 32768 - (4096 + 4096) - 1024 = 23552 input tokens after the output
        # reservation and safety margin — the same budget the qwen2.5-coder:3b runs used.
        self.assertEqual(input_budget_tokens(Config()), 23_552)

    def test_heavy_youtube_snapshot_fits_whole_under_default(self):
        # ISSUE #140: at num_ctx=32768 the diagnosed worst case (a ~45k-char YouTube watch
        # ARIA snapshot ≈ ~11k tokens) fits WHOLE in the 23552-token input budget beside a
        # realistic ~1k-token system prompt — no overflow, no truncation.
        cfg = Config()  # real defaults: num_ctx=32768
        rest = "p" * 4_000          # ~1k-token system prompt + per-turn message
        heavy_snapshot = "s" * 45_000
        b = compute_snapshot_budget(cfg, rest)
        self.assertFalse(b.overflow)
        self.assertGreater(b.budget_chars, 0)
        # The whole heavy snapshot fits inside the budget.
        self.assertGreaterEqual(b.budget_chars, len(heavy_snapshot))
        rendered = render_budgeted_snapshot(heavy_snapshot, b.budget_chars)
        self.assertNotIn("truncated", rendered)

    def test_realistic_multistep_turn_has_positive_budget(self):
        # ISSUE #140: a realistic prompt + several turns of history still leave a positive
        # snapshot budget at num_ctx=32768 (no overflow). Model it concretely at 4
        # chars/token:
        #   - realistic system prompt (web toolset docs etc.): ~12k chars (~3k tokens)
        #   - several turns of narrative history: ~4k chars (~1k tokens)
        # rest = prompt + history = ~16k chars (~4k tokens); a large budget remains.
        cfg = Config()
        prompt_chars = 12_000
        history_chars = 4_000
        rest = "p" * (prompt_chars + history_chars)   # ~16k chars ≈ 4k tokens
        heavy_snapshot = "s" * 45_000
        b = compute_snapshot_budget(cfg, rest)
        self.assertFalse(b.overflow)
        self.assertGreater(
            b.budget_chars, 0,
            msg="realistic prompt + history left no room for a snapshot at num_ctx=32768")
        # And there is still spare budget left over after a heavy snapshot — real headroom.
        spare_tokens = b.snapshot_budget_tokens - estimate_tokens(heavy_snapshot, 4.0)
        self.assertGreater(spare_tokens, 0,
                           msg="no spare input budget after heavy snapshot — #92 goal unmet")


if __name__ == "__main__":
    unittest.main()
