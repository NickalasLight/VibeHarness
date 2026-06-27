"""Context-overrun budgeting for the native stateful path (issues #170/#172/#173).

On a heavy page the native ``decide_chat`` request — ``[system] + chat_history +
[user]`` — can exceed ``num_ctx`` as history accumulates and the live page snapshot
rides along. The old budgeter sized only the latest snapshot (legacy single-message
path) and the FIFO evictor always kept >= 1 message, so a single oversized snapshot
still overflowed → Ollama silently truncated the prompt and the model rambled with no
tool call (the #170 silent-mid-turn death).

These tests pin the fix WITHOUT a live model or browser:
  * the FULL message list is budgeted (not just the snapshot);
  * the OLDEST history is evicted FIRST; the snapshot is preserved whole whenever
    eviction alone makes it fit (#173);
  * the snapshot is truncated only as a last resort, and fully dropped only when even
    that can't fit — each flagged distinctly;
  * a growing-history + large-snapshot run keeps every assembled request within
    ``num_ctx`` AND emits the loud, grep-able CONTEXT-OVERRUN line (#172).
"""
from __future__ import annotations

import unittest

from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry
from vibeharness.reporting import NullReporter
from vibeharness.snapshot_budget import (SNAPSHOT_HISTORY_MARKER, fit_chat_history,
                                         input_budget_tokens, message_tokens)

from tests._fakes import FakeValidator


# ----- a config with a TINY usable window so a couple of messages force budgeting -----
def _tiny_cfg(**over) -> Config:
    base = dict(max_steps=5, num_ctx=2000, reason_tokens=400, action_tokens=400,
                snapshot_safety_margin_tokens=0, snapshot_chars_per_token=4.0,
                chat_history_max_turns=0)
    base.update(over)
    return Config(**base)   # native_tools defaults True, two_phase False


def _snapshot_msg(body: str) -> dict:
    """A committed live-snapshot message, shaped like the agent commits it."""
    return {"role": "user",
            "content": f"{SNAPSHOT_HISTORY_MARKER} — page now\n\n{body}"}


def _hist_msg(role: str, body: str) -> dict:
    return {"role": role, "content": body}


class FitChatHistoryTest(unittest.TestCase):
    def test_no_action_when_under_budget(self):
        cfg = _tiny_cfg()
        history = [_hist_msg("user", "hi"), _snapshot_msg("small page")]
        before = [dict(m) for m in history]
        rep = fit_chat_history(history, "sys", "user", cfg)
        self.assertFalse(rep.over_budget)
        self.assertFalse(rep.acted)
        self.assertEqual(history, before)   # untouched

    def test_evicts_history_first_and_preserves_snapshot_whole(self):
        cfg = _tiny_cfg()   # input_budget = 2000 - 800 = 1200 tok (~4800 chars)
        in_budget = input_budget_tokens(cfg)
        self.assertEqual(in_budget, 1200)
        snap_body = "P" * 2000        # snapshot ~ 523 tok — comfortably fits alone
        snap = _snapshot_msg(snap_body)
        raw_snap_len = len(snap["content"])
        # Three fat OLD turns (~400 tok each) that together blow the budget.
        history = [_hist_msg("user", "H" * 1600),
                   _hist_msg("assistant", "H" * 1600),
                   _hist_msg("user", "H" * 1600),
                   snap]
        rep = fit_chat_history(history, "x" * 40, "x" * 40, cfg)

        self.assertTrue(rep.over_budget)
        self.assertTrue(rep.acted)
        # History was evicted...
        self.assertGreaterEqual(rep.evicted_msgs, 1)
        # ...BEFORE the snapshot was touched — it is preserved at full size.
        self.assertFalse(rep.snapshot_truncated)
        self.assertFalse(rep.snapshot_dropped)
        self.assertEqual(rep.snapshot_kept_chars, raw_snap_len)
        self.assertEqual(snap["content"], _snapshot_msg(snap_body)["content"])
        # The snapshot message survived; the old turns are (at least partly) gone.
        self.assertIn(snap, history)
        self.assertLess(len(history), 4)
        # And the request now fits the window.
        self.assertTrue(rep.fits_after)
        self.assertLessEqual(rep.request_tokens_after, in_budget)

    def test_truncates_snapshot_only_as_last_resort(self):
        cfg = _tiny_cfg()
        in_budget = input_budget_tokens(cfg)   # 1200 tok
        big = "P" * 8000                        # ~2023 tok — too big even alone
        snap = _snapshot_msg(big)
        history = [snap]
        rep = fit_chat_history(history, "x" * 40, "x" * 40, cfg)

        self.assertTrue(rep.over_budget)
        self.assertTrue(rep.snapshot_truncated)
        self.assertFalse(rep.snapshot_dropped)
        self.assertGreater(rep.snapshot_dropped_chars, 0)
        self.assertLess(rep.snapshot_kept_chars, rep.snapshot_raw_chars)
        # The marker is preserved at the head so the model still knows it is the page.
        self.assertIn(SNAPSHOT_HISTORY_MARKER, history[0]["content"])
        # Critically, the request now fits the window.
        self.assertTrue(rep.fits_after)
        self.assertLessEqual(rep.request_tokens_after, in_budget)

    def test_snapshot_dropped_when_even_truncation_cannot_fit(self):
        # A system prompt that alone fills the window -> nothing fits; the snapshot is
        # fully dropped and flagged as the dangerous (blind) case.
        cfg = _tiny_cfg()
        big = "P" * 8000
        snap = _snapshot_msg(big)
        history = [snap]
        rep = fit_chat_history(history, "x" * 6000, "x" * 40, cfg)
        self.assertTrue(rep.over_budget)
        self.assertTrue(rep.snapshot_dropped)
        self.assertFalse(rep.snapshot_truncated)
        self.assertEqual(rep.snapshot_kept_chars, 0)
        self.assertIn("DROPPED", history[0]["content"])

    def test_cap_evicts_but_never_the_snapshot(self):
        cfg = _tiny_cfg(num_ctx=32768, chat_history_max_turns=2)
        snap = _snapshot_msg("page")
        history = [_hist_msg("user", "a"), _hist_msg("assistant", "b"),
                   _hist_msg("user", "c"), snap]
        rep = fit_chat_history(history, "sys", "user", cfg)
        # Capped to 2 messages, and the snapshot is one of the two kept.
        self.assertLessEqual(len(history), 2)
        self.assertIn(snap, history)
        self.assertGreaterEqual(rep.evicted_msgs, 2)


# -------------------- end-to-end through the real agent loop --------------------
class _CapturingReporter(NullReporter):
    def __init__(self):
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


class _ScriptedClient(LLMClient):
    """Records the exact ``messages`` handed to each decide_chat call so we can assert
    the assembled request stayed within the window."""

    def __init__(self, decisions):
        self._d = decisions
        self._i = 0
        self.seen_messages: list[list[dict]] = []

    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        d = self._d[min(self._i, len(self._d) - 1)]
        self._i += 1
        return d

    def decide_chat(self, messages, tools, constraint, on_reason=None, on_action=None):
        self.seen_messages.append([dict(m) for m in messages])
        d = self._d[min(self._i, len(self._d) - 1)]
        self._i += 1
        return d


class EndToEndBudgetTest(unittest.TestCase):
    def _registry(self):
        return ToolRegistry(build_default_tools(FileSystem(), 1000))

    def test_growing_history_request_stays_within_num_ctx_and_logs_loudly(self):
        reg = self._registry()
        cfg = _tiny_cfg(num_ctx=4000, reason_tokens=400, action_tokens=400)
        in_budget = input_budget_tokens(cfg)   # 3200 tok
        # A huge snapshot every turn (~10000 tok) — far over the window on its own.
        huge = "P" * 40000
        reporter = _CapturingReporter()
        write = ('<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", '
                 '"content": "hi"}}</tool_call>')
        validate = '<tool_call>{"name": "validate", "arguments": {}}</tool_call>'
        client = _ScriptedClient([
            Decision("", write), Decision("", write), Decision("", validate)])
        agent = RalphAgent(client, reg, "SYS", cfg, FakeValidator(passed=True),
                           reporter=reporter, codec=get_codec("hermes"),
                           raw_snapshot_provider=lambda: huge)
        result = agent.run("do work")
        self.assertTrue(result.finished)

        # EVERY assembled request stayed within the usable window.
        for msgs in client.seen_messages:
            est = sum(message_tokens(m, cfg.snapshot_chars_per_token) for m in msgs)
            self.assertLessEqual(
                est, in_budget,
                f"assembled request {est} tok exceeded input_budget {in_budget}")

        # The later turns carried a (truncated) live snapshot — the agent was not blinded.
        later = client.seen_messages[-1]
        self.assertTrue(any(SNAPSHOT_HISTORY_MARKER in (m.get("content") or "")
                            for m in later))

        # The overrun was LOUD: a grep-able line to the reporter AND the run log.
        self.assertTrue(any("!!! CONTEXT-OVERRUN" in n for n in reporter.notes),
                        f"no loud overrun line in {reporter.notes}")
        self.assertTrue(result.context_events)
        ev = result.context_events[-1]
        self.assertIn("!!! CONTEXT-OVERRUN", ev["message"])
        self.assertTrue(ev["snapshot_truncated"] or ev["snapshot_dropped"]
                        or ev["evicted_msgs"] > 0)


if __name__ == "__main__":
    unittest.main()
