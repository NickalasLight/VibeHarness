"""Stuck-detection for mid-run escalation.

When a small local model gets wedged it tends to make NO PROGRESS in one of two shapes:

  1. It repeats the *exact same* tool call and ignores the (identical) error it keeps
     getting back. Ground-truthed on run iter-13: turns 16/17/18 were all
     ``click(target="e329")`` returning the same ``e333 [invalid]`` error.
  2. It falls into a repeating k-step **cycle** that makes zero progress — the classic
     "re-navigate → look → re-navigate" pattern (issue #191): every turn ``goto <url>``
     then a (differing) follow-up, so no single signature is N-in-a-row and a
     consecutive-only counter never fires, yet the run is plainly stuck.

This detector catches BOTH with one rule: the run is stuck when the TAIL of the recorded
action history is ``threshold`` or more repetitions of some k-length block (period
``1 <= k <= max_period``). Period ``k = 1`` reproduces the original
"N consecutive identical calls" behaviour exactly (so the default threshold of **3**
still means three identical calls in a row); ``k >= 2`` catches no-progress cycles
(e.g. a 2-step ``A,B,A,B,A,B`` loop trips at threshold=3 repeats = 6 recorded calls).

A second, independent failure mode is a *premature* ``validate`` (the model claims done
before the success signal); the validator catches it and fails, and that single event is
enough to justify escalation to the stronger API model.

This module is intentionally tiny and dependency-free so it is trivial to unit-test.
"""
from __future__ import annotations

import json


def _make_sig(tool_name: str, args: dict) -> str:
    """Stable signature for a tool call: name + canonical-JSON args."""
    try:
        arg_part = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        arg_part = repr(args)
    return f"{tool_name}::{arg_part}"


class StuckDetector:
    """Tracks recent tool calls and flags when the run makes NO PROGRESS.

    Stateful only via a small bounded list of recent signatures; :meth:`reset` clears it
    for reuse/tests. ``escalated`` is a public flag the agent sets once it has swapped
    clients, so the swap happens at most once per run.

    Stuck rule: the tail of the recorded history is ``threshold`` (or more) repetitions of
    some k-length block, for any period ``1 <= k <= max_period``. ``k = 1`` is the classic
    consecutive-identical case; ``k >= 2`` is a repeating no-progress cycle (issue #191).
    """

    # Longest cycle period considered. A real progressing UI rarely repeats an EXACT
    # >4-step signature cycle, so 4 is a safe ceiling that still catches the common
    # 2-step (re-navigate ↔ look) and 3-step loops without risking false positives.
    _MAX_PERIOD = 4

    def __init__(self, threshold: int, max_period: int | None = None):
        self._threshold = max(1, int(threshold))
        self._max_period = max(1, int(max_period if max_period is not None
                                      else self._MAX_PERIOD))
        self._history: list[str] = []
        self.escalated: bool = False

    def record(self, tool_name: str, args: dict) -> bool:
        """Record an executed action. Returns ``True`` when the tail of the action
        history is ``threshold`` repetitions of a k-step block (k = 1 is the classic
        consecutive-identical case; k >= 2 is a no-progress cycle)."""
        self._history.append(_make_sig(tool_name, args))
        # Bound the history: we only ever inspect the last max_period * threshold entries.
        cap = self._max_period * self._threshold
        if len(self._history) > cap:
            self._history = self._history[-cap:]
        return self._is_stuck()

    def _tail_cycle_repeats(self, k: int) -> int:
        """How many times the final k-length block repeats contiguously at the tail."""
        h = self._history
        n = len(h)
        if n < k:
            return 0
        block = h[n - k:n]
        repeats = 0
        i = n
        while i - k >= 0 and h[i - k:i] == block:
            repeats += 1
            i -= k
        return repeats

    def _is_stuck(self) -> bool:
        for k in range(1, self._max_period + 1):
            if self._tail_cycle_repeats(k) >= self._threshold:
                return True
        return False

    def record_premature_validate(self) -> bool:
        """A premature (failed) validate always warrants escalation. A non-validate
        action resets the no-progress run, so call this instead of :meth:`record`
        for the validate path."""
        self._history = []
        return True

    def reset(self) -> None:
        self._history = []
        self.escalated = False
