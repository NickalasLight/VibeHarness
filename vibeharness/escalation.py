"""Stuck-detection for mid-run escalation.

When a small local model gets wedged it tends to repeat the *exact same* tool call
and ignore the (identical) error it keeps getting back. Ground-truthed on run
iter-13: turns 16/17/18 were all ``click(target="e329")`` returning the same
``e333 [invalid]`` error — three consecutive identical calls. So the default stuck
threshold is **3** consecutive identical tool signatures.

A second failure mode is a *premature* ``validate`` (the model claims done before
the success signal); the validator catches it and fails, and that single event is
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
    """Tracks consecutive identical tool calls and flags when the run is stuck.

    Stateful only via a small counter dict; :meth:`reset` clears it for reuse/tests.
    ``escalated`` is a public flag the agent sets once it has swapped clients, so the
    swap happens at most once per run.
    """

    def __init__(self, threshold: int):
        self._threshold = max(1, int(threshold))
        self._consecutive_counts: dict[str, int] = {}
        self._last_sig: str | None = None
        self.escalated: bool = False

    def record(self, tool_name: str, args: dict) -> bool:
        """Record an executed action. Returns ``True`` when the number of
        consecutive identical calls reaches the threshold."""
        sig = _make_sig(tool_name, args)
        if sig == self._last_sig:
            self._consecutive_counts[sig] = self._consecutive_counts.get(sig, 0) + 1
        else:
            self._consecutive_counts = {sig: 1}
            self._last_sig = sig
        return self._consecutive_counts[sig] >= self._threshold

    def record_premature_validate(self) -> bool:
        """A premature (failed) validate always warrants escalation. A non-validate
        action resets the consecutive run, so call this instead of :meth:`record`
        for the validate path."""
        self._last_sig = None
        self._consecutive_counts = {}
        return True

    def reset(self) -> None:
        self._consecutive_counts = {}
        self._last_sig = None
        self.escalated = False
