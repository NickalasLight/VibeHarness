"""Dynamic snapshot budgeting (issue #43).

The live page snapshot (issue #24) used to be truncated to a FIXED character cap.
That is wasteful when the rest of the message is small (a big snapshot would fit
fine) and unsafe when the rest is large (even a "capped" snapshot could push the
whole message past ``num_ctx``). This module replaces the fixed cap with a DYNAMIC
budget: the snapshot may be as large as possible and is truncated ONLY when
including it whole would push the full model message past the usable input window.

Worked example from the spec:
  * a 200k-char snapshot beside a 50k-char message -> injected whole;
  * a 500k-char snapshot beside the same 50k-char message -> truncated, because
    together they would overflow the context window.

The maths (all token quantities derive from a conservative chars-per-token ratio):

    input_budget_tokens = num_ctx
                          - (reason_tokens + action_tokens)   # output reservation
                          - safety_margin_tokens              # template/role/error slack

    rest_tokens         = tokens(system_without_snapshot + user)

    snapshot_budget_tokens = input_budget_tokens - rest_tokens
    snapshot_budget_chars  = snapshot_budget_tokens * chars_per_token

If the raw snapshot already fits the char budget it is injected whole; otherwise it
is truncated to EXACTLY ``snapshot_budget_chars`` (a truncation marker is appended
by the caller's renderer, never counted against the budget). If ``rest`` already
consumes the whole input window the budget is zero -> a minimal/empty snapshot and a
warning, so we NEVER exceed ``num_ctx``.

Everything here is pure and side-effect free so it can be unit-tested without a
browser, a model, or an agent run.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class SnapshotBudget:
    """The outcome of sizing the snapshot for one turn.

    ``budget_chars`` is the maximum number of RAW snapshot characters that may be
    injected this turn (>= 0). ``overflow`` is True when the rest of the message
    already meets or exceeds the input window, so no snapshot fits at all and the
    caller should inject an empty/minimal snapshot and warn.
    """
    budget_chars: int
    input_budget_tokens: int
    rest_tokens: int
    overflow: bool

    @property
    def snapshot_budget_tokens(self) -> int:
        return max(0, self.input_budget_tokens - self.rest_tokens)


def _ceil_div_chars(n: int, chars_per_token: float) -> int:
    """Tokens for ``n`` chars, rounded UP (float-safe, no ``math`` import).

    Rounding UP means the estimate never under-counts tokens; under-counting is the
    only dangerous direction (it would let the real message exceed ``num_ctx``). A
    non-positive ratio degrades to the 1-token-per-char worst case.
    """
    if n <= 0:
        return 0
    if chars_per_token <= 0:
        return n
    q = n / chars_per_token
    iq = int(q)
    return iq + (1 if q > iq else 0)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Conservatively (ceil) estimate the token count of ``text``."""
    return _ceil_div_chars(len(text), chars_per_token)


def input_budget_tokens(config: Config) -> int:
    """The usable input window in tokens: the whole context minus the output
    reservation (reason + action tokens) minus the safety margin. Never negative."""
    reserved = config.reason_tokens + config.action_tokens + \
        config.snapshot_safety_margin_tokens
    return max(0, config.num_ctx - reserved)


def compute_snapshot_budget(config: Config, rest_text: str) -> SnapshotBudget:
    """Size the snapshot for one turn given ``rest_text`` — the FULL model message
    WITHOUT the snapshot (system prompt minus its page section, plus the per-turn
    user/history message). Returns a :class:`SnapshotBudget`.

    The returned ``budget_chars`` is the largest number of raw snapshot characters
    that, added to ``rest_text``, keeps the estimated total at or under the input
    window. It is clamped to the absolute ceiling ``web_snapshot_char_limit`` so a
    misconfiguration can never request an unbounded snapshot.
    """
    cpt = config.snapshot_chars_per_token
    in_budget = input_budget_tokens(config)
    rest_tokens = _ceil_div_chars(len(rest_text), cpt)

    snapshot_tokens = in_budget - rest_tokens
    if snapshot_tokens <= 0:
        # The rest of the message already fills (or overflows) the window: no room
        # for any snapshot. Inject nothing and let the caller warn.
        return SnapshotBudget(budget_chars=0, input_budget_tokens=in_budget,
                              rest_tokens=rest_tokens, overflow=True)

    # Tokens -> chars. Round DOWN (floor) so the chosen char budget never estimates
    # back to MORE than snapshot_tokens once re-tokenised — staying conservative.
    budget_chars = int(snapshot_tokens * cpt) if cpt > 0 else snapshot_tokens
    # Honour the absolute ceiling / safety fallback.
    budget_chars = min(budget_chars, config.web_snapshot_char_limit)
    return SnapshotBudget(budget_chars=budget_chars, input_budget_tokens=in_budget,
                          rest_tokens=rest_tokens, overflow=False)


def truncate_snapshot(raw: str, budget_chars: int) -> tuple[str, int]:
    """Truncate ``raw`` snapshot text to at most ``budget_chars`` characters.

    Returns ``(text, dropped)`` where ``dropped`` is the number of characters
    removed (0 when the snapshot fit whole). The kept body is EXACTLY
    ``budget_chars`` chars when truncation occurs, so the boundary is assertable.
    The caller is responsible for appending any human-readable truncation marker;
    the marker is intentionally NOT part of the budgeted body.
    """
    if budget_chars <= 0:
        return "", len(raw)
    if len(raw) <= budget_chars:
        return raw, 0
    return raw[:budget_chars], len(raw) - budget_chars


def render_budgeted_snapshot(raw: str, budget_chars: int) -> str:
    """Truncate ``raw`` to ``budget_chars`` and append the truncation marker iff
    truncation occurred. The marker matches web.py's existing rendering so logs read
    consistently. Returns "" when nothing fits (``budget_chars <= 0``)."""
    body, dropped = truncate_snapshot(raw, budget_chars)
    if dropped <= 0:
        return body
    if not body:
        return ""  # zero budget -> inject nothing (caller warns separately)
    return body + f"\n…[+{dropped} chars truncated]"
