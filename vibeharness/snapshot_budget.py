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

import json
from dataclasses import dataclass

from .config import Config, effective_context_window

# The header the agent stamps on the post-turn live page snapshot it commits to the
# stateful chat history (see ``RalphAgent._record`` / the page_snapshot observation).
# The fit logic below uses it to locate the LATEST snapshot message so it can be
# preserved (and only truncated as a last resort) while older history is evicted first
# (issue #173). Kept here, beside the budgeting maths, so both producer and budgeter
# agree on the marker.
SNAPSHOT_HISTORY_MARKER = "## Latest page state"


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


def input_budget_tokens(config: Config, window: int | None = None) -> int:
    """The usable input window in tokens: the whole context minus the output
    reservation (reason + action tokens) minus the safety margin. Never negative.

    ISSUE #193 — the "whole context" is the ACTIVE base model's PER-MODEL context window
    (``effective_context_window``), NOT a flat ``config.num_ctx``. For the local Ollama
    model that resolves back to ``config.num_ctx`` (qwen3:4b stays 32768 — a GPU limit,
    #77/#140); for a DeepSeek/GLM API model it is that model's much larger documented
    window, so the input/snapshot budget is no longer needlessly capped at qwen3:4b's 32768.
    ``window`` lets a caller that has already escalated to a different model (the agent) pass
    that model's window directly; when None the base role's window is resolved."""
    win = window if window is not None else effective_context_window(config)
    reserved = config.reason_tokens + config.action_tokens + \
        config.snapshot_safety_margin_tokens
    return max(0, win - reserved)


def compute_snapshot_budget(config: Config, rest_text: str,
                            window: int | None = None) -> SnapshotBudget:
    """Size the snapshot for one turn given ``rest_text`` — the FULL model message
    WITHOUT the snapshot (system prompt minus its page section, plus the per-turn
    user/history message). Returns a :class:`SnapshotBudget`.

    The returned ``budget_chars`` is the largest number of raw snapshot characters
    that, added to ``rest_text``, keeps the estimated total at or under the input
    window. It is clamped to the absolute ceiling ``web_snapshot_char_limit`` so a
    misconfiguration can never request an unbounded snapshot.
    """
    cpt = config.snapshot_chars_per_token
    in_budget = input_budget_tokens(config, window)
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


# ---------------------------------------------------------------------------
# Full stateful-request budgeting (issues #170/#172/#173)
#
# The single-message budgeter above sizes ONE snapshot against ``system + user``. The
# NATIVE stateful path (``OllamaClient.decide_chat``) instead sends
# ``[system] + chat_history + [user]`` — the accumulated multi-turn history PLUS the
# latest page snapshot (committed as a tool observation in the history). As history
# grows the assembled request can exceed ``num_ctx`` even though each individual piece
# looked fine; Ollama then silently truncates the prompt (dropping the system/task from
# the middle) and the model rambles with no tool call — the #170 silent-mid-turn death.
#
# ``fit_chat_history`` budgets the WHOLE message list. The page snapshot is the agent's
# eyes, so the ordering of cuts is (most-preferred first):
#   1. evict the OLDEST non-snapshot history messages (FIFO), respecting the optional
#      ``chat_history_max_turns`` cap, NEVER touching the latest snapshot;
#   2. only if the request STILL overflows, truncate the latest snapshot (last resort);
#   3. never silently — the caller emits a loud, grep-able CONTEXT-OVERRUN line.
# ---------------------------------------------------------------------------

# Appended to a snapshot whose tail we had to cut to fit the window (distinct from the
# single-message marker so logs make the stateful path obvious).
_HISTORY_TRUNC_MARKER = "…[+{dropped} chars truncated to fit the context window]"
# Replacement body when even a single char of snapshot won't fit (the dangerous case).
_SNAPSHOT_DROPPED_BODY = (
    "## Latest page state — [PAGE SNAPSHOT DROPPED: it exceeded the context budget even "
    "after evicting all prior history. The agent is effectively blind this turn.]")


@dataclass(frozen=True)
class BudgetFitReport:
    """What :func:`fit_chat_history` did to keep the request within ``num_ctx``.

    ``acted`` is True iff any eviction or snapshot truncation was performed (i.e. the
    request was over budget). The caller uses the fields to emit the loud overrun line
    (#172) and to assert the fit in tests (#173)."""
    over_budget: bool
    fits_after: bool
    evicted_msgs: int
    evicted_tokens: int
    snapshot_truncated: bool
    snapshot_dropped: bool
    snapshot_raw_chars: int
    snapshot_kept_chars: int
    snapshot_dropped_chars: int
    request_tokens_before: int
    request_tokens_after: int
    input_budget_tokens: int
    num_ctx: int
    history_msgs_after: int
    history_tokens_after: int
    # ISSUE #214 — on the API structured path the live page snapshot rides the TAIL user
    # message (a cache-stable prefix; it is never committed to history), so the snapshot
    # this budgeter may truncate is NOT an in-history message but the tail copy passed via
    # ``tail_snapshot``. ``tail_snapshot_out`` is the (possibly last-resort-truncated) tail
    # snapshot the caller should actually append to the live user turn; "" in native mode.
    tail_snapshot_out: str = ""

    @property
    def acted(self) -> bool:
        return (self.evicted_msgs > 0 or self.snapshot_truncated
                or self.snapshot_dropped)


def message_tokens(message: dict, chars_per_token: float) -> int:
    """Conservative token estimate for one chat message: its ``content`` plus any
    structured ``tool_calls`` (serialised the way they ride the wire)."""
    text = message.get("content") or ""
    for tc in message.get("tool_calls", []) or []:
        try:
            text += json.dumps(tc, ensure_ascii=False)
        except (TypeError, ValueError):
            text += str(tc)
    return estimate_tokens(text, chars_per_token)


def _latest_snapshot_index(chat_history: list[dict]) -> int | None:
    """Index of the most-recent message carrying the live page snapshot, or None."""
    for i in range(len(chat_history) - 1, -1, -1):
        if SNAPSHOT_HISTORY_MARKER in (chat_history[i].get("content") or ""):
            return i
    return None


def fit_chat_history(chat_history: list[dict], system: str, user: str,
                     config: Config, window: int | None = None,
                     tail_snapshot: str | None = None) -> BudgetFitReport:
    """Trim ``chat_history`` IN PLACE so ``[system] + chat_history + [user]`` fits the
    input window, evicting OLDEST history first and truncating the latest snapshot only
    as a last resort (issue #173). Returns a :class:`BudgetFitReport`.

    ISSUE #193 — the input window derives from the ACTIVE base model's PER-MODEL context
    window (``window`` when the caller has escalated to a specific model, else the base
    role's window). The report's ``num_ctx`` field carries that EFFECTIVE window so the
    loud CONTEXT-OVERRUN line reflects the model actually in use, not a flat 32768.

    ISSUE #214 — TAIL-SNAPSHOT mode (the API prompt-cache-friendly path). When
    ``tail_snapshot`` is given, the live page snapshot is NOT a committed history message
    (so the prefix stays byte-stable for prompt caching) but rides the TAIL user turn. It
    is budgeted here exactly like the in-history snapshot on the native path: every history
    message is evictable (oldest first), and only as a LAST resort is the tail snapshot
    truncated — returned via ``report.tail_snapshot_out`` for the caller to append to the
    live user message. In native (in-history) mode ``tail_snapshot`` is ``None`` and the
    behaviour is unchanged: the latest in-history snapshot is preserved/last-truncated.

    Pure except for the in-place mutation of ``chat_history`` (and the snapshot message's
    ``content`` when truncation is the last resort), so it is unit-testable without a
    model or a browser."""
    cpt = config.snapshot_chars_per_token
    eff_window = window if window is not None else effective_context_window(config)
    in_budget = input_budget_tokens(config, eff_window)
    tail_mode = tail_snapshot is not None
    snap_text = tail_snapshot or ""               # the tail snapshot (#214), "" in native mode
    base_fixed = estimate_tokens(system, cpt) + estimate_tokens(user, cpt)

    def hist_tokens() -> int:
        return sum(message_tokens(m, cpt) for m in chat_history)

    def request_tokens() -> int:
        # In tail-snapshot mode the snapshot tokens ride ``fixed`` (the tail user turn),
        # not a history message — so they shrink only when the snapshot itself is truncated.
        return base_fixed + estimate_tokens(snap_text, cpt) + hist_tokens()

    req_before = request_tokens()
    over = req_before > in_budget

    if tail_mode:
        # The snapshot is NOT in history: every history message is freely evictable, and the
        # snapshot to preserve/last-truncate is ``snap_text`` (the tail copy).
        snapshot_raw_chars = len(snap_text)

        def _oldest_evictable() -> int | None:
            return 0 if chat_history else None
    else:
        snap_idx = _latest_snapshot_index(chat_history)
        snapshot_raw_chars = (
            len(chat_history[snap_idx].get("content") or "") if snap_idx is not None else 0)

        def _oldest_evictable() -> int | None:
            si = _latest_snapshot_index(chat_history)
            for i in range(len(chat_history)):
                if i != si:
                    return i
            return None  # nothing left but the snapshot itself

    evicted_msgs = 0
    evicted_tokens = 0

    # (1a) Coarse fixed message cap (chat_history_max_turns; 0 = off). Never evicts the
    # latest snapshot — it is the agent's current view of the page.
    cap = config.chat_history_max_turns
    if cap and cap > 0:
        while len(chat_history) > cap:
            i = _oldest_evictable()
            if i is None:
                break
            evicted_tokens += message_tokens(chat_history[i], cpt)
            del chat_history[i]
            evicted_msgs += 1

    # (1b) Token budget: evict oldest non-snapshot messages until the request fits or
    # only the snapshot (and whatever can't be evicted) remains.
    while request_tokens() > in_budget:
        i = _oldest_evictable()
        if i is None:
            break
        evicted_tokens += message_tokens(chat_history[i], cpt)
        del chat_history[i]
        evicted_msgs += 1

    # (2) Last resort: the latest snapshot alone still overflows -> truncate it.
    snapshot_truncated = False
    snapshot_dropped = False
    snapshot_dropped_chars = 0
    snapshot_kept_chars = snapshot_raw_chars
    if tail_mode and request_tokens() > in_budget and snap_text:
        # TAIL-snapshot last resort (#214): truncate ``snap_text`` itself; the trimmed
        # value is returned to the caller (never written into a history message).
        snap_tok = estimate_tokens(snap_text, cpt)
        other_tok = request_tokens() - snap_tok
        allow_tok = in_budget - other_tok
        allow_chars = max(0, int(allow_tok * cpt)) if cpt > 0 else max(0, allow_tok)
        marker = "\n" + _HISTORY_TRUNC_MARKER.format(dropped=len(snap_text))
        body_budget = allow_chars - len(marker)
        if body_budget <= 0:
            snapshot_dropped_chars = len(snap_text)
            snapshot_kept_chars = 0
            snap_text = _SNAPSHOT_DROPPED_BODY
            snapshot_dropped = True
        else:
            body, dropped = truncate_snapshot(snap_text, body_budget)
            if dropped > 0:
                snapshot_dropped_chars = dropped
                snapshot_kept_chars = len(body)
                snap_text = body + marker
                snapshot_truncated = True
    elif not tail_mode:
        snap_idx = _latest_snapshot_index(chat_history)
        if request_tokens() > in_budget and snap_idx is not None:
            snap_msg = chat_history[snap_idx]
            content = snap_msg.get("content") or ""
            snap_tok = message_tokens(snap_msg, cpt)
            other_tok = request_tokens() - snap_tok           # everything but the snapshot
            allow_tok = in_budget - other_tok                 # tokens the snapshot may use
            allow_chars = max(0, int(allow_tok * cpt)) if cpt > 0 else max(0, allow_tok)
            # The truncation marker is appended to the kept body, so it MUST be reserved
            # against the char budget — otherwise the message re-tokenises to more than
            # ``allow_tok`` and the request creeps back over ``num_ctx``. We reserve the
            # marker at its longest (the full ``dropped`` count) so the bound always holds.
            marker = "\n" + _HISTORY_TRUNC_MARKER.format(dropped=len(content))
            body_budget = allow_chars - len(marker)
            if body_budget <= 0:
                # Nothing meaningful fits — drop the snapshot body but keep a LOUD
                # blind-this-turn note so the model (and the logs) know the page is
                # unavailable, not empty.
                snapshot_dropped_chars = len(content)
                snapshot_kept_chars = 0
                snap_msg["content"] = _SNAPSHOT_DROPPED_BODY
                snapshot_dropped = True
            else:
                body, dropped = truncate_snapshot(content, body_budget)
                if dropped > 0:
                    snapshot_dropped_chars = dropped
                    snapshot_kept_chars = len(body)
                    snap_msg["content"] = body + marker
                    snapshot_truncated = True

    return BudgetFitReport(
        over_budget=over,
        fits_after=request_tokens() <= in_budget,
        evicted_msgs=evicted_msgs,
        evicted_tokens=evicted_tokens,
        snapshot_truncated=snapshot_truncated,
        snapshot_dropped=snapshot_dropped,
        snapshot_raw_chars=snapshot_raw_chars,
        snapshot_kept_chars=snapshot_kept_chars,
        snapshot_dropped_chars=snapshot_dropped_chars,
        request_tokens_before=req_before,
        request_tokens_after=request_tokens(),
        input_budget_tokens=in_budget,
        num_ctx=eff_window,
        history_msgs_after=len(chat_history),
        history_tokens_after=hist_tokens(),
        tail_snapshot_out=snap_text if tail_mode else "",
    )
