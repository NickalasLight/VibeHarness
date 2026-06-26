"""Universal request-retry helper (issue #198).

A small, dependency-free retry wrapper for transient HTTP / transport failures. It is
deliberately DECOUPLED from any particular client (Dependency-Inversion): the caller
supplies the *classification* of an exception (``is_retryable`` and, optionally, how to
read a ``Retry-After`` header and a status code for logging), so the same helper can wrap
the OpenAI-compatible API path today and the Ollama path (or any other request site) later
without this module importing ``openai``/``requests``/``httpx``.

Policy (see :data:`TRANSIENT_STATUS`):
  * RETRY  — HTTP 408, 429, 500, 502, 503, 504, plus transport errors (connection
    resets, read timeouts). These are rate-limit / overload / outage / flaky-network
    conditions that typically clear on a retry.
  * SURFACE IMMEDIATELY — real client errors (400/401/403/404/409/422 …): retrying cannot
    help and would only hide a bug or an auth problem.

Backoff is exponential with full jitter and a slowly-increasing, per-wait-capped delay
(default cap 45 s), bounded to ``max_attempts`` (default 10). A provider-supplied
``Retry-After`` value, when present, takes precedence over the computed backoff (still
clamped to the cap) so we wait exactly as long as the provider asks. Every retry is logged
visibly (attempt #, status, wait) — a retried request is never silent.

References:
  * MDN HTTP 429 / Retry-After:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After
  * AWS "Exponential Backoff And Jitter" (full-jitter):
    https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
"""
from __future__ import annotations

import email.utils
import random
import sys
import time
from dataclasses import dataclass

# HTTP status codes that indicate a transient condition worth retrying.
#   408 Request Timeout, 429 Too Many Requests,
#   500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class RetryError(RuntimeError):
    """Raised when a request still fails after exhausting ``max_attempts`` retries.

    The underlying provider/transport exception is chained as ``__cause__`` so callers can
    inspect or re-wrap it (``raise ... from`` semantics)."""


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff configuration. Defaults give ~exponential waits capped at 45 s over 10
    attempts (worst-case total wait is bounded — the per-wait cap dominates)."""
    max_attempts: int = 10
    base_delay: float = 1.0       # seconds; first backoff is ~base_delay
    max_delay: float = 45.0       # per-wait cap (seconds)
    backoff_factor: float = 2.0   # geometric growth per attempt
    jitter: float = 1.0           # full jitter: wait ~ U(0, computed) when 1.0


DEFAULT_POLICY = RetryPolicy()


def parse_retry_after(value: "str | int | float | None",
                      *, now: "float | None" = None) -> "float | None":
    """Parse a ``Retry-After`` header into a non-negative number of seconds.

    Accepts the two RFC 9110 forms: a delta-seconds integer (e.g. ``"5"``) or an
    HTTP-date (e.g. ``"Wed, 21 Oct 2025 07:28:00 GMT"``). Returns ``None`` when the value
    is missing or unparseable, so the caller falls back to computed backoff."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    s = str(value).strip()
    if not s:
        return None
    try:
        return max(0.0, float(s))            # delta-seconds form
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(s)  # HTTP-date form
    except (ValueError, TypeError):
        return None
    if dt is None:
        return None
    target = dt.timestamp()
    base = time.time() if now is None else now
    return max(0.0, target - base)


def compute_backoff(policy: RetryPolicy, attempt: int,
                    retry_after: "float | None" = None,
                    *, rng: "random.Random | None" = None) -> float:
    """Return the wait (seconds) before ``attempt``'s retry.

    A provider ``retry_after`` wins (clamped to ``max_delay``); otherwise an exponential
    ``base_delay * backoff_factor**(attempt-1)`` is capped at ``max_delay`` and then
    full-jittered to spread out concurrent retriers. ``attempt`` is 1-based (1 = the wait
    after the first failure)."""
    if retry_after is not None:
        return min(float(retry_after), policy.max_delay)
    raw = policy.base_delay * (policy.backoff_factor ** max(0, attempt - 1))
    capped = min(raw, policy.max_delay)
    r = rng or random
    # Full jitter (jitter==1.0): U(0, capped). Partial jitter keeps a (1-jitter) floor.
    floor = capped * (1.0 - policy.jitter)
    return floor + (capped - floor) * r.random()


def _default_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def retry_request(fn,
                  *,
                  is_retryable,
                  retry_after=None,
                  status_of=None,
                  policy: RetryPolicy = DEFAULT_POLICY,
                  description: str = "request",
                  log=_default_log,
                  sleep=time.sleep,
                  rng: "random.Random | None" = None):
    """Call ``fn()`` with retries on transient failures.

    Parameters (all classification is injected — this module knows no client type):
      * ``fn`` — zero-arg callable performing the request; its return value is returned.
      * ``is_retryable(exc) -> bool`` — REQUIRED. ``True`` => transient (retry); ``False``
        => surface ``exc`` immediately (e.g. a 4xx client error).
      * ``retry_after(exc) -> float | None`` — optional; seconds the provider asked us to
        wait (already parsed). ``None`` => use computed backoff.
      * ``status_of(exc) -> int | None`` — optional; only used to enrich the retry log.
      * ``policy`` — :class:`RetryPolicy`.
      * ``description`` — human label for logs / the exhaustion error.
      * ``log`` / ``sleep`` / ``rng`` — injectable for testing.

    Returns ``fn()``'s result on success. Raises the original exception immediately when
    it is not retryable; raises :class:`RetryError` (chaining the last exception) once
    ``max_attempts`` is reached."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classification decides retry vs surface
            if not is_retryable(exc):
                raise
            if attempt >= policy.max_attempts:
                raise RetryError(
                    f"{description} failed after {attempt} attempts "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            ra = None
            if retry_after is not None:
                try:
                    ra = retry_after(exc)
                except Exception:  # never let header parsing break the retry loop
                    ra = None
            wait = compute_backoff(policy, attempt, ra, rng=rng)
            status = None
            if status_of is not None:
                try:
                    status = status_of(exc)
                except Exception:
                    status = None
            status_txt = f"HTTP {status}" if status is not None else type(exc).__name__
            ra_txt = " (Retry-After honored)" if ra is not None else ""
            log(
                f"[retry] {description}: {status_txt} on attempt "
                f"{attempt}/{policy.max_attempts}; retrying in {wait:.1f}s{ra_txt}"
            )
            sleep(wait)
