"""Unit tests for the universal retry helper (issue #198). No network, no openai."""
import random
import unittest

from vibeharness.retry import (RetryError, RetryPolicy, TRANSIENT_STATUS,
                               compute_backoff, parse_retry_after, retry_request)


class _Status(Exception):
    """A toy HTTP-ish error carrying a status code (stands in for any client's error)."""
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


def _is_retryable(exc):
    return isinstance(exc, _Status) and exc.status in TRANSIENT_STATUS


# Tiny policy: zero waits so tests are instant; capture sleeps instead of sleeping.
FAST = RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0)


class RetryRequestTest(unittest.TestCase):
    def setUp(self):
        self.slept = []
        self.logs = []

    def _run(self, fn, **kw):
        return retry_request(
            fn, is_retryable=_is_retryable, status_of=lambda e: getattr(e, "status", None),
            policy=kw.pop("policy", FAST), sleep=self.slept.append,
            log=self.logs.append, **kw)

    def test_success_first_try_no_retry(self):
        calls = []
        out = self._run(lambda: calls.append(1) or "ok")
        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.slept, [])

    def test_retry_then_success_on_429(self):
        seq = [_Status(429), _Status(429), "done"]
        def fn():
            v = seq.pop(0)
            if isinstance(v, Exception):
                raise v
            return v
        out = self._run(fn)
        self.assertEqual(out, "done")
        self.assertEqual(len(self.slept), 2)        # two retries waited
        self.assertEqual(len(self.logs), 2)         # both logged visibly
        self.assertIn("HTTP 429", self.logs[0])

    def test_no_retry_on_4xx_client_error(self):
        calls = []
        def fn():
            calls.append(1)
            raise _Status(400)
        with self.assertRaises(_Status) as ctx:
            self._run(fn)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(calls), 1)             # surfaced immediately, no retry
        self.assertEqual(self.slept, [])

    def test_max_attempts_exhaustion_raises_retry_error(self):
        calls = []
        def fn():
            calls.append(1)
            raise _Status(503)
        with self.assertRaises(RetryError) as ctx:
            self._run(fn)
        self.assertEqual(len(calls), 10)            # max_attempts
        self.assertEqual(len(self.slept), 9)        # waited between the 10 tries
        self.assertIsInstance(ctx.exception.__cause__, _Status)
        self.assertIn("after 10 attempts", str(ctx.exception))

    def test_retry_after_header_is_honored(self):
        class _RA(_Status):
            def __init__(self):
                super().__init__(429)
        seq = [_RA(), "ok"]
        def fn():
            v = seq.pop(0)
            if isinstance(v, Exception):
                raise v
            return v
        # retry_after returns 12s; with max_delay >= 12 the wait must equal it exactly.
        self._run(fn, policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=30.0),
                  retry_after=lambda e: 12.0)
        self.assertEqual(self.slept, [12.0])
        self.assertIn("Retry-After honored", self.logs[0])

    def test_retry_after_clamped_to_max_delay(self):
        def fn():
            raise _Status(429)
        with self.assertRaises(RetryError):
            self._run(fn, policy=RetryPolicy(max_attempts=2, max_delay=5.0),
                      retry_after=lambda e: 999.0)
        self.assertEqual(self.slept, [5.0])         # clamped to cap


class BackoffTest(unittest.TestCase):
    def test_exponential_growth_capped(self):
        p = RetryPolicy(base_delay=1.0, max_delay=10.0, backoff_factor=2.0, jitter=0.0)
        # jitter 0 => deterministic: 1, 2, 4, 8, then capped at 10
        self.assertEqual([compute_backoff(p, a) for a in range(1, 7)],
                         [1.0, 2.0, 4.0, 8.0, 10.0, 10.0])

    def test_full_jitter_within_bounds(self):
        p = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=1.0)
        rng = random.Random(0)
        for a in range(1, 8):
            w = compute_backoff(p, a, rng=rng)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 10.0)

    def test_retry_after_overrides_backoff(self):
        p = RetryPolicy(base_delay=1.0, max_delay=10.0)
        self.assertEqual(compute_backoff(p, 5, retry_after=3.0), 3.0)


class ParseRetryAfterTest(unittest.TestCase):
    def test_delta_seconds(self):
        self.assertEqual(parse_retry_after("7"), 7.0)
        self.assertEqual(parse_retry_after(7), 7.0)

    def test_none_and_blank(self):
        self.assertIsNone(parse_retry_after(None))
        self.assertIsNone(parse_retry_after("  "))
        self.assertIsNone(parse_retry_after("garbage"))

    def test_http_date(self):
        # 60s in the future relative to a fixed "now".
        secs = parse_retry_after("Wed, 21 Oct 2099 07:29:00 GMT", now=0)
        self.assertIsNotNone(secs)
        self.assertGreater(secs, 0)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(parse_retry_after(-5), 0.0)


if __name__ == "__main__":
    unittest.main()
