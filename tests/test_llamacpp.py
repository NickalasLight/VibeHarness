import json
import unittest
import urllib.error
import urllib.request

from vibeharness.codec import DecodeConstraint
from vibeharness.config import Config
from vibeharness.llamacpp import LlamaCppClient, LlamaCppUnavailable


def sse(content="", stop=False, extra=None):
    """Encode one llama.cpp streamed server-sent event line."""
    obj = {"content": content, "stop": stop}
    if extra:
        obj.update(extra)
    return ("data: " + json.dumps(obj) + "\n").encode("utf-8")


class FakeResponse:
    """Stand-in for the urlopen() context manager: iterating yields raw lines."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


class FakeUrlopen:
    """Records each request body and replays one scripted FakeResponse per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bodies = []
        self._i = 0

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


class LlamaCppClientTest(unittest.TestCase):
    def setUp(self):
        self._orig_urlopen = urllib.request.urlopen
        self.cfg = Config(backend="llamacpp", llamacpp_url="http://test:8080")

    def tearDown(self):
        urllib.request.urlopen = self._orig_urlopen

    def _install(self, responses):
        fake = FakeUrlopen(responses)
        urllib.request.urlopen = fake
        return fake

    def test_decide_returns_phase2_content_as_action_json(self):
        phase1 = FakeResponse([sse("thinking "), sse("hard", stop=True)])
        phase2 = FakeResponse([sse('{"tool":'), sse(' "x"}'), sse("", stop=True)])
        self._install([phase1, phase2])

        reason_tokens, action_tokens = [], []
        constraint = DecodeConstraint(gbnf='root ::= "x"')
        decision = LlamaCppClient(self.cfg).decide(
            "SYS", "USER", constraint,
            on_reason=reason_tokens.append, on_action=action_tokens.append,
        )

        self.assertEqual(decision.action_json, '{"tool": "x"}')
        self.assertEqual(decision.reasoning, "thinking hard")
        self.assertEqual("".join(action_tokens), '{"tool": "x"}')
        self.assertEqual("".join(reason_tokens), "thinking hard")

    def test_grammar_forwarded_in_phase2_body(self):
        phase1 = FakeResponse([sse("r", stop=True)])
        phase2 = FakeResponse([sse("ok", stop=True)])
        fake = self._install([phase1, phase2])

        gbnf = 'root ::= "a" | "b"'
        LlamaCppClient(self.cfg).decide("S", "U", DecodeConstraint(gbnf=gbnf))

        # Two requests: phase 1 (reasoning), phase 2 (constrained action).
        self.assertEqual(len(fake.bodies), 2)
        p1, p2 = fake.bodies
        self.assertNotIn("grammar", p1)
        self.assertEqual(p1["stop"], ["</think>"])
        self.assertEqual(p2["grammar"], gbnf)
        self.assertNotIn("json_schema", p2)
        self.assertEqual(p2["stop"][0], "<|im_end|>")
        self.assertTrue(p2["stream"])
        self.assertTrue(p2["cache_prompt"])

    def test_json_schema_forwarded_when_only_schema_set(self):
        phase1 = FakeResponse([sse("r", stop=True)])
        phase2 = FakeResponse([sse("ok", stop=True)])
        fake = self._install([phase1, phase2])

        schema = {"type": "object"}
        LlamaCppClient(self.cfg).decide("S", "U", DecodeConstraint(json_schema=schema))

        p2 = fake.bodies[1]
        self.assertEqual(p2["json_schema"], schema)
        self.assertNotIn("grammar", p2)

    def test_extra_stops_appended(self):
        phase1 = FakeResponse([sse("r", stop=True)])
        phase2 = FakeResponse([sse("ok", stop=True)])
        fake = self._install([phase1, phase2])

        LlamaCppClient(self.cfg).decide(
            "S", "U", DecodeConstraint(stop=("###",)))

        self.assertEqual(fake.bodies[1]["stop"], ["<|im_end|>", "###"])

    def test_unreachable_server_raises_llamacpp_unavailable(self):
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        urllib.request.urlopen = boom
        with self.assertRaises(LlamaCppUnavailable) as ctx:
            LlamaCppClient(self.cfg).decide("S", "U", DecodeConstraint())
        self.assertIn("llama-server", str(ctx.exception))
        self.assertIn("http://test:8080", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
