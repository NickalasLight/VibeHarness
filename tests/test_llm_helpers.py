import json
import os
import unittest
import urllib.request

from vibeharness.codec import DecodeConstraint
from vibeharness.config import Config
from vibeharness.llm import OllamaClient, ensure_single_runner_env


class LLMHelperTest(unittest.TestCase):
    def test_render_chatml_structure(self):
        prompt = OllamaClient._render_chatml("SYS", "USER")
        self.assertIn("<|im_start|>system\nSYS<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nUSER<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_continue_closes_open_think(self):
        out = OllamaClient._continue_after_reasoning("<think>reasoning so far")
        self.assertEqual(out, "<think>reasoning so far</think>\n")

    def test_continue_empty_reasoning(self):
        self.assertEqual(OllamaClient._continue_after_reasoning("   "), "")

    def test_continue_already_closed(self):
        out = OllamaClient._continue_after_reasoning("<think>x</think>answer")
        self.assertEqual(out, "<think>x</think>answer\n")


# ---- ISSUE #77: single-runner request shape ----

class _FakeResponse:
    """urlopen() context manager: iterating yields scripted NDJSON byte lines."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


class _RecordingUrlopen:
    """Records every request body and replays one scripted _FakeResponse per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bodies = []
        self._i = 0

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


def _line(obj):
    return (json.dumps(obj) + "\n").encode("utf-8")


class NumCtxPinnedTest(unittest.TestCase):
    def test_default_num_ctx_is_pinned_to_a_gpu_fitting_value(self):
        # #77: pinned to one value that actually fits the 8 GB GPU (was 131072,
        # which the GPU never delivered -> varying auto-fit sizes -> stacked runners).
        self.assertEqual(Config().num_ctx, 32768)

    def test_default_keep_alive_is_constant_string(self):
        self.assertEqual(Config().ollama_keep_alive, "30m")


class RequestShapeTest(unittest.TestCase):
    """Every Ollama request must carry the SAME num_ctx and keep_alive so only one
    (model, context-size) runner is ever requested. Asserted on the recorded payloads
    a full decide() builds — no live server."""

    def setUp(self):
        self._orig = urllib.request.urlopen
        self.cfg = Config(ollama_url="http://test:11434")

    def tearDown(self):
        urllib.request.urlopen = self._orig

    def _install(self, responses):
        fake = _RecordingUrlopen(responses)
        urllib.request.urlopen = fake
        return fake

    def test_decide_sends_num_ctx_and_keep_alive_on_every_request(self):
        # TWO-phase path (VibeThinker / mythos): phase 1 (/api/chat) then phase 2
        # (/api/generate) -> two recorded bodies, each carrying the runner-shape options.
        cfg = Config(ollama_url="http://test:11434", two_phase=True)
        phase1 = _FakeResponse([_line({"message": {"content": "think"}}),
                                _line({"done": True})])
        phase2 = _FakeResponse([_line({"response": '{"tool":"x"}'}),
                                _line({"done": True})])
        fake = self._install([phase1, phase2])

        OllamaClient(cfg).decide("SYS", "USER", DecodeConstraint())

        self.assertEqual(len(fake.bodies), 2)  # both phases issued a request
        for body in fake.bodies:
            self.assertEqual(body["keep_alive"], "30m",
                             msg="keep_alive missing/wrong on a request")
            self.assertEqual(body["options"]["num_ctx"], 32768,
                             msg="num_ctx missing/wrong on a request")

    def test_decide_single_phase_sends_one_chat_request(self):
        # SINGLE-phase path (#125, qwen3coder default two_phase=False): decide() issues
        # exactly ONE /api/chat request, still stamped with num_ctx + keep_alive.
        cfg = Config(ollama_url="http://test:11434")  # two_phase defaults to False
        self.assertFalse(cfg.two_phase)
        chat = _FakeResponse([_line({"message": {"content": '{"name":"x","arguments":{}}'}}),
                              _line({"done": True})])
        fake = self._install([chat])

        d = OllamaClient(cfg).decide("SYS", "USER", DecodeConstraint())

        self.assertEqual(len(fake.bodies), 1)
        self.assertEqual(d.reasoning, "")  # no separate reasoning pass
        self.assertEqual(fake.bodies[0]["keep_alive"], "30m")
        self.assertEqual(fake.bodies[0]["options"]["num_ctx"], 32768)

    def test_generate_sends_num_ctx_and_keep_alive(self):
        resp = _FakeResponse([_line({"response": "hi"}), _line({"done": True})])
        fake = self._install([resp])
        OllamaClient(self.cfg).generate("prompt")
        self.assertEqual(len(fake.bodies), 1)
        self.assertEqual(fake.bodies[0]["keep_alive"], "30m")
        self.assertEqual(fake.bodies[0]["options"]["num_ctx"], 32768)

    def test_keep_alive_follows_config_value(self):
        from dataclasses import replace
        cfg = replace(self.cfg, ollama_keep_alive="5m", num_ctx=16384)
        resp = _FakeResponse([_line({"response": "hi"}), _line({"done": True})])
        fake = self._install([resp])
        OllamaClient(cfg).generate("prompt")
        self.assertEqual(fake.bodies[0]["keep_alive"], "5m")
        self.assertEqual(fake.bodies[0]["options"]["num_ctx"], 16384)


class SingleRunnerEnvTest(unittest.TestCase):
    """OLLAMA_MAX_LOADED_MODELS must be set to "1" on startup, WITHOUT clobbering a
    value the user already exported (setdefault semantics)."""

    def setUp(self):
        self._key = "OLLAMA_MAX_LOADED_MODELS"
        self._saved = os.environ.get(self._key)
        os.environ.pop(self._key, None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._saved

    def test_sets_one_when_unset(self):
        ensure_single_runner_env()
        self.assertEqual(os.environ[self._key], "1")

    def test_does_not_override_user_set_value(self):
        os.environ[self._key] = "3"
        ensure_single_runner_env()
        self.assertEqual(os.environ[self._key], "3")

    def test_constructing_client_sets_the_env(self):
        OllamaClient(Config())
        self.assertEqual(os.environ[self._key], "1")


if __name__ == "__main__":
    unittest.main()
