import types
import unittest
from unittest import mock

from vibeharness.api_llm import (ApiLLMClient, ApiUnavailable, _strip_fences,
                                 _ApiFunction, _ApiToolCall)
from vibeharness.llm import Decision
from vibeharness.providers import ApiProviderConfig

PROVIDER = ApiProviderConfig("zhipuai", "https://api.z.ai/api/paas/v4/",
                             "ZHIPUAI_API_KEY", "glm-5.2")
SCHEMA = {"type": "array", "items": {"type": "object"}}


def _stream_chunk(text):
    delta = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice])


def _full_response(text):
    msg = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


class FakeCompletions:
    def __init__(self, *, stream_pieces=None, full_text=None, error=None):
        self.stream_pieces = stream_pieces
        self.full_text = full_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and kwargs.get("stream"):
            raise self.error
        if kwargs.get("stream"):
            return iter([_stream_chunk(p) for p in (self.stream_pieces or [])])
        return _full_response(self.full_text or "")


class FakeOpenAI:
    last_kwargs = None

    def __init__(self, completions):
        FakeOpenAI.last_kwargs = None
        self._completions = completions

    def __call__(self, **kwargs):
        FakeOpenAI.last_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=self._completions)
        return self


def _client(completions):
    fake = FakeOpenAI(completions)
    with mock.patch("openai.OpenAI", fake):
        return ApiLLMClient(provider=PROVIDER, api_key="secret", model="glm-5.2"), fake


class ApiLLMClientTest(unittest.TestCase):
    def test_constructor_passes_key_and_base_url_to_openai(self):
        _, fake = _client(FakeCompletions(stream_pieces=["[]"]))
        self.assertEqual(fake.last_kwargs["api_key"], "secret")
        self.assertEqual(fake.last_kwargs["base_url"], PROVIDER.base_url)

    def test_decide_streams_and_returns_decision(self):
        comp = FakeCompletions(stream_pieces=['[{"tool":', ' "click"}]'])
        client, _ = _client(comp)
        seen = []
        d = client.decide("SYS", "USER", SCHEMA, on_action=seen.append)
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.action_json, '[{"tool": "click"}]')
        self.assertEqual(d.reasoning, "")
        self.assertEqual(seen, ['[{"tool":', ' "click"}]'])   # streamed chunk-by-chunk

    def test_decide_injects_schema_instruction_into_user(self):
        comp = FakeCompletions(stream_pieces=["[]"])
        client, _ = _client(comp)
        client.decide("SYS", "ORIGINAL_USER", SCHEMA)
        sent = comp.calls[0]["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "SYS"})
        self.assertIn("ORIGINAL_USER", sent[1]["content"])
        self.assertIn("JSON Schema", sent[1]["content"])

    def test_decide_strips_markdown_fences(self):
        comp = FakeCompletions(stream_pieces=['```json\n[{"tool":"x"}]\n```'])
        client, _ = _client(comp)
        d = client.decide("S", "U", SCHEMA)
        self.assertEqual(d.action_json, '[{"tool":"x"}]')

    def test_streaming_unsupported_falls_back_to_non_streaming(self):
        err = RuntimeError("streaming is not supported for this model")
        comp = FakeCompletions(error=err, full_text='[{"tool":"y"}]')
        client, _ = _client(comp)
        d = client.decide("S", "U", SCHEMA)
        self.assertEqual(d.action_json, '[{"tool":"y"}]')
        self.assertEqual(len(comp.calls), 2)                  # stream attempt + fallback

    def test_generic_api_error_raises_api_unavailable(self):
        comp = FakeCompletions(error=RuntimeError("401 unauthorized"))
        client, _ = _client(comp)
        with self.assertRaises(ApiUnavailable) as ctx:
            client.decide("S", "U", SCHEMA)
        self.assertIn("zhipuai", str(ctx.exception))

    def test_decide_used_as_validator(self):
        comp = FakeCompletions(stream_pieces=['{"verdict":"pass","reason":"ok"}'])
        client, _ = _client(comp)
        d = client.decide("VALIDATOR_SYS", "history", {"type": "object"})
        self.assertEqual(d.action_json, '{"verdict":"pass","reason":"ok"}')


class AdapterShapeTest(unittest.TestCase):
    def test_tool_call_adapters_match_ollama_shape(self):
        tc = _ApiToolCall(function=_ApiFunction(name="click", arguments={"target": "e1"}))
        self.assertEqual(tc.function.name, "click")
        self.assertEqual(dict(tc.function.arguments), {"target": "e1"})


class StripFencesTest(unittest.TestCase):
    def test_plain_passthrough(self):
        self.assertEqual(_strip_fences('  [1,2] '), "[1,2]")

    def test_json_fence(self):
        self.assertEqual(_strip_fences("```json\n[1]\n```"), "[1]")

    def test_bare_fence(self):
        self.assertEqual(_strip_fences("```\n{}\n```"), "{}")


if __name__ == "__main__":
    unittest.main()
