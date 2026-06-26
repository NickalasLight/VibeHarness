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
    return types.SimpleNamespace(choices=[choice], usage=None)


def _usage(prompt_tokens, completion_tokens):
    return types.SimpleNamespace(prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens)


def _openai_usage_chunk(prompt_tokens, completion_tokens):
    """OpenAI/z.ai shape: a FINAL chunk with empty ``choices`` carrying the usage object."""
    return types.SimpleNamespace(choices=[],
                                 usage=_usage(prompt_tokens, completion_tokens))


def _deepseek_final_chunk(text, prompt_tokens, completion_tokens):
    """DeepSeek shape: the LAST content chunk carries BOTH ``choices`` and ``usage``
    (verified live: choices_len=1 + CompletionUsage(...)). Issue #187."""
    delta = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice],
                                 usage=_usage(prompt_tokens, completion_tokens))


def _reasoning_chunk(text):
    delta = types.SimpleNamespace(content=None, reasoning_content=text)
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice], usage=None)


class FakeChunkCompletions:
    """Streams a caller-supplied list of raw chunk objects (so a test can inject a usage
    chunk of the exact provider shape)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._chunks)


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

    # ---- issue #207: structured multi-turn decide_messages ----
    def test_supports_structured_history(self):
        client, _ = _client(FakeCompletions(stream_pieces=["[]"]))
        self.assertTrue(client.supports_structured_history())

    def test_decide_messages_sends_full_history_and_instructs_last_user(self):
        comp = FakeCompletions(stream_pieces=['[{"tool":"validate","args":{}}]'])
        client, _ = _client(comp)
        history = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "TURN1"},
            {"role": "assistant", "content": '[{"tool":"click","args":{"target":"e1"}}]'},
            {"role": "user", "content": "<tool_response>\nclicked e1\n</tool_response>"},
            {"role": "user", "content": "TURN2"},
        ]
        d = client.decide_messages(history, SCHEMA)
        self.assertEqual(d.action_json, '[{"tool":"validate","args":{}}]')
        sent = comp.calls[0]["messages"]
        # the FULL multi-turn array is sent (system/user/assistant/user/user), not a
        # flattened [system, user] pair.
        self.assertEqual([m["role"] for m in sent],
                         ["system", "user", "assistant", "user", "user"])
        # the schema instruction is appended ONLY to the LAST user turn (the live turn)…
        self.assertIn("JSON Schema", sent[-1]["content"])
        self.assertIn("TURN2", sent[-1]["content"])
        # …and NOT to the earlier user turn (so stored history stays clean).
        self.assertNotIn("JSON Schema", sent[1]["content"])
        self.assertEqual(sent[1]["content"], "TURN1")

    def test_decide_messages_does_not_mutate_caller_history(self):
        comp = FakeCompletions(stream_pieces=["[]"])
        client, _ = _client(comp)
        history = [{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}]
        client.decide_messages(history, SCHEMA)
        # the caller's list/messages are untouched (schema appended on a copy).
        self.assertEqual(history[1]["content"], "U")

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


class UsageAccountingTest(unittest.TestCase):
    """Issue #187: the streamed usage chunk MUST be tallied so [API_USAGE] is nonzero for
    GLM and DeepSeek. The chunk shape differs by provider; both must be captured."""

    def test_openai_style_trailing_usage_chunk_tallied(self):
        # z.ai/OpenAI: usage rides a trailing chunk whose ``choices`` is empty.
        chunks = [_stream_chunk("[]"),
                  _openai_usage_chunk(prompt_tokens=120, completion_tokens=34)]
        client, _ = _client(FakeChunkCompletions(chunks))
        client.decide("S", "U", SCHEMA)
        self.assertEqual(client.tokens_in, 120)
        self.assertEqual(client.tokens_out, 34)
        self.assertGreater(client.usage_summary()["estimated_cost_usd"], 0)

    def test_deepseek_style_usage_on_final_content_chunk_tallied(self):
        # DeepSeek: usage rides the LAST content chunk (choices non-empty). The pre-#187
        # ``not chunk.choices`` guard dropped this, logging tokens_in=0.
        chunks = [_stream_chunk('[{"tool":'),
                  _deepseek_final_chunk(' "click"}]', prompt_tokens=11, completion_tokens=5)]
        comp = FakeChunkCompletions(chunks)
        client, _ = _client(comp)
        d = client.decide("S", "U", SCHEMA)
        # Content on the usage-bearing chunk is STILL processed (not dropped).
        self.assertEqual(d.action_json, '[{"tool": "click"}]')
        self.assertEqual(client.tokens_in, 11)
        self.assertEqual(client.tokens_out, 5)

    def test_reasoning_chunks_do_not_break_usage_tally(self):
        # Reasoning models stream reasoning_content; completion_tokens (incl. reasoning, per
        # the provider's usage object) is what we trust for tokens_out.
        chunks = [_reasoning_chunk("thinking..."),
                  _stream_chunk("[]"),
                  _openai_usage_chunk(prompt_tokens=200, completion_tokens=88)]
        client, _ = _client(FakeChunkCompletions(chunks))
        d = client.decide("S", "U", SCHEMA)
        self.assertEqual(d.reasoning, "thinking...")
        self.assertEqual(client.tokens_in, 200)
        self.assertEqual(client.tokens_out, 88)

    def test_usage_is_cumulative_and_counted_once_per_call(self):
        client, _ = _client(FakeChunkCompletions(
            [_stream_chunk("[]"), _openai_usage_chunk(10, 5)]))
        client.decide("S", "U", SCHEMA)
        # Second call reuses the same fake chunk list -> tallies again (cumulative).
        client._client.chat.completions = FakeChunkCompletions(
            [_stream_chunk("[]"), _openai_usage_chunk(7, 3)])
        client.decide("S", "U", SCHEMA)
        self.assertEqual(client.tokens_in, 17)
        self.assertEqual(client.tokens_out, 8)

    def test_non_streaming_fallback_tallies_usage(self):
        # The non-stream fallback path reads resp.usage (issue #187 parity).
        err = RuntimeError("streaming is not supported for this model")
        comp = FakeCompletions(error=err, full_text='[]')
        client, _ = _client(comp)
        # Attach a usage object to the non-stream response.
        orig_create = comp.create

        def create(**kwargs):
            if kwargs.get("stream"):
                return orig_create(**kwargs)
            resp = orig_create(**kwargs)
            resp.usage = _usage(40, 9)
            return resp
        comp.create = create
        client.decide("S", "U", SCHEMA)
        self.assertEqual(client.tokens_in, 40)
        self.assertEqual(client.tokens_out, 9)


class AdapterShapeTest(unittest.TestCase):
    def test_tool_call_adapters_match_ollama_shape(self):
        tc = _ApiToolCall(function=_ApiFunction(name="click", arguments={"target": "e1"}))
        self.assertEqual(tc.function.name, "click")
        self.assertEqual(dict(tc.function.arguments), {"target": "e1"})


class _RetryCompletions:
    """Fake completions whose ``create`` raises a queued sequence of errors, then streams
    ``final_pieces``. Used to drive the issue-#198 retry path with REAL openai exceptions."""

    def __init__(self, errors, final_pieces=("[]",)):
        self._errors = list(errors)
        self._final = list(final_pieces)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return iter([_stream_chunk(p) for p in self._final])


def _http_error(cls, status, headers=None):
    """Build a real openai ``APIStatusError`` subclass instance with a status + headers."""
    import httpx
    req = httpx.Request("POST", "https://api.z.ai/api/paas/v4/chat/completions")
    resp = httpx.Response(status, headers=headers or {}, request=req)
    return cls(f"HTTP {status}", response=resp, body=None)


# Zero-wait policy so retry tests are instant.
from vibeharness.retry import RetryError, RetryPolicy
_FAST = RetryPolicy(max_attempts=10, base_delay=0.0, max_delay=0.0)


def _retry_client(completions):
    fake = FakeOpenAI(completions)
    with mock.patch("openai.OpenAI", fake):
        c = ApiLLMClient(provider=PROVIDER, api_key="secret", model="glm-5.2",
                         retry_policy=_FAST)
    return c, fake


class BrowserUserAgentTest(unittest.TestCase):
    def test_default_browser_ua_header_on_client(self):
        from vibeharness.config import BROWSER_USER_AGENT
        _, fake = _client(FakeCompletions(stream_pieces=["[]"]))
        headers = fake.last_kwargs["default_headers"]
        self.assertEqual(headers["User-Agent"], BROWSER_USER_AGENT)
        self.assertTrue(headers["User-Agent"].startswith("Mozilla/5.0"))

    def test_custom_ua_override_forwarded(self):
        fake = FakeOpenAI(FakeCompletions(stream_pieces=["[]"]))
        with mock.patch("openai.OpenAI", fake):
            ApiLLMClient(provider=PROVIDER, api_key="k", model="m",
                         user_agent="Custom/1.0")
        self.assertEqual(fake.last_kwargs["default_headers"]["User-Agent"], "Custom/1.0")


class RetryIntegrationTest(unittest.TestCase):
    def test_retry_then_success_on_429(self):
        import openai
        err = _http_error(openai.RateLimitError, 429, {"retry-after": "0"})
        comp = _RetryCompletions([err, err], final_pieces=['[{"tool":"x"}]'])
        client, _ = _retry_client(comp)
        d = client.decide("S", "U", SCHEMA)
        self.assertEqual(d.action_json, '[{"tool":"x"}]')
        self.assertEqual(len(comp.calls), 3)        # 2 failures + 1 success

    def test_no_retry_on_4xx_client_error(self):
        import openai
        err = _http_error(openai.BadRequestError, 400)
        comp = _RetryCompletions([err])
        client, _ = _retry_client(comp)
        with self.assertRaises(ApiUnavailable):
            client.decide("S", "U", SCHEMA)
        self.assertEqual(len(comp.calls), 1)        # surfaced immediately, no retry

    def test_persistent_429_exhausts_and_raises_clear_error(self):
        import openai
        err = _http_error(openai.RateLimitError, 429, {"retry-after": "0"})
        comp = _RetryCompletions([err] * 12)
        client, _ = _retry_client(comp)
        with self.assertRaises(ApiUnavailable) as ctx:
            client.decide("S", "U", SCHEMA)
        self.assertEqual(len(comp.calls), 10)       # max_attempts
        self.assertIn("after 10 attempts", str(ctx.exception))

    def test_transport_error_is_retried(self):
        import httpx, openai
        req = httpx.Request("POST", "https://api.z.ai/")
        err = openai.APITimeoutError(request=req)
        comp = _RetryCompletions([err], final_pieces=["[]"])
        client, _ = _retry_client(comp)
        d = client.decide("S", "U", SCHEMA)
        self.assertEqual(d.action_json, "[]")
        self.assertEqual(len(comp.calls), 2)


class StripFencesTest(unittest.TestCase):
    def test_plain_passthrough(self):
        self.assertEqual(_strip_fences('  [1,2] '), "[1,2]")

    def test_json_fence(self):
        self.assertEqual(_strip_fences("```json\n[1]\n```"), "[1]")

    def test_bare_fence(self):
        self.assertEqual(_strip_fences("```\n{}\n```"), "{}")


if __name__ == "__main__":
    unittest.main()
