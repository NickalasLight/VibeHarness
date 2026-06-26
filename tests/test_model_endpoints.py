"""Per-role pluggable LLM endpoints + the repaired ApiLLMClient signature (issue #163).

Covers the new seam end to end WITHOUT a live model:
  * the unified provider registry (kinds, lookups, legacy view),
  * ModelSpec + resolve_role_spec legacy-key fallback and config.models override,
  * build_client dispatch per kind (model + sampling resolution),
  * LLMClient.supports_native_tools() capability + the codec auto-degrade,
  * ApiLLMClient.decide on the real DecodeConstraint interface (schema + stop) and the
    GLM reasoning_content fallback,
  * settings persistence of the nested models.<role>.<field> keys.
"""
from __future__ import annotations

import os
import types
import unittest
from dataclasses import replace
from unittest import mock

from vibeharness import providers
from vibeharness.api_llm import ApiLLMClient, _unpack_constraint
from vibeharness.clients import build_client, select_execution_codec
from vibeharness.codec import DecodeConstraint, get_codec
from vibeharness.config import Config, ModelSpec, resolve_role_spec
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llamacpp import LlamaCppClient
from vibeharness.llm import Decision, LLMClient, OllamaClient
from vibeharness.providers import (LLAMACPP, OLLAMA, OPENAI_COMPATIBLE, Provider,
                                   get_endpoint, is_local)
from vibeharness.registry import ToolRegistry
from vibeharness.settings import Settings


def _registry() -> ToolRegistry:
    return ToolRegistry(build_default_tools(FileSystem(), 1000))


# --------------------------------------------------------------------------- #
# Unified provider registry.
# --------------------------------------------------------------------------- #
class RegistryTest(unittest.TestCase):
    def test_builtin_kinds(self):
        self.assertEqual(get_endpoint("ollama").kind, OLLAMA)
        self.assertEqual(get_endpoint("llamacpp").kind, LLAMACPP)
        self.assertEqual(get_endpoint("zhipuai").kind, OPENAI_COMPATIBLE)

    def test_local_kinds_have_no_secret_coords(self):
        for name in ("ollama", "llamacpp"):
            p = get_endpoint(name)
            self.assertIsNone(p.base_url)
            self.assertIsNone(p.api_key_env)
            self.assertTrue(is_local(p))

    def test_api_kind_carries_key_env_not_secret(self):
        p = get_endpoint("zhipuai")
        self.assertFalse(is_local(p))
        self.assertEqual(p.api_key_env, "ZHIPUAI_API_KEY")
        self.assertTrue(p.base_url.startswith("https://"))
        for v in (p.name, p.base_url, p.api_key_env, p.model):
            self.assertNotIn("sk-", str(v).lower())

    def test_unknown_endpoint_lists_known_names(self):
        with self.assertRaises(KeyError) as ctx:
            get_endpoint("nope")
        self.assertIn("zhipuai", str(ctx.exception))

    def test_legacy_get_provider_view_unchanged(self):
        # The legacy API-provider surface keeps returning an ApiProviderConfig view.
        p = providers.get_provider("zhipuai")
        self.assertEqual((p.name, p.model, p.api_key_env), ("zhipuai", "glm-5.2", "ZHIPUAI_API_KEY"))

    def test_legacy_get_provider_rejects_local_kind(self):
        with self.assertRaises(KeyError):
            providers.get_provider("ollama")

    def test_add_provider_is_one_entry(self):
        # Open/Closed: a new registry entry is immediately resolvable, nothing else edited.
        extra = Provider("acme", OPENAI_COMPATIBLE, base_url="https://acme/v1/",
                         api_key_env="ACME_KEY", model="acme-1")
        with mock.patch.dict(providers.REGISTRY, {"acme": extra}, clear=False):
            self.assertEqual(get_endpoint("acme").model, "acme-1")


# --------------------------------------------------------------------------- #
# ModelSpec + resolve_role_spec.
# --------------------------------------------------------------------------- #
class ResolveRoleSpecTest(unittest.TestCase):
    def test_legacy_fallback_base(self):
        spec = resolve_role_spec(Config(), "base")
        self.assertEqual(spec.provider, "ollama")     # Config.backend
        self.assertEqual(spec.model, "qwen3:4b")       # Config.model

    def test_legacy_fallback_validator_and_escalation(self):
        cfg = Config()
        v = resolve_role_spec(cfg, "validator")
        self.assertEqual((v.provider, v.model), ("zhipuai", "glm-5.2"))
        # ISSUE #197: escalation default moved to deepseek/deepseek-v4-flash (zhipuai key
        # is account rate-limited → 429, so the old default silently no-oped).
        e = resolve_role_spec(cfg, "escalation")
        self.assertEqual((e.provider, e.model), ("deepseek", "deepseek-v4-flash"))

    def test_advisor_empty_model_falls_back_to_base_model(self):
        cfg = replace(Config(), advisor_model="")
        a = resolve_role_spec(cfg, "advisor")
        self.assertEqual(a.model, cfg.model)
        self.assertEqual(a.provider, "ollama")

    def test_explicit_models_entry_wins(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7-flash")
        cfg = replace(Config(), models={"base": spec})
        self.assertIs(resolve_role_spec(cfg, "base"), spec)
        # Other roles still fall back to legacy keys.
        self.assertEqual(resolve_role_spec(cfg, "validator").provider, "zhipuai")

    def test_unknown_role_raises(self):
        with self.assertRaises(KeyError):
            resolve_role_spec(Config(), "frobnicate")


# --------------------------------------------------------------------------- #
# build_client dispatch.
# --------------------------------------------------------------------------- #
class BuildClientTest(unittest.TestCase):
    def test_ollama_kind_builds_ollama_client_with_model_override(self):
        spec = ModelSpec(provider="ollama", model="some-model:1b")
        client = build_client(spec, Config())
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client._cfg.model, "some-model:1b")
        self.assertTrue(client.supports_native_tools())

    def test_ollama_sampling_override_applies_to_action_temperature(self):
        spec = ModelSpec(provider="ollama", model="m", temperature=0.1, top_p=0.5, top_k=7)
        client = build_client(spec, Config())
        self.assertEqual(client._cfg.action_temperature, 0.1)
        self.assertEqual(client._cfg.top_p, 0.5)
        self.assertEqual(client._cfg.top_k, 7)

    def test_llamacpp_kind_builds_llamacpp_client(self):
        spec = ModelSpec(provider="llamacpp", model="m")
        client = build_client(spec, Config())
        self.assertIsInstance(client, LlamaCppClient)
        self.assertFalse(client.supports_native_tools())

    def test_openai_compatible_routes_through_make_api_client(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7-flash")
        sentinel = object()
        with mock.patch.object(providers, "make_api_client", return_value=sentinel) as mk:
            client = build_client(spec, Config())
        self.assertIs(client, sentinel)
        mk.assert_called_once_with("zhipuai", "glm-4.7-flash")

    def test_openai_compatible_forwards_temperature_override(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7-flash", temperature=0.2)
        with mock.patch.object(providers, "make_api_client") as mk:
            build_client(spec, Config())
        mk.assert_called_once_with("zhipuai", "glm-4.7-flash", temperature=0.2)

    def test_openai_compatible_missing_key_raises(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7-flash")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                build_client(spec, Config())


# --------------------------------------------------------------------------- #
# Capability gating + codec auto-degrade.
# --------------------------------------------------------------------------- #
class _SingleShotClient(LLMClient):
    """A non-native client: implements only decide()."""
    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        return Decision("", "[]")


class _NativeFake(LLMClient):
    """A native-capable client: overrides decide_chat (so the base default reports True)."""
    def decide(self, system, user, constraint, on_reason=None, on_action=None):
        return Decision("", "[]")

    def decide_chat(self, messages, tools, constraint, on_reason=None, on_action=None):
        return Decision("", "[]")


class CapabilityTest(unittest.TestCase):
    def test_supports_native_tools_defaults(self):
        self.assertFalse(_SingleShotClient().supports_native_tools())
        self.assertTrue(_NativeFake().supports_native_tools())

    def test_native_kept_for_native_client_and_hermes(self):
        reg = _registry()
        codec = get_codec("hermes")
        out_codec, native, note = select_execution_codec(Config(), _NativeFake(), codec, reg)
        self.assertTrue(native)
        self.assertIs(out_codec, codec)
        self.assertIsNone(note)

    def test_api_client_degrades_hermes_to_json(self):
        reg = _registry()
        codec = get_codec("hermes")
        out_codec, native, note = select_execution_codec(Config(), _SingleShotClient(), codec, reg)
        self.assertFalse(native)
        self.assertEqual(out_codec.name, "json")
        self.assertIsNotNone(note)

    def test_api_client_keeps_already_constrained_json_codec(self):
        reg = _registry()
        codec = get_codec("json")
        out_codec, native, note = select_execution_codec(Config(), _SingleShotClient(), codec, reg)
        self.assertFalse(native)
        self.assertEqual(out_codec.name, "json")
        self.assertIsNone(note)


# --------------------------------------------------------------------------- #
# ApiLLMClient.decide on the real DecodeConstraint interface.
# --------------------------------------------------------------------------- #
def _stream_chunk(content=None, reasoning=None):
    delta = types.SimpleNamespace(content=content, reasoning_content=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class _RecordingCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._chunks)


class _FakeOpenAI:
    def __init__(self, completions):
        self._completions = completions

    def __call__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=self._completions)
        return self


def _api_client(completions):
    with mock.patch("openai.OpenAI", _FakeOpenAI(completions)):
        return ApiLLMClient(provider=get_endpoint("zhipuai"), api_key="secret",
                            model="glm-4.7-flash")


class ApiDecideSignatureTest(unittest.TestCase):
    def test_decode_constraint_schema_is_instructed(self):
        comp = _RecordingCompletions([_stream_chunk('{"verdict":"pass","reason":"ok"}')])
        client = _api_client(comp)
        schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
        d = client.decide("SYS", "USER", DecodeConstraint(json_schema=schema))
        self.assertEqual(d.action_json, '{"verdict":"pass","reason":"ok"}')
        user_msg = comp.calls[0]["messages"][1]["content"]
        self.assertIn("USER", user_msg)
        self.assertIn("JSON Schema", user_msg)
        self.assertIn("verdict", user_msg)

    def test_decode_constraint_stop_is_forwarded(self):
        comp = _RecordingCompletions([_stream_chunk("[]")])
        client = _api_client(comp)
        client.decide("S", "U", DecodeConstraint(json_schema={"type": "array"},
                                                 stop=("</think>", "STOP")))
        self.assertEqual(comp.calls[0]["stop"], ["</think>", "STOP"])

    def test_none_schema_omits_instruction_clause(self):
        comp = _RecordingCompletions([_stream_chunk("free text")])
        client = _api_client(comp)
        client.decide("S", "U", DecodeConstraint(json_schema=None))
        user_msg = comp.calls[0]["messages"][1]["content"]
        self.assertNotIn("JSON Schema", user_msg)
        self.assertEqual(user_msg, "U")
        self.assertNotIn("stop", comp.calls[0])   # no stop strings, no stop kwarg

    def test_reasoning_content_fallback_when_content_empty(self):
        # GLM reasoning model: the verdict arrives only in reasoning_content.
        comp = _RecordingCompletions([
            _stream_chunk(content="", reasoning='{"verdict":"pass",'),
            _stream_chunk(content="", reasoning='"reason":"done"}'),
        ])
        client = _api_client(comp)
        seen_reason = []
        d = client.decide("S", "U", DecodeConstraint(json_schema={"type": "object"}),
                          on_reason=seen_reason.append)
        self.assertEqual(d.action_json, '{"verdict":"pass","reason":"done"}')
        self.assertTrue(seen_reason)   # reasoning tokens streamed to on_reason

    def test_unpack_constraint_accepts_dict_and_none(self):
        self.assertEqual(_unpack_constraint({"type": "object"}), ({"type": "object"}, ()))
        self.assertEqual(_unpack_constraint(None), (None, ()))
        c = DecodeConstraint(json_schema={"a": 1}, stop=("x",))
        self.assertEqual(_unpack_constraint(c), ({"a": 1}, ("x",)))


# --------------------------------------------------------------------------- #
# Settings persistence of nested models.<role>.<field> keys.
# --------------------------------------------------------------------------- #
class NestedSettingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = mock.patch.dict(os.environ, {}, clear=False)
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("VIBEHARNESS_HOME")
        os.environ["VIBEHARNESS_HOME"] = self._dir.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VIBEHARNESS_HOME", None)
        else:
            os.environ["VIBEHARNESS_HOME"] = self._prev
        self._dir.cleanup()

    def test_set_and_apply_nested_spec(self):
        Settings.set("models.base.provider", "zhipuai")
        Settings.set("models.base.model", "glm-4.7-flash")
        Settings.set("models.base.temperature", "0.2")
        cfg = Settings.apply(Config())
        spec = cfg.models["base"]
        self.assertEqual((spec.provider, spec.model, spec.temperature),
                         ("zhipuai", "glm-4.7-flash", 0.2))

    def test_partial_spec_fills_from_legacy(self):
        # Only the provider is set: the model is filled from the role's legacy resolution.
        Settings.set("models.validator.provider", "ollama")
        cfg = Settings.apply(Config())
        spec = cfg.models["validator"]
        self.assertEqual(spec.provider, "ollama")
        self.assertEqual(spec.model, "glm-5.2")   # legacy validation_model

    def test_unknown_nested_field_raises(self):
        with self.assertRaises(KeyError):
            Settings.set("models.base.bogus", "x")


if __name__ == "__main__":
    unittest.main()
