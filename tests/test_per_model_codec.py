"""Per-MODEL tool-call codec + per-turn tool-call cap resolution (issues #179 + #178).

Covers, WITHOUT a live model:
  * the per-model policy registry (qwen3:4b → {hermes, 3}; GLM models → {json, cap});
  * resolve_model_codec / resolve_model_limit precedence (spec override → registry →
    Config fallback);
  * the CLI resolution (resolve_run_codec / resolve_max_actions) keyed off the BASE model,
    with explicit --codec / --max-actions-per-turn and saved settings still winning;
  * ApiLLMClient: thinking (reasoning_content) surfaced as reasoning SEPARATE from the
    constrained action, plus the empty-content JSON recovery (_recover_json).
"""
from __future__ import annotations

import os
import tempfile
import types
import unittest
from dataclasses import replace
from unittest import mock

from vibeharness import cli
from vibeharness.api_llm import ApiLLMClient, _recover_json
from vibeharness.config import (Config, ModelSpec, ModelToolPolicy, model_tool_policy,
                                resolve_model_codec, resolve_model_limit)
from vibeharness.providers import get_endpoint


# --------------------------------------------------------------------------- #
# Per-model policy registry.
# --------------------------------------------------------------------------- #
class ModelPolicyTest(unittest.TestCase):
    def test_qwen_is_hermes_3(self):
        p = model_tool_policy("qwen3:4b")
        self.assertEqual((p.codec, p.max_actions_per_turn), ("hermes", 3))

    def test_glm_flash_is_json_5(self):
        p = model_tool_policy("glm-4.7-flash")
        self.assertEqual((p.codec, p.max_actions_per_turn), ("json", 5))

    def test_glm_flagships_are_json_8(self):
        for m in ("glm-4.7", "glm-5.2"):
            p = model_tool_policy(m)
            self.assertEqual((p.codec, p.max_actions_per_turn), ("json", 8), m)

    def test_case_insensitive_exact_match(self):
        self.assertIs(model_tool_policy("QWEN3:4B"), model_tool_policy("qwen3:4b"))

    def test_unknown_glm_family_falls_back_to_json(self):
        # A future GLM variant still gets a schema-constrained codec (never native hermes).
        p = model_tool_policy("glm-4.7-air")
        self.assertEqual(p.codec, "json")

    def test_unknown_non_glm_returns_none(self):
        self.assertIsNone(model_tool_policy("llama3:8b"))
        self.assertIsNone(model_tool_policy(""))
        self.assertIsNone(model_tool_policy(None))


# --------------------------------------------------------------------------- #
# resolve_model_codec / resolve_model_limit precedence.
# --------------------------------------------------------------------------- #
class ResolveCodecLimitTest(unittest.TestCase):
    def test_spec_field_wins_over_registry(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7-flash",
                         codec="tagged_json", max_actions_per_turn=2)
        self.assertEqual(resolve_model_codec(Config(), spec), "tagged_json")
        self.assertEqual(resolve_model_limit(Config(), spec), 2)

    def test_registry_used_when_spec_unset(self):
        spec = ModelSpec(provider="zhipuai", model="glm-4.7")
        self.assertEqual(resolve_model_codec(Config(), spec), "json")
        self.assertEqual(resolve_model_limit(Config(), spec), 8)

    def test_config_fallback_for_unknown_model(self):
        spec = ModelSpec(provider="ollama", model="mystery:1b")
        cfg = replace(Config(), codec="xml", max_actions_per_turn=4)
        self.assertEqual(resolve_model_codec(cfg, spec), "xml")
        self.assertEqual(resolve_model_limit(cfg, spec), 4)

    def test_qwen_resolves_to_hermes_3(self):
        spec = ModelSpec(provider="ollama", model="qwen3:4b")
        self.assertEqual(resolve_model_codec(Config(), spec), "hermes")
        self.assertEqual(resolve_model_limit(Config(), spec), 3)


# --------------------------------------------------------------------------- #
# CLI resolution keyed off the BASE model.
# --------------------------------------------------------------------------- #
class CliResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("VIBEHARNESS_HOME")
        os.environ["VIBEHARNESS_HOME"] = self.tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VIBEHARNESS_HOME", None)
        else:
            os.environ["VIBEHARNESS_HOME"] = self._prev
        self.tmp.cleanup()

    def _cfg(self, *argv):
        return cli.resolve_config(cli.build_parser().parse_args(list(argv)))

    def test_default_qwen_base_is_hermes_3(self):
        cfg = self._cfg("task")
        self.assertEqual(cfg.codec, "hermes")
        self.assertEqual(cfg.max_actions_per_turn, 3)

    def test_glm_base_provider_switches_to_json_and_cap(self):
        cfg = self._cfg("task", "--base-provider", "zhipuai",
                        "--base-model", "glm-4.7-flash")
        self.assertEqual(cfg.codec, "json")
        self.assertEqual(cfg.max_actions_per_turn, 5)

    def test_glm_flagship_base_cap_is_8(self):
        cfg = self._cfg("task", "--base-provider", "zhipuai", "--base-model", "glm-4.7")
        self.assertEqual(cfg.codec, "json")
        self.assertEqual(cfg.max_actions_per_turn, 8)

    def test_explicit_codec_flag_wins_over_per_model(self):
        cfg = self._cfg("task", "--base-provider", "zhipuai",
                        "--base-model", "glm-4.7-flash", "--codec", "hermes")
        self.assertEqual(cfg.codec, "hermes")

    def test_explicit_max_actions_flag_wins_over_per_model(self):
        cfg = self._cfg("task", "--base-provider", "zhipuai",
                        "--base-model", "glm-4.7", "--max-actions-per-turn", "2")
        self.assertEqual(cfg.max_actions_per_turn, 2)

    def test_saved_codec_setting_wins_over_per_model(self):
        from vibeharness.settings import Settings
        Settings.set("codec", "tagged_json")
        cfg = self._cfg("task", "--base-provider", "zhipuai", "--base-model", "glm-4.7")
        self.assertEqual(cfg.codec, "tagged_json")

    def test_model_flag_drives_local_base_policy(self):
        # --model sets the local base model; an unknown local model keeps Config defaults.
        cfg = self._cfg("task", "--model", "qwen3:4b")
        self.assertEqual(cfg.codec, "hermes")
        self.assertEqual(cfg.max_actions_per_turn, 3)


# --------------------------------------------------------------------------- #
# ApiLLMClient: reasoning separate from the constrained action.
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


class ApiReasoningSeparationTest(unittest.TestCase):
    def test_thinking_is_reasoning_action_is_content(self):
        # The model streams thinking as reasoning_content and the JSON tool call as content.
        comp = _RecordingCompletions([
            _stream_chunk(reasoning="Let me think about the steps... "),
            _stream_chunk(content='[{"tool": "goto", "args": {"url": "x"}}]'),
        ])
        client = _api_client(comp)
        seen_reason, seen_action = [], []
        from vibeharness.codec import DecodeConstraint
        d = client.decide("S", "U", DecodeConstraint(json_schema={"type": "array"}),
                          on_reason=seen_reason.append, on_action=seen_action.append)
        # The action is the CONSTRAINED content only — thinking never reaches it.
        self.assertEqual(d.action_json, '[{"tool": "goto", "args": {"url": "x"}}]')
        self.assertIn("think about the steps", d.reasoning)
        self.assertNotIn("think about the steps", d.action_json)
        self.assertTrue(seen_reason)   # thinking streamed to on_reason
        self.assertTrue(seen_action)   # action streamed to on_action

    def test_empty_content_recovers_json_from_reasoning(self):
        # Verdict embedded (with prose) inside reasoning_content, content empty.
        comp = _RecordingCompletions([
            _stream_chunk(content="", reasoning='I conclude: {"verdict": "pass", '
                                                '"reason": "done"} that is final.'),
        ])
        client = _api_client(comp)
        from vibeharness.codec import DecodeConstraint
        d = client.decide("S", "U", DecodeConstraint(json_schema={"type": "object"}))
        self.assertEqual(d.action_json, '{"verdict": "pass", "reason": "done"}')

    def test_recover_json_object_and_array(self):
        self.assertEqual(_recover_json('noise {"a": 1} tail'), '{"a": 1}')
        self.assertEqual(_recover_json('x [1, 2, 3] y'), '[1, 2, 3]')
        self.assertEqual(_recover_json('brace in "str }" {"k": "v}"}'),
                         '{"k": "v}"}')

    def test_recover_json_returns_empty_for_pure_prose(self):
        # Free-form thinking with no JSON must NOT be treated as a tool call.
        self.assertEqual(_recover_json("Okay, let's tackle this task step by step."), "")
        self.assertEqual(_recover_json(""), "")


if __name__ == "__main__":
    unittest.main()
