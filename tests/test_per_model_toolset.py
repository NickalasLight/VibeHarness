"""Per-MODEL toolsets + per-MODEL system-prompt augmentation (issue #203).

Each model (and per-role spec) can declare, via the existing per-model policy infra:
  * ``tool_omit``  — a DENYLIST of tools removed from the model's view,
  * ``tool_allow`` — an optional ALLOWLIST (intersected with the run's loaded toolset),
  * ``system_prompt_augmentation`` — capability guidance appended to the system prompt.

Resolved by ``config.resolve_model_tool_omit / _tool_allow / _prompt_augmentation``
(spec → MODEL_TOOL_POLICIES → empty), composed onto the run-loaded registry by
``ToolRegistry.filtered`` (SUBTRACTION only → a model can never invoke a tool the run did
not load), and re-resolved on escalation take-over (``RalphAgent._escalate``).

First payloads:
  * qwen3:4b OMITS navigate_back / navigate_forward / evaluate (lean, safe; #67 preserved).
  * GLM + DeepSeek keep evaluate AND get the evaluate-preference prompt guidance.

All hermetic — no live browser, no Ollama.
"""
from __future__ import annotations

import unittest
from unittest import mock

from vibeharness import providers
from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import (Config, ModelSpec, ModelToolPolicy,
                                resolve_model_prompt_augmentation,
                                resolve_model_tool_allow, resolve_model_tool_omit)
from vibeharness.registry import ToolRegistry
from vibeharness.tools import Tool, ToolResult
from vibeharness.toolset import apply_model_toolset, default_catalog

from tests._fakes import FakeLLMClient, FakeValidator


# --------------------------------------------------------------------------- #
# Tool doubles for the registry-filter / composition tests.
# --------------------------------------------------------------------------- #
class _Stub(Tool):
    def __init__(self, name):
        self._n = name

    @property
    def name(self):           # noqa: D401
        return self._n

    description = "stub"

    @property
    def parameters(self):
        return []

    def run(self, args) -> ToolResult:
        return ToolResult(True, "ok")


def _reg(*names) -> ToolRegistry:
    return ToolRegistry([_Stub(n) for n in names])


# --------------------------------------------------------------------------- #
# 1) Resolvers
# --------------------------------------------------------------------------- #
class ResolverTest(unittest.TestCase):
    def _spec(self, model, **kw):
        return ModelSpec(provider="x", model=model, **kw)

    def test_qwen_omits_nav_and_evaluate(self):
        omit = resolve_model_tool_omit(Config(), self._spec("qwen3:4b"))
        self.assertEqual(omit, frozenset({"navigate_back", "navigate_forward", "evaluate"}))

    def test_qwen_has_no_augmentation(self):
        self.assertEqual(resolve_model_prompt_augmentation(Config(), self._spec("qwen3:4b")), "")

    def test_capable_models_omit_nothing(self):
        for m in ("glm-4.7", "glm-4.7-flash", "glm-5.2", "deepseek-chat",
                  "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"):
            self.assertEqual(resolve_model_tool_omit(Config(), self._spec(m)), frozenset(),
                             f"{m} must not omit any tool")

    def test_capable_models_get_evaluate_guidance(self):
        for m in ("glm-4.7", "deepseek-chat", "deepseek-v4-flash",
                  "glm-9-future", "deepseek-v9"):   # incl. family fallbacks
            aug = resolve_model_prompt_augmentation(Config(), self._spec(m))
            self.assertIn("evaluate", aug.lower(), f"{m} must be guided to use evaluate")

    def test_unknown_model_is_a_noop(self):
        spec = self._spec("mystery-1b")
        self.assertEqual(resolve_model_tool_omit(Config(), spec), frozenset())
        self.assertIsNone(resolve_model_tool_allow(Config(), spec))
        self.assertEqual(resolve_model_prompt_augmentation(Config(), spec), "")

    def test_spec_override_wins(self):
        # An explicit spec field beats the registry, both for omit and augmentation.
        spec = self._spec("qwen3:4b", tool_omit=frozenset(),
                          system_prompt_augmentation="custom note")
        self.assertEqual(resolve_model_tool_omit(Config(), spec), frozenset())
        self.assertEqual(resolve_model_prompt_augmentation(Config(), spec), "custom note")

    def test_policy_field_defaults_are_noop(self):
        p = ModelToolPolicy(codec="json", max_actions_per_turn=1)
        self.assertEqual(p.tool_omit, frozenset())
        self.assertIsNone(p.tool_allow)
        self.assertEqual(p.system_prompt_augmentation, "")


# --------------------------------------------------------------------------- #
# 2) ToolRegistry.filtered — composition + safety invariant
# --------------------------------------------------------------------------- #
class RegistryFilterTest(unittest.TestCase):
    def test_omit_removes(self):
        reg = _reg("validate", "a", "b", "c").filtered(omit=frozenset({"b"}))
        self.assertEqual(set(reg.names()), {"validate", "a", "c"})

    def test_allow_intersects_loaded_only(self):
        # An allowlist naming a tool the run did NOT load adds nothing (safety invariant).
        reg = _reg("validate", "a", "b").filtered(allow=frozenset({"a", "ghost"}))
        self.assertEqual(set(reg.names()), {"validate", "a"})
        self.assertNotIn("ghost", reg.names())

    def test_result_is_always_a_subset(self):
        loaded = {"validate", "a", "b", "c"}
        reg = _reg(*loaded).filtered(allow=frozenset({"a", "b", "x", "y"}),
                                     omit=frozenset({"b"}))
        self.assertTrue(set(reg.names()).issubset(loaded))
        self.assertEqual(set(reg.names()), {"validate", "a"})

    def test_validate_always_kept(self):
        # validate is core and survives even an omit naming it / a narrow allowlist.
        self.assertIn("validate", _reg("validate", "a").filtered(omit=frozenset({"validate"})).names())
        self.assertIn("validate", _reg("validate", "a").filtered(allow=frozenset({"a"})).names())

    def test_default_is_identity(self):
        reg = _reg("validate", "a", "b").filtered()
        self.assertEqual(set(reg.names()), {"validate", "a", "b"})


# --------------------------------------------------------------------------- #
# 3) Per-model exposure through the real web toolset
# --------------------------------------------------------------------------- #
class WebExposureTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        cat = default_catalog()
        self.full = cat.build_registry(cat.select(["web"]), self.cfg)

    def _view(self, provider, model):
        return apply_model_toolset(self.full, self.cfg,
                                   ModelSpec(provider=provider, model=model))

    def test_qwen_view_lacks_evaluate(self):
        # nav tools are not loaded yet (#206 owns that); evaluate IS loaded and gated off.
        view = self._view("ollama", "qwen3:4b")
        self.assertNotIn("evaluate", view.names())
        self.assertNotIn("navigate_back", view.names())

    def test_capable_views_keep_evaluate(self):
        for provider, model in (("zhipuai", "glm-4.7"), ("deepseek", "deepseek-v4-flash")):
            self.assertIn("evaluate", self._view(provider, model).names())

    def test_qwen_native_tool_schema_excludes_evaluate(self):
        view = apply_model_toolset(self.full, self.cfg,
                                   ModelSpec(provider="ollama", model="qwen3:4b"))
        tools = get_codec("hermes").tools(view)
        self.assertNotIn("evaluate", str(tools))


# --------------------------------------------------------------------------- #
# 4) Prompt augmentation present for capable models, absent for qwen
# --------------------------------------------------------------------------- #
class PromptAugmentationTest(unittest.TestCase):
    def _prompt(self, provider, model, codec_name):
        from vibeharness.cli import _augmented_guidance
        from vibeharness.prompt import SystemPromptBuilder
        cfg = Config()
        cat = default_catalog()
        toolsets = cat.select(["web"])
        spec = ModelSpec(provider=provider, model=model)
        registry = apply_model_toolset(cat.build_registry(toolsets, cfg), cfg, spec)
        guidance = _augmented_guidance(toolsets, cfg, spec)
        native = codec_name == "hermes"
        return SystemPromptBuilder(registry, cfg.max_actions_per_turn,
                                   get_codec(codec_name),
                                   guidance=guidance).build("task", native_tools=native)

    def test_deepseek_prompt_has_evaluate_guidance(self):
        sp = self._prompt("deepseek", "deepseek-v4-flash", "json").lower()
        self.assertIn("evaluate", sp)
        self.assertIn("date picker", sp)

    def test_glm_prompt_has_evaluate_guidance(self):
        self.assertIn("evaluate", self._prompt("zhipuai", "glm-4.7", "json").lower())

    def test_qwen_prompt_unchanged_no_augmentation(self):
        # qwen's prompt must NOT carry the capable-model evaluate guidance.
        sp = self._prompt("ollama", "qwen3:4b", "hermes").lower()
        self.assertNotIn("you are a capable model", sp)
        self.assertNotIn("prefer `evaluate", sp)


# --------------------------------------------------------------------------- #
# 5) Escalation take-over re-resolves the toolset + prompt for the new model
# --------------------------------------------------------------------------- #
class EscalationTest(unittest.TestCase):
    def test_takeover_restores_evaluate_for_deepseek(self):
        # Base qwen3:4b (evaluate omitted) gets stuck and escalates to DeepSeek (evaluate
        # kept). The agent must re-derive its ACTIVE registry from the FULL run-loaded set so
        # the escalator regains evaluate.
        cat = default_catalog()
        cfg = Config(model="qwen3:4b", max_steps=8, escalation_enabled=True,
                     escalation_stuck_threshold=3, escalation_provider="deepseek",
                     escalation_model="deepseek-v4-flash", validation_provider="")
        full = cat.build_registry(cat.select(["web"]), cfg)
        from vibeharness.config import resolve_role_spec
        base_view = apply_model_toolset(full, cfg, resolve_role_spec(cfg, "base"))
        self.assertNotIn("evaluate", base_view.names())

        base_client = FakeLLMClient([{"tool": "noop_missing", "args": {}}])
        swapped = FakeLLMClient([{"tool": "validate", "args": {}}])
        agent = RalphAgent(base_client, base_view, "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("json"),
                           full_registry=full)
        self.assertNotIn("evaluate", agent._registry.names())   # qwen view: no evaluate
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            agent.run("go")
        self.assertIs(agent._client, swapped)                   # take-over happened
        self.assertIn("evaluate", agent._registry.names())      # #203: evaluate restored


if __name__ == "__main__":
    unittest.main()
