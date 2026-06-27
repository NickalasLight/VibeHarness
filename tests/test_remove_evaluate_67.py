"""Issue #67 (per-MODEL gating after #203): the `evaluate` (run-JS) web tool must NEVER
be reachable by the limited local 3B agent (qwen3:4b) — but IS exposed to the capable API
models (GLM / DeepSeek) so they can drive complex widgets (#203 payload B).

History: #67 originally removed `evaluate` ENTIRELY (it was excluded from
``_WEB_TOOL_CLASSES``). Issue #203 re-loads it into the web toolset and instead enforces
#67's guarantee PER-MODEL: ``evaluate`` is in the run-loaded registry, but qwen3:4b's
per-model toolset view (``config.MODEL_TOOL_POLICIES`` → ``ToolRegistry.filtered``) OMITS
it, so the small model's exposure is unchanged (it still hits the unknown-tool ERROR path
if it tries to call it), while GLM/DeepSeek keep it AND are prompt-guided to use it.

This test therefore asserts the GUARANTEE at the layer that now owns it — the qwen3:4b
per-model VIEW — plus the complementary fact that capable models DO get it.

All hermetic — no live browser, no Ollama. A fake CLI stands in for playwright-cli.
"""
from __future__ import annotations

import unittest

from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import Config, ModelSpec, resolve_role_spec
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.toolset import apply_model_toolset, default_catalog
from vibeharness.web import _WEB_TOOL_CLASSES, WebToolset

from tests._fakes import FakeCli, FakeLLMClient, FakeValidator


# The discrete subtools that MUST remain available to the limited 3B agent.
EXPECTED_SUBTOOLS = {
    "goto", "click", "fill", "type", "press_key", "select_option", "check",
    "uncheck", "hover", "drag", "upload", "reload",
}


def _full_web_registry(config: Config | None = None) -> ToolRegistry:
    """The run-loaded web registry (the FULL set — evaluate IS loaded after #203)."""
    cfg = config or Config()
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(["web"]), cfg)


def _qwen_web_registry(config: Config | None = None) -> ToolRegistry:
    """The qwen3:4b per-MODEL VIEW of the web registry — evaluate omitted (#203/#67)."""
    cfg = config or Config()           # default model is qwen3:4b on this branch
    base_spec = resolve_role_spec(cfg, "base")
    return apply_model_toolset(_full_web_registry(cfg), cfg, base_spec)


class EvaluateGatedFromQwenTest(unittest.TestCase):
    """`evaluate` must be absent from qwen3:4b's view, its codec call-schema, and the
    small-model prompt — even though it is loaded in the full run registry for capable models."""

    def test_evaluate_loaded_in_full_registry(self):
        # #203: evaluate is re-loaded so capable models can use it.
        self.assertIn("evaluate", _full_web_registry().names())
        self.assertIn("evaluate", {cls.name for cls in _WEB_TOOL_CLASSES})

    def test_evaluate_not_in_qwen_view(self):
        self.assertNotIn("evaluate", _qwen_web_registry().names())

    def test_qwen_view_keeps_discrete_subtools(self):
        names = set(_qwen_web_registry().names())
        self.assertTrue(EXPECTED_SUBTOOLS.issubset(names))

    def test_evaluate_not_in_qwen_codec_call_schema(self):
        # The json codec constrains decoding to a oneOf of the qwen-view tools' call schemas.
        registry = _qwen_web_registry()
        codec = get_codec("json")
        constraint = codec.constraint(registry, max_actions=4)
        consts = {
            branch["properties"]["tool"]["const"]
            for branch in constraint.json_schema["items"]["oneOf"]
        }
        self.assertNotIn("evaluate", consts)
        self.assertTrue(EXPECTED_SUBTOOLS.issubset(consts))


class EvaluateExposedToCapableModelsTest(unittest.TestCase):
    """The capable API models (GLM/DeepSeek) DO get evaluate in their view (#203 payload B)."""

    def test_glm_and_deepseek_views_include_evaluate(self):
        cfg = Config()
        full = _full_web_registry(cfg)
        for provider, model in (("zhipuai", "glm-4.7"), ("deepseek", "deepseek-v4-flash"),
                                ("deepseek", "deepseek-chat")):
            view = apply_model_toolset(full, cfg, ModelSpec(provider=provider, model=model))
            self.assertIn("evaluate", view.names(), f"{model} must see evaluate")


class EvaluateCallIsInvalidToolErrorForQwenTest(unittest.TestCase):
    """When qwen3:4b (its view omits evaluate) tries to call `evaluate`, it must hit the
    invalid/unknown-tool ERROR path — it must NOT execute JavaScript."""

    def _agent(self, actions):
        client = FakeLLMClient(actions)
        cfg = Config(max_steps=5)
        return RalphAgent(client, _qwen_web_registry(), "sys", cfg, FakeValidator(passed=True))

    def test_evaluate_call_returns_invalid_tool_error(self):
        validate = {"tool": "validate", "args": {"summary": "done"}}
        result = self._agent(
            [{"tool": "evaluate", "args": {"expression": "() => document.cookie"}}, validate]
        ).run("t")
        action = result.turns[0].actions[0]
        self.assertFalse(action.ok)
        obs = action.observation
        self.assertIn("ERROR", obs)
        self.assertIn("evaluate", obs)
        self.assertIn("not a valid tool", obs)
        self.assertIn("goto", obs)   # the error names tools that DO exist so it can recover
        self.assertTrue(result.finished)

    def test_remaining_subtools_still_callable_via_registry(self):
        cli = FakeCli(ok=True, output="### Page\n[e1] button\n[e2] button\nok")
        # Build from the qwen view's tools (evaluate excluded).
        tools = [cls(cli, 1000) for cls in _WEB_TOOL_CLASSES if cls.name in EXPECTED_SUBTOOLS]
        registry = ToolRegistry(tools)
        sample_args = {
            "goto": {"url": "https://example.com"},
            "click": {"target": "e1"},
            "fill": {"target": "e1", "text": "x"},
            "type": {"text": "x"},
            "press_key": {"key": "Enter"},
            "select_option": {"target": "e1", "value": "v"},
            "check": {"target": "e1"},
            "uncheck": {"target": "e1"},
            "hover": {"target": "e1"},
            "drag": {"target": "e1", "end": "e2"},
            "upload": {"target": "e1", "file": "/tmp/f"},
            "reload": {},
        }
        for name in EXPECTED_SUBTOOLS:
            tool = registry.get(name)
            self.assertIsNotNone(tool, f"{name} not in registry")
            res = tool.run(sample_args[name])
            self.assertTrue(res.ok, f"{name} did not run ok: {res.observation}")


class NoEvaluateAdvertisedToQwenTest(unittest.TestCase):
    """The small-model (qwen3:4b) system prompt must not advertise a callable `evaluate`
    or invite it to run JavaScript — that guidance is reserved for capable models (#203)."""

    def _qwen_system_prompt(self) -> str:
        cfg = Config()
        toolsets = default_catalog().select(["web"])
        base_spec = resolve_role_spec(cfg, "base")
        registry = _qwen_web_registry(cfg)
        guidance = SystemPromptBuilder.assemble_guidance(toolsets)
        return SystemPromptBuilder(registry, guidance=guidance).build("task")

    def test_no_evaluate_tool_heading_in_qwen_prompt(self):
        sp = self._qwen_system_prompt()
        self.assertNotIn("### `evaluate`", sp)

    def test_no_evaluate_or_runjs_in_qwen_guidance_or_descriptions(self):
        sp = self._qwen_system_prompt().lower()
        for forbidden in ("`evaluate`", "evaluate tool", "call evaluate",
                          "use evaluate", "run js", "run javascript",
                          "execute javascript", "evaluate javascript"):
            self.assertNotIn(forbidden, sp,
                             f"qwen prompt advertises evaluate/JS: {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
