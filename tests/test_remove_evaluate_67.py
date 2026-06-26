"""Issue #67: the `evaluate` (run-JS) web tool is removed entirely.

The limited 3B agent must accomplish web tasks using only the discrete web
subtools (goto/click/fill/type/press_key/select_option/hover/navigate_back/…).
It must NEVER be able to execute arbitrary JavaScript. Calling `evaluate` now
hits the standard invalid/unknown-tool ERROR path (issue #51) instead of running
JS — and `evaluate` must not appear anywhere it could be advertised: the web
toolset's tool list, the registry, the codec call-schema, or any agent-facing
description/guidance string.

All hermetic — no live browser, no Ollama. A fake CLI stands in for playwright-cli.
"""
from __future__ import annotations

import unittest

from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import Config
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.toolset import default_catalog
from vibeharness.web import _WEB_TOOL_CLASSES, WebToolset

from tests._fakes import FakeCli, FakeLLMClient, FakeValidator


# The discrete subtools that MUST remain after `evaluate` is removed.
# screenshot / navigate_back / navigate_forward were later trimmed from the
# registered set (see ``_WEB_TOOL_CLASSES``), so they are no longer expected.
EXPECTED_SUBTOOLS = {
    "goto", "click", "fill", "type", "press_key", "select_option", "check",
    "uncheck", "hover", "drag", "upload", "reload",
}


def _web_registry(config: Config | None = None) -> ToolRegistry:
    cfg = config or Config()
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(["web"]), cfg)


class EvaluateToolRemovedTest(unittest.TestCase):
    """The `evaluate`/JS tool is gone from the toolset, the registry, the codec
    call-schema, and the subtool class list — and no class is named `evaluate`."""

    def test_evaluate_not_in_toolset_tool_list(self):
        names = {t.name for t in WebToolset().create_tools(Config())}
        self.assertNotIn("evaluate", names)

    def test_evaluate_not_in_web_registry_names(self):
        self.assertNotIn("evaluate", _web_registry().names())

    def test_no_subtool_class_is_named_evaluate(self):
        names = {cls.name for cls in _WEB_TOOL_CLASSES}
        self.assertNotIn("evaluate", names)

    def test_evaluatetool_class_is_gone(self):
        # The class may survive as dead code (like OpenBrowserTool/SnapshotTool),
        # but it MUST be excluded from the registered web tool classes so the agent
        # can never reach it — i.e. no registered subtool is named `evaluate`.
        self.assertNotIn("evaluate", {cls.name for cls in _WEB_TOOL_CLASSES})

    def test_evaluate_not_in_codec_call_schema(self):
        # The json codec constrains decoding to a oneOf of the registry tools' call
        # schemas. `evaluate` must not appear as a permitted tool const.
        registry = _web_registry()
        codec = get_codec("json")
        constraint = codec.constraint(registry, max_actions=4)
        consts = {
            branch["properties"]["tool"]["const"]
            for branch in constraint.json_schema["items"]["oneOf"]
        }
        self.assertNotIn("evaluate", consts)
        # Sanity: the surviving discrete subtools DO appear as permitted consts.
        self.assertTrue(EXPECTED_SUBTOOLS.issubset(consts))


class EvaluateCallIsInvalidToolErrorTest(unittest.TestCase):
    """Calling `evaluate` must hit the invalid/unknown-tool ERROR path — it must
    NOT execute JavaScript and must NOT silently succeed."""

    def _agent(self, actions):
        client = FakeLLMClient(actions)
        cfg = Config(max_steps=5)
        return RalphAgent(client, _web_registry(), "sys", cfg, FakeValidator(passed=True))

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
        # The error names tools that DO exist so the model can recover.
        self.assertIn("goto", obs)
        # The loop kept going and finished via validate — no JS was executed.
        self.assertTrue(result.finished)

    def test_remaining_subtools_still_callable_via_registry(self):
        # Output doubles as the snapshot the issue-#73 target guard reads, so it must
        # list the refs (e1/e2) the targeted subtools are smoked with.
        cli = FakeCli(ok=True, output="### Page\n[e1] button\n[e2] button\nok")
        tools = [cls(cli, 1000) for cls in _WEB_TOOL_CLASSES]
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


class NoEvaluateAdvertisedTest(unittest.TestCase):
    """No agent-facing description/guidance/prompt string advertises a callable
    `evaluate` or invites the agent to run JavaScript."""

    def _web_system_prompt(self) -> str:
        toolsets = default_catalog().select(["web"])
        guidance = SystemPromptBuilder.assemble_guidance(toolsets)
        return SystemPromptBuilder(_web_registry(), guidance=guidance).build("task")

    def test_no_evaluate_tool_heading_in_prompt(self):
        sp = self._web_system_prompt()
        self.assertNotIn("### `evaluate`", sp)

    def test_no_evaluate_or_runjs_in_guidance_or_descriptions(self):
        sp = self._web_system_prompt().lower()
        for forbidden in ("`evaluate`", "evaluate tool", "call evaluate",
                          "use evaluate", "run js", "run javascript",
                          "execute javascript", "evaluate javascript"):
            self.assertNotIn(forbidden, sp,
                             f"agent-facing text advertises evaluate/JS: {forbidden!r}")

    def test_toolset_description_does_not_advertise_evaluate(self):
        desc = WebToolset.description.lower()
        self.assertNotIn("evaluate", desc)
        self.assertNotIn("run js", desc)
        self.assertNotIn("javascript", desc)

    def test_subtool_descriptions_do_not_mention_evaluate(self):
        for cls in _WEB_TOOL_CLASSES:
            desc = (cls.description or "").lower()
            self.assertNotIn("`evaluate`", desc)
            self.assertNotIn("evaluate tool", desc)


if __name__ == "__main__":
    unittest.main()
