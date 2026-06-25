"""Issue #51: discrete web subtools, full removal of the agent-facing `snapshot`
tool, an explicit invalid-tool error, and a build-identity readout.

All hermetic — no live browser, no Ollama. The web subtools are exercised through
the registry/codec exactly as the agent reaches them, with a fake CLI standing in
for playwright-cli.
"""
from __future__ import annotations

import unittest

from vibeharness.agent import RalphAgent
from vibeharness.cli import build_identity, build_parser, main
from vibeharness.codec import get_codec
from vibeharness.config import Config
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.toolset import default_catalog
from vibeharness.web import _WEB_TOOL_CLASSES, WebToolset

from tests._fakes import FakeCli, FakeLLMClient, FakeValidator


# The discrete subtools we expect the web toolset to expose (issue #51).
EXPECTED_SUBTOOLS = {
    "goto", "click", "fill", "type", "press_key", "select_option", "check",
    "uncheck", "hover", "drag", "upload", "screenshot",
    "navigate_back", "navigate_forward", "reload",
}


def _web_registry(config: Config | None = None) -> ToolRegistry:
    cfg = config or Config()
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(["web"]), cfg)


class SnapshotToolRemovedTest(unittest.TestCase):
    """B: the agent-facing `snapshot` tool is gone from the registry, the available
    tools, AND the codec call-schema."""

    def test_snapshot_not_in_web_registry_names(self):
        names = _web_registry().names()
        self.assertNotIn("snapshot", names)
        # And the old monolithic browse tool is gone too.
        self.assertNotIn("browse", names)

    def test_snapshot_not_in_toolset_tool_list(self):
        names = {t.name for t in WebToolset().create_tools(Config())}
        self.assertNotIn("snapshot", names)
        self.assertNotIn("browse", names)

    def test_snapshot_not_in_codec_call_schema(self):
        # The json codec constrains decoding to a oneOf of the registry tools' call
        # schemas. `snapshot` must not appear as a permitted tool const.
        registry = _web_registry()
        codec = get_codec("json")
        constraint = codec.constraint(registry, max_actions=4)
        schema = constraint.json_schema
        consts = {
            branch["properties"]["tool"]["const"]
            for branch in schema["items"]["oneOf"]
        }
        self.assertNotIn("snapshot", consts)
        self.assertNotIn("browse", consts)
        # Sanity: the discrete subtools DO appear as permitted tool consts.
        self.assertTrue(EXPECTED_SUBTOOLS.issubset(consts))

    def test_no_subtool_class_is_named_snapshot(self):
        names = {cls.name for cls in _WEB_TOOL_CLASSES}
        self.assertNotIn("snapshot", names)
        self.assertNotIn("browse", names)


class SubtoolsRegisteredAndCallableTest(unittest.TestCase):
    """A/E: every discrete subtool is registered and callable through the registry,
    smoke-tested against a fake CLI (no browser)."""

    def test_all_expected_subtools_registered(self):
        names = set(_web_registry().names())
        missing = EXPECTED_SUBTOOLS - names
        self.assertEqual(missing, set(), f"subtools missing from registry: {missing}")

    def test_each_subtool_is_callable_via_registry(self):
        # Build a registry whose web tools are backed by a FakeCli so .run() works
        # without a browser, then smoke each subtool with minimal valid args. The
        # output doubles as the snapshot the issue-#73 target guard reads, so it must
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
            "upload": {"file": "/tmp/f"},
            "screenshot": {},
            "navigate_back": {},
            "navigate_forward": {},
            "reload": {},
        }
        for name in EXPECTED_SUBTOOLS:
            tool = registry.get(name)
            self.assertIsNotNone(tool, f"{name} not in registry")
            res = tool.run(sample_args[name])
            self.assertTrue(res.ok, f"{name} did not run ok: {res.observation}")

    def test_subtool_docs_appear_in_system_prompt(self):
        sp = SystemPromptBuilder(_web_registry()).build("do it")
        for name in EXPECTED_SUBTOOLS:
            self.assertIn(f"### `{name}`", sp, f"{name} missing from system prompt docs")


class InvalidToolErrorTest(unittest.TestCase):
    """C/E: calling a removed/unknown tool (snapshot, or anything) returns an explicit
    ERROR action — never silently succeeds — and the loop continues."""

    def _agent(self, actions):
        client = FakeLLMClient(actions)
        cfg = Config(max_steps=5)
        return RalphAgent(client, _web_registry(), "sys", cfg, FakeValidator(passed=True))

    def test_snapshot_call_is_explicit_error(self):
        validate = {"tool": "validate", "args": {"summary": "done"}}
        result = self._agent([{"tool": "snapshot", "args": {}}, validate]).run("t")
        obs = result.turns[0].actions[0].observation
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("ERROR", obs)
        self.assertIn("snapshot", obs)
        self.assertIn("not a valid tool", obs)
        # The error names the tools that DO exist so the model can recover.
        self.assertIn("goto", obs)

    def test_arbitrary_unknown_tool_is_explicit_error(self):
        validate = {"tool": "validate", "args": {"summary": "done"}}
        result = self._agent([{"tool": "teleport", "args": {}}, validate]).run("t")
        obs = result.turns[0].actions[0].observation
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("ERROR", obs)
        self.assertIn("teleport", obs)
        self.assertIn("not a valid tool", obs)
        # ...and the loop kept going to validate+finish.
        self.assertTrue(result.finished)


class NoAgentFacingSnapshotStringTest(unittest.TestCase):
    """B/E: no agent-facing prompt/guidance/description string advertises a CALLABLE
    `snapshot`. The only permitted mentions describe the AUTO-INJECTED page section."""

    def _web_system_prompt(self) -> str:
        toolsets = default_catalog().select(["web"])
        guidance = SystemPromptBuilder.assemble_guidance(toolsets)
        return SystemPromptBuilder(_web_registry(), guidance=guidance).build("task")

    def test_no_snapshot_tool_heading(self):
        sp = self._web_system_prompt()
        # A callable tool is documented as "### `name`". snapshot/browse must not be.
        self.assertNotIn("### `snapshot`", sp)
        self.assertNotIn("### `browse`", sp)

    def test_no_guidance_tells_agent_to_take_a_snapshot(self):
        # Guidance/descriptions must not instruct the agent to call/request/take a
        # snapshot — the page is provided automatically.
        sp = self._web_system_prompt().lower()
        for forbidden in ("call snapshot", "take a snapshot", "use snapshot",
                          "request a snapshot", "snapshot action", "snapshot tool",
                          "`snapshot`"):
            self.assertNotIn(forbidden, sp, f"agent-facing text advertises snapshot: {forbidden!r}")

    def test_page_section_is_labelled_provided_automatically(self):
        # Issue #146: build() no longer renders the page section (the snapshot moved to
        # the user turn). The validator-context section renderer keeps the labelled
        # heading so the validator still sees the page is provided automatically.
        from vibeharness.prompt import render_page_section
        sp = SystemPromptBuilder(_web_registry()).build("task", page="### live page content")
        self.assertNotIn("# Current page (live snapshot", sp)
        section = render_page_section("### live page content")
        self.assertIn("# Current page (live snapshot — provided automatically)", section)

    def test_subtool_descriptions_do_not_advertise_snapshot_as_tool(self):
        # The discrete tools' descriptions may REFERENCE the live page snapshot
        # section (for refs), but must never present snapshot as a callable tool.
        for cls in _WEB_TOOL_CLASSES:
            desc = (cls.description or "").lower()
            self.assertNotIn("`snapshot`", desc)
            self.assertNotIn("snapshot tool", desc)
            self.assertNotIn("snapshot action", desc)


class BuildIdentityTest(unittest.TestCase):
    """D/E: `vibe --version` exposes the package version AND a build identity."""

    def test_build_identity_has_version_and_build(self):
        ident = build_identity()
        from vibeharness import __version__
        self.assertIn(__version__, ident)
        self.assertIn("build", ident)
        self.assertTrue(ident.startswith("vibe "))

    def test_version_flag_parses(self):
        args = build_parser().parse_args(["--version"])
        self.assertTrue(args.version)

    def test_version_command_prints_identity(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["--version"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        from vibeharness import __version__
        self.assertIn(__version__, out)
        self.assertIn("build", out)


if __name__ == "__main__":
    unittest.main()
