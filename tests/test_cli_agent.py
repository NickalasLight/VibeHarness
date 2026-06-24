"""CLI tests for --agent / --list-agents (issue #22).

An "agent type" is a named default toolset selection: the agent's prompt is derived
from the active toolsets' system_guidance (#19), so these tests assert (a) the right
toolset(s) become active and (b) the built system prompt carries that toolset's
guidance. No model / Ollama is required — we parse args, call helpers, or call main
on a path that exits before any model work.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from vibeharness import cli
from vibeharness.config import Config
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.toolset import agent_default_toolsets, default_catalog


def _system_prompt_for(names: list[str]) -> str:
    """Build the system prompt for the given active toolset names (no model)."""
    catalog = default_catalog()
    toolsets = catalog.select(names)
    return SystemPromptBuilder(
        catalog.build_registry(toolsets, Config()),
        guidance=SystemPromptBuilder.assemble_guidance(toolsets)).build("a task")


class AgentDefaultToolsetMappingTest(unittest.TestCase):
    def test_each_agent_maps_to_same_named_toolset(self):
        mapping = agent_default_toolsets()
        self.assertEqual(mapping["web"], ["web"])
        self.assertEqual(mapping["fs"], ["fs"])

    def test_mapping_stays_in_lockstep_with_catalog(self):
        self.assertEqual(set(agent_default_toolsets()), set(default_catalog().names()))


class AgentResolutionTest(unittest.TestCase):
    """--toolset wins; else --agent's default; else today's default (fs)."""

    def test_agent_web_selects_web_toolset(self):
        args = cli.build_parser().parse_args(["task", "--agent", "web"])
        self.assertEqual(cli.selected_toolset_names(args), ["web"])

    def test_agent_fs_selects_fs_toolset(self):
        args = cli.build_parser().parse_args(["task", "--agent", "fs"])
        self.assertEqual(cli.selected_toolset_names(args), ["fs"])

    def test_toolset_overrides_and_augments_agent(self):
        args = cli.build_parser().parse_args(
            ["task", "--agent", "web", "--toolset", "web,fs"])
        self.assertEqual(cli.selected_toolset_names(args), ["web", "fs"])

    def test_no_agent_keeps_default(self):
        args = cli.build_parser().parse_args(["task"])
        self.assertEqual(cli.selected_toolset_names(args), ["fs"])


class AgentPromptGuidanceTest(unittest.TestCase):
    """The active toolset's guidance must reach the built system prompt (#19)."""

    def test_agent_web_prompt_has_web_worker_guidance(self):
        args = cli.build_parser().parse_args(["task", "--agent", "web"])
        prompt = _system_prompt_for(cli.selected_toolset_names(args))
        # web-worker guidance: the live-snapshot + consent-banner wording.
        self.assertIn("live snapshot", prompt)
        self.assertIn("consent banner", prompt)

    def test_agent_fs_prompt_has_fs_guidance(self):
        args = cli.build_parser().parse_args(["task", "--agent", "fs"])
        prompt = _system_prompt_for(cli.selected_toolset_names(args))
        self.assertIn("create_file", prompt)
        # No web guidance should be present for a pure fs agent.
        self.assertNotIn("live snapshot", prompt)

    def test_agent_web_with_both_toolsets_has_both_guidances(self):
        args = cli.build_parser().parse_args(
            ["task", "--agent", "web", "--toolset", "web,fs"])
        prompt = _system_prompt_for(cli.selected_toolset_names(args))
        self.assertIn("live snapshot", prompt)   # web
        self.assertIn("create_file", prompt)     # fs


class AgentValidationTest(unittest.TestCase):
    def test_known_agent_has_no_error(self):
        self.assertIsNone(cli.agent_error("web"))
        self.assertIsNone(cli.agent_error("fs"))

    def test_unknown_agent_reports_helpful_error(self):
        msg = cli.agent_error("bogus")
        self.assertIsNotNone(msg)
        self.assertIn("unknown agent 'bogus'", msg)
        self.assertIn("Available:", msg)
        self.assertIn("web", msg)
        self.assertIn("fs", msg)


class AgentCliExitTest(unittest.TestCase):
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

    def test_invalid_agent_exits_2_with_message(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["do something", "--agent", "bogus"])
        self.assertEqual(rc, 2)
        out = buf.getvalue()
        self.assertIn("unknown agent 'bogus'", out)
        self.assertIn("Available:", out)


class ListAgentsTest(unittest.TestCase):
    def test_list_agents_exits_0_and_lists_web_and_fs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["--list-agents"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("web", out)
        self.assertIn("fs", out)


class AgentHelpTest(unittest.TestCase):
    def test_help_mentions_agent_and_list_agents(self):
        help_text = cli.build_parser().format_help()
        self.assertIn("--agent", help_text)
        self.assertIn("--list-agents", help_text)


if __name__ == "__main__":
    unittest.main()
