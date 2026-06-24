"""Per-agent-type max-actions-per-turn cap (issue #52).

The number of tool calls allowed PER TURN is configurable by agent type, not just
by the global ``Config.max_actions_per_turn``. Defaults: fs keeps the global
(multiple) default; web == 4 (raised from 1 now that snapshot-ref enforcement #73
guards stale refs); validator == 1.

These tests assert ONE source of truth: the cap the CLI resolves for an agent is
the cap the prompt STATES *and* the cap the agent loop ENFORCES. No model / Ollama
is required — we resolve config from parsed args and run the loop with a fake client.
"""
import os
import tempfile
import unittest

from vibeharness import cli
from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.toolset import (
    agent_default_max_actions,
    default_catalog,
)

from tests._fakes import FakeLLMClient, FakeValidator


def _resolved_cap(argv: list[str]) -> int:
    """Resolve the per-turn cap the CLI would use for these args."""
    args = cli.build_parser().parse_args(argv)
    return cli.resolve_config(args).max_actions_per_turn


def _prompt_cap_text(cap: int) -> str:
    """The system-prompt text a build with this cap would contain (json codec)."""
    catalog = default_catalog()
    toolsets = catalog.select(["fs"])
    registry = catalog.build_registry(toolsets, Config())
    return SystemPromptBuilder(registry, cap).build("a task")


class IsolatedSettings(unittest.TestCase):
    """Each test gets a private, empty settings store so a saved
    max_actions_per_turn on the dev machine can't perturb resolution."""

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


class MappingTest(IsolatedSettings):
    def test_defaults_are_fs_multiple_web_four_validator_one(self):
        mapping = agent_default_max_actions()
        # fs keeps the GLOBAL (multiple) default — the single source of truth.
        self.assertEqual(mapping["fs"], Config.max_actions_per_turn)
        self.assertGreater(mapping["fs"], 1)  # "multiple"
        self.assertEqual(mapping["web"], 4)
        self.assertEqual(mapping["validator"], 1)

    def test_fs_default_tracks_supplied_global_default(self):
        # The fs entry is driven by the passed-in global default, not hard-coded.
        self.assertEqual(agent_default_max_actions(7)["fs"], 7)


class ResolutionPrecedenceTest(IsolatedSettings):
    def test_agent_web_resolves_to_four(self):
        self.assertEqual(_resolved_cap(["task", "--agent", "web"]), 4)

    def test_agent_validator_resolves_to_one(self):
        self.assertEqual(_resolved_cap(["task", "--agent", "validator"]), 1)

    def test_agent_fs_resolves_to_multiple_default(self):
        self.assertEqual(
            _resolved_cap(["task", "--agent", "fs"]), Config.max_actions_per_turn)

    def test_no_agent_resolves_to_global_default(self):
        self.assertEqual(_resolved_cap(["task"]), Config.max_actions_per_turn)

    def test_explicit_flag_overrides_agent_default_for_web(self):
        # Explicit --max-actions-per-turn beats the agent's web=1 default.
        self.assertEqual(
            _resolved_cap(["task", "--agent", "web", "--max-actions-per-turn", "5"]), 5)

    def test_explicit_flag_overrides_agent_default_for_fs(self):
        self.assertEqual(
            _resolved_cap(["task", "--agent", "fs", "--max-actions-per-turn", "5"]), 5)

    def test_saved_setting_overrides_agent_default(self):
        from vibeharness.settings import Settings
        Settings.set("max-actions-per-turn", "3")
        # A saved setting is an explicit user choice; it beats the web=1 agent default.
        self.assertEqual(_resolved_cap(["task", "--agent", "web"]), 3)

    def test_explicit_flag_beats_saved_setting(self):
        from vibeharness.settings import Settings
        Settings.set("max-actions-per-turn", "3")
        self.assertEqual(
            _resolved_cap(["task", "--agent", "web", "--max-actions-per-turn", "5"]), 5)


class PromptStatesResolvedCapTest(IsolatedSettings):
    """For each agent the prompt's STATED cap equals the ENFORCED (resolved) cap."""

    def _assert_prompt_states(self, argv: list[str], cap: int):
        prompt = _prompt_cap_text(cap)
        self.assertEqual(_resolved_cap(argv), cap)
        self.assertIn(f"at most {cap} actions per turn", prompt)

    def test_web_prompt_states_four(self):
        self._assert_prompt_states(["task", "--agent", "web"], 4)

    def test_validator_prompt_states_one(self):
        self._assert_prompt_states(["task", "--agent", "validator"], 1)

    def test_fs_prompt_states_multiple_default(self):
        self._assert_prompt_states(["task", "--agent", "fs"], Config.max_actions_per_turn)

    def test_explicit_flag_prompt_states_that_cap(self):
        self._assert_prompt_states(
            ["task", "--agent", "web", "--max-actions-per-turn", "5"], 5)


class LoopEnforcesResolvedCapTest(IsolatedSettings):
    """Given a model turn emitting MORE actions than the cap, the loop runs only the
    first ``cap`` and tells the model it was capped — driven by the same resolved cap."""

    def setUp(self):
        super().setUp()
        self.workdir = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def tearDown(self):
        self.workdir.cleanup()
        super().tearDown()

    def _p(self, name: str) -> str:
        return os.path.join(self.workdir.name, name)

    def _run_with_cap(self, cap: int, n_actions: int):
        # A single turn emitting n_actions create_file calls, then validate.
        writes = [
            {"tool": "create_file",
             "args": {"path": self._p(f"f{i}.txt"), "content": f"content {i} {i} {i}"}}
            for i in range(n_actions)
        ]
        actions = [writes, {"tool": "validate", "args": {"summary": "done"}}]
        client = FakeLLMClient(actions)
        config = Config(max_steps=5, max_actions_per_turn=cap)
        agent = RalphAgent(client, self.registry, "SYSTEM", config,
                           FakeValidator(passed=True))
        return agent.run("a task")

    def test_web_cap_one_runs_only_first_of_two(self):
        result = self._run_with_cap(cap=1, n_actions=2)
        # Only the first write happened.
        self.assertTrue(os.path.exists(self._p("f0.txt")))
        self.assertFalse(os.path.exists(self._p("f1.txt")))
        # And the model was told it was capped.
        first_turn = result.turns[0]
        notes = " ".join(a.observation for a in first_turn.actions)
        self.assertIn("per-turn limit of 1", notes)

    def test_fs_cap_runs_up_to_cap(self):
        cap = Config.max_actions_per_turn
        result = self._run_with_cap(cap=cap, n_actions=cap)
        # All `cap` writes ran (none dropped) since n_actions == cap.
        for i in range(cap):
            self.assertTrue(os.path.exists(self._p(f"f{i}.txt")))
        notes = " ".join(a.observation for a in result.turns[0].actions)
        self.assertNotIn("per-turn limit", notes)

    def test_over_cap_drops_excess(self):
        result = self._run_with_cap(cap=2, n_actions=5)
        self.assertTrue(os.path.exists(self._p("f0.txt")))
        self.assertTrue(os.path.exists(self._p("f1.txt")))
        self.assertFalse(os.path.exists(self._p("f2.txt")))
        notes = " ".join(a.observation for a in result.turns[0].actions)
        self.assertIn("per-turn limit of 2", notes)
        self.assertIn("3 ignored", notes)


if __name__ == "__main__":
    unittest.main()
