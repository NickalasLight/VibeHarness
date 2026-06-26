import itertools
import os
import tempfile
import unittest

from vibeharness.agent import Action, RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.registry import ToolRegistry
from vibeharness.reporting import NullReporter
from vibeharness.tools import Tool, ToolResult

from tests._fakes import FakeLLMClient, FakeValidator, RecordingLLMClient


class BoomTool(Tool):
    """A tool that raises an unexpected (non-ToolResult) error when run, to model a
    mid-turn crash that escapes the normal tool-error handling."""

    name = "boom"
    description = "raises mid-turn"

    @property
    def parameters(self):
        return []

    def run(self, args) -> ToolResult:
        raise RuntimeError("kaboom from a tool mid-turn")


class RecordingReporter(NullReporter):
    """Captures the validator-stream hooks the agent drives."""

    def __init__(self):
        self.events = []

    def validator_start(self):
        self.events.append(("start", None))

    def validator_reasoning_token(self, text):
        self.events.append(("reason", text))

    def validator_verdict_token(self, text):
        self.events.append(("verdict", text))


VALIDATE = {"tool": "validate", "args": {}}


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.dir, name)

    def _agent(self, actions, validator=None, max_steps=10, reporter=None):
        client = FakeLLMClient(actions)
        # Escalation (swap to the API model on stuck / premature-validate) is enabled by
        # default in Config but is exercised in test_escalation_agent.py — disable it here
        # so these local-loop tests stay hermetic and never reach for an API client.
        return RalphAgent(client, self.registry, "SYSTEM",
                          Config(max_steps=max_steps, escalation_enabled=False),
                          validator or FakeValidator(passed=True), reporter=reporter)

    def test_anti_loop_skips_repeated_successful_action(self):
        # #125: an identical, already-successful action is NOT re-executed; the model is
        # steered instead. Guards the small-model loop (e.g. re-filling a filled field).
        class CountingTool(Tool):
            name = "ping"
            description = "counts its runs"

            def __init__(self):
                self.calls = 0

            @property
            def parameters(self):
                return []

            def run(self, args) -> ToolResult:
                self.calls += 1
                return ToolResult(True, f"you pinged (#{self.calls}).")

        tool = CountingTool()
        registry = ToolRegistry([tool])
        client = FakeLLMClient([
            {"tool": "ping", "args": {}},   # turn 1: executes
            {"tool": "ping", "args": {}},   # turn 2: identical -> steered, NOT executed
            VALIDATE,
        ])
        agent = RalphAgent(client, registry, "SYS", Config(max_steps=10),
                           FakeValidator(passed=True))
        result = agent.run("ping")
        self.assertEqual(tool.calls, 1)  # the duplicate did not run
        self.assertIn("was already successfully set to", result.transcript())

    def test_anti_loop_steers_repeated_failed_action(self):
        # #125 iter 6: a repeated FAILING action (e.g. an invalid ref) must also be steered,
        # not retried forever. Runs once, then the identical retry is blocked.
        class FailTool(Tool):
            name = "boomf"
            description = "always fails"

            def __init__(self):
                self.calls = 0

            @property
            def parameters(self):
                return []

            def run(self, args) -> ToolResult:
                self.calls += 1
                return ToolResult(False, "you tried boomf but it failed.")

        tool = FailTool()
        registry = ToolRegistry([tool])
        client = FakeLLMClient([
            {"tool": "boomf", "args": {}},   # turn 1: runs, fails
            {"tool": "boomf", "args": {}},   # turn 2: identical -> steered (already failed)
            VALIDATE,
        ])
        agent = RalphAgent(client, registry, "SYS", Config(max_steps=10),
                           FakeValidator(passed=True))
        result = agent.run("boom")
        self.assertEqual(tool.calls, 1)  # the failed action was not retried
        self.assertIn("and it FAILED", result.transcript())

    def test_soft_repeat_click_reruns_after_success(self):
        # iter-1 fix: `click` is a SOFT-REPEAT tool. Clicking the SAME button again is the
        # correct recovery after a validation-blocked submit, so a repeated SUCCESSFUL click
        # must RUN again (not be blocked as an "already set" no-op). This is the bug that
        # froze the job form on page 1 (re-click Continue was permanently blocked).
        class ClickTool(Tool):
            name = "click"
            description = "clicks an element"

            def __init__(self):
                self.calls = 0

            @property
            def parameters(self):
                return []

            def run(self, args) -> ToolResult:
                self.calls += 1
                return ToolResult(True, f"you clicked {args.get('target')} (#{self.calls}).")

        tool = ClickTool()
        registry = ToolRegistry([tool])
        client = FakeLLMClient([
            {"tool": "click", "args": {"target": "e81"}},   # turn 1: runs
            {"tool": "click", "args": {"target": "e81"}},   # turn 2: identical -> still RUNS
            VALIDATE,
        ])
        agent = RalphAgent(client, registry, "SYS", Config(max_steps=10),
                           FakeValidator(passed=True))
        agent.run("click twice")
        self.assertEqual(tool.calls, 2)  # the repeated click DID run again

    def test_soft_repeat_click_is_bounded(self):
        # iter-1 fix: a soft-repeat click is allowed to repeat, but a no-op click (e.g. on a
        # heading that changes nothing) must NOT loop forever — after _SOFT_REPEAT_LIMIT
        # identical repeats it is steered, so the tool runs at most 1 + LIMIT times.
        class ClickTool(Tool):
            name = "click"
            description = "clicks an element"

            def __init__(self):
                self.calls = 0

            @property
            def parameters(self):
                return []

            def run(self, args) -> ToolResult:
                self.calls += 1
                return ToolResult(True, f"you clicked {args.get('target')} (#{self.calls}).")

        tool = ClickTool()
        registry = ToolRegistry([tool])
        # Emit the SAME click MORE times than the limit; it must be capped at
        # 1 + _SOFT_REPEAT_LIMIT runs (first run + the bounded soft repeats).
        n = RalphAgent._SOFT_REPEAT_LIMIT + 4
        actions = [{"tool": "click", "args": {"target": "e35"}} for _ in range(n)]
        actions.append(VALIDATE)
        client = FakeLLMClient(actions)
        agent = RalphAgent(client, registry, "SYS",
                           Config(max_steps=n + 2, escalation_enabled=False),
                           FakeValidator(passed=True))
        agent.run("loop heading")
        self.assertEqual(tool.calls, 1 + RalphAgent._SOFT_REPEAT_LIMIT)

    def test_loop_guard_exemption_only_advancing_navigation(self):
        # #125 iter 9: open_browser / goto reset state on repeat -> must be guarded
        # (signature returned). Only navigate_back/forward advance on repeat -> exempt.
        agent = self._agent([VALIDATE])
        self.assertIsNone(agent._action_signature("navigate_back", {}))
        self.assertIsNone(agent._action_signature("navigate_forward", {}))
        self.assertIsNotNone(agent._action_signature("open_browser", {}))
        self.assertIsNotNone(agent._action_signature("goto", {"url": "u"}))

    def test_anti_loop_allows_different_args(self):
        # Same tool with DIFFERENT args is a different action and must still run.
        actions = [
            {"tool": "create_file", "args": {"path": self.p("a.txt"), "content": "x"}},
            {"tool": "create_file", "args": {"path": self.p("b.txt"), "content": "y"}},
            VALIDATE,
        ]
        self._agent(actions).run("two files")
        self.assertTrue(os.path.exists(self.p("a.txt")))
        self.assertTrue(os.path.exists(self.p("b.txt")))

    def test_sequential_turns_then_validate_passes(self):
        actions = [
            {"tool": "create_file", "args": {"path": self.p("a.txt"), "content": "hello hello hello"}},
            {"tool": "read_file", "args": {"path": self.p("a.txt")}},
            VALIDATE,
        ]
        validator = FakeValidator(passed=True, reason="file created and verified")
        result = self._agent(actions, validator).run("make a file")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns), 3)
        self.assertEqual(result.final_summary, "file created and verified")
        self.assertEqual(len(result.validations), 1)
        self.assertTrue(result.validations[0]["passed"])
        # Issue #57: the validator is fed (context, history) — no self-claim. With no
        # context provider wired here, context is "" and history is the action account.
        self.assertEqual(validator.calls[0]["context"], "")
        self.assertNotIn("claim", validator.calls[0])
        self.assertIn("you created the file", validator.calls[0]["history"])

    def test_over_limit_batch_runs_only_first_n(self):
        # Five writes in one batch, but the cap is 2 -> only the first 2 files appear,
        # and the model is told the extras were ignored. The run never crashes.
        batch = [{"tool": "create_file",
                  "args": {"path": self.p(f"f{n}.txt"), "content": "x"}} for n in range(5)]
        client = FakeLLMClient([batch])
        agent = RalphAgent(client, self.registry, "SYSTEM",
                           Config(max_steps=1, max_actions_per_turn=2,
                                  escalation_enabled=False),
                           FakeValidator(passed=True))
        result = agent.run("over limit")
        self.assertTrue(os.path.exists(self.p("f0.txt")))
        self.assertTrue(os.path.exists(self.p("f1.txt")))
        self.assertFalse(os.path.exists(self.p("f2.txt")))
        # Excess actions are now silently dropped: only the first 2 run and are
        # recorded; no "per-turn limit" note is surfaced to the model.
        obs = [a.observation for a in result.turns[0].actions]
        self.assertFalse(any("per-turn limit" in o for o in obs))
        executed = [a for a in result.turns[0].actions if a.tool == "create_file"]
        self.assertEqual(len(executed), 2)

    def test_multiple_actions_in_one_turn(self):
        actions = [[
            {"tool": "create_file", "args": {"path": self.p("a.txt"), "content": "batched"}},
            {"tool": "read_file", "args": {"path": self.p("a.txt")}},
            VALIDATE,
        ]]
        result = self._agent(actions).run("batch it")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(len(result.turns[0].actions), 3)

    def test_actions_after_validate_pass_are_ignored(self):
        actions = [[VALIDATE,
                    {"tool": "write_file", "args": {"path": self.p("nope.txt"), "content": "x"}}]]
        result = self._agent(actions).run("t")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns[0].actions), 1)
        self.assertFalse(os.path.exists(self.p("nope.txt")))

    def test_validation_failure_continues_the_loop(self):
        validator = FakeValidator(passed=False, reason="the file is missing")
        result = self._agent([VALIDATE], validator, max_steps=3).run("t")
        self.assertFalse(result.finished)
        self.assertEqual(len(result.turns), 3)            # kept trying, never passed
        self.assertEqual(len(result.validations), 3)
        self.assertIn("FAILED", result.turns[0].actions[0].observation)
        self.assertIn("missing", result.turns[0].actions[0].observation)

    def test_malformed_turn_is_reported_and_loop_continues(self):
        # A bad first turn (unparseable JSON or an unknown tool) is reported as a
        # not-ok action; the loop does not crash and goes on to validate+finish.
        cases = [
            ("{ not json", "invalid"),                       # unparseable JSON
            ({"tool": "teleport", "args": {}}, "is not a valid tool"),  # unknown tool
        ]
        for bad_action, expected in cases:
            with self.subTest(bad_action=bad_action):
                result = self._agent([bad_action, VALIDATE], max_steps=5).run("t")
                self.assertFalse(result.turns[0].actions[0].ok)
                self.assertIn(expected, result.turns[0].actions[0].observation)
                self.assertTrue(result.finished)

    def test_stops_at_step_budget_without_validating(self):
        actions = [{"tool": "list_directory", "args": {"path": self.dir}}]
        result = self._agent(actions, max_steps=3).run("loop")
        self.assertFalse(result.finished)
        self.assertEqual(len(result.turns), 3)

    def test_on_turn_checkpoint_is_called_each_turn(self):
        seen = []
        self._agent([VALIDATE]).run("t", on_turn=lambda r: seen.append(len(r.turns)))
        self.assertEqual(seen, [1])   # one turn, checkpointed once

    def test_unexpected_mid_turn_error_flushes_partial_turn_then_reraises(self):
        # A turn that completes one real action and then hits an unexpected tool
        # error (#16). The agent must: keep the completed action on the turn, call
        # on_turn ONCE so the partial result reaches a checkpoint, then re-raise so
        # the caller still sees the crash. Without the failsafe, on_turn never fires
        # for this turn and the completed action is lost.
        registry = ToolRegistry(build_default_tools(FileSystem(), 1000) + [BoomTool()])
        actions = [[
            {"tool": "list_directory", "args": {"path": self.dir}},
            {"tool": "boom", "args": {}},
        ]]
        agent = RalphAgent(FakeLLMClient(actions), registry, "SYS",
                           Config(max_steps=5), FakeValidator())
        checkpoints = []
        with self.assertRaises(RuntimeError):
            agent.run("t", on_turn=lambda r: checkpoints.append(
                [a.tool for a in r.turns[-1].actions]))
        # the failsafe flushed exactly once, and that snapshot already held the
        # completed list_directory action (plus the recorded failure marker)
        self.assertEqual(len(checkpoints), 1)
        self.assertIn("list_directory", checkpoints[0])

    def test_error_before_any_action_still_flushes_a_turn(self):
        # If the crash strikes before the Turn is even appended (e.g. the system
        # prompt provider raises), the failsafe must still create a turn, flush it,
        # and re-raise — so the run is never silently turn-less on a crash.
        def boom_provider():
            raise RuntimeError("provider blew up before turn 1 produced anything")

        client = FakeLLMClient([VALIDATE])
        agent = RalphAgent(client, self.registry, "SYS", Config(max_steps=5),
                           FakeValidator(), system_prompt_provider=boom_provider)
        seen = []
        with self.assertRaises(RuntimeError):
            agent.run("t", on_turn=lambda r: seen.append(len(r.turns)))
        self.assertEqual(seen, [1])   # one turn was synthesised and flushed

    def test_validator_stream_reaches_reporter(self):
        reporter = RecordingReporter()
        self._agent([VALIDATE], reporter=reporter).run("t")
        # the agent emitted the start marker, then forwarded the validator's
        # streamed reasoning and verdict tokens to the reporter.
        self.assertEqual(reporter.events[0], ("start", None))
        self.assertIn(("reason", "judging"), reporter.events)
        self.assertIn(("verdict", '{"verdict":"pass"}'), reporter.events)

    def test_provider_refreshes_system_each_turn(self):
        client = RecordingLLMClient([{"tool": "list_directory", "args": {"path": self.dir}}])
        counter = itertools.count(1)
        provider = lambda: f"SYS-{next(counter)}"
        agent = RalphAgent(client, self.registry, "STATIC-SYS",
                           Config(max_steps=3), FakeValidator(passed=True),
                           system_prompt_provider=provider)
        agent.run("t")
        # each turn saw a freshly regenerated system prompt, not the static one
        self.assertEqual(client.systems, ["SYS-1", "SYS-2", "SYS-3"])
        self.assertNotIn("STATIC-SYS", client.systems)

    def test_one_arg_provider_receives_user_message(self):
        # Issue #43: a provider that accepts the per-turn user message is called WITH
        # it (so it can budget the page snapshot against the full message), while the
        # zero-arg legacy provider above still works. The agent detects the arity.
        client = RecordingLLMClient([{"tool": "list_directory", "args": {"path": self.dir}}])
        seen_users = []

        def provider(user):
            seen_users.append(user)
            return "SYS-WITH-USER"

        agent = RalphAgent(client, self.registry, "STATIC-SYS",
                           Config(max_steps=1), FakeValidator(passed=True),
                           system_prompt_provider=provider)
        agent.run("my-task")
        # The provider was handed the exact per-turn user message the model also saw.
        self.assertEqual(len(seen_users), 1)
        self.assertEqual(seen_users[0], client.users[0])
        self.assertIn("my-task", seen_users[0])
        self.assertEqual(client.systems, ["SYS-WITH-USER"])

    def test_transcript_and_to_dict(self):
        result = self._agent([VALIDATE]).run("t")
        text = result.transcript()
        self.assertIn("TASK: t", text)
        self.assertIn("FINISHED: True", text)
        self.assertIn("VALIDATIONS:", text)
        d = result.to_dict()
        self.assertIn("validations", d)
        self.assertIn("reasoning", d["turns"][0])


if __name__ == "__main__":
    unittest.main()
