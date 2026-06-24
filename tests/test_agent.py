import json
import os
import tempfile
import unittest

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry
from vibeharness.reporting import NullReporter
from vibeharness.validation import Validator, Verdict


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


class FakeLLMClient(LLMClient):
    """Returns scripted actions instead of calling a model. When the script is
    exhausted it repeats the last action (so loops can run to the step budget)."""

    def __init__(self, actions):
        self._actions = actions
        self._i = 0

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        payload = action if isinstance(action, str) else json.dumps(action)
        return Decision(reasoning="", action_json=payload)


class FakeValidator(Validator):
    def __init__(self, passed=True, reason="looks complete"):
        self._passed, self._reason = passed, reason
        self.calls = []

    def validate(self, task, history, claim, on_reason=None, on_action=None):
        self.calls.append({"task": task, "history": history, "claim": claim})
        # Stream a little so a reporter observing the validator can be exercised.
        if on_reason:
            on_reason("judging")
        if on_action:
            on_action('{"verdict":"pass"}')
        return Verdict(self._passed, self._reason)


VALIDATE = {"tool": "validate", "args": {"summary": "done"}}


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
        return RalphAgent(client, self.registry, "SYSTEM", Config(max_steps=max_steps),
                          validator or FakeValidator(passed=True), reporter=reporter)

    def test_sequential_turns_then_validate_passes(self):
        actions = [
            {"tool": "write_file", "args": {"path": self.p("a.txt"), "content": "hello hello hello"}},
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
        # the validator received the task + the agent's claim
        self.assertEqual(validator.calls[0]["claim"], "done")
        self.assertIn("make a file", validator.calls[0]["task"])

    def test_multiple_actions_in_one_turn(self):
        actions = [[
            {"tool": "write_file", "args": {"path": self.p("a.txt"), "content": "batched"}},
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

    def test_invalid_json_is_reported_and_loop_continues(self):
        result = self._agent(["{ not json", VALIDATE], max_steps=5).run("t")
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("invalid", result.turns[0].actions[0].observation)
        self.assertTrue(result.finished)

    def test_unknown_tool_is_reported(self):
        result = self._agent([{"tool": "teleport", "args": {}}, VALIDATE], max_steps=5).run("t")
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("not a real tool", result.turns[0].actions[0].observation)

    def test_stops_at_step_budget_without_validating(self):
        actions = [{"tool": "list_directory", "args": {"path": self.dir}}]
        result = self._agent(actions, max_steps=3).run("loop")
        self.assertFalse(result.finished)
        self.assertEqual(len(result.turns), 3)

    def test_on_turn_checkpoint_is_called_each_turn(self):
        seen = []
        self._agent([VALIDATE]).run("t", on_turn=lambda r: seen.append(len(r.turns)))
        self.assertEqual(seen, [1])   # one turn, checkpointed once

    def test_validator_stream_reaches_reporter(self):
        reporter = RecordingReporter()
        self._agent([VALIDATE], reporter=reporter).run("t")
        # the agent emitted the start marker, then forwarded the validator's
        # streamed reasoning and verdict tokens to the reporter.
        self.assertEqual(reporter.events[0], ("start", None))
        self.assertIn(("reason", "judging"), reporter.events)
        self.assertIn(("verdict", '{"verdict":"pass"}'), reporter.events)

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
