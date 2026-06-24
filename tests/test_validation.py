import json
import unittest

from vibeharness.llm import Decision, LLMClient
from vibeharness.validation import (LLMValidator, ValidateTool, Verdict,
                                    build_validator_prompt)


class ScriptedClient(LLMClient):
    """Returns a fixed verdict JSON, and records the prompt it was given."""
    def __init__(self, verdict_json: str):
        self._verdict = verdict_json
        self.last_system = None
        self.last_user = None
        self.last_on_reason = None
        self.last_on_action = None

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        self.last_system, self.last_user = system, user
        self.last_on_reason, self.last_on_action = on_reason, on_action
        # Drive the streaming callbacks the way a real streaming client would.
        if on_reason:
            on_reason("judging")
        if on_action:
            on_action(self._verdict)
        return Decision(reasoning="<think>judging</think>", action_json=self._verdict)


class ValidatorTest(unittest.TestCase):
    def test_pass_verdict(self):
        client = ScriptedClient('{"verdict":"pass","reason":"all steps done"}')
        v = LLMValidator(client).validate("do X", "First, you did X.", "I did X")
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "all steps done")
        self.assertIn("judging", v.reasoning)

    def test_fail_verdict(self):
        client = ScriptedClient('{"verdict":"fail","reason":"step 2 missing"}')
        v = LLMValidator(client).validate("do X then Y", "First, you did X.", "done")
        self.assertFalse(v.passed)
        self.assertIn("missing", v.reason)

    def test_unparseable_verdict_is_treated_as_fail(self):
        v = LLMValidator(ScriptedClient("not json at all")).validate("t", "h", "c")
        self.assertFalse(v.passed)

    def test_validator_prompt_includes_task_history_and_claim(self):
        client = ScriptedClient('{"verdict":"pass","reason":"ok"}')
        LLMValidator(client).validate("ORIGINAL TASK", "AGENT HISTORY", "AGENT CLAIM")
        self.assertIn("ORIGINAL TASK", client.last_user)
        self.assertIn("AGENT HISTORY", client.last_user)
        self.assertIn("AGENT CLAIM", client.last_user)

    def test_build_prompt_handles_missing_claim(self):
        prompt = build_validator_prompt("t", "h", "")
        self.assertIn("no summary", prompt)

    def test_streaming_callbacks_are_forwarded_to_client(self):
        client = ScriptedClient('{"verdict":"pass","reason":"ok"}')
        reasons, actions = [], []
        on_reason = lambda t: reasons.append(t)
        on_action = lambda t: actions.append(t)
        LLMValidator(client).validate("t", "h", "c",
                                      on_reason=on_reason, on_action=on_action)
        # the validator handed our exact callbacks straight to client.decide,
        # and the client streamed through them.
        self.assertIs(client.last_on_reason, on_reason)
        self.assertIs(client.last_on_action, on_action)
        self.assertEqual(reasons, ["judging"])
        self.assertEqual(actions, ['{"verdict":"pass","reason":"ok"}'])


class ValidateToolTest(unittest.TestCase):
    def test_schema_and_params(self):
        tool = ValidateTool()
        self.assertEqual(tool.name, "validate")
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "validate")
        self.assertIn("summary", schema["properties"]["args"]["properties"])


if __name__ == "__main__":
    unittest.main()
