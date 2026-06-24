import unittest

from vibeharness.validation import (LLMValidator, ValidateTool,
                                    build_validator_prompt)

from tests._fakes import ScriptedVerdictClient as ScriptedClient


class ValidatorTest(unittest.TestCase):
    def test_verdict_pass_and_fail(self):
        # The validator's verdict and reason are taken from the parsed model JSON.
        cases = [
            ('{"verdict":"pass","reason":"all steps done"}', True, "all steps done"),
            ('{"verdict":"fail","reason":"step 2 missing"}', False, "missing"),
        ]
        for verdict_json, expected_pass, reason_fragment in cases:
            with self.subTest(verdict_json=verdict_json):
                v = LLMValidator(ScriptedClient(verdict_json)).validate(
                    "do X then Y", "First, you did X.", "claim")
                self.assertEqual(v.passed, expected_pass)
                self.assertIn(reason_fragment, v.reason)
                if expected_pass:
                    self.assertIn("judging", v.reasoning)

    def test_unparseable_verdict_is_treated_as_fail(self):
        v = LLMValidator(ScriptedClient("not json at all")).validate("t", "h", "c")
        self.assertFalse(v.passed)

    def test_validator_passes_a_decode_constraint_not_a_raw_schema(self):
        # Regression (#13): the codec seam changed LLMClient.decide to take a
        # DecodeConstraint. LLMValidator must wrap VERDICT_SCHEMA, not pass the raw
        # dict — passing the dict crashed the real OllamaClient with
        # "AttributeError: 'dict' object has no attribute 'stop'" on every validate.
        from vibeharness.codec import DecodeConstraint
        from vibeharness.llm import Decision, LLMClient
        from vibeharness.validation import VERDICT_SCHEMA

        captured = {}

        class RecordingClient(LLMClient):
            def decide(self, system, user, constraint, on_reason=None, on_action=None):
                captured["constraint"] = constraint
                return Decision(reasoning="", action_json='{"verdict":"pass","reason":"ok"}')

        LLMValidator(RecordingClient()).validate("t", "h", "c")
        self.assertIsInstance(captured["constraint"], DecodeConstraint)
        self.assertEqual(captured["constraint"].json_schema, VERDICT_SCHEMA)

    def test_validator_prompt_includes_task_history_and_claim(self):
        client = ScriptedClient('{"verdict":"pass","reason":"ok"}')
        LLMValidator(client).validate("ORIGINAL TASK", "AGENT HISTORY", "AGENT CLAIM")
        self.assertIn("ORIGINAL TASK", client.last_user)
        self.assertIn("AGENT HISTORY", client.last_user)
        self.assertIn("AGENT CLAIM", client.last_user)

    def test_build_prompt_handles_missing_claim(self):
        prompt = build_validator_prompt("t", "h", "")
        self.assertIn("no summary", prompt)


class ValidateToolTest(unittest.TestCase):
    def test_schema_and_params(self):
        tool = ValidateTool()
        self.assertEqual(tool.name, "validate")
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "validate")
        self.assertIn("summary", schema["properties"]["args"]["properties"])


if __name__ == "__main__":
    unittest.main()
