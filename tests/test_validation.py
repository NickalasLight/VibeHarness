import unittest

from vibeharness.config import Config
from vibeharness.toolset import (Toolset, agent_default_toolsets,
                                 default_catalog)
from vibeharness.validation import (LLMValidator, ValidateTool,
                                    ValidatorToolset, VALIDATOR_SYSTEM,
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


class ValidatorAgentTypeTest(unittest.TestCase):
    """The validator is declared as a first-class agent type via the SAME
    framework as web/fs (issue #31): prompt via system_guidance, verdict tool via
    create_tools, registered in the catalog so it is a recognized agent type."""

    def test_validator_toolset_is_a_toolset(self):
        self.assertIsInstance(ValidatorToolset(), Toolset)
        self.assertEqual(ValidatorToolset().name, "validator")

    def test_validator_is_a_recognized_agent_type(self):
        # agent_default_toolsets() (and therefore --agent / --list-agents) knows it.
        mapping = agent_default_toolsets()
        self.assertIn("validator", mapping)
        self.assertEqual(mapping["validator"], ["validator"])
        self.assertIn("validator", default_catalog().names())

    def test_prompt_is_exposed_via_the_framework(self):
        # The validator's PROMPT is surfaced through #19's system_guidance hook,
        # and it is the very same VALIDATOR_SYSTEM that drives live validation.
        self.assertEqual(ValidatorToolset().system_guidance(), VALIDATOR_SYSTEM)

    def test_verdict_tool_is_exposed_via_the_framework(self):
        tools = ValidatorToolset().create_tools(Config())
        self.assertEqual([t.name for t in tools], ["validate"])
        self.assertIsInstance(tools[0], ValidateTool)

    def test_selecting_validator_toolset_builds_a_registry_with_validate(self):
        # Even though the validator toolset declares `validate` AND the catalog
        # injects it as the core tool, the registry holds exactly one (de-duped).
        catalog = default_catalog()
        registry = catalog.build_registry(catalog.select(["validator"]), Config())
        self.assertEqual(registry.names(), ["validate"])


class ValidatorExecutionUnchangedTest(unittest.TestCase):
    """The single-shot validate execution must be UNCHANGED by the #31 declaration:
    LLMValidator.validate still issues one pass/fail decision using VALIDATOR_SYSTEM
    directly — it is NOT routed through the main agent's tool loop."""

    def test_validate_is_single_shot_using_validator_system_prompt(self):
        from vibeharness.llm import Decision, LLMClient

        calls = {"count": 0, "system": None}

        class RecordingClient(LLMClient):
            def decide(self, system, user, constraint, on_reason=None, on_action=None):
                calls["count"] += 1
                calls["system"] = system
                return Decision(reasoning="", action_json='{"verdict":"pass","reason":"ok"}')

        verdict = LLMValidator(RecordingClient()).validate("task", "history", "claim")
        # Exactly one model call (single-shot, not a loop) ...
        self.assertEqual(calls["count"], 1)
        # ... driven by the SAME prompt the framework now declares ...
        self.assertEqual(calls["system"], VALIDATOR_SYSTEM)
        self.assertEqual(ValidatorToolset().system_guidance(), calls["system"])
        # ... and the verdict behavior is the parsed pass/fail.
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reason, "ok")


if __name__ == "__main__":
    unittest.main()
