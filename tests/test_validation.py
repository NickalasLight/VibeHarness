import glob
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vibeharness.config import Config
from vibeharness.runlog import RunLogger
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
                    "# YOUR ASSIGNED TASK\ndo X then Y", "First, you did X.")
                self.assertEqual(v.passed, expected_pass)
                self.assertIn(reason_fragment, v.reason)
                if expected_pass:
                    self.assertIn("judging", v.reasoning)

    def test_unparseable_verdict_is_treated_as_fail(self):
        v = LLMValidator(ScriptedClient("not json at all")).validate("ctx", "h")
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

        LLMValidator(RecordingClient()).validate("ctx", "h")
        self.assertIsInstance(captured["constraint"], DecodeConstraint)
        self.assertEqual(captured["constraint"].json_schema, VERDICT_SCHEMA)

    def test_validator_prompt_includes_context_and_history_but_not_claim(self):
        # Issue #57: the validator's user message carries the SAME context the agent
        # had (the tool-less main prompt) + the action history — and NO self-claim.
        client = ScriptedClient('{"verdict":"pass","reason":"ok"}')
        context = ("# YOUR ASSIGNED TASK\nORIGINAL TASK\n\n---\n\n"
                   "# Current page (live snapshot)\nbutton \"Submit\" [ref=e7]\n\n---\n\n")
        LLMValidator(client).validate(context, "AGENT HISTORY")
        # the page snapshot + task context appear, BEFORE the history
        self.assertIn("ORIGINAL TASK", client.last_user)
        self.assertIn("# Current page (live snapshot)", client.last_user)
        self.assertIn("[ref=e7]", client.last_user)
        self.assertIn("AGENT HISTORY", client.last_user)
        self.assertLess(client.last_user.index("ORIGINAL TASK"),
                        client.last_user.index("AGENT HISTORY"))
        # no self-claim section anymore
        self.assertNotIn("completion claim", client.last_user)

    def test_build_prompt_places_context_before_history(self):
        prompt = build_validator_prompt("CONTEXT BLOCK", "HISTORY BLOCK")
        self.assertIn("CONTEXT BLOCK", prompt)
        self.assertIn("HISTORY BLOCK", prompt)
        self.assertLess(prompt.index("CONTEXT BLOCK"), prompt.index("HISTORY BLOCK"))
        self.assertNotIn("claim", prompt.lower())


class ValidatorLoggingTest(unittest.TestCase):
    """Issue #47: every validate() call is persisted to its own
    validator_<guid>.json in the run's .vibe/ folder."""

    def _logger(self, workspace):
        return RunLogger(workspace, datetime(2026, 6, 24, 12, 0, 0))

    def _validator_files(self, vibe_dir):
        return sorted(glob.glob(str(Path(vibe_dir) / "validator_*.json")))

    def test_validate_writes_validator_file_with_inputs_reasoning_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            v = LLMValidator(ScriptedClient('{"verdict":"pass","reason":"all good"}'),
                             logger=logger, config=Config()).validate(
                "CONTEXT WITH TASK AND PAGE", "AGENT HISTORY")
            self.assertTrue(v.passed)

            files = self._validator_files(logger.dir)
            self.assertEqual(len(files), 1)
            self.assertTrue(os.path.basename(files[0]).startswith("validator_"))

            data = json.loads(Path(files[0]).read_text(encoding="utf-8"))  # valid JSON
            # inputs: the richer #57 context is recorded, not a self-claim
            self.assertEqual(data["inputs"]["context"], "CONTEXT WITH TASK AND PAGE")
            self.assertEqual(data["inputs"]["history"], "AGENT HISTORY")
            self.assertNotIn("claim", data["inputs"])
            # private reasoning captured
            self.assertIn("judging", data["reasoning"])
            # verdict
            self.assertTrue(data["verdict"]["passed"])
            self.assertEqual(data["verdict"]["reason"], "all good")
            # timestamp + model/config
            self.assertIn("timestamp", data)
            self.assertEqual(data["model"], Config().model)
            self.assertEqual(data["config"]["temperature"], Config().temperature)

    def test_two_validate_calls_write_two_distinct_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            v = LLMValidator(ScriptedClient('{"verdict":"fail","reason":"nope"}'),
                             logger=logger, config=Config())
            v.validate("ctx", "h1")
            v.validate("ctx", "h2")
            files = self._validator_files(logger.dir)
            self.assertEqual(len(files), 2)
            self.assertNotEqual(files[0], files[1])  # unique guids, no clobber
            histories = {json.loads(Path(f).read_text(encoding="utf-8"))["inputs"]["history"]
                         for f in files}
            self.assertEqual(histories, {"h1", "h2"})

    def test_no_logger_writes_nothing_and_still_validates(self):
        # Back-compat: LLMValidator(client) with no logger works and writes no file.
        with tempfile.TemporaryDirectory() as tmp:
            v = LLMValidator(ScriptedClient('{"verdict":"pass","reason":"ok"}')).validate(
                "ctx", "h")
            self.assertTrue(v.passed)
            self.assertEqual(self._validator_files(Path(tmp) / ".vibe"), [])

    def test_logging_failure_does_not_break_validation(self):
        # An unwritable log dir must not stop the verdict from being returned.
        class ExplodingLogger:
            def log_validator(self, **kwargs):
                raise OSError("disk full")
        v = LLMValidator(ScriptedClient('{"verdict":"pass","reason":"ok"}'),
                         logger=ExplodingLogger(), config=Config()).validate("ctx", "h")
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "ok")

    def test_runlogger_log_validator_swallows_write_errors(self):
        # The logger's own write is guarded too: pointing .vibe/ at an existing
        # FILE makes mkdir() raise, which must be swallowed (no exception).
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            Path(logger.dir).write_text("not a dir", encoding="utf-8")  # block mkdir
            logger.log_validator(context="ctx", history="h", reasoning="r",
                                 passed=True, reason="ok", config=Config())  # must not raise


class ValidateToolTest(unittest.TestCase):
    def test_schema_takes_no_args(self):
        # Issue #57: `validate` takes NO arguments. Its call-schema accepts an empty
        # args object — no properties, nothing required — and `summary` is gone.
        tool = ValidateTool()
        self.assertEqual(tool.name, "validate")
        self.assertEqual(tool.parameters, [])
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "validate")
        args_schema = schema["properties"]["args"]
        self.assertEqual(args_schema.get("properties", {}), {})
        self.assertNotIn("summary", args_schema.get("properties", {}))
        # an empty args object must be schema-valid (no `required` key)
        self.assertNotIn("required", args_schema)

    def test_no_arg_validate_runs(self):
        # A no-arg validate call executes the safe fallback without error.
        result = ValidateTool().run({})
        self.assertTrue(result.ok)


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

        verdict = LLMValidator(RecordingClient()).validate("context", "history")
        # Exactly one model call (single-shot, not a loop) ...
        self.assertEqual(calls["count"], 1)
        # ... driven by the SAME prompt the framework now declares ...
        self.assertEqual(calls["system"], VALIDATOR_SYSTEM)
        self.assertEqual(ValidatorToolset().system_guidance(), calls["system"])
        # ... and the verdict behavior is the parsed pass/fail.
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reason, "ok")


class ValidatorSystemPromptTest(unittest.TestCase):
    """Issue #57: VALIDATOR_SYSTEM must reflect judging-from-context (no self-claim)
    and steer the agent toward elements that actually exist in the page snapshot."""

    def test_reflects_judging_from_snapshot_and_history_no_self_claim(self):
        sys = VALIDATOR_SYSTEM.lower()
        self.assertIn("snapshot", sys)
        self.assertIn("history", sys)
        # explicitly disclaims relying on a self-claim
        self.assertIn("no self-claim", sys)

    def test_steers_toward_real_existing_snapshot_elements(self):
        sys = VALIDATOR_SYSTEM.lower()
        self.assertIn("actually exist", sys)
        # warns against guessed/invented selectors
        self.assertTrue("guessed" in sys or "invented" in sys)


if __name__ == "__main__":
    unittest.main()
