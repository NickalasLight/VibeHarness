"""System-prompt genericity guards (prompt cleanup).

The web guidance and templates were made model- and domain-agnostic: the Qwen-only
``/no_think`` token was removed from the native template, the job-application-specific
web guidance was replaced with a generic "Web Agent" block (no application flow, no
German/locale field mappings, no CV/cover-letter or login policy), and the validator
prompt no longer hard-codes a "small 3B model" assumption (wrong for e.g. GLM).

These tests PIN that cleanup so a future edit cannot silently reintroduce the
non-generic content. They are pure string/structure checks — no model, no browser.
"""
from __future__ import annotations

import unittest

from vibeharness.config import Config
from vibeharness.prompt import _SYSTEM_TEMPLATE_NATIVE, SystemPromptBuilder
from vibeharness.toolset import default_catalog
from vibeharness.validation import VALIDATOR_SYSTEM
from vibeharness.web import WebToolset


def _web_guidance() -> str:
    return WebToolset().system_guidance() or ""


class TestNativeTemplateGeneric(unittest.TestCase):
    def test_no_qwen_no_think_token(self):
        # /no_think is a Qwen-only control token — meaningless/misleading for other
        # models (e.g. GLM, a reasoning model). It must not live in the generic template;
        # thinking is controlled at the payload level instead.
        self.assertNotIn("/no_think", _SYSTEM_TEMPLATE_NATIVE)


class TestWebGuidanceGeneric(unittest.TestCase):
    def setUp(self):
        self.g = _web_guidance()

    def test_is_generic_web_agent_not_job_application(self):
        self.assertIn("# Web Agent", self.g)
        self.assertNotIn("Job Application Agent", self.g)
        self.assertNotIn("Application Flow", self.g)

    def test_no_locale_specific_terms(self):
        # German / locale-specific tokens removed in favour of language-agnostic phrasing.
        for term in ("Weiter", "DD.MM.YYYY"):
            self.assertNotIn(term, self.g)

    def test_no_job_application_field_mappings(self):
        for term in ("candidate profile", "cover_letter", "salary_expectation",
                     "first_name", "date_of_birth"):
            self.assertNotIn(term, self.g)

    def test_keeps_core_browser_mechanics(self):
        # The genuinely useful, generic guidance must survive the cleanup.
        self.assertIn("SNAPSHOT IS GROUND TRUTH", self.g)
        for tool in ("fill", "select_option", "set_spinbutton", "upload", "click", "validate"):
            self.assertIn(tool, self.g)

    def test_refers_to_registered_goto_not_unavailable_open_browser(self):
        # open_browser is NOT in the default web toolset (goto opens the browser
        # automatically); the guidance must not tell the agent to call a tool it lacks.
        self.assertNotIn("open_browser", self.g)
        self.assertIn("goto", self.g)


class TestValidatorPromptGeneric(unittest.TestCase):
    def test_no_model_size_assumption(self):
        # The validator must be model-agnostic; the old "small 3B model that freezes"
        # framing is wrong for non-3B models (e.g. GLM).
        self.assertNotIn("3B", VALIDATOR_SYSTEM)

    def test_keeps_strict_verdict_contract(self):
        self.assertIn("pass", VALIDATOR_SYSTEM)
        self.assertIn("fail", VALIDATOR_SYSTEM)


class TestAssembledWebPromptGeneric(unittest.TestCase):
    def test_full_native_web_prompt_has_no_residual_specifics(self):
        cfg = Config()
        catalog = default_catalog()
        toolsets = catalog.select(["web"])
        registry = catalog.build_registry(toolsets, cfg)
        builder = SystemPromptBuilder(
            registry, cfg.max_actions_per_turn,
            guidance=SystemPromptBuilder.assemble_guidance(toolsets))
        prompt = builder.build(native_tools=True)
        for term in ("/no_think", "Job Application Agent", "Weiter", "candidate profile"):
            self.assertNotIn(term, prompt)


if __name__ == "__main__":
    unittest.main()
