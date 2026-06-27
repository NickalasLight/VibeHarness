"""Per-model live-page snapshot PROSE compaction (issue #218).

The snapshot-prose transform (``snapshot_prose.aria_yaml_to_prose``, wired in
``cli._snapshot_provider``) used to be a single global toggle (``Config.web_snapshot_prose``)
applied for EVERY model. It is structurally worth it for the small local ``qwen3:4b`` (its
32768-token window — #77/#140 — NEEDS the ~2x compaction), but the capable API models
(GLM / DeepSeek, 128K-1M ctx) want the fuller, LOSSLESS raw ARIA snapshot — prose drops
alert/status/log node text, which stalled a live deepseek run (#218).

This is now per-model, built on the existing policy registry (mirrors #179 codec / #201
dedup / #203 toolsets):
  * ``ModelToolPolicy.snapshot_prose`` (True only for qwen3:4b; False for GLM/DeepSeek),
  * resolved by ``config.resolve_model_snapshot_prose`` (spec → registry → global default),
  * with an explicit ``--web-snapshot-prose`` flag / saved setting still winning
    (``cli.resolve_run_snapshot_prose`` / ``explicit_snapshot_prose_override``),
  * RE-resolved on escalation take-over (the ``snapshot_prose_on_escalate`` callback the
    agent invokes in ``_escalate``) so a qwen3:4b(ON) → deepseek(OFF) swap drops the
    compaction mid-run.
"""
from __future__ import annotations

import argparse
import unittest
from unittest import mock

from vibeharness import cli, providers
from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import (Config, ModelSpec, ModelToolPolicy,
                                resolve_model_snapshot_prose)
from vibeharness.registry import ToolRegistry
from vibeharness.tools import Tool, ToolResult

from tests._fakes import FakeLLMClient, FakeValidator


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class CountingClick(Tool):
    name = "click"
    description = "counts clicks"

    def __init__(self):
        self.calls = 0

    @property
    def parameters(self):
        return []

    def run(self, args) -> ToolResult:
        self.calls += 1
        return ToolResult(True, f"you clicked target #{self.calls}")


def _registry() -> ToolRegistry:
    return ToolRegistry([CountingClick()])


# --------------------------------------------------------------------------- #
# 1) Resolver — the per-model policy value
# --------------------------------------------------------------------------- #
class ResolverTest(unittest.TestCase):
    def _spec(self, model):
        return ModelSpec(provider="x", model=model)

    def test_qwen_prose_on(self):
        # Small local model NEEDS the compaction for its 32768-token window.
        self.assertTrue(resolve_model_snapshot_prose(Config(), self._spec("qwen3:4b")))

    def test_api_models_prose_off(self):
        for m in ("glm-4.7", "glm-4.7-flash", "glm-5.2", "deepseek-chat",
                  "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"):
            self.assertFalse(
                resolve_model_snapshot_prose(Config(), self._spec(m)),
                f"{m} must NOT get prose (wants lossless raw ARIA)")

    def test_family_fallbacks_prose_off(self):
        # An unrecognised GLM / DeepSeek variant resolves to its family policy (prose off).
        self.assertFalse(resolve_model_snapshot_prose(Config(), self._spec("glm-9-future")))
        self.assertFalse(resolve_model_snapshot_prose(Config(), self._spec("deepseek-v9")))

    def test_unknown_model_defaults_to_global(self):
        # An unconfigured model inherits the legacy GLOBAL default (the A/B seam), BOTH ways.
        self.assertTrue(resolve_model_snapshot_prose(
            Config(web_snapshot_prose=True), self._spec("mystery-1b")))
        self.assertFalse(resolve_model_snapshot_prose(
            Config(web_snapshot_prose=False), self._spec("mystery-1b")))

    def test_spec_override_wins(self):
        # An explicit spec field beats the registry both ways.
        off = ModelSpec(provider="x", model="qwen3:4b", snapshot_prose=False)
        on = ModelSpec(provider="x", model="deepseek-chat", snapshot_prose=True)
        self.assertFalse(resolve_model_snapshot_prose(Config(), off))
        self.assertTrue(resolve_model_snapshot_prose(Config(), on))

    def test_policy_field_defaults_true(self):
        # The dataclass default is the conservative True (only API entries set it False).
        self.assertTrue(ModelToolPolicy(codec="json", max_actions_per_turn=1).snapshot_prose)


# --------------------------------------------------------------------------- #
# 2) cli.resolve_run_snapshot_prose — explicit flag / saved setting overrides
# --------------------------------------------------------------------------- #
class RunResolverTest(unittest.TestCase):
    def _args(self, **kw):
        ns = argparse.Namespace(web_snapshot_prose=False)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def _spec(self, model):
        return ModelSpec(provider="x", model=model)

    def test_per_model_when_no_explicit(self):
        # No flag, no saved setting -> resolve from the active model.
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertTrue(cli.resolve_run_snapshot_prose(
                self._args(), Config(), self._spec("qwen3:4b")))
            self.assertFalse(cli.resolve_run_snapshot_prose(
                self._args(), Config(), self._spec("deepseek-chat")))

    def test_explicit_flag_forces_on_even_for_deepseek(self):
        # --web-snapshot-prose (store_true) wins over the deepseek prose-OFF policy.
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertTrue(cli.resolve_run_snapshot_prose(
                self._args(web_snapshot_prose=True), Config(), self._spec("deepseek-chat")))

    def test_saved_setting_overrides_policy_both_ways(self):
        # A saved setting pins prose explicitly, beating the per-model policy.
        with mock.patch.object(cli.Settings, "load", return_value={"web_snapshot_prose": False}):
            # qwen would be ON by policy, but the saved OFF wins.
            self.assertFalse(cli.resolve_run_snapshot_prose(
                self._args(), Config(), self._spec("qwen3:4b")))
        with mock.patch.object(cli.Settings, "load", return_value={"web_snapshot_prose": True}):
            # deepseek would be OFF by policy, but the saved ON wins.
            self.assertTrue(cli.resolve_run_snapshot_prose(
                self._args(), Config(), self._spec("deepseek-chat")))

    def test_explicit_override_helper(self):
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertIsNone(cli.explicit_snapshot_prose_override(self._args()))
            self.assertTrue(cli.explicit_snapshot_prose_override(
                self._args(web_snapshot_prose=True)))
        with mock.patch.object(cli.Settings, "load", return_value={"web_snapshot_prose": False}):
            self.assertFalse(cli.explicit_snapshot_prose_override(self._args()))


# --------------------------------------------------------------------------- #
# 3) Escalation take-over re-resolves prose mid-run (qwen3:4b ON -> deepseek OFF)
# --------------------------------------------------------------------------- #
class EscalationTest(unittest.TestCase):
    def test_takeover_to_deepseek_drops_prose(self):
        # Mirror the shared snapshot-provider cell + callback wired in cli._run_locked: the
        # cell starts ON for the qwen base and the callback re-resolves it per active model.
        prose_state = {"on": resolve_model_snapshot_prose(
            Config(), ModelSpec(provider="ollama", model="qwen3:4b"))}
        self.assertTrue(prose_state["on"])  # qwen base -> prose ON

        def _on_escalate(spec):
            prose_state["on"] = resolve_model_snapshot_prose(Config(), spec)

        tool = CountingClick()
        base = FakeLLMClient([{"tool": "noop_missing", "args": {}}])  # stuck loop -> escalate
        swapped = FakeLLMClient([{"tool": "validate", "args": {}}])
        cfg = Config(
            model="qwen3:4b", max_steps=8, escalation_enabled=True,
            escalation_stuck_threshold=3,
            escalation_provider="deepseek", escalation_model="deepseek-chat",
            validation_provider="")
        agent = RalphAgent(base, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("json"),
                           snapshot_prose_on_escalate=_on_escalate)
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            agent.run("go")
        self.assertIs(agent._client, swapped)   # take-over happened
        self.assertFalse(prose_state["on"])     # #218: prose dropped on take-over to deepseek

    def test_explicit_override_survives_escalation(self):
        # With an explicit --web-snapshot-prose, the callback (resolve_run_snapshot_prose) keeps
        # prose ON across the swap even when escalating to a prose-OFF deepseek.
        args = argparse.Namespace(web_snapshot_prose=True)
        prose_state = {"on": True}

        def _on_escalate(spec):
            with mock.patch.object(cli.Settings, "load", return_value={}):
                prose_state["on"] = cli.resolve_run_snapshot_prose(args, Config(), spec)

        tool = CountingClick()
        base = FakeLLMClient([{"tool": "noop_missing", "args": {}}])
        swapped = FakeLLMClient([{"tool": "validate", "args": {}}])
        cfg = Config(
            model="qwen3:4b", max_steps=8, escalation_enabled=True,
            escalation_stuck_threshold=3,
            escalation_provider="deepseek", escalation_model="deepseek-chat",
            validation_provider="")
        agent = RalphAgent(base, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("json"),
                           snapshot_prose_on_escalate=_on_escalate)
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            agent.run("go")
        self.assertIs(agent._client, swapped)
        self.assertTrue(prose_state["on"])      # explicit override wins across the swap


if __name__ == "__main__":
    unittest.main()
