"""Per-model multi-target click schema (issue #222).

The ``click`` tool gains a ``targets`` array param — a LIST of ``{target, repeat?}`` objects
clicked IN ORDER, each its own count — generalising #206's same-target ``repeat`` to N
different refs. It is gated by a PER-MODEL flag (mirrors #218 snapshot-prose):
  * ``ModelToolPolicy.web_click_multi_target`` (True for GLM/DeepSeek; False for qwen3:4b),
  * resolved by ``config.resolve_model_click_multi_target`` (spec → registry → global default),
  * with an explicit ``--web-click-multi-target`` flag / saved setting still winning
    (``cli.resolve_run_click_multi_target`` / ``explicit_click_multi_target_override``),
  * RE-resolved on escalation take-over (the ``click_multi_target_on_escalate`` callback the
    agent invokes in ``_escalate``) so a qwen3:4b(OFF) → deepseek(ON) swap advertises the
    array mid-run.

Flag OFF ⇒ the advertised schema AND run() behaviour are byte-identical to the single-target
tool. The post-click ``_POST_CLICK_SETTLE_S`` settle is preserved after EVERY physical click
(between repeats of one ref AND between different refs).
"""
from __future__ import annotations

import argparse
import unittest
from unittest import mock

from vibeharness import cli, providers
from vibeharness.agent import RalphAgent
from vibeharness.codec import get_codec
from vibeharness.config import (Config, ModelSpec, ModelToolPolicy,
                                resolve_model_click_multi_target)
from vibeharness.registry import ToolRegistry
from vibeharness.web import ClickTool

from tests._fakes import FakeCli, FakeLLMClient, FakeValidator


# A snapshot whose interactables include the refs the multi-target tests click, so the
# issue-#73 per-click target guard (which captures a fresh snapshot and checks the ref is
# present) lets these calls proceed.
_SNAP = (
    'Page: "Form"\n'
    '  - button "A" [ref=e10] [cursor=pointer]: a\n'
    '  - button "B" [ref=e22] [cursor=pointer]: b\n'
    '  - button "C" [ref=e30] [cursor=pointer]: c\n'
)


def _click(multi: bool = True, output: str = _SNAP) -> "tuple[ClickTool, FakeCli]":
    cli_dbl = FakeCli(ok=True, output=output)
    tool = ClickTool(cli_dbl, observation_limit=2000)
    tool._multi_target = multi
    return tool, cli_dbl


def _clicks(cli_dbl: FakeCli) -> list:
    return [c for c in cli_dbl.calls if c and c[0] == "click"]


# --------------------------------------------------------------------------- #
# 1) Resolver — the per-model policy value
# --------------------------------------------------------------------------- #
class ResolverTest(unittest.TestCase):
    def _spec(self, model):
        return ModelSpec(provider="x", model=model)

    def test_qwen_multi_off(self):
        # Small local model's 3-call accuracy peak (#206/#210) keeps single-target only.
        self.assertFalse(
            resolve_model_click_multi_target(Config(), self._spec("qwen3:4b")))

    def test_api_models_multi_on(self):
        for m in ("glm-4.7", "glm-4.7-flash", "glm-5.2", "deepseek-chat",
                  "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"):
            self.assertTrue(
                resolve_model_click_multi_target(Config(), self._spec(m)),
                f"{m} must get the multi-target schema")

    def test_family_fallbacks_multi_on(self):
        # An unrecognised GLM / DeepSeek variant resolves to its family policy (multi ON).
        self.assertTrue(
            resolve_model_click_multi_target(Config(), self._spec("glm-9-future")))
        self.assertTrue(
            resolve_model_click_multi_target(Config(), self._spec("deepseek-v9")))

    def test_unknown_model_defaults_to_global(self):
        # An unconfigured model inherits the GLOBAL default (the A/B seam), BOTH ways.
        self.assertTrue(resolve_model_click_multi_target(
            Config(web_click_multi_target=True), self._spec("mystery-1b")))
        self.assertFalse(resolve_model_click_multi_target(
            Config(web_click_multi_target=False), self._spec("mystery-1b")))

    def test_spec_override_wins(self):
        on = ModelSpec(provider="x", model="qwen3:4b", web_click_multi_target=True)
        off = ModelSpec(provider="x", model="deepseek-chat", web_click_multi_target=False)
        self.assertTrue(resolve_model_click_multi_target(Config(), on))
        self.assertFalse(resolve_model_click_multi_target(Config(), off))

    def test_policy_field_defaults_false(self):
        # The dataclass default is the conservative False (only API entries set it True).
        self.assertFalse(
            ModelToolPolicy(codec="json", max_actions_per_turn=1).web_click_multi_target)


# --------------------------------------------------------------------------- #
# 2) cli.resolve_run_click_multi_target — explicit flag / saved setting overrides
# --------------------------------------------------------------------------- #
class RunResolverTest(unittest.TestCase):
    def _args(self, **kw):
        ns = argparse.Namespace(web_click_multi_target=False)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def _spec(self, model):
        return ModelSpec(provider="x", model=model)

    def test_per_model_when_no_explicit(self):
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertFalse(cli.resolve_run_click_multi_target(
                self._args(), Config(), self._spec("qwen3:4b")))
            self.assertTrue(cli.resolve_run_click_multi_target(
                self._args(), Config(), self._spec("deepseek-chat")))

    def test_explicit_flag_forces_on_even_for_qwen(self):
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertTrue(cli.resolve_run_click_multi_target(
                self._args(web_click_multi_target=True), Config(), self._spec("qwen3:4b")))

    def test_saved_setting_overrides_policy_both_ways(self):
        with mock.patch.object(cli.Settings, "load",
                               return_value={"web_click_multi_target": True}):
            # qwen would be OFF by policy, but the saved ON wins.
            self.assertTrue(cli.resolve_run_click_multi_target(
                self._args(), Config(), self._spec("qwen3:4b")))
        with mock.patch.object(cli.Settings, "load",
                               return_value={"web_click_multi_target": False}):
            # deepseek would be ON by policy, but the saved OFF wins.
            self.assertFalse(cli.resolve_run_click_multi_target(
                self._args(), Config(), self._spec("deepseek-chat")))

    def test_explicit_override_helper(self):
        with mock.patch.object(cli.Settings, "load", return_value={}):
            self.assertIsNone(cli.explicit_click_multi_target_override(self._args()))
            self.assertTrue(cli.explicit_click_multi_target_override(
                self._args(web_click_multi_target=True)))
        with mock.patch.object(cli.Settings, "load",
                               return_value={"web_click_multi_target": False}):
            self.assertFalse(cli.explicit_click_multi_target_override(self._args()))


# --------------------------------------------------------------------------- #
# 3) Schema — flag OFF byte-identical; flag ON exposes the nested object-array
# --------------------------------------------------------------------------- #
class SchemaTest(unittest.TestCase):
    def test_flag_off_schema_byte_identical(self):
        tool, _ = _click(multi=False)
        self.assertEqual([p.name for p in tool.parameters], ["target", "repeat"])
        schema = tool._args_schema()
        self.assertEqual(set(schema["properties"]), {"target", "repeat"})
        # target stays REQUIRED on the single-target path (pre-#222 behaviour).
        self.assertEqual(schema["required"], ["target"])
        self.assertNotIn("targets", schema["properties"])

    def test_flag_on_exposes_targets_array(self):
        tool, _ = _click(multi=True)
        self.assertIn("targets", [p.name for p in tool.parameters])
        targets = next(p for p in tool.parameters if p.name == "targets")
        self.assertEqual(targets.type, "array")
        self.assertFalse(targets.required)
        items = targets.schema()["items"]
        self.assertEqual(items["type"], "object")
        self.assertEqual(set(items["properties"]), {"target", "repeat"})
        self.assertEqual(items["required"], ["target"])
        self.assertEqual(items["properties"]["repeat"]["default"], 1)
        # The worked example is IN the description (best practice for tool schemas).
        self.assertIn("e10", targets.description)

    def test_json_codec_action_schema_includes_nested_object_array(self):
        # The CRITICAL path: GLM/DeepSeek run the `json` codec; its decode constraint is
        # registry.action_schema(). Assert the nested {target, repeat} object-array is there.
        tool, _ = _click(multi=True)
        reg = ToolRegistry([tool])
        action_schema = get_codec("json").constraint(reg, max_actions=10).json_schema
        click_schema = next(
            opt for opt in action_schema["items"]["oneOf"]
            if opt["properties"]["tool"]["const"] == "click")
        targets = click_schema["properties"]["args"]["properties"]["targets"]
        self.assertEqual(targets["type"], "array")
        self.assertEqual(targets["items"]["type"], "object")
        self.assertEqual(targets["items"]["required"], ["target"])


# --------------------------------------------------------------------------- #
# 4) Behaviour — ordered per-target counts, per-click settle, ceilings, errors
# --------------------------------------------------------------------------- #
class MultiTargetRunTest(unittest.TestCase):
    def test_ordered_per_target_counts_and_settle_count(self):
        # targets=[{e10,8},{e22,2}] ⇒ e10 ×8 THEN e22 ×2, in order, with 10 settles total.
        tool, cli_dbl = _click()
        with mock.patch("vibeharness.web.time.sleep") as slp:
            res = tool.run({"targets": [{"target": "e10", "repeat": 8},
                                        {"target": "e22", "repeat": 2}]})
        self.assertTrue(res.ok)
        self.assertEqual(_clicks(cli_dbl),
                         [["click", "e10"]] * 8 + [["click", "e22"]] * 2)
        # PRESERVE THE DELAY: one 2s settle after EVERY physical click (8 + 2 = 10).
        self.assertEqual(slp.call_args_list, [mock.call(2.0)] * 10)
        self.assertIn("e10×8", res.observation)
        self.assertIn("e22×2", res.observation)

    def test_per_item_default_repeat_is_one(self):
        tool, cli_dbl = _click()
        with mock.patch("vibeharness.web.time.sleep") as slp:
            res = tool.run({"targets": [{"target": "e10"}, {"target": "e22"}]})
        self.assertTrue(res.ok)
        self.assertEqual(_clicks(cli_dbl), [["click", "e10"], ["click", "e22"]])
        self.assertEqual(slp.call_args_list, [mock.call(2.0)] * 2)

    def test_targets_precedence_over_single_target(self):
        # Both target AND targets present ⇒ targets WINS (documented precedence).
        tool, cli_dbl = _click()
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"target": "e30", "repeat": 5,
                            "targets": [{"target": "e10", "repeat": 2}]})
        self.assertTrue(res.ok)
        self.assertEqual(_clicks(cli_dbl), [["click", "e10"]] * 2)  # e30 never clicked

    def test_flag_off_ignores_targets_uses_single_path(self):
        # Flag OFF: a model that emits `targets` is ignored; the single `target` path runs.
        tool, cli_dbl = _click(multi=False)
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"target": "e30", "targets": [{"target": "e10"}]})
        self.assertTrue(res.ok)
        self.assertEqual(_clicks(cli_dbl), [["click", "e30"]])  # single-target path

    def test_stop_early_on_failure_reports_progress(self):
        # e10 ×2 land, then e22 fails on its first click → stop, report which item/iter.
        cli_dbl = FakeCli(ok=True, output=_SNAP)
        orig = cli_dbl.run

        def flaky(*args):
            if args and args[0] == "click" and args[1] == "e22":
                return (False, "Error: element not found")
            return orig(*args)
        cli_dbl.run = flaky
        tool = ClickTool(cli_dbl, observation_limit=2000)
        tool._multi_target = True
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"targets": [{"target": "e10", "repeat": 2},
                                        {"target": "e22", "repeat": 3}]})
        self.assertFalse(res.ok)
        self.assertIn("2 time", res.observation)      # the 2 e10 clicks landed
        self.assertIn("targets[1]", res.observation)  # the failing item index
        self.assertIn("e22", res.observation)

    def test_per_item_repeat_clamped_to_max(self):
        tool, cli_dbl = _click()
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"targets": [{"target": "e10", "repeat": 10_000}]})
        self.assertTrue(res.ok)
        self.assertEqual(len(_clicks(cli_dbl)), ClickTool._MAX_REPEAT)

    def test_list_length_capped(self):
        tool, cli_dbl = _click()
        big = [{"target": "e10"} for _ in range(ClickTool._MAX_TARGETS + 1)]
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"targets": big})
        self.assertFalse(res.ok)
        self.assertEqual(_clicks(cli_dbl), [])  # rejected before any click
        self.assertIn(str(ClickTool._MAX_TARGETS), res.observation)

    def test_total_click_ceiling(self):
        # A few items whose repeats SUM past the total ceiling are rejected up front.
        tool, cli_dbl = _click()
        per = ClickTool._MAX_REPEAT  # 100 each
        n = ClickTool._MAX_TOTAL_CLICKS // per + 1
        targets = [{"target": "e10", "repeat": per} for _ in range(n)]
        with mock.patch("vibeharness.web.time.sleep"):
            res = tool.run({"targets": targets})
        self.assertFalse(res.ok)
        self.assertEqual(_clicks(cli_dbl), [])
        self.assertIn("total", res.observation.lower())

    def test_non_list_targets_rejected(self):
        tool, cli_dbl = _click()
        res = tool.run({"targets": "e10"})
        self.assertFalse(res.ok)
        self.assertIn("list", res.observation)
        self.assertEqual(_clicks(cli_dbl), [])

    def test_empty_targets_rejected(self):
        tool, cli_dbl = _click()
        res = tool.run({"targets": []})
        self.assertFalse(res.ok)
        self.assertIn("EMPTY", res.observation)

    def test_item_missing_target_rejected(self):
        tool, cli_dbl = _click()
        res = tool.run({"targets": [{"repeat": 2}]})
        self.assertFalse(res.ok)
        self.assertIn("target", res.observation)
        self.assertEqual(_clicks(cli_dbl), [])

    def test_item_non_integer_repeat_rejected(self):
        tool, cli_dbl = _click()
        res = tool.run({"targets": [{"target": "e10", "repeat": "lots"}]})
        self.assertFalse(res.ok)
        self.assertIn("targets[0]", res.observation)
        self.assertEqual(_clicks(cli_dbl), [])

    def test_non_dict_item_rejected(self):
        tool, cli_dbl = _click()
        res = tool.run({"targets": ["e10"]})
        self.assertFalse(res.ok)
        self.assertIn("object", res.observation)
        self.assertEqual(_clicks(cli_dbl), [])


# --------------------------------------------------------------------------- #
# 5) Escalation take-over re-resolves the schema mid-run (qwen3:4b OFF → deepseek ON)
# --------------------------------------------------------------------------- #
class EscalationTest(unittest.TestCase):
    def test_takeover_to_deepseek_turns_multi_on(self):
        # Mirror the shared ClickTool flag + callback wired in cli._run_locked: the flag
        # starts OFF for the qwen base and the callback re-resolves it per active model.
        tool = ClickTool(FakeCli(ok=True, output=_SNAP), observation_limit=2000)
        tool._multi_target = resolve_model_click_multi_target(
            Config(), ModelSpec(provider="ollama", model="qwen3:4b"))
        self.assertFalse(tool._multi_target)  # qwen base → OFF

        def _on_escalate(spec):
            tool._multi_target = resolve_model_click_multi_target(Config(), spec)

        base = FakeLLMClient([{"tool": "noop_missing", "args": {}}])  # stuck → escalate
        swapped = FakeLLMClient([{"tool": "validate", "args": {}}])
        cfg = Config(
            model="qwen3:4b", max_steps=8, escalation_enabled=True,
            escalation_stuck_threshold=3,
            escalation_provider="deepseek", escalation_model="deepseek-chat",
            validation_provider="")
        agent = RalphAgent(base, ToolRegistry([tool]), "SYS", cfg,
                           FakeValidator(passed=True), codec=get_codec("json"),
                           click_multi_target_on_escalate=_on_escalate)
        with mock.patch.object(providers, "make_api_client", return_value=swapped):
            agent.run("go")
        self.assertIs(agent._client, swapped)   # take-over happened
        self.assertTrue(tool._multi_target)      # #222: schema turned ON for deepseek


if __name__ == "__main__":
    unittest.main()
