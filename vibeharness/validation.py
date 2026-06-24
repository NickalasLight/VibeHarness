"""Validator subagent + the core `validate` tool.

The main agent calls `validate` when it believes it has finished the task. That
hands the original task, the agent's action history, and the agent's claim to a
second VibeThinker instance (the validator) with a dedicated system prompt. The
validator returns a constrained pass/fail verdict with a reason:

  - pass -> the run finishes (final summary = the validator's reason)
  - fail -> the reason is returned to the main agent as the `validate` result so
            it knows what still needs doing, and the loop continues.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .codec import DecodeConstraint
from .config import Config
from .llm import LLMClient, TokenSink
from .tools import Param, Tool, ToolResult
from .toolset import Toolset

VALIDATOR_SYSTEM = """\
You are a strict validation agent. Another agent has been working to complete a \
user's task using tools. Your job is to decide whether it has FULLY and CORRECTLY \
completed that task, judging only from the evidence you are given.

You receive:
- the ORIGINAL TASK the user asked for,
- a plain-English account of every action the agent took and what each returned,
- the agent's own claim that it is finished.

Rules:
- Judge ONLY from the account. Do not assume a step happened unless the account \
shows it succeeding. Treat errors, skipped steps, and unverified claims as incomplete.
- Be strict but fair. If ANY required part of the task is missing, not done, or not \
supported by the account, the verdict is "fail".
- When you fail it, state specifically what is missing or wrong so the agent knows \
exactly what to do next.

Output exactly one JSON object: {"verdict": "pass" or "fail", "reason": "<1-3 sentences>"}.
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str
    reasoning: str = ""   # the validator's private reasoning (kept for the log)


def build_validator_prompt(task: str, history: str, claim: str) -> str:
    return (
        f"# Original task\n{task}\n\n"
        f"# Account of what the agent did\n{history}\n\n"
        f"# The agent's completion claim\n{claim or '(the agent gave no summary)'}\n\n"
        f"# Your verdict\nDecide whether the original task is fully and correctly "
        f"complete. Respond with one JSON object: a verdict of pass or fail and a reason."
    )


class Validator(ABC):
    @abstractmethod
    def validate(self, task: str, history: str, claim: str,
                 on_reason: "TokenSink | None" = None,
                 on_action: "TokenSink | None" = None) -> Verdict:
        ...


class LLMValidator(Validator):
    """Validator backed by an LLM (the same model, a different system prompt)."""

    def __init__(self, client: LLMClient):
        self._client = client

    def validate(self, task: str, history: str, claim: str,
                 on_reason: "TokenSink | None" = None,
                 on_action: "TokenSink | None" = None) -> Verdict:
        user = build_validator_prompt(task, history, claim)
        decision = self._client.decide(
            VALIDATOR_SYSTEM, user, DecodeConstraint(json_schema=VERDICT_SCHEMA),
            on_reason=on_reason, on_action=on_action)
        try:
            data = json.loads(decision.action_json)
        except json.JSONDecodeError:
            return Verdict(False, "validator output could not be parsed; treating as incomplete.",
                           decision.reasoning)
        return Verdict(passed=(data.get("verdict") == "pass"),
                       reason=data.get("reason", ""), reasoning=decision.reasoning)


class ValidateTool(Tool):
    """Core tool (present in every toolset) that triggers the validator. The agent
    loop intercepts it; `run` is only a safe fallback."""
    name = "validate"
    description = (
        "Call ONLY when the task is FULLY done. A separate validator reviews your work: "
        "if it agrees the run ends, otherwise you get feedback on what is missing — fix it "
        "and validate again. Never call speculatively."
    )

    @property
    def parameters(self):
        return [Param("summary", "string",
                      "Brief summary of what you accomplished and why the task is complete.")]

    def run(self, args: dict) -> ToolResult:
        return ToolResult(True, "validation requested.")


class ValidatorToolset(Toolset):
    """The validator declared as a first-class agent type — exactly like the web/fs
    toolsets (issue #31).

    This unifies *declaration only*: the validator's PROMPT and its verdict TOOL are
    now exposed through the same per-agent-type framework (``system_guidance`` from
    #19 + the toolset catalog from #22), so ``--agent validator`` and ``--list-agents``
    discover it like any other agent.

    EXECUTION is intentionally unchanged: validation still runs as a SINGLE-SHOT
    pass/fail verdict via :class:`LLMValidator.validate`, which uses ``VALIDATOR_SYSTEM``
    directly — it is NOT routed through the main agent's multi-turn tool loop. The
    ``system_guidance`` here returns that very same ``VALIDATOR_SYSTEM`` text so the
    prompt lives in one place and the declaration cannot drift from what actually runs.
    """
    name = "validator"
    description = ("Review another agent's completed work and return a strict "
                   "single-shot pass/fail verdict (the `validate` tool).")

    def system_guidance(self) -> str | None:
        # The validator's own system prompt, surfaced through the #19 mechanism.
        # The SAME constant drives LLMValidator.validate, so the framework
        # declaration and the live single-shot execution share one source of truth.
        return VALIDATOR_SYSTEM

    def create_tools(self, config: Config) -> list[Tool]:
        # The verdict tool. `validate` is also the core tool injected into every
        # registry by ToolsetCatalog.build_registry, so declaring it here makes the
        # validator's toolset explicit/consistent with web/fs without changing what
        # tools any registry ends up holding (the catalog de-duplicates by name).
        return [ValidateTool()]
