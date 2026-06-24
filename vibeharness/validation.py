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
from .tools import Tool, ToolResult
from .toolset import Toolset

VALIDATOR_SYSTEM = """\
You are a strict validation agent. Another agent has been working to complete a \
user's task using tools. Your job is to decide whether it has FULLY and CORRECTLY \
completed that task, judging only from the evidence you are given.

You receive the EXACT context the working agent had:
- the ORIGINAL TASK the user asked for,
- a snapshot of its workspace and, when it was driving a browser, a live snapshot \
of the CURRENT page (the `# Current page (live snapshot)` section),
- a plain-English account of every action the agent took and what each returned.

There is NO self-claim of completion: do not look for one and do not take any \
agent's word for it. Judge the task ONLY against the page snapshot and the action \
history above.

Rules:
- Judge ONLY from the provided snapshot and action history. Do not assume a step \
happened unless the history shows it succeeding. Treat errors, skipped steps, and \
unverified actions as incomplete.
- Be strict but fair. If ANY required part of the task is missing, not done, or not \
supported by the snapshot/history, the verdict is "fail".
- When you fail it, do NOT merely diagnose what is missing. Your `reason` MUST give \
a CONCRETE, PRIORITIZED next-step recommendation that unblocks the working agent (it \
is a small 3B model that freezes when it cannot decide which element to act on, even \
though the snapshot names it). Make the decision FOR it:
  * Name the SINGLE most important next action AND its EXACT ref (the one thing that \
unblocks progress) first, then 1-2 concrete follow-up steps.
  * Use ONLY real refs/text that ACTUALLY EXIST in the page snapshot you see. Never \
tell it to target guessed or invented selectors. If a control is mislabeled (e.g. a \
search box appears as `dropdown [e78]`), STILL name that ref and say exactly what to \
do with it.
  * Keep it SHORT and IMPERATIVE — direction, not prose. The agent reads your `reason` \
verbatim as its instruction.
  * Example shape: "A consent dialog (e87) is blocking the page — click its Accept \
button [e88] first. Then the search field is [e78] — fill it with the query and click \
Search [e79]. Then click the first result's ref."

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


def build_validator_prompt(main_system_prompt_minus_tools: str, history: str) -> str:
    """Build the validator's USER message (issue #57).

    The validator no longer gets a self-reported summary. Instead it receives the
    SAME context the main agent had: the main agent's latest system prompt WITH the
    tool descriptions / format-instructions stripped (so it keeps the task, the
    workspace, and the `# Current page (live snapshot)` section), followed by the
    action history. ``main_system_prompt_minus_tools`` is produced by
    ``SystemPromptBuilder.build(..., include_tool_guidance=False)``.
    """
    context = (main_system_prompt_minus_tools or "").strip()
    return (
        f"{context}\n\n---\n\n"
        f"# Account of what the agent did\n{history}\n\n"
        f"# Your verdict\nDecide whether the task above is fully and correctly "
        f"complete, judging ONLY from the page snapshot and the action history. "
        f"Respond with one JSON object: a verdict of pass or fail and a reason."
    )


class Validator(ABC):
    @abstractmethod
    def validate(self, main_system_prompt_minus_tools: str, history: str,
                 on_reason: "TokenSink | None" = None,
                 on_action: "TokenSink | None" = None) -> Verdict:
        ...


class LLMValidator(Validator):
    """Validator backed by an LLM (the same model, a different system prompt).

    If a :class:`~vibeharness.runlog.RunLogger` is supplied, every ``validate`` call
    is persisted to its own ``validator_<guid>.json`` in the run's ``.vibe/`` folder
    (issue #47) — inputs, the validator's private reasoning, and the verdict. The
    logger is optional/None-safe so existing usage and tests that construct
    ``LLMValidator(client)`` keep working and simply log nothing.
    """

    def __init__(self, client: LLMClient, logger=None, config: "Config | None" = None):
        self._client = client
        self._logger = logger
        self._config = config

    def validate(self, main_system_prompt_minus_tools: str, history: str,
                 on_reason: "TokenSink | None" = None,
                 on_action: "TokenSink | None" = None) -> Verdict:
        user = build_validator_prompt(main_system_prompt_minus_tools, history)
        decision = self._client.decide(
            VALIDATOR_SYSTEM, user, DecodeConstraint(json_schema=VERDICT_SCHEMA),
            on_reason=on_reason, on_action=on_action)
        try:
            data = json.loads(decision.action_json)
            verdict = Verdict(passed=(data.get("verdict") == "pass"),
                              reason=data.get("reason", ""), reasoning=decision.reasoning)
        except json.JSONDecodeError:
            verdict = Verdict(False, "validator output could not be parsed; treating as incomplete.",
                              decision.reasoning)
        self._log(main_system_prompt_minus_tools, history, verdict)
        return verdict

    def _log(self, context: str, history: str, verdict: Verdict) -> None:
        """Hand this invocation to the run logger (best-effort; never throws).

        Mirrors the #37 diagnostics contract: the logger's own write is already
        guarded, and this wrapper guards the call site too so even a missing/odd
        logger can never break the verdict the agent depends on. Issue #57: the
        logged ``context`` is now the richer tool-less main prompt (task + workspace
        + page snapshot) the validator actually judged from — no self-claim.
        """
        if self._logger is None:
            return
        try:
            self._logger.log_validator(
                context=context, history=history,
                reasoning=verdict.reasoning, passed=verdict.passed,
                reason=verdict.reason, config=self._config)
        except Exception:
            pass


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
        # `validate` takes NO arguments (issue #57). The validator no longer reads a
        # self-reported summary; it judges from the page snapshot + action history it
        # is fed directly. An empty parameter list yields a call-schema whose `args`
        # is an empty object, so a no-arg `validate` call is accepted.
        return []

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
