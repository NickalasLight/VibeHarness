"""The Ralph loop.

Each turn: render the task + narrative -> ask the model for one or more
constrained actions (a JSON array) -> execute them in order -> append a
natural-language observation for each. Repeat until `finish` or the step budget.

Batching several actions in one turn is allowed (the model decides them together,
without seeing intermediate results); a turn that needs a result before deciding
the next move simply emits a single action.

The agent depends only on abstractions: an LLMClient, a ToolRegistry, a
NarrativeMemory and a Reporter.
"""
from __future__ import annotations

import inspect
import itertools
import json
import threading
from dataclasses import asdict, dataclass, field

from typing import Callable

from .codec import ToolCallCodec, get_codec
from .config import Config
from .llm import Decision, LLMClient
from .memory import NarrativeMemory
from .prompt import build_turn_prompt
from .registry import ToolRegistry
from .reporting import NullReporter, Reporter
from .validation import Validator


@dataclass
class Action:
    """One executed tool call within a turn."""
    tool: str | None
    args: dict
    observation: str
    ok: bool
    final: bool = False


@dataclass
class Turn:
    """One model turn: its reasoning, the raw action payload, and the executed actions."""
    index: int
    reasoning: str
    raw_action: str
    actions: list[Action] = field(default_factory=list)


@dataclass
class RunResult:
    task: str
    turns: list[Turn] = field(default_factory=list)
    finished: bool = False
    final_summary: str = ""
    validations: list[dict] = field(default_factory=list)  # each validator verdict
    # Set when the run ended for a reason other than finishing/exhausting the step
    # budget (e.g. a turn exceeded its wall-clock generation budget). Empty otherwise.
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def transcript(self) -> str:
        out = [f"TASK: {self.task}", ""]
        for turn in self.turns:
            out.append(f"--- Turn {turn.index} ---")
            if turn.reasoning.strip():
                out.append(f"reasoning:\n{turn.reasoning.strip()}")
            for a in turn.actions:
                out.append(f"action: {a.tool} {json.dumps(a.args, ensure_ascii=False)}")
                out.append(f"result: {a.observation}")
            out.append("")
        if self.validations:
            out.append("VALIDATIONS:")
            for v in self.validations:
                out.append(f"  turn {v['turn']}: "
                           f"{'PASS' if v['passed'] else 'FAIL'} — {v['reason']}")
            out.append("")
        out.append(f"FINISHED: {self.finished}")
        if self.final_summary:
            out.append(f"SUMMARY: {self.final_summary}")
        return "\n".join(out)


def _accepts_one_positional(fn: Callable) -> bool:
    """True if ``fn`` can be called with exactly one positional argument.

    Used to distinguish a snapshot-budgeting system-prompt provider (takes the
    per-turn user message) from a legacy zero-arg provider, without ever swallowing
    a TypeError that the provider itself might raise.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = 0
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            return True  # *args accepts one positional
    return positional >= 1


class RalphAgent:
    def __init__(self, client: LLMClient, registry: ToolRegistry, system_prompt: str,
                 config: Config, validator: Validator, reporter: Reporter | None = None,
                 system_prompt_provider: Callable[[], str] | None = None,
                 codec: ToolCallCodec | None = None,
                 validator_context_provider: Callable[[str], str] | None = None):
        self._client = client
        self._registry = registry
        self._system = system_prompt
        self._cfg = config
        self._validator = validator
        self._reporter = reporter or NullReporter()
        # Optional per-turn refresh hook. When set, it is called at the start of
        # each turn to regenerate the system prompt (e.g. with a fresh workspace
        # tree); when None, the static system_prompt above is reused every turn.
        self._system_provider = system_prompt_provider
        # Optional per-turn provider of the TOOL-LESS main system prompt fed to the
        # validator (issue #57): the task + workspace + live page snapshot the main
        # agent had, WITHOUT tool descriptions / format instructions. When None, the
        # validator falls back to seeing only the action history (no page context).
        self._validator_context_provider = validator_context_provider
        # Cached arity of the provider (does it accept the per-turn user message?).
        # Resolved lazily on first use so the inspection happens once per run.
        self._provider_wants_user: bool | None = None
        # The tool-call codec owns the action wire format: the decode constraint
        # and how the raw payload is parsed back into (tool, args) pairs.
        self._codec = codec or get_codec("json")

    def run(self, task: str, on_turn: Callable[["RunResult"], None] | None = None) -> RunResult:
        memory = NarrativeMemory()
        result = RunResult(task=task)
        limit = self._cfg.max_actions_per_turn
        constraint = self._codec.constraint(self._registry, limit)

        # max_steps <= 0 means run until validation passes.
        turns = (itertools.count(1) if self._cfg.max_steps <= 0
                 else range(1, self._cfg.max_steps + 1))
        for i in turns:
            self._reporter.turn_start(i)
            # The turn's Turn object is created as soon as we have a decision, but a
            # mid-turn crash can strike before/after that. Track it so the failsafe
            # below can attach the failure to the right (possibly already-appended)
            # turn and never lose the actions that already completed this turn.
            turn: Turn | None = None
            try:
                # Build the per-turn user message FIRST so a system-prompt provider can
                # size the live page snapshot against the FULL message it will share the
                # context window with (issue #43 dynamic snapshot budget). Providers may
                # be zero-arg (legacy: workspace-only refresh) or accept the user message;
                # _build_system bridges both so older wirings keep working unchanged.
                user = build_turn_prompt(task, memory.render(), self._codec.turn_action_hint())
                system = self._build_system(user)
                decision = self._decide(system, user, constraint)
                if decision is None:
                    # The turn exceeded its wall-clock generation budget. Record a
                    # failed turn so the abort is observable, then end gracefully.
                    budget = self._cfg.turn_timeout_seconds
                    reason = (f"turn {i} exceeded the {budget}s generation budget; aborting")
                    turn = Turn(index=i, reasoning="", raw_action="")
                    result.turns.append(turn)
                    self._record(turn, Action(None, {}, reason, ok=False), memory)
                    result.stop_reason = reason
                    if on_turn is not None:
                        on_turn(result)
                    break

                turn = Turn(index=i, reasoning=decision.reasoning, raw_action=decision.action_json)
                result.turns.append(turn)

                actions, error = self._codec.parse(decision.action_json)
                if error is not None:
                    self._record(turn, Action(None, {}, f"your last response was invalid and "
                                              f"could not be run: {error}.", ok=False), memory)
                else:
                    # Defensive guard: even if a batch slips past the schema cap, only the
                    # first `limit` actions run. Record a brief note so the model knows.
                    if limit > 0 and len(actions) > limit:
                        dropped = len(actions) - limit
                        actions = actions[:limit]
                        self._record(turn, Action(None, {}, f"you emitted more than the "
                                                  f"per-turn limit of {limit} actions; only the "
                                                  f"first {limit} were run ({dropped} ignored).",
                                                  ok=False), memory)
                    for tool_name, args in actions:
                        if tool_name == "validate":
                            self._validate(args, turn, memory, result, user)
                            if result.finished:
                                break
                            continue
                        self._record(turn, self._execute(tool_name, args), memory)
            except BaseException as exc:
                # Failsafe: any unexpected mid-turn error (an uncaught tool/validator/
                # decode exception, a KeyboardInterrupt, …) must NOT discard the work
                # already done this turn. Tool-level errors are still caught and
                # recorded as before; this only fires for genuinely unexpected ones.
                # Record the failure on the current turn (creating one if the crash
                # struck before the Turn was appended), flush the partial result via
                # on_turn so it reaches disk, then re-raise so the caller still sees
                # the crash and exit codes are unchanged.
                if turn is None:
                    turn = Turn(index=i, reasoning="", raw_action="")
                    result.turns.append(turn)
                reason = f"turn {i} aborted by an unexpected error: {type(exc).__name__}: {exc}"
                self._record(turn, Action(None, {}, reason, ok=False), memory)
                result.stop_reason = reason
                if on_turn is not None:
                    on_turn(result)
                raise

            if on_turn is not None:          # stream the log after every turn
                on_turn(result)
            if result.finished:
                break

        return result

    def _build_system(self, user: str) -> str:
        """Produce this turn's system prompt.

        With no provider, the static system prompt is reused. With a provider, call
        it to regenerate the prompt fresh (workspace tree, live page snapshot, …).
        The provider may optionally accept the per-turn ``user`` message so it can
        budget the page snapshot against the full message (issue #43); zero-arg
        providers (the legacy workspace-refresh seam) are still supported.
        """
        provider = self._system_provider
        if provider is None:
            return self._system
        # Inspect the provider's arity ONCE so we never confuse a legacy zero-arg
        # provider with one that raised TypeError internally (a bare try/except would).
        if self._provider_wants_user is None:
            self._provider_wants_user = _accepts_one_positional(provider)
        return provider(user) if self._provider_wants_user else provider()

    def _validator_context(self, user: str) -> str:
        """The tool-less main system prompt to hand the validator (issue #57).

        Reuses the same per-turn rendering as the main prompt — task + workspace +
        the already-#43-budgeted live page snapshot — but with the tool descriptions
        and format instructions stripped. The ``user`` message is forwarded so the
        snapshot is budgeted against the same full message. When no provider is wired
        (e.g. unit tests / fs runs), returns "" so the validator simply sees the
        action history with no page context.
        """
        provider = self._validator_context_provider
        if provider is None:
            return ""
        return provider(user)

    # ---- per-turn generation, bounded by a wall-clock budget ----
    def _decide(self, system: str, user: str, constraint) -> Decision | None:
        """Run one turn's blocking decide() call.

        When ``turn_timeout_seconds <= 0`` the call is inline — identical to the
        original behaviour (no thread, no guard). When > 0, decide() (a blocking,
        streaming call) is run in a daemon worker thread and joined with the
        budget. Returns the Decision on success, or ``None`` if the budget was
        exceeded. A blown budget leaves the worker thread running detached; it is a
        daemon so it cannot keep the process alive once the run returns.
        """
        budget = self._cfg.turn_timeout_seconds
        if budget <= 0:
            return self._client.decide(
                system, user, constraint,
                on_reason=self._reporter.reasoning_token,
                on_action=self._reporter.action_token,
            )

        box: dict = {}

        def work() -> None:
            try:
                box["decision"] = self._client.decide(
                    system, user, constraint,
                    on_reason=self._reporter.reasoning_token,
                    on_action=self._reporter.action_token,
                )
            except BaseException as exc:  # surface, don't swallow, generation errors
                box["error"] = exc

        worker = threading.Thread(target=work, name="vibe-turn-decide", daemon=True)
        worker.start()
        worker.join(budget)
        if worker.is_alive():
            return None                  # budget exceeded; abandon the daemon worker
        if "error" in box:
            raise box["error"]
        return box["decision"]

    # ---- validation ----
    def _validate(self, args: dict, turn: Turn, memory: NarrativeMemory,
                  result: RunResult, user: str = "") -> None:
        # `validate` takes NO args (issue #57); the validator no longer reads a
        # self-reported summary. It is fed the SAME context the main agent had — the
        # tool-less main system prompt (task + workspace + live page snapshot) — plus
        # the action history, and judges purely from that.
        self._reporter.note("validating — checking the task against a validator…")
        self._reporter.validator_start()
        context = self._validator_context(user)
        verdict = self._validator.validate(
            context, memory.render(),
            on_reason=self._reporter.validator_reasoning_token,
            on_action=self._reporter.validator_verdict_token,
        )
        result.validations.append({"turn": turn.index, "passed": verdict.passed,
                                   "reason": verdict.reason, "reasoning": verdict.reasoning})
        if verdict.passed:
            obs = f"validation PASSED — {verdict.reason}"
            self._record(turn, Action("validate", args, obs, ok=True, final=True), memory)
            result.finished = True
            result.final_summary = verdict.reason
        else:
            obs = (f"validation FAILED — {verdict.reason} "
                   f"Keep working to address this, then call validate again.")
            self._record(turn, Action("validate", args, obs, ok=False), memory)

    # ---- helpers ----
    def _execute(self, tool_name: str | None, args: dict) -> Action:
        tool = self._registry.get(tool_name) if tool_name else None
        if tool is None:
            # An unknown/invalid tool name (e.g. a removed tool like `snapshot`, or a
            # hallucinated one) must NEVER silently succeed: surface an explicit error
            # back into the loop naming the bad tool and the tools that DO exist, so
            # the model can correct course next turn (issue #51).
            available = ", ".join(self._registry.names())
            return Action(tool_name, args,
                          f"ERROR: '{tool_name}' is not a valid tool — it is unknown or "
                          f"unavailable, so nothing was done. Use only these tools: "
                          f"{available}.", ok=False)
        result = tool.run(args)
        return Action(tool_name, args, result.observation, ok=result.ok, final=result.is_final)

    def _record(self, turn: Turn, action: Action, memory: NarrativeMemory) -> None:
        turn.actions.append(action)
        memory.record(action.observation)
        self._reporter.action_result(action)
