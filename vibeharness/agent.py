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
import re
import threading
from dataclasses import asdict, dataclass, field

from typing import Callable

# Matches the first JSON/Hermes tool-call block in a model response so we can
# extract the free-text preamble the model emitted before it.
_TOOL_BLOCK_RE = re.compile(
    r"<tool_call>|```(?:json)?",
    re.DOTALL,
)

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

    def run(self, task: str, on_turn: Callable[["RunResult"], None] | None = None,
            advice_provider: Callable[[int], "str | None"] | None = None) -> RunResult:
        memory = NarrativeMemory()
        result = RunResult(task=task)
        limit = self._cfg.max_actions_per_turn
        constraint = self._codec.constraint(self._registry, limit)
        # Anti-loop guard (#125): every action ATTEMPTED this run, mapped to whether it
        # succeeded. A small model (esp. single-phase + greedy over a near-static page
        # snapshot) re-emits the exact same action forever — re-filling an already-filled
        # field (success loop) OR retrying an impossible one (failure loop: an invalid/
        # hallucinated ref, or `fill` on a non-input). Neither makes progress, so we do NOT
        # re-run an identical (tool, args); we steer to a different element / advance the
        # form. The steer differs by prior outcome. Navigation tools are exempt
        # (see _action_signature). Iter 6 showed failure loops (click e163 ×12, fill e68
        # ×11) dominated when only successes were deduped — hence both outcomes are tracked.
        attempted: dict[str, bool] = {}
        # Refs the model has ALREADY successfully acted on this run. Surfaced in the steer
        # message (#125 iter 7) so a stuck model is told concretely which targets are done
        # and to pick an UNHANDLED ref from the snapshot, instead of cycling among the same
        # few. Targets the "can't find the next field" failure mode.
        handled_refs: set[str] = set()
        # Escalating block pressure: count how many times any blocked action on the same
        # TARGET has fired. When this reaches 3, we emit a HARD STOP that names the exact
        # alternative call (select_option / click / evaluate) so a low-capability model
        # that ignores soft warnings is forced to see the concrete fix every turn.
        target_block_counts: dict[str, int] = {}

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
                # Inject advisor hint when available (injected by caller via advice_provider).
                if advice_provider is not None:
                    hint = advice_provider(i)
                    if hint:
                        user += (f"\n\n<user_advice>The human user gives you the following "
                                 f"hint: '{hint}'</user_advice>")
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

                # For single-phase models (two_phase=False, e.g. Qwen) the reasoning
                # field is always "". The model's preamble — whatever it wrote before
                # the <tool_call> block — lives in raw_action instead. Record it into
                # the narrative so the model can reference its own prior thinking on
                # the next turn (helps catch "I already did X" confusions).
                if not decision.reasoning:
                    m = _TOOL_BLOCK_RE.search(decision.action_json or "")
                    preamble = (decision.action_json[:m.start()].strip() if m
                                else "").strip()
                    if preamble:
                        memory.record(f"you reasoned: {preamble[:400]}")

                actions, error = self._codec.parse(decision.action_json)
                if error is not None:
                    self._record(turn, Action(None, {}, f"your last response was invalid and "
                                              f"could not be run: {error}.", ok=False), memory)
                else:
                    # Defensive guard: even if a batch slips past the schema cap, only the
                    # first `limit` actions run. Explicitly list the dropped calls so the
                    # model can re-issue them with the SAME tools on the next turn — silent
                    # dropping caused the Country combobox to be re-attempted with fill
                    # instead of select_option in iter3 (13 wasted turns).
                    if limit > 0 and len(actions) > limit:
                        dropped_calls = actions[limit:]
                        actions = actions[:limit]
                        dropped_desc = "; ".join(
                            f"{t}({', '.join(f'{k}={repr(v)[:40]}' for k, v in (a or {}).items())})"
                            for t, a in dropped_calls
                        )
                        self._record(turn, Action(None, {}, (
                            f"ERROR: you emitted {limit + len(dropped_calls)} actions but the "
                            f"per-turn limit is {limit}. The following {len(dropped_calls)} "
                            f"action(s) were NOT executed — re-issue them on your NEXT turn "
                            f"with the SAME tools and exact values: {dropped_desc}"
                        ), ok=False), memory)
                    for tool_name, args in actions:
                        if tool_name == "validate":
                            self._validate(args, turn, memory, result, user)
                            if result.finished:
                                break
                            continue
                        sig = self._action_signature(tool_name, args)
                        if sig is not None and sig in attempted:
                            target = args.get("target", "")
                            value = args.get("text") or args.get("value") or ""
                            if attempted[sig]:
                                obs = (f"WARNING: '{target}' was already successfully set to "
                                       f"'{value}' earlier this run. You are now trying to set "
                                       f"it to '{value}' again — the same value. This is a "
                                       f"no-op: the field already holds that value. If you "
                                       f"intended a different value, call {tool_name} with the "
                                       f"corrected value instead. Otherwise move to the NEXT "
                                       f"unfilled field or click Next/Continue/Submit.")
                            else:
                                # Escalating pressure for failure loops on the same target.
                                count = target_block_counts.get(target, 0) + 1
                                target_block_counts[target] = count
                                if count >= 3:
                                    obs = (
                                        f"HARD STOP: {tool_name}('{target}') has been blocked "
                                        f"{count} times. Text-input tools (fill/type/press_key) "
                                        f"will NEVER work on this element. You are REQUIRED to "
                                        f"use one of these RIGHT NOW:\n"
                                        f"  • select_option(target='{target}', value='{value}') "
                                        f"— if it is a native <select> or custom dropdown\n"
                                        f"  • click(target='{target}') — to open a custom "
                                        f"combobox, then click the matching option from the "
                                        f"updated snapshot\n"
                                        f"  • evaluate(expression='el => el.tagName + \" \" + "
                                        f"(el.getAttribute(\"role\")||el.type)', target='{target}') "
                                        f"— to inspect the element type before choosing\n"
                                        f"Do NOT call {tool_name} on '{target}' again. "
                                        f"Pick one of the three options above NOW."
                                    )
                                else:
                                    obs = (f"WARNING: you already attempted {tool_name} on "
                                           f"'{target}' (value: '{value}') and it FAILED. "
                                           f"That element may be a dropdown, combobox, date "
                                           f"picker, file upload, or other non-text component "
                                           f"that does not accept a direct fill. Try a different "
                                           f"approach: use select_option to choose from a list, "
                                           f"click '{target}' to open it first, use evaluate to "
                                           f"inspect what kind of element it is, or use "
                                           f"press_key after clicking it. "
                                           f"Do NOT call {tool_name} on '{target}' again.")
                            if handled_refs:
                                obs += (" You have already successfully handled these refs: "
                                        f"{', '.join(sorted(handled_refs))} — pick a fillable/"
                                        "clickable ref from the snapshot that is NOT in that list.")
                            self._record(turn, Action(tool_name, args, obs, ok=False), memory)
                            continue
                        action = self._execute(tool_name, args)
                        self._record(turn, action, memory)
                        if sig is not None:
                            attempted[sig] = action.ok
                        if action.ok and isinstance(args.get("target"), str):
                            handled_refs.add(args["target"])
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

    # Only tools whose identical repeat ADVANCES state (each call goes one step further)
    # are EXEMPT from the anti-loop guard. open_browser / goto(same URL) / reload do NOT
    # advance on repeat — they RESET (a fresh blank page, or a reload that wipes a
    # half-filled form), so an identical repeat is a destructive loop and MUST be steered.
    # (#125 iter 9: the model called open_browser 63x because it was exempt; each call
    # reset the page to blank and never reached `goto`.)
    _LOOP_EXEMPT_TOOLS = frozenset({"navigate_back", "navigate_forward"})

    def _action_signature(self, tool_name: str | None, args: dict) -> str | None:
        """A stable signature for an executed action, or ``None`` if this tool is exempt
        from the anti-loop guard. Identical ``(tool, args)`` -> identical signature, so a
        re-emitted duplicate of a previously-successful action can be detected and steered
        rather than blindly re-run (#125)."""
        if not tool_name or tool_name in self._LOOP_EXEMPT_TOOLS:
            return None
        try:
            payload = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = repr(args)
        return f"{tool_name}|{payload}"

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
