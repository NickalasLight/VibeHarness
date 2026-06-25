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
import time
from dataclasses import asdict, dataclass, field, replace as _dc_replace

from typing import Callable

# Matches the first JSON/Hermes tool-call block in a model response so we can
# extract the free-text preamble the model emitted before it.
_TOOL_BLOCK_RE = re.compile(
    r"<tool_call>|```(?:json)?",
    re.DOTALL,
)

from .codec import ToolCallCodec, get_codec
from .config import Config
from .escalation import StuckDetector
from .llm import Decision, LLMClient
from .memory import NarrativeMemory
from .prompt import build_turn_prompt
from .registry import ToolRegistry
from .reporting import NullReporter, Reporter
from .snapshot_budget import estimate_tokens, input_budget_tokens
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
                 validator_context_provider: Callable[[str], str] | None = None,
                 raw_snapshot_provider: Callable[[], str] | None = None):
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
        # Zero-arg callable that captures the live page snapshot after tool execution.
        # When None (fs-only runs / unit tests), no post-turn snapshot is recorded.
        self._raw_snapshot_provider = raw_snapshot_provider
        # Cached arity of the system_prompt_provider (does it take the user message?).
        # Resolved lazily on first use (see _build_system).
        self._provider_wants_user: bool | None = None
        # The tool-call codec owns the action wire format: the decode constraint
        # and how the raw payload is parsed back into (tool, args) pairs.
        self._codec = codec or get_codec("json")
        # NATIVE stateful tool calling (issue #129/#130/#131). Active only when (a) the
        # run opts in (config.native_tools), (b) the model is single-phase (the native
        # path is for the non-thinking base agent, not VibeThinker's two-phase <think>
        # flow), and (c) the codec actually speaks native tools (codec.tools() non-None —
        # only ``hermes`` today). Otherwise we use the legacy single-message decide(),
        # so json/xml/etc codecs and VibeThinker are completely unaffected.
        self._native = bool(
            getattr(config, "native_tools", False)
            and not config.two_phase
            and self._codec.tools(registry) is not None
        )
        # The native enveloped tool schemas, computed once (stateless). Sent in every
        # native /api/chat request's ``tools:`` field so Ollama applies the model's own
        # trained tool template. None when the native path is inactive.
        self._tools = self._codec.tools(registry) if self._native else None

    def run(self, task: str, on_turn: Callable[["RunResult"], None] | None = None,
            advice_provider: Callable[[int], "str | None"] | None = None) -> RunResult:
        memory = NarrativeMemory()
        # Stateful multi-turn message history (issue #129/#130/#131): the REAL
        # user/assistant/tool messages exchanged this run, WITHOUT the system message
        # (which is regenerated fresh each turn — it carries the live page snapshot — and
        # prepended at request time). Empty and unused on the legacy single-message path.
        # NarrativeMemory is kept in parallel: the advisor and the human-readable
        # transcript read its prose; this list is the model's transport-level memory.
        chat_history: list[dict] = []
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
        # SOFT-REPEAT counter (iter-1): how many times each identical click/upload signature
        # has been re-emitted. A click/upload is allowed to repeat (re-click Continue after a
        # validation block; re-open a combobox), but a successful-yet-pointless repeat (e.g.
        # clicking a HEADING that does nothing) would otherwise loop forever now that clicks
        # are not no-op-blocked. After _SOFT_REPEAT_LIMIT identical repeats we steer instead.
        soft_repeat_counts: dict[str, int] = {}
        detector = StuckDetector(self._cfg.escalation_stuck_threshold)

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
                # In NATIVE mode the real chat_history carries past turns, so the user
                # message omits the prose narrative (which would duplicate the history and
                # waste the window); it keeps the task reminder + action hint. In legacy
                # mode the narrative is embedded as before.
                if self._native:
                    user = build_turn_prompt(task, "", self._codec.turn_action_hint())
                else:
                    user = build_turn_prompt(task, memory.render(),
                                             self._codec.turn_action_hint())
                # Inject advisor hint when available (injected by caller via advice_provider).
                if advice_provider is not None:
                    hint = advice_provider(i)
                    if hint:
                        user += (f"\n\n<user_advice>The human user gives you the following "
                                 f"hint: '{hint}'</user_advice>")
                system = self._build_system(user)
                if self._native:
                    decision = self._decide_chat(system, chat_history, user, constraint)
                else:
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

                # Prefer STRUCTURED tool_calls when Ollama parsed them (native path); the
                # 3B model usually returns none and the call arrives as text, so fall back
                # to the codec's tolerant parse() of the content. Either way we get
                # (name, args) pairs and a possible error string.
                if decision.tool_calls:
                    actions = self._codec.parse_tool_calls(list(decision.tool_calls))
                    error = None if actions else "structured tool_calls were empty/invalid"
                    if error:  # fall back to parsing the text content
                        actions, error = self._codec.parse(decision.action_json)
                else:
                    actions, error = self._codec.parse(decision.action_json)
                if error is not None:
                    self._record(turn, Action(None, {}, f"your last response was invalid and "
                                              f"could not be run: {error}.", ok=False), memory)
                else:
                    # Defensive guard: if the model emits more tool calls than the per-turn
                    # limit, silently keep only the first `limit` and discard the rest.
                    # The excess calls are also stripped from `decision` so chat history
                    # appears as if the model only ever requested the executed calls —
                    # no error is surfaced, no "re-issue" noise added to the context.
                    if limit > 0 and len(actions) > limit:
                        actions = actions[:limit]
                        if decision.tool_calls:
                            decision = _dc_replace(
                                decision, tool_calls=decision.tool_calls[:limit])
                        elif decision.action_json:
                            # Text-based path: strip excess <tool_call> blocks from content.
                            _tc_blocks = re.findall(
                                r"<tool_call>[\s\S]*?</tool_call>", decision.action_json)
                            if len(_tc_blocks) > limit:
                                _end = decision.action_json.rindex(_tc_blocks[limit - 1]) + len(
                                    _tc_blocks[limit - 1])
                                decision = _dc_replace(
                                    decision, action_json=decision.action_json[:_end])
                    for _ti, (tool_name, args) in enumerate(actions):
                        if _ti > 0:
                            time.sleep(1)  # 1-second safety gap between batched calls
                        if tool_name == "validate":
                            self._validate(args, turn, memory, result, user,
                                           detector=detector)
                            if result.finished:
                                break
                            continue
                        sig = self._action_signature(tool_name, args)
                        # Stuck detection fires on EVERY occurrence of this (tool, args),
                        # including early-continue paths below, so we record BEFORE any
                        # dedup guards can short-circuit the loop.
                        if detector.record(tool_name, args):
                            self._escalate(
                                detector,
                                f"stuck after {self._cfg.escalation_stuck_threshold} "
                                f"identical '{tool_name}' calls",
                            )
                        # SOFT-REPEAT (iter-1 fix): click/upload are not value-sets. A repeat
                        # is legitimate (re-click Continue after a validation block; re-open a
                        # combobox; retry an upload once the chooser is ready). Let them RUN on
                        # repeat — but BOUND it: after _SOFT_REPEAT_LIMIT identical repeats,
                        # steer instead, so a successful-yet-pointless click (e.g. on a heading
                        # that does nothing) cannot loop forever. This keeps the page-1 deadlock
                        # fix (re-click Continue) while preventing the heading-click loop.
                        if (sig is not None and sig in attempted
                                and tool_name in self._SOFT_REPEAT_TOOLS):
                            tgt = args.get("target", "")
                            reps = soft_repeat_counts.get(sig, 0)
                            if reps < self._SOFT_REPEAT_LIMIT:
                                soft_repeat_counts[sig] = reps + 1
                                action = self._execute(tool_name, args)
                                self._record(turn, action, memory)
                                attempted[sig] = action.ok
                                if not action.ok and tgt:
                                    target_block_counts[tgt] = target_block_counts.get(tgt, 0) + 1
                                elif action.ok and isinstance(tgt, str) and tgt:
                                    handled_refs.add(tgt)
                                continue
                            # Exceeded the repeat budget: fall through to the steer message
                            # so the model is told this exact click is going nowhere.
                        if sig is not None and sig in attempted:
                            target = args.get("target", "")
                            value = args.get("text") or args.get("value") or ""
                            if attempted[sig]:
                                if tool_name in ("goto", "reload", "open_browser"):
                                    url = args.get("url", "")
                                    obs = (f"WARNING: you already navigated to '{url}' "
                                           f"successfully. The page is already loaded — "
                                           f"READ the current page snapshot and interact with "
                                           f"the elements shown there. Do NOT navigate again.")
                                else:
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

                # After ALL tool calls finish: wait 1 s then capture ONE snapshot and
                # record it as the final tool observation for this turn. The model will
                # see it as a role:tool message labelled "page_snapshot" that says what
                # the page looks like NOW — after every action has taken effect.
                # Only the LATEST snapshot is kept in history (earlier ones are evicted
                # before committing so they never accumulate and contradict each other).
                if self._raw_snapshot_provider is not None and not result.finished:
                    time.sleep(1)
                    snap = self._raw_snapshot_provider()
                    obs = (
                        "## Latest page state — this is what the page looks like NOW "
                        "after all your actions this turn\n\n" + snap
                        if snap else
                        "## Latest page state — the page is currently blank or still loading. "
                        "There are NO interactive elements available right now. "
                        "Try navigating to the target URL again or wait and retry."
                    )
                    self._record(turn, Action("page_snapshot", {}, obs, ok=True), memory)

                # NATIVE stateful history (issue #129/#130/#131): commit THIS turn to the
                # real message history so the next turn's request replays it verbatim.
                # We append the user turn, the assistant response, and one role:"tool"
                # message per executed action (its observation) — exactly the shape
                # Ollama's chat template feeds back via <tool_response>. Done here, after
                # the action loop, so every observation recorded on the turn is captured.
                # Then FIFO-evict the oldest non-system messages to stay within the token
                # budget. Skipped on the legacy path (chat_history stays empty/unused).
                if self._native:
                    # Evict any previous page_snapshot observation before committing the
                    # new turn so only the LATEST snapshot is visible in history.
                    self._evict_old_page_snapshot(chat_history)
                    self._commit_turn_to_history(chat_history, user, decision, turn)
                    self._evict_history(chat_history, system, user)
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

        # Log API usage if escalation happened (self._client was swapped to ApiLLMClient)
        try:
            from .api_llm import ApiLLMClient as _ApiLLMClient
            if isinstance(self._client, _ApiLLMClient):
                usage = self._client.usage_summary()
                self._reporter.note(
                    f"[API_USAGE] tokens_in={usage['tokens_in']} "
                    f"tokens_out={usage['tokens_out']} "
                    f"estimated_cost_usd={usage['estimated_cost_usd']:.6f}"
                )
        except Exception:
            pass

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

    def _evict_old_page_snapshot(self, chat_history: list[dict]) -> None:
        """Remove the page_snapshot <tool_response> block from all prior user messages.

        Only the snapshot from the CURRENT turn (about to be committed) should be visible.
        Stale snapshots from earlier turns contradict the current page state and waste
        the context window. The new snapshot will be added by _commit_turn_to_history.

        After the batched-tool-result change (#151), page_snapshot observations live inside
        a batched role:user message as a <tool_response> block rather than as standalone
        role:tool messages. We strip that block from older entries by regex."""
        import re
        snapshot_pattern = re.compile(
            r"\n?<tool_response>\n## Latest page state.*?</tool_response>",
            re.DOTALL
        )
        for m in chat_history:
            if m.get("role") != "user":
                continue
            content = m.get("content") or ""
            if "## Latest page state" not in content:
                continue
            m["content"] = snapshot_pattern.sub("", content).strip()

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

    def _decide_chat(self, system: str, chat_history: list[dict], user: str,
                     constraint) -> Decision | None:
        """Run one turn's NATIVE stateful decide_chat() call (issue #129/#130/#131).

        Assembles the request messages as ``[system] + chat_history + [current user]``
        and calls the client's native ``decide_chat`` with the enveloped tool schemas.
        Mirrors :meth:`_decide` exactly for the wall-clock budget: inline when
        ``turn_timeout_seconds <= 0``, else a joined daemon worker that returns ``None``
        on overrun. The system message is rebuilt EACH turn (it carries the live page
        snapshot), so it is sent fresh and is never stored in ``chat_history``."""
        messages = [{"role": "system", "content": system}] + list(chat_history) + [
            {"role": "user", "content": user}]

        def call() -> Decision:
            return self._client.decide_chat(
                messages, self._tools, constraint,
                on_reason=self._reporter.reasoning_token,
                on_action=self._reporter.action_token,
            )

        budget = self._cfg.turn_timeout_seconds
        if budget <= 0:
            return call()
        box: dict = {}

        def work() -> None:
            try:
                box["decision"] = call()
            except BaseException as exc:
                box["error"] = exc

        worker = threading.Thread(target=work, name="vibe-turn-decide", daemon=True)
        worker.start()
        worker.join(budget)
        if worker.is_alive():
            return None
        if "error" in box:
            raise box["error"]
        return box["decision"]

    # ---- native stateful chat history (issue #129/#130/#131) ----
    def _commit_turn_to_history(self, chat_history: list[dict], user: str,
                                decision: "Decision", turn: "Turn") -> None:
        """Append this completed turn to the stateful message history.

        Records, in order:
          * the current ``user`` turn message,
          * the assistant message — carrying the STRUCTURED ``tool_calls`` when Ollama
            returned them, else the raw response content (the common case for this 3B
            model, where the call came back as text),
          * a single batched ``role: "user"`` message containing ALL action observations,
            each pre-wrapped in ``<tool_response>...</tool_response>`` tags (see below).

        Tool observations are emitted even for error/blocked actions (their observation is
        the steer/error text) so the model SEES the consequence of its last move in the
        history, not just in a one-off injection.

        BATCHED TOOL RESULTS (#151 — Qwen3 training format): Qwen3's HuggingFace chat
        template batches ALL consecutive tool responses for a turn into a SINGLE
        ``<|im_start|>user`` block, each wrapped in ``<tool_response>...</tool_response>``.
        Ollama's modelfile template handles ``role: "tool"`` by emitting SEPARATE
        ``<|im_start|>user`` blocks per message — one per tool result — which diverges from
        training. To match the training format exactly we pre-wrap every observation in
        ``<tool_response>`` tags ourselves and send a SINGLE ``role: "user"`` message.
        Because we send ``role: "user"``, Ollama passes the content through as-is without
        adding extra wrapping."""
        chat_history.append({"role": "user", "content": user})
        assistant: dict = {"role": "assistant"}
        if decision.tool_calls:
            assistant["content"] = ""
            assistant["tool_calls"] = [
                tc for tc in decision.tool_calls if isinstance(tc, dict)]
        else:
            assistant["content"] = decision.action_json or ""
        chat_history.append(assistant)
        # Qwen3 training format: batch ALL tool responses into ONE user message,
        # each wrapped in <tool_response>...</tool_response>. The HF chat template
        # emits a single <|im_start|>user block for all consecutive tool results.
        # We pre-wrap and send as role:user so Ollama doesn't emit separate blocks.
        if turn.actions:
            responses = "\n".join(
                f"<tool_response>\n{action.observation or ''}\n</tool_response>"
                for action in turn.actions
            )
            chat_history.append({"role": "user", "content": responses})

    def _evict_history(self, chat_history: list[dict], system: str, user: str) -> None:
        """FIFO-evict the OLDEST non-system messages until the history fits the budget.

        Two bounds, both applied (whichever is tighter wins):
          * a TOKEN budget — the input window (``num_ctx`` minus the output reservation
            and safety margin, via :func:`input_budget_tokens`) minus what the freshly
            rebuilt system + the next user message will cost; the remainder is what the
            history may occupy. Tokens are estimated with the same conservative
            chars-per-token heuristic the snapshot budget uses (#43), so the two agree.
          * an optional fixed message cap (``chat_history_max_turns``, 0 = off).

        The system message is regenerated each turn and is NEVER evicted (it is not in
        ``chat_history``). We always keep at least the most recent message so a single
        oversized turn cannot empty the history entirely."""
        cfg = self._cfg
        cpt = cfg.snapshot_chars_per_token
        in_budget = input_budget_tokens(cfg)
        # Reserve room for the system + the user message that will accompany the history
        # on the NEXT request (the current user turn is already in chat_history).
        fixed = estimate_tokens(system, cpt) + estimate_tokens(user, cpt)
        history_budget = max(0, in_budget - fixed)

        def msg_tokens(m: dict) -> int:
            text = m.get("content") or ""
            for tc in m.get("tool_calls", []) or []:
                text += json.dumps(tc, ensure_ascii=False)
            return estimate_tokens(text, cpt)

        # Optional fixed cap first (coarse), then the token budget.
        cap = cfg.chat_history_max_turns
        if cap and cap > 0:
            while len(chat_history) > cap:
                chat_history.pop(0)
        total = sum(msg_tokens(m) for m in chat_history)
        while total > history_budget and len(chat_history) > 1:
            total -= msg_tokens(chat_history.pop(0))

    # ---- validation ----
    def _validate(self, args: dict, turn: Turn, memory: NarrativeMemory,
                  result: RunResult, user: str = "",
                  detector: "StuckDetector | None" = None) -> None:
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
            if (detector is not None and self._cfg.escalation_on_premature_validate
                    and detector.record_premature_validate()):
                self._escalate(detector, "premature validate (validator rejected the claim)")

    # ---- escalation ----
    def _escalate(self, detector: StuckDetector, why: str) -> None:
        """Swap self._client to the configured API model — same browser, same session.

        At most once per run (detector.escalated guards re-entry). A no-op with a
        warning log when escalation is disabled or the provider key is absent, so the
        run degrades gracefully rather than crashing.
        """
        if detector.escalated or not self._cfg.escalation_enabled:
            return
        detector.escalated = True
        try:
            from .providers import make_api_client
            new_client = make_api_client(
                self._cfg.escalation_provider,
                self._cfg.escalation_model or None,
            )
            self._client = new_client
            self._reporter.note(
                f"[ESCALATION] {why} — switching to "
                f"{self._cfg.escalation_provider}:{self._cfg.escalation_model}"
            )
        except Exception as exc:
            self._reporter.note(
                f"[ESCALATION] {why} — escalation disabled: {exc}"
            )

    # Only tools whose identical repeat ADVANCES state (each call goes one step further)
    # are EXEMPT from the anti-loop guard.
    _LOOP_EXEMPT_TOOLS = frozenset({"navigate_back", "navigate_forward"})

    # Tools whose repetition is NOT a value-set no-op and must NOT be hard-blocked on the
    # second attempt. Includes navigation tools (goto, reload, open_browser): after a page
    # goes blank or a session dies the model MUST be allowed to re-navigate. We allow up to
    # _SOFT_REPEAT_LIMIT repeats before steering — enough to recover without infinite loops.
    # Also includes click/upload as before (re-clicking Continue after validation failure is
    # the correct recovery; upload has no value to compare).
    _SOFT_REPEAT_TOOLS = frozenset({"click", "upload", "goto", "reload", "open_browser"})
    # Max identical re-runs allowed for a soft-repeat tool before it is steered. Raised
    # 2 -> 12 (iter-2): a NUMBER-STEPPER (the spinbutton +/- buttons on the Experience step)
    # CANNOT be filled (Playwright: "Element is not an <input>") — the ONLY way to set
    # "9 years" is to click the Increase button 9 times, so a tight limit of 2 made those
    # scored fields unreachable (iter-2: the model looped clicking e630/e636 and never got
    # past 2). 12 lets a realistic stepper value be reached (and matches max_actions_per_turn
    # so a full increment batch can run in one turn) while still bounding a genuinely useless
    # click loop (e.g. clicking a heading) to 12 reps — well within a run. True FAILURE loops
    # (ok=False, e.g. re-clicking an invalid ref) are caught separately and faster by the
    # escalating target_block_counts HARD STOP below, independent of this limit.
    _SOFT_REPEAT_LIMIT = 12

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
