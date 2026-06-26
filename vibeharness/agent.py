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

# ISSUE #183: a <think>…</think> reasoning trace (closed, or budget-truncated and left
# open) must never be persisted into the stateful chat history — only the action is.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove any <think>…</think> reasoning trace from assistant content so it is not
    replayed into the model's stateful history. Handles a budget-truncated trace with an
    unclosed <think> by dropping everything from that tag onward."""
    out = _THINK_BLOCK_RE.sub("", text or "")
    m = _THINK_OPEN_RE.search(out)
    if m and not re.search(r"</think\s*>", out, re.IGNORECASE):
        out = out[:m.start()]
    return out.strip()

from .codec import ToolCallCodec, get_codec
from .config import Config
from .escalation import StuckDetector
from .llm import Decision, LLMClient
from .memory import NarrativeMemory
from .prompt import build_turn_prompt
from .registry import ToolRegistry
from .reporting import NullReporter, Reporter
from .snapshot_budget import fit_chat_history, BudgetFitReport
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
    # Context-budget events (issues #170/#172/#173): one entry every time the assembled
    # request would exceed num_ctx and the harness evicted history and/or truncated the
    # live page snapshot to fit. Persisted in the run log so an agent that "went blind"
    # is never a silent budgeting event. Each entry mirrors the loud stdout line.
    context_events: list[dict] = field(default_factory=list)
    # Escalation events (issue #191): one entry every time mid-run escalation is
    # ATTEMPTED — success (the stronger API model took over) OR failure (the escalation
    # provider key is missing/unreachable). Persisted in the run JSON/.md, reusing the
    # #180 ``context_events`` mechanism, so a non-functional escalation can never
    # masquerade as "the model just looped" (the #191 root cause).
    escalation_events: list[dict] = field(default_factory=list)

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
        if self.escalation_events:
            out.append("ESCALATION:")
            for e in self.escalation_events:
                out.append(f"  {e.get('message', '')}")
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
                 raw_snapshot_provider: Callable[[], str] | None = None,
                 turn_input_logger: "Callable[[int, str, list, str, float], None] | None" = None,
                 turn_output_logger: "Callable[[int, str, str], None] | None" = None,
                 escalation_system_prompt_provider: Callable[[str], str] | None = None):
        self._client = client
        self._registry = registry
        self._system = system_prompt
        self._cfg = config
        # ISSUE #193 — the model spec currently driving the run, used to size the input/
        # snapshot budget against the ACTIVE model's per-model context window. Starts at the
        # base role's spec and is updated by ``_escalate`` when the run switches to the API
        # model, so the budget tracks the live client (e.g. DeepSeek/GLM's large window
        # instead of qwen3:4b's GPU-pinned 32768) rather than a flat num_ctx.
        from .config import resolve_role_spec
        try:
            self._active_spec = resolve_role_spec(config, "base")
        except Exception:
            self._active_spec = None
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
        # Optional per-turn IO logging hooks (issue #170/#171). Split into two so the
        # request payload is persisted BEFORE the (blocking) model call — a crash during
        # generation then still leaves the crashing turn's exact input on disk:
        #   turn_input_logger(turn, system, messages, user, chars_per_token) -> None
        #   turn_output_logger(turn, reasoning, action) -> None
        self._turn_input_logger = turn_input_logger
        self._turn_output_logger = turn_output_logger
        # Cached arity of the system_prompt_provider (does it take the user message?).
        # Resolved lazily on first use (see _build_system).
        self._provider_wants_user: bool | None = None
        # The tool-call codec owns the action wire format: the decode constraint
        # and how the raw payload is parsed back into (tool, args) pairs.
        self._codec = codec or get_codec("json")
        # The per-turn tool-call cap actually in force. Starts at the run's configured
        # cap (the BASE model's, resolved by the CLI) and is re-pointed at the ESCALATOR
        # model's cap on take-over (issue #191) via ``resolve_model_limit``.
        self._max_actions = config.max_actions_per_turn
        # Optional escalation-only system-prompt provider (issue #191). Built native-OFF
        # (so the `# Tools` block + JSON format instructions ARE present) and bound to the
        # escalator's json codec, it is swapped in by ``_escalate`` when escalating to a
        # single-shot API model. None on the fs/test path (a static system prompt is used).
        self._escalation_system_provider = escalation_system_prompt_provider
        # NATIVE stateful tool calling (issue #129/#130/#131). Active only when (a) the
        # run opts in (config.native_tools), (b) the model is single-phase (the native
        # path is for the non-thinking base agent, not VibeThinker's two-phase <think>
        # flow), and (c) the codec actually speaks native tools (codec.tools() non-None —
        # only ``hermes`` today). Otherwise we use the legacy single-message decide(),
        # so json/xml/etc codecs and VibeThinker are completely unaffected.
        # The client capability is the final gate (issue #163): a non-native client (the
        # single-shot API client) is ALWAYS driven via _decide, never decide_chat, even if
        # the codec speaks native tools — the CLI separately auto-degrades the codec to a
        # constrained-JSON one for such clients (see clients.select_execution_codec). The
        # test doubles that implement decide_chat report native via the base default, so the
        # existing native-path tests are unaffected.
        self._native = bool(
            getattr(config, "native_tools", False)
            and not config.two_phase
            and self._codec.tools(registry) is not None
            and self._client.supports_native_tools()
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
            # Recompute the per-turn cap + decode constraint from the LIVE codec/cap each
            # turn: mid-run escalation (issue #191) re-points self._codec and
            # self._max_actions at the ESCALATOR model's per-model policy, and that switch
            # must take effect on the turns AFTER the swap. On a run with no escalation this
            # is identical to computing them once up front.
            limit = self._max_actions
            constraint = self._codec.constraint(self._registry, limit)
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
                # NATIVE stateful budget guard (issues #170/#172/#173). BEFORE the request
                # is assembled/logged/sent, ensure [system] + chat_history + [user] fits
                # num_ctx: evict the OLDEST history first, and truncate the live page
                # snapshot ONLY as a last resort (the snapshot is the agent's eyes). Any
                # action is logged LOUDLY (never a silent "the agent went blind"). Done
                # here so the input dump below records the ACTUAL request that is sent.
                if self._native:
                    self._fit_request_to_context(chat_history, system, user, result, i)
                # Persist the EXACT request BEFORE the blocking model call, so a crash
                # DURING generation still leaves this turn's full input (incl. the
                # accumulated history and its token size) on disk (#170/#171). Legacy
                # single-message path has no stateful history, so log an empty list.
                if self._turn_input_logger is not None:
                    try:
                        self._turn_input_logger(
                            i, system, list(chat_history) if self._native else [], user,
                            self._cfg.snapshot_chars_per_token)
                    except Exception:
                        pass
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

                if self._turn_output_logger is not None:
                    try:
                        self._turn_output_logger(
                            i, decision.reasoning or "", decision.action_json or "")
                    except Exception:
                        pass
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
                    # CONSECUTIVE-DUPLICATE filter: if the model emits the same action
                    # back-to-back — either within this turn's batch OR as the first
                    # action of this turn matching the last action of the prior turn —
                    # execute only the first occurrence. The duplicate is silently
                    # dropped from decision.tool_calls / decision.action_json so chat
                    # history records only what was actually executed, as if the second
                    # call never happened.
                    _dedup_actions: list[tuple[str, dict]] = []
                    _dedup_kept: list[int] = []
                    _prev_sig: str | None = None
                    for _di, (_dn, _da) in enumerate(actions):
                        _ds = self._action_signature(_dn, _da)
                        if _ds is not None and _ds == _prev_sig:
                            self._reporter.note(
                                f"[DEDUP] dropping consecutive duplicate {_dn}")
                            continue
                        _dedup_actions.append((_dn, _da))
                        _dedup_kept.append(_di)
                        if _ds is not None:
                            _prev_sig = _ds
                    if len(_dedup_actions) < len(actions):
                        actions = _dedup_actions
                        if decision.tool_calls:
                            decision = _dc_replace(
                                decision,
                                tool_calls=[decision.tool_calls[j] for j in _dedup_kept])
                        elif decision.action_json:
                            _tc_all = re.findall(
                                r"<tool_call>[\s\S]*?</tool_call>", decision.action_json)
                            if _tc_all:
                                _tc_kept = [_tc_all[j] for j in _dedup_kept
                                            if j < len(_tc_all)]
                                if len(_tc_kept) < len(_tc_all):
                                    _tc_start = decision.action_json.index(_tc_all[0])
                                    _tc_end = (decision.action_json.rindex(_tc_all[-1])
                                               + len(_tc_all[-1]))
                                    decision = _dc_replace(
                                        decision,
                                        action_json=(decision.action_json[:_tc_start]
                                                     + "\n".join(_tc_kept)
                                                     + decision.action_json[_tc_end:]))
                    # SAME-TURN duplicate suppression (#162): generalise the consecutive
                    # filter above to NON-adjacent repeats, but only for web tools the run
                    # explicitly opted in via config.web_dedup_same_turn_tools. No-op when
                    # that set is empty (the default) or off the web flow, so existing
                    # behaviour is unchanged. Both `actions` (what runs) and `decision`
                    # (the assistant block committed to history) are trimmed together.
                    actions, decision = self._suppress_same_turn_duplicates(actions, decision)
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
                                f"stuck: no-progress loop involving '{tool_name}' "
                                f"(threshold {self._cfg.escalation_stuck_threshold})",
                                result,
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
                # The token/snapshot budget is NOT enforced here: it is enforced at the
                # TOP of the next turn by ``_fit_request_to_context`` against the ACTUAL
                # assembled request (system + history + user), where it can evict oldest
                # history first and truncate the live snapshot only as a last resort, and
                # log it loudly (issues #170/#172/#173). Keeping a single budgeter there
                # avoids two policies fighting over the history. Skipped on the legacy path.
                if self._native:
                    # Evict any previous page_snapshot observation before committing the
                    # new turn so only the LATEST snapshot is visible in history.
                    self._evict_old_page_snapshot(chat_history)
                    self._commit_turn_to_history(chat_history, user, decision, turn)
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
            # ISSUE #183: the REASONING trace must NEVER be persisted into the stateful
            # history — only the assistant's action belongs there. On the native
            # think-then-act path the thinking arrives in a SEPARATE channel
            # (Decision.reasoning) and is already excluded, but a thinking model can leak
            # a <think>…</think> block into the content; strip it so the stored assistant
            # turn is the tool call alone (the trace is kept for logs/display, not replayed).
            assistant["content"] = _strip_think(decision.action_json or "")
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

    def _fit_request_to_context(self, chat_history: list[dict], system: str, user: str,
                                result: "RunResult", turn_index: int) -> BudgetFitReport:
        """Make the assembled native request fit ``num_ctx`` (issues #170/#172/#173).

        Delegates the maths to :func:`fit_chat_history`, which mutates ``chat_history``
        in place: it evicts the OLDEST non-snapshot messages first (respecting
        ``chat_history_max_turns``) and truncates the live page snapshot ONLY as a last
        resort, so the latest snapshot — the agent's eyes — is preserved whole whenever
        evicting stale history alone makes the request fit.

        When the request was over budget (anything was evicted or the snapshot was
        touched) an EXTREMELY VISIBLE, grep-able line is emitted to BOTH stdout (the
        reporter, flushed) and the run log (``result.context_events``) so a budgeting
        event can never masquerade as a model failure. Returns the report (for tests)."""
        # ISSUE #193 — size the fit against the ACTIVE model's per-model context window
        # (base spec, or the escalated API model). A local model resolves back to
        # config.num_ctx (qwen3:4b stays 32768); an API model gets its larger documented
        # window so the latest page snapshot is preserved whole instead of truncated.
        from .config import effective_context_window
        window = effective_context_window(self._cfg, self._active_spec)
        report = fit_chat_history(chat_history, system, user, self._cfg, window)
        if report.over_budget or report.acted:
            self._log_context_overrun(report, result, turn_index)
        return report

    def _log_context_overrun(self, report: BudgetFitReport, result: "RunResult",
                             turn_index: int) -> None:
        """Emit the loud, single-line, grep-able CONTEXT-OVERRUN signal (issue #172).

        Marker ``!!! CONTEXT-OVERRUN`` so it is trivially searchable. The fully-dropped
        snapshot (the agent is blind) is flagged even more prominently than a partial
        truncation; an eviction-only event (snapshot preserved) is reported as the
        healthy, preferred outcome. Written to stdout via the reporter (flushed) AND
        appended to the run log so a subsequent crash can never hide it."""
        common = (
            f"request ~{report.request_tokens_before}->{report.request_tokens_after}"
            f"/{report.num_ctx} tok, input_budget {report.input_budget_tokens} tok, "
            f"history {report.history_msgs_after} msgs/{report.history_tokens_after} tok, "
            f"evicted {report.evicted_msgs} msgs/{report.evicted_tokens} tok")
        if report.snapshot_dropped:
            kind = "snapshot-dropped"
            line = (f"!!! CONTEXT-OVERRUN [SNAPSHOT FULLY DROPPED — AGENT IS BLIND THIS "
                    f"TURN] turn {turn_index}: the live page snapshot "
                    f"({report.snapshot_raw_chars} chars) did not fit even after evicting "
                    f"all evictable history — {common} !!!")
        elif report.snapshot_truncated:
            kind = "snapshot-truncated"
            line = (f"!!! CONTEXT-OVERRUN [snapshot truncated] turn {turn_index}: kept "
                    f"{report.snapshot_kept_chars}/{report.snapshot_raw_chars} chars "
                    f"(dropped {report.snapshot_dropped_chars}) — {common} !!!")
        elif report.evicted_msgs:
            kind = "history-evicted"
            line = (f"!!! CONTEXT-OVERRUN [history evicted, snapshot preserved] turn "
                    f"{turn_index}: {common} !!!")
        else:
            # Over budget but nothing evictable and no snapshot to trim — still loud.
            kind = "over-budget"
            line = (f"!!! CONTEXT-OVERRUN [over budget, nothing to evict] turn "
                    f"{turn_index}: {common} !!!")
        # stdout (flushed by the reporter) + the run log, so neither path can hide it.
        self._reporter.note(line)
        result.context_events.append({
            "turn": turn_index, "kind": kind,
            "request_tokens_before": report.request_tokens_before,
            "request_tokens_after": report.request_tokens_after,
            "num_ctx": report.num_ctx,
            "input_budget_tokens": report.input_budget_tokens,
            "evicted_msgs": report.evicted_msgs,
            "evicted_tokens": report.evicted_tokens,
            "snapshot_raw_chars": report.snapshot_raw_chars,
            "snapshot_kept_chars": report.snapshot_kept_chars,
            "snapshot_dropped_chars": report.snapshot_dropped_chars,
            "snapshot_truncated": report.snapshot_truncated,
            "snapshot_dropped": report.snapshot_dropped,
            "fits_after": report.fits_after,
            "message": line,
        })

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
                self._escalate(detector, "premature validate (validator rejected the claim)",
                               result)

    # ---- escalation ----
    def _escalate(self, detector: StuckDetector, why: str,
                  result: "RunResult | None" = None) -> None:
        """Hand the run over to the ESCALATOR model — same browser, same session (#191).

        At most once per run (``detector.escalated`` guards re-entry). On take-over the
        agent's ACTIVE codec + call PATH + per-turn cap are re-pointed at the ESCALATOR
        model's per-model policy (issues #163/#179): escalating from the local qwen3:4b
        (hermes, native ``decide_chat``) to a z.ai/GLM API model flips the agent to the
        ``json`` codec + the single-shot ``_decide`` path + the escalator's tool-call cap —
        otherwise the API client would run under the native-only ``hermes`` codec and emit
        un-parseable free prose (the #179 failure). Both SUCCESS and FAILURE (missing /
        unreachable provider key) are RECORDED to the run JSON/.md via ``result`` so a
        non-functional escalation can never masquerade as "the model just looped" (#191).
        """
        if detector.escalated or not self._cfg.escalation_enabled:
            return
        detector.escalated = True
        from .clients import build_client, select_execution_codec
        from .config import (resolve_role_spec, resolve_model_codec,
                             resolve_model_limit)
        spec = resolve_role_spec(self._cfg, "escalation")
        try:
            # Build the escalation client through the unified factory + the escalation role
            # spec (issue #163), so escalation honours config.models["escalation"] (or the
            # legacy escalation_provider/escalation_model keys) and its model + sampling
            # come from its ModelSpec/MODEL_TOOL_POLICIES. A missing key raises here.
            new_client = build_client(spec, self._cfg)
        except Exception as exc:
            # LOUD + RECORDED failure (#191): the key is missing/unreachable. Record a
            # DISTINCT observable event so the artifact shows escalation was attempted and
            # could not take over — never a silent terminal-only degrade.
            self._record_escalation_event(
                result, success=False, why=why, spec=spec, error=str(exc))
            self._reporter.note(
                f"[ESCALATION] {why} — FAILED, could NOT take over: {exc}")
            return
        self._client = new_client
        # ISSUE #193 — the budget now follows the escalated model's context window so the
        # live page snapshot is no longer truncated to the local model's cap post-escalation.
        self._active_spec = spec
        # Switch the ACTIVE codec + call path + cap to the escalator model's (#179/#178).
        codec_name = resolve_model_codec(self._cfg, spec)
        escalator_codec = get_codec(codec_name)
        exec_codec, native, _note = select_execution_codec(
            self._cfg, new_client, escalator_codec, self._registry)
        self._codec = exec_codec
        self._native = native
        self._tools = exec_codec.tools(self._registry) if native else None
        self._max_actions = resolve_model_limit(self._cfg, spec)
        # Swap to the escalation system-prompt provider (native-OFF: the `# Tools` block +
        # JSON format instructions ARE present) so the single-shot API model is actually
        # told the tools + wire format. None on the fs/test path (static prompt kept).
        if self._escalation_system_provider is not None:
            self._system_provider = self._escalation_system_provider
            self._provider_wants_user = None   # re-resolve the new provider's arity
        self._record_escalation_event(
            result, success=True, why=why, spec=spec,
            codec=exec_codec.name, native=native, max_actions=self._max_actions)
        self._reporter.note(
            f"[ESCALATION] {why} — TAKING OVER with {spec.provider}:{spec.model} "
            f"(codec={exec_codec.name}, path={'native' if native else 'single-shot'}, "
            f"max_actions={self._max_actions})")

    def _record_escalation_event(self, result: "RunResult | None", *, success: bool,
                                 why: str, spec, error: str | None = None,
                                 codec: str | None = None, native: bool | None = None,
                                 max_actions: int | None = None) -> None:
        """Append a single escalation event to the run log (issue #191).

        Mirrors the #180 ``context_events`` mechanism: a grep-able ``[ESCALATION]`` line is
        recorded in ``result.escalation_events`` (persisted to the run JSON/.md) so both a
        successful take-over and a failed one (missing/unreachable key) are on the permanent
        record, not just a transient terminal note. ``result`` is None only on legacy call
        paths that pass no run result; the event is then dropped (stdout note still fires)."""
        if success:
            message = (f"[ESCALATION] TOOK OVER ({why}) — {spec.provider}:{spec.model} "
                       f"codec={codec} path={'native' if native else 'single-shot'} "
                       f"max_actions={max_actions}")
        else:
            message = (f"[ESCALATION] FAILED to take over ({why}) — "
                       f"{spec.provider}:{spec.model}: {error}")
        if result is not None:
            result.escalation_events.append({
                "success": success,
                "why": why,
                "provider": spec.provider,
                "model": spec.model,
                "codec": codec,
                "native": native,
                "max_actions": max_actions,
                "error": error,
                "message": message,
            })

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

    # ---- same-turn duplicate suppression (#162) ----
    def _suppress_same_turn_duplicates(
        self, actions: "list[tuple[str, dict]]", decision: Decision,
    ) -> "tuple[list[tuple[str, dict]], Decision]":
        """Drop EXACT duplicate calls within this one turn for opted-in web tools (#162).

        A call is a duplicate when an EARLIER call in the SAME ``actions`` batch had the
        same ``(tool, args)`` signature — at any position, not only back-to-back (the
        consecutive filter upstream already covers adjacency). Only tools named in
        ``config.web_dedup_same_turn_tools`` participate, and only on the web-agent flow
        (a live snapshot provider is wired). The dropped call is removed from BOTH the
        executed ``actions`` and the ``decision`` assistant block, so it never runs and
        leaves no trace in the replayed history.

        Returns ``(actions, decision)`` UNCHANGED when: the opt-in set is empty, this is
        not the web flow, nothing duplicates, or the assistant block cannot be trimmed to
        match the survivors (see :meth:`_trim_decision_to_kept`). The last case is the
        consistency guarantee: we never drop a result while leaving its call in the
        assistant block (or vice versa) — if we cannot do both cleanly, we do neither.
        """
        dedup_tools = set(getattr(self._cfg, "web_dedup_same_turn_tools", ()) or ())
        if not dedup_tools or self._raw_snapshot_provider is None:
            return actions, decision
        seen: set[str] = set()
        kept: list[int] = []
        for idx, (name, args) in enumerate(actions):
            if name in dedup_tools:
                sig = self._action_signature(name, args)
                if sig is not None and sig in seen:
                    self._reporter.note(
                        f"[DEDUP] dropping same-turn duplicate {name}")
                    continue
                if sig is not None:
                    seen.add(sig)
            kept.append(idx)
        if len(kept) == len(actions):
            return actions, decision
        trimmed_decision, ok = self._trim_decision_to_kept(decision, len(actions), kept)
        if not ok:
            # Could not safely realign the assistant block with the survivors — abandon
            # the suppression entirely so calls and results stay consistent.
            return actions, decision
        return [actions[j] for j in kept], trimmed_decision

    def _trim_decision_to_kept(
        self, decision: Decision, n_before: int, kept: "list[int]",
    ) -> "tuple[Decision, bool]":
        """Return ``(decision, ok)`` with the assistant tool-call block trimmed to ``kept``
        (indices into the ``n_before``-length action list).

        ``ok`` is ``False`` when the block could not be trimmed to match — the caller must
        then drop NOTHING, keeping calls and results aligned (#162). In LEGACY (non-native)
        mode there is no committed assistant block at all, so trimming is unnecessary and
        always safe (``ok=True``, decision returned untouched). In NATIVE mode the call may
        live in structured ``decision.tool_calls`` OR as ``<tool_call>`` text blocks in
        ``decision.action_json``; either is trimmed by index, but ONLY when its count
        matches ``n_before`` (otherwise some calls were skipped during parse and an
        index-based trim would misalign — so we report ``ok=False`` and bail)."""
        if not self._native:
            return decision, True
        if decision.tool_calls:
            if len(decision.tool_calls) != n_before:
                return decision, False
            return _dc_replace(
                decision, tool_calls=[decision.tool_calls[j] for j in kept]), True
        blocks = re.findall(r"<tool_call>[\s\S]*?</tool_call>", decision.action_json or "")
        if not blocks or len(blocks) != n_before:
            return decision, False
        kept_blocks = [blocks[j] for j in kept]
        start = decision.action_json.index(blocks[0])
        end = decision.action_json.rindex(blocks[-1]) + len(blocks[-1])
        trimmed = (decision.action_json[:start] + "\n".join(kept_blocks)
                   + decision.action_json[end:])
        return _dc_replace(decision, action_json=trimmed), True

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
