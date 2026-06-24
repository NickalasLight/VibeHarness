"""Periodic free-text hint advisor for the Qwen base agent.

Every N Qwen turns the advisor is called with the task + the last N action turns and
returns concise plain-English advice: what the agent is doing wrong, what to try next, etc.

The advice is injected into the Qwen agent's next-turn user message as:
  <user_advice>The human user gives you the following hint: '...'</user_advice>

When advisor_model is "" (default), the SAME model as the base agent is used — Qwen
self-advises.  This avoids model-swap overhead and keeps only one model in VRAM.  Set
advisor_model to "vibethinker:latest" to use VibeThinker as the advisor instead.

The advisor always uses two_phase=False (single-phase, like the base Qwen): it speaks the
same Hermes <tool_call> dialect but its output is treated as free text — the advisor just
generates advice tokens and we take the raw output as the hint.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .config import Config
from .llm import OllamaClient

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SYSTEM = (
    "You are an expert web-form automation observer. "
    "You watch an AI agent filling a job-application form and give it precise, actionable advice.\n\n"
    "The full task description (including ALL expected field values such as State, Country, "
    "phone, address, etc.) is provided in the user message below under 'TASK'. "
    "You have full visibility into what values SHOULD be entered — use them when advising.\n\n"
    "ADVICE PRINCIPLES:\n"
    "1. Be concrete and specific. When you recommend a tool call, write it out exactly: "
    "e.g. 'call select_option(target=\"e65\", value=\"TX\")' or "
    "'call evaluate(expression=\"el => el.tagName\", target=\"e65\")'. "
    "Never leave the agent guessing at the tool name, ref, or value.\n"
    "2. Observe before concluding. Do not assume why something failed — "
    "recommend evaluate or click to confirm the element type first if uncertain.\n"
    "3. Reason from evidence. Cite turn numbers, tool names, refs, and error text.\n"
    "4. Detect and break loops. If you see the same tool+target pair appearing 2 or "
    "more times in the recent history — especially with FAILED or WARNING results — "
    "the agent is stuck in a loop. Name the exact looping call, explain why it will "
    "never work, and give 2-3 concrete alternative calls the agent should try instead.\n"
    "5. Distinguish stuck from done. If fields are already filled, direct the agent "
    "to the next UNFILLED field or page navigation, not backward.\n"
    "6. Be concise. 3-6 plain-English sentences. No JSON, no code blocks, no bullet lists. "
    "Speak directly to the agent as 'you'."
)


def _build_user(task: str, history: str) -> str:
    return (
        f"TASK:\n{task}\n\n"
        f"RECENT AGENT ACTIONS (last turns):\n{history}\n\n"
        "What is going wrong (if anything) and what should the agent do next?"
    )


_TOOL_CALL_RE = re.compile(
    r"<tool_call>.*?</tool_call>|```json.*?```|```.*?```|\{[^{}]*\"name\"[^{}]*\}",
    re.DOTALL,
)


def _extract_preamble(raw_action: str) -> str:
    """Return the free-text preamble Qwen emitted before its tool call, if any.

    Qwen's single-phase output is: [optional preamble text] <tool_call>...</tool_call>.
    The preamble is whatever came before the first JSON/Hermes block — it is the model's
    reasoning in plain English and is valuable context for the advisor.  Returns "" if
    the output is pure tool-call with no preamble, or if raw_action is empty.
    """
    if not raw_action:
        return ""
    m = _TOOL_CALL_RE.search(raw_action)
    if m:
        preamble = raw_action[:m.start()].strip()
    else:
        preamble = raw_action.strip()
    return preamble[:300] if preamble else ""


def format_turns_for_advisor(turns: list, n: int) -> str:
    """Render the last ``n`` turns as readable action history for the advisor prompt.

    Includes each turn's:
    - Qwen's preamble reasoning (text it emitted before the tool call, if any)
    - The parsed action (tool name + args)
    - The observation (what the tool returned)
    This gives VibeThinker full visibility into what Qwen was "thinking" before each action.
    """
    import json
    recent = turns[-n:] if len(turns) >= n else turns
    lines = []
    for turn in recent:
        lines.append(f"Turn {turn.index}:")
        # Include Qwen's preamble reasoning if present (two_phase=False path).
        preamble = _extract_preamble(getattr(turn, "raw_action", "") or "")
        if preamble:
            lines.append(f"  [reasoning] {preamble}")
        for a in turn.actions:
            status = "OK" if a.ok else "FAIL"
            tool = a.tool or "(none)"
            args_s = json.dumps(a.args, ensure_ascii=False)[:120] if a.args else ""
            obs = (a.observation or "")[:200]
            lines.append(f"  [{status}] {tool} {args_s}")
            lines.append(f"         -> {obs}")
    return "\n".join(lines)


class VibeThinkerAdvisor:
    """Calls VibeThinker as a free-text advisor alongside the Qwen base agent.

    Uses a separate OllamaClient scoped to the advisor model + temperature, derived
    from the run Config via ``dataclasses.replace``.  OLLAMA_MAX_LOADED_MODELS must
    be 2 (set by the caller before constructing OllamaClient) so both models stay hot.
    """

    def __init__(self, config: Config) -> None:
        # Resolve empty advisor_model to the base model (Qwen self-advising).
        # Empty string = same model as the base agent; this keeps only one model in VRAM
        # and avoids model-swap overhead.  Explicit model name = use that model instead
        # (e.g. "vibethinker:latest" for the VibeThinker advisor — requires model-swap).
        resolved_model = config.advisor_model or config.model
        self._self_advising = (resolved_model == config.model)
        # num_ctx=4096: the advisor only sees ~5 turns of history (~1-2k tokens).
        # Cutting from 32768 → 4096 saves ~1.2 GB of KV-cache VRAM on a separate advisor
        # model.  When self-advising, Qwen is already loaded so ctx size matters less, but
        # we keep it small so the advisor call is fast (short prompt → quick response).
        advisor_cfg = replace(
            config,
            model=resolved_model,
            temperature=config.advisor_temperature,
            action_temperature=config.advisor_temperature,
            reason_tokens=1024,   # advisor reasoning cap
            action_tokens=512,    # advice text cap
            two_phase=False,      # Qwen is single-phase; VibeThinker callers set two_phase=True
            num_ctx=4096,         # small ctx: fast + low VRAM
        )
        self._client = OllamaClient(advisor_cfg)
        self._interval = config.advisor_interval
        mode = "self-advising" if self._self_advising else f"model={resolved_model}"
        print(f"[advisor] active — {mode} "
              f"temp={config.advisor_temperature} every={config.advisor_interval} turns "
              f"ctx=4096",
              flush=True)

    def advise(self, task: str, turns: list, reporter=None) -> str:
        """Generate free-text advice from the last ``self._interval`` turns.

        When ``reporter`` is supplied (a :class:`~vibeharness.reporting.Reporter`), the
        advisor's token stream is rendered live in the terminal via the blue advisor lane.
        Without a reporter the generation is silent (test / headless use).
        """
        history = format_turns_for_advisor(turns, self._interval)
        system = _SYSTEM
        user = _build_user(task, history)
        label = "self" if self._self_advising else "advisor"
        print(f"\n[advisor] {label} model generating advice...", flush=True)
        if reporter is not None:
            reporter.advisor_start()
            on_token = reporter.advisor_token
        else:
            on_token = None
        # Single-phase: one /api/chat call, free-text output (no JSON schema constraint).
        advice = self._client._reason(system, user, on_token=on_token)
        if reporter is not None:
            reporter.advisor_end()
        # Strip any Hermes tool-call blocks or <think> tags that may appear.
        advice = _THINK_RE.sub("", advice).strip()
        advice = _TOOL_CALL_RE.sub("", advice).strip()
        if not advice:
            advice = "(no advice generated)"
        print(f"[advisor] advice stored — {len(advice)} chars sent to base agent.", flush=True)
        return advice
