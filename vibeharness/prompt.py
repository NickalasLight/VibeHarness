"""Prompt construction.

The system prompt is deliberately lean and is assembled from the registry, so the
documented tools never drift from the tools that exist. Each tool and its
parameters are described once in plain English.

How to emit a tool call (the JSON array, an XML form, code, …) is owned by the
active :class:`~vibeharness.codec.ToolCallCodec`, which supplies the format block
via ``format_instructions``. The full action grammar is intentionally NOT printed
here: output is already constrained at decode time (see ``OllamaClient._act``), so
re-printing it would be large, redundant, and would push the tool docs toward the
low-attention middle of the context.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from .codec import ToolCallCodec, get_codec
from .config import Config
from .registry import ToolRegistry


class _GuidanceSource(Protocol):
    """Anything that can advertise short system-prompt guidance — e.g. a Toolset.

    Kept structural (a Protocol) so :class:`SystemPromptBuilder` depends only on the
    ``system_guidance`` capability, not on the concrete ``Toolset`` class (DIP).
    """

    def system_guidance(self) -> str | None: ...

_SYSTEM_TEMPLATE = """\
You are a capable task-execution agent operating a computer through a small set \
of tools. Work in a loop: each turn, read the account of what you have done, then \
choose one or more tools to make progress. Keep going until the task is fully done, \
then call `validate`.

# How the loop works
{format_instructions}
- Each action's result is added to the account. Use it to decide your next turn. On \
an error, adapt — never repeat the same failing call.
- When the task is genuinely done, end your turn with `validate` plus a short \
summary. A validator checks your work: if it agrees the run ends; otherwise you get \
feedback on what is missing — fix it and validate again.

# Tools
{docs}

{tool_guidance}# Guidance
- Treat the task as exact ground truth: do not paraphrase, invent, or drift from it. \
Re-read it before each action.
- Prefer the simplest tool for the step; use relative paths unless an absolute one \
is required.
- Verify before validating (e.g. read a file back after writing it).
"""


# NATIVE-tools system template (issue #129/#130/#131). When the harness sends the tools
# in Ollama's ``tools:`` field, Ollama injects the model's OWN trained `# Tools` block
# (the enveloped function schemas) AND the call-format / anti-fence instructions from the
# model's chat template. So this template OMITS both the `{docs}` `# Tools` section and the
# codec's `{format_instructions}` (re-stating them would fight the model's native template
# and waste the window) — it keeps only the loop description, the per-toolset guidance, and
# the general guidance. It still tells the model that tools are provided to it natively.
_SYSTEM_TEMPLATE_NATIVE = """\
You are a capable task-execution agent operating a computer through a small set \
of tools. Work in a loop: each turn, look at the current state and the result of \
your previous actions, then call one or more of the tools available to you to make \
progress. Keep going until the task is fully done, then call `validate`.

# How the loop works
- The tools you may call are provided to you directly. Call them using the function-\
calling format — do NOT paste tool definitions or wrap calls in markdown.
- Each action's result is returned to you as a tool message. Use it to decide your \
next turn. On an error, adapt — never repeat the same failing call.
- When the task is genuinely done, end your turn with `validate` plus a short \
summary. A validator checks your work: if it agrees the run ends; otherwise you get \
feedback on what is missing — fix it and validate again.

{tool_guidance}# Guidance
- Treat the task as exact ground truth: do not paraphrase, invent, or drift from it. \
Re-read it before each action.
- Prefer the simplest tool for the step; use relative paths unless an absolute one \
is required.
- Verify before validating (e.g. read a file back after writing it).
"""


class SystemPromptBuilder:
    def __init__(self, registry: ToolRegistry,
                 max_actions_per_turn: int = Config().max_actions_per_turn,
                 codec: ToolCallCodec | None = None,
                 guidance: str = ""):
        self._registry = registry
        self._max_actions = max_actions_per_turn
        self._codec = codec or get_codec("json")
        # Pre-assembled, role-specific guidance for the ACTIVE toolset(s). Empty by
        # default so existing callers (e.g. SystemPromptBuilder(registry)) are
        # unaffected and no empty section is rendered.
        self._guidance = guidance

    @staticmethod
    def assemble_guidance(sources: Iterable[_GuidanceSource]) -> str:
        """Collect the non-empty ``system_guidance`` of each source, in order, with
        duplicates removed. The result is ready to pass as the ``guidance`` argument.

        Order-stable (preserves the order toolsets were selected) and de-duplicated so
        two toolsets sharing a note never repeat it. Returns "" when nothing applies.
        """
        seen: set[str] = set()
        notes: list[str] = []
        for source in sources:
            text = (source.system_guidance() or "").strip()
            if text and text not in seen:
                seen.add(text)
                notes.append(text)
        return "\n".join(f"- {note}" for note in notes)

    def build(self, task: str = "", workspace: str = "", page: str = "",
              include_tool_guidance: bool = True, native_tools: bool = False) -> str:
        """Render the system prompt: header (task + workspace) + body.

        Issue #146: the live page snapshot is NO LONGER part of the system prompt.
        Every major web-agent benchmark (WebArena, VisualWebArena, SeeAct, AgentBench,
        WorkArena, browser-use) places the live page observation in a USER turn, never
        the system prompt; "Lost in the Middle" (arXiv:2307.03172) shows LLMs attend
        best to the START and END of the input, so the fast-changing snapshot is moved
        to the very END of the user turn (the recency slot — see ``RalphAgent.run``).
        The ``page`` parameter is KEPT for backwards compatibility but is now IGNORED:
        no ``# Current page`` section is added here. Callers that still need the
        snapshot rendered as a section (the validator context — issue #57) append it
        themselves around this builder (see ``cli.render_page_section``).

        When ``include_tool_guidance`` is False, the body — the `# How the loop works`
        / codec format-instruction block, the `# Tools` docs, AND the
        `# Working with your tools` guidance — is omitted entirely, leaving ONLY the
        header (task + workspace). This is the validator's view (issue #57): it must
        see the SAME task/workspace context the main agent had, but NOT the tool
        descriptions or format rules, because it is judging the work, not producing
        tool calls.

        When ``native_tools`` is True (issue #129/#130/#131), the harness sends the tool
        schemas in Ollama's ``tools:`` field, so Ollama injects the model's OWN `# Tools`
        block and call-format instructions from its chat template. The body then OMITS
        both the `# Tools` docs and the codec's `{format_instructions}` (re-stating them
        would duplicate/fight the model's native template) — it keeps the loop
        description, the per-toolset guidance, and the general guidance. Ignored when
        ``include_tool_guidance`` is False (the validator view has no tool body anyway).
        """
        body = ""
        if include_tool_guidance:
            # Render the per-toolset guidance section only when there is guidance to show,
            # so a no-guidance build leaves no empty "# Working with your tools" heading.
            tool_guidance = ""
            if self._guidance.strip():
                tool_guidance = f"# Working with your tools\n{self._guidance.strip()}\n\n"
            if native_tools:
                # Ollama injects the tools + format instructions from the model's own
                # template; omit the harness's `# Tools` docs and format block.
                body = _SYSTEM_TEMPLATE_NATIVE.format(tool_guidance=tool_guidance)
            else:
                # The active codec may supply its own tool-definition rendering (issue
                # #105 / #123): a Hermes/Qwen model reads tools as a <tools> function-schema
                # block, not Markdown. Fall back to the registry's Markdown docs when the
                # codec has no opinion (every codec but `hermes` returns None), so the other
                # formats render exactly as before.
                docs = self._codec.tool_definitions(self._registry)
                if docs is None:
                    docs = self._registry.docs()
                body = _SYSTEM_TEMPLATE.format(
                    docs=docs,
                    tool_guidance=tool_guidance,
                    format_instructions=self._codec.format_instructions(self._max_actions))
        header = ""
        if task:
            # Anchor the task at the very front of the context (primacy / authoritative
            # system instruction). Combined with the recency reminder in the turn prompt,
            # the task is pinned at both high-attention ends, resisting mid-context drift.
            # The closing sentence references "the tools and rules below", which only
            # exist when the body is rendered; drop it for the tool-less validator view.
            tail = (" Everything below explains the tools and rules for accomplishing it."
                    if include_tool_guidance else "")
            header = (
                f"# YOUR ASSIGNED TASK\n{task}\n\n"
                f"Keep this EXACT task in mind at all times — do not paraphrase, summarize, "
                f"or drift from it.{tail}\n\n---\n\n"
            )
        if workspace:
            # A snapshot of the working directory, refreshed every turn so newly
            # created files show up next turn. Sits right after the task block.
            header += f"# Workspace\n{workspace}\n\n---\n\n"
        # Issue #146: the live page snapshot is intentionally NOT rendered here anymore.
        # It now lives at the END of the user turn (the high-attention recency slot — see
        # ``RalphAgent.run`` and ``cli.render_page_section``). ``page`` is accepted for
        # backwards compatibility but ignored; passing it has no effect on the prompt.
        return header + body if header else body


# Issue #146: the marker that identifies a live-snapshot block appended to a USER turn.
# History pruning scans for this exact substring to strip stale snapshots, and the user-
# turn appender / validator section renderer build their headings from it, so the three
# stay in lock-step. Changing the wording here updates all three at once.
SNAPSHOT_USER_MARKER = "## Current page (live snapshot"

# The placeholder that replaces a pruned snapshot block in older user turns, so old
# observations (tool results) survive in history while only the LATEST snapshot is shown.
SNAPSHOT_PRUNED_PLACEHOLDER = (
    "[page snapshot removed — only the latest snapshot is shown in the current turn]")


def append_snapshot_to_user(user: str, snapshot: str) -> str:
    """Append the live page snapshot to the END of a user-turn message (issue #146).

    Placed at the very end so it occupies the high-attention recency slot ("Lost in the
    Middle", arXiv:2307.03172) — the model attends most strongly to the most recent
    content, which is exactly where the fast-changing page state belongs. Returns ``user``
    unchanged when there is no snapshot this turn (web inactive, or the budget left no
    room — issue #43), so non-web runs and overflow turns are unaffected."""
    if not snapshot:
        return user
    return (f"{user}\n\n---\n"
            f"{SNAPSHOT_USER_MARKER} — what the page looks like RIGHT NOW after your "
            f"last actions)\n{snapshot}")


def render_page_section(page: str) -> str:
    """Render the standalone `# Current page` section for the validator context (#57).

    The validator no longer reads the system prompt's page section (issue #146 moved the
    snapshot out of the system prompt). It still needs the SAME budgeted snapshot the
    agent saw, so the validator-context provider appends this section to the tool-less
    prompt. Returns "" when there is no snapshot, so an fs-only / overflow turn renders
    no page section."""
    if not page:
        return ""
    return ("# Current page (live snapshot — provided automatically)\n"
            f"{page}\n\n---\n\n")


def build_turn_prompt(task: str, narrative: str,
                      action_hint: str = "Respond with a JSON array of one or more actions.") -> str:
    """The per-turn user message.

    The task is anchored in the two high-attention zones only: the FRONT (the system
    prompt, via SystemPromptBuilder.build(task)) and the END (a short reminder right
    before the model generates). Transformers attend most strongly to the start and
    end of the context and weakest to the middle ("lost in the middle"), so these two
    placements pin the task without the bloat of a third copy in the low-attention
    middle. The growing history sits in the middle, where it is reference, not the goal.

    ``action_hint`` is the active codec's end-of-turn format reminder, so the recency
    nudge always matches the wire format the model is being asked to produce.
    """
    return (
        f"# What you have done so far\n{narrative}\n\n"
        f"# Reminder — your exact task (verbatim) is:\n{task}\n\n"
        f"# Your next action\n"
        f"Choose the next action (or several, as a batch) to make progress on the task "
        f"above, ending with `validate` once you believe it is complete. {action_hint}"
    )
