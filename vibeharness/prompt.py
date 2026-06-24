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

    def build(self, task: str = "", workspace: str = "", page: str = "") -> str:
        # Render the per-toolset guidance section only when there is guidance to show,
        # so a no-guidance build leaves no empty "# Working with your tools" heading.
        tool_guidance = ""
        if self._guidance.strip():
            tool_guidance = f"# Working with your tools\n{self._guidance.strip()}\n\n"
        body = _SYSTEM_TEMPLATE.format(
            docs=self._registry.docs(),
            tool_guidance=tool_guidance,
            format_instructions=self._codec.format_instructions(self._max_actions))
        header = ""
        if task:
            # Anchor the task at the very front of the context (primacy / authoritative
            # system instruction). Combined with the recency reminder in the turn prompt,
            # the task is pinned at both high-attention ends, resisting mid-context drift.
            header = (
                f"# YOUR ASSIGNED TASK\n{task}\n\n"
                f"Keep this EXACT task in mind at all times — do not paraphrase, summarize, "
                f"or drift from it. Everything below explains the tools and rules for "
                f"accomplishing it.\n\n---\n\n"
            )
        if workspace:
            # A snapshot of the working directory, refreshed every turn so newly
            # created files show up next turn. Sits right after the task block.
            header += f"# Workspace\n{workspace}\n\n---\n\n"
        if page:
            # A fresh snapshot of the live browser page, regenerated every turn
            # (issue #24) so the model always sees the CURRENT page state (consent
            # banners, modals, …). This is PROVIDED AUTOMATICALLY: there is no agent
            # tool to request it (the old `snapshot` action was removed, issue #51).
            # Because the whole system prompt is rebuilt each turn, only the latest
            # snapshot is ever present — the prior one disappears (stale-dropping by
            # regeneration, never accumulated in narrative memory).
            header += ("# Current page (live snapshot — provided automatically)\n"
                       f"{page}\n\n---\n\n")
        return header + body if header else body


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
