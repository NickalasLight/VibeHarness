"""Prompt construction.

The system prompt is deliberately lean and is assembled from the registry, so the
documented tools never drift from the tools that exist. Each tool and its
parameters are described once in plain English.

The full JSON action-schema is intentionally NOT printed here. Output is already
constrained at decode time by Ollama's ``format`` grammar (see
``OllamaClient._act``, which passes ``registry.action_schema()`` as ``format``):
every emitted action is guaranteed structurally valid regardless of the prompt.
Re-printing that schema would be large, redundant, and would push the tool docs
toward the low-attention middle of the context. A one-line shape reminder is enough.
"""
from __future__ import annotations

from .registry import ToolRegistry

_SYSTEM_TEMPLATE = """\
You are a capable task-execution agent operating a computer through a small set \
of tools. Work in a loop: each turn, read the account of what you have done, then \
choose one or more tools to make progress. Keep going until the task is fully done, \
then call `validate`.

# How the loop works
- Each turn, output a JSON ARRAY of one or more actions of the form \
{{"tool": <name>, "args": {{...}}}}; they run in order. Output only that array — \
no prose. Use only the tools listed below.
- Batch independent or predictable actions in one turn (e.g. write a file then read \
it back); emit a single action when you must see its result before deciding.
- Each action's result is added to the account. Use it to decide your next turn. On \
an error, adapt — never repeat the same failing call.
- When the task is genuinely done, end your turn with `validate` plus a short \
summary. A validator checks your work: if it agrees the run ends; otherwise you get \
feedback on what is missing — fix it and validate again.

# Tools
{docs}

# Guidance
- Treat the task as exact ground truth: do not paraphrase, invent, or drift from it. \
Re-read it before each action.
- Prefer the simplest tool for the step; use relative paths unless an absolute one \
is required.
- Verify before validating (e.g. read a file back after writing it).
"""


class SystemPromptBuilder:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def build(self, task: str = "") -> str:
        body = _SYSTEM_TEMPLATE.format(docs=self._registry.docs())
        if not task:
            return body
        # Anchor the task at the very front of the context (primacy / authoritative
        # system instruction). Combined with the recency reminder in the turn prompt,
        # the task is pinned at both high-attention ends, resisting mid-context drift.
        header = (
            f"# YOUR ASSIGNED TASK\n{task}\n\n"
            f"Keep this EXACT task in mind at all times — do not paraphrase, summarize, "
            f"or drift from it. Everything below explains the tools and rules for "
            f"accomplishing it.\n\n---\n\n"
        )
        return header + body


def build_turn_prompt(task: str, narrative: str) -> str:
    """The per-turn user message.

    The task is anchored in the two high-attention zones only: the FRONT (the system
    prompt, via SystemPromptBuilder.build(task)) and the END (a short reminder right
    before the model generates). Transformers attend most strongly to the start and
    end of the context and weakest to the middle ("lost in the middle"), so these two
    placements pin the task without the bloat of a third copy in the low-attention
    middle. The growing history sits in the middle, where it is reference, not the goal.
    """
    return (
        f"# What you have done so far\n{narrative}\n\n"
        f"# Reminder — your exact task (verbatim) is:\n{task}\n\n"
        f"# Your next action\n"
        f"Choose the next action (or several, as a batch) to make progress on the task "
        f"above, ending with `validate` once you believe it is complete. Respond with a "
        f"JSON array of one or more actions."
    )
