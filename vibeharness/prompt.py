"""Prompt construction.

The system prompt is deliberately concise and is assembled from the registry so
the documented tools and the enforced schema can never disagree. It follows the
common system-prompt convention: describe each tool and its parameters in plain
English, then give the formal JSON schema.
"""
from __future__ import annotations

import json

from .registry import ToolRegistry

_SYSTEM_TEMPLATE = """\
You are a capable task-execution agent operating a computer through a small set \
of tools. You work in a loop: on each turn you read what you have done so far, \
then choose one or more tools to make progress. Keep going until the task is fully \
done, then call `validate`.

# How the loop works
- You are given the task and a plain-English account of the actions you have \
already taken and what each returned.
- Each turn, output a JSON ARRAY of one or more actions, each of the form \
{{"tool": <tool name>, "args": {{ ... }}}}. The actions run in order.
- Batch several actions in one turn when they are independent or you are confident \
of the outcome (e.g. write a file then read it back). Emit a single action when \
you must see its result before deciding the next move.
- Output nothing except that JSON array. Do not invent tools or parameters. \
Only use the tools listed below.
- After your actions run you will see each result described in the account. Use it \
to decide your next turn. If an action returns an error, adapt — do not repeat the \
same failing call.
- When you believe the task is complete, end with a `validate` action and a short \
summary of what you accomplished. A validator will check your work: if it agrees, \
the run ends; if not, you will be told what is still missing — fix it and validate \
again. Do not call `validate` until you have genuinely attempted the whole task.

# Tools
{docs}

# Action schema
Your output each turn must validate against this JSON schema (an array of actions):
{schema}

# Guidance
- The user's task is given to you verbatim. Treat it as the exact ground truth: do \
not paraphrase, summarize, invent, or drift from it, even after long reasoning. \
Re-read the task before deciding each action.
- Prefer the simplest tool that accomplishes the step.
- Verify your work before validating (e.g. after writing a file, read it back).
- Use relative paths unless an absolute path is required.
"""


class SystemPromptBuilder:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def build(self, task: str = "", workspace: str = "") -> str:
        schema = json.dumps(self._registry.action_schema(), indent=2)
        body = _SYSTEM_TEMPLATE.format(docs=self._registry.docs(), schema=schema)
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
        return header + body if header else body


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
