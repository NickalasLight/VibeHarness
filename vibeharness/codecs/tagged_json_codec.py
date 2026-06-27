"""The tagged-JSON tool-call codec.

Each tool call is a JSON object ``{"tool": <name>, "args": {...}}`` wrapped in its
own ``<local_toolcall>...</local_toolcall>`` tag block; the model emits one block per
action, in order. The tag wrapper cannot be expressed as a JSON-schema ``format``, so
the action phase is left unconstrained and parsed purely by this codec — surrounding
prose between (or around) the blocks is tolerated and ignored.
"""
from __future__ import annotations

import json
import re

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry

# Match every <local_toolcall>...</local_toolcall> block. The closing tag is optional
# so a final block whose closing tag the model forgot is still recovered (the inner
# text then runs to end-of-string). DOTALL so a block may span multiple lines.
_BLOCK_RE = re.compile(
    r"<local_toolcall>(.*?)(?:</local_toolcall>|\Z)",
    re.DOTALL,
)


class TaggedJsonCodec(ToolCallCodec):
    name = "tagged_json"

    def format_instructions(self, max_actions: int) -> str:
        cap = (f" You may emit at most {max_actions} actions per turn."
               if max_actions and max_actions > 0 else "")
        return (
            '- Each turn, output one or more actions. Wrap EACH action as a JSON object '
            'of the form {"tool": <name>, "args": {...}} inside its own '
            '<local_toolcall></local_toolcall> tag block; the blocks run in order. '
            'Use only the tools listed below.\n'
            '- Batch independent or predictable actions in one turn (e.g. write a file then '
            'read it back); emit a single action when you must see its result before deciding.'
            + cap
        )

    def turn_action_hint(self) -> str:
        return "Wrap each action's JSON object in <local_toolcall></local_toolcall> tags."

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        # The tag wrapper isn't expressible as a JSON-schema `format`, so decoding is
        # left unconstrained and validated by parse().
        return DecodeConstraint(json_schema=None, stop=())

    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        blocks = _BLOCK_RE.findall(raw or "")
        # Drop any trailing block that is pure whitespace (e.g. a stray opening tag at
        # end of output) so it doesn't masquerade as an empty, invalid action.
        blocks = [b for b in blocks if b.strip()]
        if not blocks:
            return None, "no <local_toolcall>...</local_toolcall> blocks found"
        actions: list[ToolCall] = []
        for inner in blocks:
            try:
                obj = json.loads(inner.strip())
            except json.JSONDecodeError as e:
                return None, f"invalid JSON inside <local_toolcall> block ({e})"
            if not isinstance(obj, dict) or "tool" not in obj:
                return None, "each <local_toolcall> block must contain an object with a 'tool' field"
            args = obj.get("args", {})
            if not isinstance(args, dict):
                return None, "'args' must be an object"
            actions.append((obj["tool"], args))
        return actions, None


CODEC = TaggedJsonCodec()
