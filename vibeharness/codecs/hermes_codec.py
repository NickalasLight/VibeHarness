"""The Hermes / Qwen2.5 native tool-call codec (issue #105).

This codec speaks the format ``Shadow0482/mythos_fast`` (a VibeThinker-3B fine-tune
on ~2M tool-call samples) was trained on — the standard Qwen2.5 / Hermes convention:

  * tool DEFINITIONS are presented as OpenAI-nested JSON function schemas wrapped in a
    ``<tools>...</tools>`` block (supplied via :meth:`tool_definitions`, which the
    SystemPromptBuilder substitutes for the default Markdown docs), and
  * each tool CALL is a JSON object ``{"name": <tool>, "arguments": {...}}`` wrapped in
    its own ``<tool_call>...</tool_call>`` tag; the model emits one or more consecutive
    blocks per turn.

The model card warns that tool-use performance "depends heavily on the format and
structure of tool definitions provided at inference time", so aligning BOTH the tool
definitions and the call wire-format to the trained dialect is the whole point of this
codec (see ``MYTHOS_FAST_ANALYSIS.md`` sections B and C).

Like ``xml`` and ``tagged_json``, the ``<tool_call>`` tag wrapper is not expressible as
a JSON-schema ``format``, so the action phase is left UNCONSTRAINED and this codec's
:meth:`parse` does the structural work. (A GBNF grammar that locks the output to
``<tool_call>\n{json}\n</tool_call>`` is an optional, backend-gated upgrade — honoured
only by a llama.cpp backend, not Ollama's ``format`` — so it is not the baseline; the
model is fine-tuned to emit this shape natively.)

This codec lives in its own isolated module and shares no code file with the other
formats, so it merges without conflict (see :func:`vibeharness.codec.get_codec`).
"""
from __future__ import annotations

import json
import re

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry

# Match every <tool_call>...</tool_call> block. The closing tag is optional so a final
# block whose closing tag the model forgot is still recovered (the inner text then runs
# to end-of-string), mirroring tagged_json's tolerance. DOTALL so a JSON object may span
# multiple lines (the native template puts the object on its own line).
_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)(?:</tool_call>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


class HermesCodec(ToolCallCodec):
    name = "hermes"

    def format_instructions(self, max_actions: int) -> str:
        cap = (f" You may emit at most {max_actions} tool calls per turn."
               if max_actions and max_actions > 0 else "")
        return (
            "- Each turn, emit one or more tool calls. Return EACH call as a JSON object "
            'of the form {"name": <tool-name>, "arguments": {...}} wrapped in its own '
            "<tool_call></tool_call> tags, like:\n"
            "    <tool_call>\n"
            '    {"name": "write_file", "arguments": {"path": "notes/todo.txt", '
            '"content": "buy milk"}}\n'
            "    </tool_call>\n"
            "  Emit consecutive <tool_call> blocks to make several calls; they run in "
            "order. Use only the functions listed in the <tools> block below.\n"
            "- Batch independent or predictable calls in one turn (e.g. write a file then "
            "read it back); emit a single call when you must see its result before deciding."
            + cap
        )

    def turn_action_hint(self) -> str:
        return ('Respond with one or more <tool_call>{"name": ..., "arguments": {...}}'
                "</tool_call> blocks.")

    def tool_definitions(self, registry: ToolRegistry) -> str | None:
        """Render tools as the Hermes ``<tools>`` function-schema block instead of the
        default Markdown docs — the exact tool-definition format mythos_fast expects."""
        return registry.tools_block(style="hermes")

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        # The <tool_call> tag wrapper isn't expressible as a JSON-schema `format`, so
        # decoding is left unconstrained and validated by parse(). The model is
        # fine-tuned to emit this shape natively. (GBNF would be a backend-gated upgrade.)
        return DecodeConstraint(json_schema=None)

    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        blocks = _BLOCK_RE.findall(raw or "")
        # Drop any block that is pure whitespace (e.g. a stray opening tag at end of
        # output) so it doesn't masquerade as an empty, invalid call.
        blocks = [b for b in blocks if b.strip()]
        if not blocks:
            return None, "no <tool_call>...</tool_call> blocks found"
        actions: list[ToolCall] = []
        for inner in blocks:
            try:
                obj = json.loads(inner.strip())
            except json.JSONDecodeError as e:
                return None, f"invalid JSON inside <tool_call> block ({e})"
            if not isinstance(obj, dict) or "name" not in obj:
                return None, "each <tool_call> block must contain an object with a 'name' field"
            name = obj["name"]
            if not isinstance(name, str) or not name:
                return None, "'name' must be a non-empty string"
            args = obj.get("arguments", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return None, "'arguments' must be an object"
            actions.append((name, args))
        return actions, None


CODEC = HermesCodec()
