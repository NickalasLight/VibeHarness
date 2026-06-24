"""The Hermes / Qwen2.5 native tool-call codec.

This codec speaks the standard Qwen2.5 / Hermes tool-calling convention — the format
`qwen2.5-coder:3b-instruct` (the default model on branch ``beta_qwen3coder``, issue
#123) and the broader Qwen2.5 family were trained on:

  * tool DEFINITIONS are presented as **bare** JSON function schemas
    ``{"name", "description", "parameters"}`` — ONE per line — wrapped in a
    ``<tools>...</tools>`` block (supplied via :meth:`tool_definitions`, which the
    SystemPromptBuilder substitutes for the default Markdown docs), and
  * each tool CALL is a JSON object ``{"name": <tool>, "arguments": {...}}`` wrapped in
    its own ``<tool_call>...</tool_call>`` tag; the model emits one or more consecutive
    blocks per turn.

This was ground-truthed (NOT assumed) from the model's authoritative chat template:
``Qwen/Qwen2.5-Coder-3B-Instruct`` ``tokenizer_config.json`` renders each tool with a
BARE ``tool | tojson`` (no ``type``/``function`` envelope), instructs the model to
"return a json object with function name and arguments within <tool_call></tool_call>
XML tags", and feeds results back wrapped in ``<tool_response>`` (see
``QWEN3CODER_ANALYSIS.md`` for the captured fragments). The Qwen card warns that
tool-use performance "depends heavily on the format and structure of tool definitions
provided at inference time", so aligning BOTH the tool definitions and the call
wire-format to the trained dialect is the whole point of this codec.

Like ``xml`` and ``tagged_json``, the ``<tool_call>`` tag wrapper is not expressible as
a JSON-schema ``format``, so the action phase is left UNCONSTRAINED and this codec's
:meth:`parse` does the structural work. (A GBNF grammar that locks the output to
``<tool_call>\n{json}\n</tool_call>`` is an optional, backend-gated upgrade — honoured
only by a llama.cpp backend, not Ollama's ``format`` — so it is not the baseline; the
model emits this shape natively.)

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
        # Lead with the model's NATIVE instruction wording, verbatim from its chat
        # template. Under the harness's hand-rolled ChatML transport the native template
        # never fires, so the model only sees what we put here; reproducing the trained
        # instruction string keeps the model on its native single-pass <tools>/<tool_call>
        # distribution. The harness-specific batching guidance follows, clearly separated.
        cap = (f" You may emit at most {max_actions} tool calls per turn."
               if max_actions and max_actions > 0 else "")
        return (
            "You may call one or more functions to assist with the user query. You are "
            "provided with function signatures within <tools></tools> XML tags (see the "
            "# Tools section below).\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n"
            "- Emit consecutive <tool_call> blocks to make several calls; they run in "
            "order. Use only the functions listed in the <tools> block below.\n"
            "- Batch independent or predictable calls in one turn (e.g. write a file then "
            "read it back); emit a single call when you must see its result before deciding."
            + cap
        )

    def turn_action_hint(self) -> str:
        return ('Respond with one or more <tool_call>{"name": ..., "arguments": {...}}'
                "</tool_call> blocks.")

    def tool_definitions(self, registry: ToolRegistry) -> str | None:
        """Render tools as the Hermes ``<tools>`` block (BARE per-line function schemas)
        instead of the default Markdown docs — the exact tool-definition format the
        Qwen2.5/Qwen2.5-Coder native template renders via ``tool | tojson``."""
        return registry.tools_block(style="hermes")

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        # The <tool_call> tag wrapper isn't expressible as a JSON-schema `format`, so
        # decoding is left unconstrained and validated by parse(). The model emits this
        # shape natively. (A GBNF grammar would be a backend-gated llama.cpp upgrade; do
        # NOT re-introduce a `format` constraint here — it would fight the native dialect.)
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
