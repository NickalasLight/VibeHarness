"""A pure-XML tool-call codec (Anthropic-style tags, zero JSON escaping).

Tool calls are one or more ``<tool_call name="...">`` blocks, each holding
``<arg name="...">value</arg>`` children whose text is the *raw* string content —
no JSON quoting or escaping. Because XML cannot be expressed as a JSON-schema
``format``, the action phase is left unconstrained at decode time and the parser
does all the structural work: it pulls each block out with a tolerant regex,
unescapes the five XML entities, and coerces obvious int/float/bool literals so a
tool that wants an int ``page`` gets an int.

This codec lives in its own isolated module and shares no code file with the
other formats, so it merges without conflict (see :func:`vibeharness.codec.get_codec`).
"""
from __future__ import annotations

import re

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry

# A whole <tool_call ...> ... </tool_call> block. DOTALL so values may span lines.
_BLOCK_RE = re.compile(
    r"<tool_call\b([^>]*)>(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# An <arg name="...">value</arg> child inside a block.
_ARG_RE = re.compile(
    r'<arg\b[^>]*?\bname\s*=\s*"([^"]*)"[^>]*>(.*?)</arg>',
    re.DOTALL | re.IGNORECASE,
)
# The name="..." attribute on a tool_call open tag.
_NAME_RE = re.compile(r'\bname\s*=\s*"([^"]*)"', re.IGNORECASE)


def _unescape(text: str) -> str:
    """Unescape the five core XML entities. ``&amp;`` is resolved last so an
    already-correct ``&amp;lt;`` does not collapse into ``<``."""
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


_INT_RE = re.compile(r"[+-]?\d+$")
_FLOAT_RE = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def _coerce(value: str):
    """Turn an obvious int/float/bool literal into the native type; otherwise
    keep the (already-unescaped) string. Whitespace around the literal is ignored
    for the type test but trimmed only for the typed result, never for plain text."""
    stripped = value.strip()
    low = stripped.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(stripped):
        try:
            return int(stripped)
        except ValueError:
            return value
    if _FLOAT_RE.match(stripped) and any(c in stripped for c in ".eE"):
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


class XmlCodec(ToolCallCodec):
    name = "xml"

    def format_instructions(self, max_actions: int) -> str:
        cap = (f" You may emit at most {max_actions} actions per turn."
               if max_actions and max_actions > 0 else "")
        return (
            "- Each turn, output one or more tool calls as XML blocks, in the order "
            "they should run. Output only the blocks — no prose. Use only the tools "
            "listed below. Each block looks like:\n"
            '    <tool_call name="write_file">\n'
            '      <arg name="path">notes/todo.txt</arg>\n'
            '      <arg name="content">buy milk</arg>\n'
            "    </tool_call>\n"
            "- Each <arg> value is the raw string content: write it literally, with no "
            "JSON quoting or escaping. (Only the XML entities &amp;amp; &amp;lt; &amp;gt; "
            "&amp;quot; need escaping if those characters appear in a value.)\n"
            "- Batch independent or predictable actions in one turn (e.g. write a file "
            "then read it back); emit a single block when you must see its result before "
            "deciding."
            + cap
        )

    def turn_action_hint(self) -> str:
        return 'Respond with one or more <tool_call name="..."> blocks.'

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        # XML is not expressible as a JSON-schema `format`; leave the action phase
        # unconstrained and let parse() do the structural work.
        return DecodeConstraint(json_schema=None)

    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        if raw is None:
            return None, "no <tool_call> blocks found"
        actions: list[ToolCall] = []
        for attrs, body in _BLOCK_RE.findall(raw):
            name_match = _NAME_RE.search(attrs)
            if not name_match:
                return None, '<tool_call> is missing its name="..." attribute'
            tool = _unescape(name_match.group(1)).strip()
            if not tool:
                return None, '<tool_call> has an empty name'
            args: dict = {}
            for arg_name, arg_value in _ARG_RE.findall(body):
                key = _unescape(arg_name).strip()
                args[key] = _coerce(_unescape(arg_value))
            actions.append((tool, args))
        if not actions:
            return None, "no <tool_call> blocks found"
        return actions, None


CODEC = XmlCodec()
