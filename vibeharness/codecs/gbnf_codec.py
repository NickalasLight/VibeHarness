"""A GBNF-constrained JSON tool-call codec.

Same wire format as the JSON codec — a JSON array of ``{"tool": <name>, "args": {...}}``
objects — but the decode-time constraint is a hand-built GBNF grammar (llama.cpp's
grammar format) carried in ``DecodeConstraint.gbnf`` rather than a JSON schema.

A llama.cpp backend honours the grammar and can only emit text the grammar accepts;
Ollama ignores ``gbnf``, so under Ollama this codec generates freely and the parser
(identical in robustness to the JSON codec) recovers the JSON regardless. This file is
intentionally self-contained — it shares no code with ``json_codec``.
"""
from __future__ import annotations

import json

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry


def _gbnf_string_literal(s: str) -> str:
    r"""A GBNF terminal matching the exact JSON string ``"s"`` (quotes included).

    The tool name is embedded as a quoted, escaped literal so the grammar only
    accepts that one spelling. GBNF string literals use the same backslash escapes
    as JSON, so escaping ``\`` and ``"`` is sufficient for tool names.
    """
    body = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"\\"{body}\\""'


# ---------------------------------------------------------------------------
# Grammar structure (top to bottom):
#   root        -> a JSON array of 1..N action objects (N = max_actions cap)
#   action      -> { "tool": <one of the tool-name literals> , "args": object }
#   toolname    -> alternation of the registry's exact tool-name string literals
#   value/object/array/string/number/ws -> a general JSON-value subgrammar so
#                  "args" (and any nested value) accepts any well-formed JSON object.
# The array's item count is bounded by emitting the first action, then a fixed
# number of optional ``, action`` groups (max_actions - 1 of them) when capped,
# or an unbounded ``( "," ws action )*`` when uncapped.
# ---------------------------------------------------------------------------
def _build_grammar(registry: ToolRegistry, max_actions: int) -> str:
    tool_names = [t.name for t in registry.all()]
    toolname_alt = " | ".join(_gbnf_string_literal(n) for n in tool_names)

    if max_actions and max_actions > 0:
        # First action, then up to (max_actions - 1) further optional actions.
        extra = "".join(' ( "," ws action )?' for _ in range(max_actions - 1))
        root_items = f"action{extra}"
    else:
        root_items = 'action ( "," ws action )*'

    lines = [
        f'root        ::= ws "[" ws {root_items} ws "]" ws',
        '',
        'action      ::= "{" ws "\\"tool\\"" ws ":" ws toolname ws "," ws '
        '"\\"args\\"" ws ":" ws object ws "}"',
        '',
        f'toolname    ::= {toolname_alt}',
        '',
        '# --- general JSON value subgrammar (args accepts any JSON object) ---',
        'value       ::= object | array | string | number | '
        '"true" | "false" | "null"',
        '',
        'object      ::= "{" ws ( string ws ":" ws value '
        '( ws "," ws string ws ":" ws value )* )? ws "}"',
        '',
        'array       ::= "[" ws ( value ( ws "," ws value )* )? ws "]"',
        '',
        'string      ::= "\\"" ( [^"\\\\] | "\\\\" ["\\\\/bfnrt] | '
        '"\\\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\\""',
        '',
        'number      ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? '
        '( [eE] [-+]? [0-9]+ )?',
        '',
        'ws          ::= [ \\t\\n\\r]*',
    ]
    return "\n".join(lines)


class GbnfCodec(ToolCallCodec):
    name = "gbnf"

    def format_instructions(self, max_actions: int) -> str:
        cap = (f" You may emit at most {max_actions} actions per turn."
               if max_actions and max_actions > 0 else "")
        return (
            '- Each turn, output a JSON ARRAY of one or more actions of the form '
            '{"tool": <name>, "args": {...}}; they run in order. Output only that array — '
            'no prose. Use only the tools listed below.\n'
            '- Batch independent or predictable actions in one turn (e.g. write a file then '
            'read it back); emit a single action when you must see its result before deciding.'
            + cap
        )

    def turn_action_hint(self) -> str:
        return "Respond with a JSON array of one or more actions."

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        return DecodeConstraint(gbnf=_build_grammar(registry, max_actions), json_schema=None)

    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"not valid JSON ({e})"
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list) or not obj:
            return None, "expected a non-empty JSON array of actions"
        actions: list[ToolCall] = []
        for item in obj:
            if not isinstance(item, dict) or "tool" not in item:
                return None, "each action must be an object with a 'tool' field"
            args = item.get("args", {})
            if not isinstance(args, dict):
                return None, "'args' must be an object"
            actions.append((item["tool"], args))
        return actions, None


CODEC = GbnfCodec()
