"""The code-as-action (CodeAct) tool-call codec.

Tool calls are written as Python-style function calls inside a fenced ```python
code block — one call per top-level statement, executed in order::

    ```python
    create_file(path="notes/todo.txt", content="buy milk")
    read_file(path="notes/todo.txt")
    ```

Each statement is a call to a tool by name, with arguments passed as keyword
arguments whose values are ordinary Python literals (strings, ints, bools,
lists, dicts, ...). Decoding is unconstrained — Python source is not a
JSON-schema shape — so the whole burden of validity falls on :meth:`parse`.

SAFETY: model output is NEVER executed. Parsing is done purely by static
analysis with the :mod:`ast` module (``ast.parse``) plus
:func:`ast.literal_eval` on individual argument values. No ``exec``/``eval`` of
model text ever happens.
"""
from __future__ import annotations

import ast
import re

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry

# Prefer a fenced ```python ... ``` block; otherwise any ``` ... ``` block.
_FENCE_PY = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_ANY = re.compile(r"```[ \t]*\r?\n(.*?)```", re.DOTALL)


class CodeActCodec(ToolCallCodec):
    name = "codeact"

    def format_instructions(self, max_actions: int) -> str:
        cap = (f" Emit at most {max_actions} calls per turn."
               if max_actions and max_actions > 0 else "")
        return (
            "- Each turn, respond with a ```python code block containing one or more "
            "tool calls written as `tool_name(arg=value, ...)`, one call per line. "
            "The calls run top to bottom, in order. Output only that code block — no "
            "prose. Use only the tools listed below.\n"
            "- Pass every argument as a keyword argument (`name=value`); positional "
            "arguments are not allowed. Each value must be a plain Python literal: a "
            "string, int, float, bool, None, list, or dict.\n"
            "- Batch independent or predictable calls in one block (e.g. write a file "
            "then read it back); emit a single call when you must see its result before "
            "deciding."
            + cap
        )

    def turn_action_hint(self) -> str:
        return "Respond with a ```python code block of tool calls, one per line."

    def constraint(self, registry: ToolRegistry, max_actions: int) -> DecodeConstraint:
        # Python source is not a JSON-schema-shaped format, so the action phase is
        # left unconstrained; parse() bears the full validity burden.
        return DecodeConstraint(json_schema=None)

    def _extract_block(self, raw: str) -> str:
        m = _FENCE_PY.search(raw)
        if m:
            return m.group(1)
        m = _FENCE_ANY.search(raw)
        if m:
            return m.group(1)
        return raw

    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        source = self._extract_block(raw)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return None, f"not valid Python ({e.msg})"

        actions: list[ToolCall] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                return None, (
                    "each line must be a single tool call of the form "
                    "tool_name(arg=value, ...)"
                )
            call = stmt.value
            if not isinstance(call.func, ast.Name):
                return None, "the call target must be a plain tool name, e.g. read_file(...)"
            tool = call.func.id
            if call.args:
                return None, (
                    f"{tool}(...) used positional arguments; pass every argument as a "
                    "keyword argument, e.g. path=\"...\""
                )
            args: dict = {}
            for kw in call.keywords:
                if kw.arg is None:
                    return None, f"{tool}(...) used **kwargs unpacking; pass explicit keyword arguments"
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError, TypeError):
                    return None, (
                        f"argument '{kw.arg}' of {tool}(...) is not a plain literal; "
                        "use a string, int, float, bool, None, list, or dict"
                    )
            actions.append((tool, args))

        if not actions:
            return None, "no tool calls found in the python code block"
        return actions, None


CODEC = CodeActCodec()
