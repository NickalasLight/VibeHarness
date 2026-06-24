"""The default JSON tool-call codec.

Tool calls are a JSON array of ``{"tool": <name>, "args": {...}}`` objects, and the
action phase is constrained at decode time by the registry's JSON action-schema
(Ollama's ``format`` field). This is the format the harness shipped with; it is the
baseline the other codecs are benchmarked against.
"""
from __future__ import annotations

import json

from ..codec import DecodeConstraint, ToolCall, ToolCallCodec
from ..registry import ToolRegistry


class JSONCodec(ToolCallCodec):
    name = "json"

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
        limit = max_actions if max_actions and max_actions > 0 else None
        return DecodeConstraint(json_schema=registry.action_schema(max_items=limit))

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


CODEC = JSONCodec()
