"""ToolRegistry: the single source of truth that the rest of the harness reads.

Adding a tool requires no changes to the agent or prompt code (Open/Closed):
docs and the constrained-decoding schema are both built from the registered tools.
"""
from __future__ import annotations

from .tools import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        if not tools:
            raise ValueError("registry needs at least one tool")
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"duplicate tool name: {t.name}")
            self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def action_schema(self, max_items: int | None = None) -> dict:
        """An array of one or more actions; each must match one tool's call schema.
        Forces every emitted action to be structurally valid. When ``max_items`` is
        given, adds ``maxItems`` so the model literally cannot emit more than that
        many actions in a single turn."""
        schema = {
            "type": "array",
            "minItems": 1,
            "items": {"oneOf": [t.call_schema() for t in self._tools.values()]},
        }
        if max_items is not None:
            schema["maxItems"] = max_items
        return schema

    def docs(self) -> str:
        return "\n\n".join(t.doc() for t in self._tools.values())

    def tools_block(self, style: str = "hermes") -> str:
        """A tool-definition block in an ALTERNATIVE wire style (issue #105 / #123).

        The default Markdown rendering lives in :meth:`docs`; this is the seam a
        codec uses when the model was fine-tuned to read tool definitions in a
        non-Markdown shape. ``style="hermes"`` emits the Qwen2.5 / Hermes
        ``<tools>...</tools>`` block: one **bare** function schema per line —
        ``{"name", "description", "parameters":<args schema>}`` — built from each
        tool's ``_args_schema()`` (the SAME parameter source the JSON-schema
        constraint uses, so the two can never drift).

        The shape is the BARE ``{"name","description","parameters"}`` object with NO
        ``{"type":"function","function":{...}}`` envelope. This was ground-truthed
        from the model's authoritative chat template — for ``beta_qwen3coder`` the
        Qwen2.5-Coder-3B-Instruct ``tokenizer_config.json`` renders each tool with a
        BARE ``tool | tojson`` (see ``QWEN3CODER_ANALYSIS.md``); the same bare shape
        was confirmed for the mythos_fast fine-tune (#105). Qwen's model card warns
        tool-use quality "depends heavily on the format of tool definitions", so we
        emit exactly the shape the model's template renders.

        Open/closed: adding a style adds a branch here; the existing ``docs()``
        Markdown path and the other codecs are untouched.
        """
        if style == "hermes":
            import json

            lines = ["<tools>"]
            for t in self._tools.values():
                # bare {"name","description","parameters"} — the exact shape the
                # model's native template renders via `tool | tojson`.
                fn = {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t._args_schema(),
                }
                lines.append(json.dumps(fn, ensure_ascii=False))
            lines.append("</tools>")
            return "\n".join(lines)
        raise ValueError(f"unknown tools_block style: {style!r}")
