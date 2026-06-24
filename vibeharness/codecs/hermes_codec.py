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

# FALLBACK recovery (#125 iter 1): an instruct model NOT trained on the <tool_call> tags
# (e.g. qwen2.5-coder:3b-instruct) emits the correct {"name","arguments"} JSON but wraps
# it in a ```json ... ``` markdown fence — or bare — with no <tool_call> tags at all. A
# well-formed tool call must never be discarded over a missing wrapper, so when no
# <tool_call> block is present we recover JSON objects from fences / raw text instead.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _unwrap_arg_value(v):
    """Repair the SCHEMA-LEAK argument bug a weak 3B model emits under native tools.

    Ground-truthed (NOT assumed) from live ``/api/chat`` runs against
    ``qwen2.5-coder:3b-instruct`` on Ollama 0.30.8 with the ``tools:`` field set: the
    model sometimes echoes the JSON-Schema describing the parameter and tucks the real
    value under a ``value`` key, e.g. ``{"type": "string", "description": "...",
    "value": "Paris"}``. Recover the ``value``. Anything that is not a recognised
    schema-leak wrapper is returned unchanged, so well-formed values (strings, numbers,
    legitimate nested objects) pass through untouched."""
    if isinstance(v, dict) and "value" in v and (
            "type" in v or "description" in v) and not (
            set(v) - {"type", "description", "value", "enum", "default"}):
        return v["value"]
    return v


def _repair_arguments(name: str, args: dict) -> dict:
    """Best-effort repair of an arguments object emitted by a weak 3B model.

    Fixes the two argument-shape malformations observed live (see module-level
    ground-truth note and ``_unwrap_arg_value``):

      * SCHEMA-LEAK: a per-value ``{"type"/"description"/"value": ...}`` wrapper.
      * NESTED-ARG:  the whole value is a one-key object echoing the argument name,
        ``{"city": {"city": "Paris"}}`` -> ``{"city": "Paris"}``.

    Conservative: only these two empirically observed shapes are touched; every other
    value is preserved verbatim so correct calls are never altered."""
    repaired: dict = {}
    for k, val in args.items():
        # NESTED-ARG: {"city": {"city": "Paris"}} -> "Paris"
        if isinstance(val, dict) and len(val) == 1 and k in val:
            val = val[k]
        repaired[k] = _unwrap_arg_value(val)
    return repaired


def _iter_json_values(text: str):
    """Yield JSON values parsed from ``text``, tolerating surrounding prose.

    Tries a whole-string parse first (handles a lone object or a top-level array of
    calls), then falls back to a brace-balanced scan that extracts and parses each
    top-level ``{...}`` region (string-/escape-aware so braces inside string values
    don't throw off the depth count)."""
    text = (text or "").strip()
    if not text:
        return
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    yield json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                start = None


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
            "within <tool_call></tool_call> XML tags with NO other text. "
            "Do not include any backticks or ```json:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n"
            "- Emit consecutive <tool_call> blocks to make several calls; they run in "
            "order. Use only the functions listed in the <tools> block below.\n"
            "- Batch independent or predictable calls in one turn (e.g. fill several "
            "fields, then click Next); emit a single call when you must see its result "
            "before deciding."
            + cap
        )

    def turn_action_hint(self) -> str:
        # Concrete multi-call example in the recency zone so the 3B model learns the
        # pattern from an in-context demonstration, not just an abstract instruction.
        return (
            "Respond with one or more CONSECUTIVE <tool_call> blocks — one block per action.\n"
            "Example (two actions in one turn):\n"
            '<tool_call>\n{"name": "fill", "arguments": {"target": "e12", "text": "Alice"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "fill", "arguments": {"target": "e14", "text": "alice@example.com"}}\n</tool_call>\n'
            "ALWAYS batch independent form-field fills, clicks, and selections together. "
            "Use a single <tool_call> ONLY when you must see a result before deciding the next step."
        )

    # ---- native Ollama tool calling (#129/#130/#131) ----
    #
    # GROUND TRUTH (live `/api/chat` against qwen2.5-coder:3b-instruct, Ollama 0.30.8,
    # with the `tools:` field set): passing tools natively makes Ollama inject its OWN
    # `# Tools` block — the ENVELOPED `{"type":"function","function":{...}}` schemas the
    # model was trained on, PLUS the anti-fence clause ("Do not include any backticks or
    # ```json") — straight from the model's chat template. That is strictly better than
    # the harness hand-injecting a BARE <tools> block. BUT: this 3B model still does NOT
    # reliably emit clean <tool_call> tags, and Ollama 0.30.8 does NOT parse its output
    # into structured `message.tool_calls` (observed null across runs); the call lands in
    # `message.content` as <tool_call>, a ```json fence, or bare/array JSON, sometimes
    # with the schema-leak / nested-arg bugs. So the native path is: send enveloped
    # `tools:`, then read `tool_calls` IF present, else fall back to parse(content).

    def tools(self, registry: "ToolRegistry") -> list[dict]:
        """The enveloped tool schemas for Ollama's native ``tools:`` request field.

        Each tool is ``{"type": "function", "function": {"name", "description",
        "parameters": <args schema>}}`` — the OpenAI/Ollama envelope. ``parameters``
        comes from each tool's ``_args_schema()`` — the SAME source the bare ``<tools>``
        block and the JSON-schema constraint use, so the three can never drift.

        ENVELOPE vs BARE (reconciling CORRECTIONS.md / PR #135): that audit says the
        *wire shape the MODEL sees* must be BARE (``{"name","description","parameters"}``)
        because the raw HF template renders ``tool | tojson``. That is true — and it is
        exactly what happens here. The qwen2.5-coder **Ollama** modelfile template (ground
        truth, captured from ``ollama show --modelfile``) renders each tool as
        ``{"type": "function", "function": {{ .Function }}}`` where ``.Function`` is the
        BARE object. So Ollama's ``tools:`` API contract takes the ENVELOPED schema and
        Ollama itself emits the bare shape into the prompt. Passing the envelope to the
        ``tools:`` field is therefore correct AND yields the bare wire shape the model was
        trained on; do NOT strip the envelope here (that BARE rule is for the *manual*
        ``<tools>`` text injection in ``tool_definitions``/``tools_block``, which is kept
        bare and is used only on the non-native path). Verified end-to-end live."""
        return [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t._args_schema(),
            }}
            for t in registry.all()
        ]

    def parse_tool_calls(self, tool_calls: list) -> "list[ToolCall]":
        """Convert Ollama's structured ``message.tool_calls`` into ``(name, args)`` pairs.

        Ollama returns ``[{"function": {"name": ..., "arguments": {...}}}, ...]`` (the
        ``arguments`` may already be a dict, or a JSON string for some backends). Applies
        the same ``_repair_arguments`` repair as the text path so the schema-leak / nested
        bugs are fixed regardless of how the call arrived. Skips entries with no usable
        name. Used by the LLM client's native path when ``tool_calls`` is non-empty;
        ``parse(content)`` remains the fallback for this model, which usually returns
        ``tool_calls: null`` and the call as text."""
        out: list[ToolCall] = []
        for tc in tool_calls or []:
            fn = (tc or {}).get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append((name, _repair_arguments(name, args)))
        return out

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
        raw = raw or ""
        # 1) Preferred: explicit <tool_call>...</tool_call> blocks (the trained wrapper).
        blocks = [b.strip() for b in _BLOCK_RE.findall(raw) if b.strip()]
        # 2) Fallback (#125 iter 1): no wrapper tags -> recover the {"name","arguments"}
        #    JSON the model DID emit, from ```json ... ``` fences, else from the raw text.
        #    Keeps a valid call from being thrown away just because the tags are missing.
        if not blocks:
            fenced = [m.strip() for m in _FENCE_RE.findall(raw) if m.strip()]
            blocks = fenced if fenced else ([raw.strip()] if raw.strip() else [])
        if not blocks:
            return None, ('no tool call found (expected a <tool_call>{...}</tool_call> '
                          'block or a {"name", "arguments"} JSON object)')

        actions: list[ToolCall] = []
        saw_json = False
        for inner in blocks:
            for value in _iter_json_values(inner):
                saw_json = True
                # A top-level array is treated as several calls; a lone object as one.
                items = value if isinstance(value, list) else [value]
                for obj in items:
                    if not isinstance(obj, dict) or "name" not in obj:
                        return None, ("each tool call must be a JSON object with a "
                                      "'name' field")
                    name = obj["name"]
                    if not isinstance(name, str) or not name:
                        return None, "'name' must be a non-empty string"
                    # Qwen sometimes uses "parameters" instead of "arguments" (the key
                    # its <tools> schema uses); accept either so a near-miss is not lost.
                    args = obj.get("arguments")
                    if args is None:
                        args = obj.get("parameters", {})
                    if args is None:
                        args = {}
                    if not isinstance(args, dict):
                        return None, "'arguments' must be an object"
                    actions.append((name, _repair_arguments(name, args)))
        if not saw_json:
            return None, ('could not parse a tool call: expected a '
                          '<tool_call>{...}</tool_call> block or a {"name", "arguments"} '
                          "JSON object")
        if not actions:
            return None, "no valid tool call (need a JSON object with a 'name' field)"
        return actions, None


CODEC = HermesCodec()
