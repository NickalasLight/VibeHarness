"""External-API LLM client (OpenAI-compatible).

A drop-in :class:`~vibeharness.llm.LLMClient` that talks to any OpenAI-compatible
HTTP API (default provider: ZhipuAI / z.ai ``glm-5.2``) instead of the local
Ollama backend. It is used in two places:

  - escalation — the agent swaps its client to this mid-run when it detects a
    stuck loop, keeping the same browser/session (see :mod:`vibeharness.escalation`);
  - validation — the validator can always run on the API model for a stronger,
    independent second opinion.

Design notes / clean-architecture rules honoured here:
  - This module imports NOTHING from ``agent.py`` or ``web.py``; the dependency
    direction is ``agent -> api_llm <- providers <- config``.
  - The api key is passed in (read from the environment by :mod:`providers`); it is
    never read, logged, or stored by this module beyond the live client object.
  - The harness speaks a constrained JSON-array action protocol (not provider-native
    tool calls), so :meth:`decide` asks the model to emit exactly that JSON, matching
    :class:`vibeharness.llm.OllamaClient` semantics and the existing agent loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .codec import DecodeConstraint
from .llm import Decision, LLMClient, TokenSink
from .providers import ApiProviderConfig


class ApiUnavailable(RuntimeError):
    """Raised when the external API cannot be reached or returns an error."""


# --- Ollama-shaped tool_call adapters -------------------------------------
# The agent dispatch path (in feature-rich harnesses) reads ``tc.function.name``
# and ``dict(tc.function.arguments)``. These adapters satisfy that shape so a
# provider-native tool call can be surfaced identically. This barebones harness
# parses a JSON-array action instead, but the adapters are kept for parity and so
# future native-tool wiring needs no client changes.
@dataclass
class _ApiFunction:
    name: str
    arguments: dict          # already parsed from the JSON string


@dataclass
class _ApiToolCall:
    function: _ApiFunction


_ACTION_INSTRUCTION = (
    "Respond with ONLY a JSON value that conforms to this JSON Schema — no prose, "
    "no markdown fences, no explanation:\n{schema}\n"
    "Output the JSON value and nothing else."
)


class ApiLLMClient(LLMClient):
    """:class:`LLMClient` backed by an OpenAI-compatible API (e.g. z.ai GLM)."""

    def __init__(self, provider: ApiProviderConfig, api_key: str, model: str,
                 temperature: float = 0.3, timeout: int = 600,
                 price_per_1k_in: float = 0.00015,
                 price_per_1k_out: float = 0.00015):
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._price_per_1k_in = price_per_1k_in
        self._price_per_1k_out = price_per_1k_out
        # cumulative token counters (updated after each call)
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        try:
            from openai import OpenAI
        except ImportError as e:   # pragma: no cover - exercised via providers tests
            raise ImportError(
                "the 'openai' package is required for the API LLM client; "
                "install it with `pip install openai` (or `pip install vibeharness[api]`)."
            ) from e
        self._client = OpenAI(api_key=api_key, base_url=provider.base_url,
                              timeout=timeout)

    def cost_usd(self) -> float:
        """Estimated USD cost for all API calls made via this client so far."""
        return (self.tokens_in / 1000.0 * self._price_per_1k_in
                + self.tokens_out / 1000.0 * self._price_per_1k_out)

    def usage_summary(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "estimated_cost_usd": round(self.cost_usd(), 6),
        }

    def supports_native_tools(self) -> bool:
        """The OpenAI-compatible client is SINGLE-SHOT (issue #163): it does not speak
        Ollama's native ``tools:`` /api/chat protocol, so it reports ``False``. The agent
        therefore routes API-backed roles to ``_decide`` (single-shot) and the harness
        auto-selects a constrained-JSON codec for them, rather than the stateful
        ``decide_chat`` path. ``decide_chat`` stays inherited from the base default
        (flatten → ``decide``) for any caller that still invokes it."""
        return False

    # ---- LLMClient interface ----
    def decide(self, system: str, user: str,
               constraint: "DecodeConstraint | dict | None",
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        """Single-shot decision matching the real :class:`~vibeharness.llm.LLMClient`
        interface (issue #163 — repairs the stale ``action_schema`` signature that broke
        the validator and escalation API paths).

        ``constraint`` is a :class:`~vibeharness.codec.DecodeConstraint`. Its
        ``json_schema`` (when not ``None``) becomes the JSON-instruction appended to the
        user turn so the model emits exactly that shape — the agent's constrained-action
        array, or the validator's verdict schema; the schema clause is OMITTED when
        ``json_schema`` is ``None`` (an unconstrained codec). Its ``stop`` strings are
        forwarded to the API's ``stop`` parameter.

        For backward compatibility a bare ``dict`` (a raw JSON schema) or ``None`` is also
        accepted, so historical callers/tests that passed a schema directly keep working.
        Returns the emitted text as ``action_json``; ``reasoning`` carries any
        reasoning-trace tokens the model exposed separately (see :meth:`_chat`)."""
        schema, stop = _unpack_constraint(constraint)
        instructed_user = user
        if schema is not None:
            instructed_user = f"{user}\n\n" + _ACTION_INSTRUCTION.format(
                schema=json.dumps(schema, ensure_ascii=False))
        text, reasoning = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": instructed_user},
        ], on_token=on_action, on_reason=on_reason, stop=stop)
        # The action is parsed ONLY from the constrained ``content`` (issue #179): the
        # model's thinking trace is streamed to ``on_reason`` and stored in
        # ``Decision.reasoning`` SEPARATELY, so it never reaches the tool-call parser. With
        # a JSON-schema constraint present (the json codec for GLM) the model emits the
        # tool-call JSON as ``content`` and its thinking as ``reasoning_content`` — the
        # #179 root cause (thinking prose landing in the action channel) cannot recur.
        action = text
        if not action.strip():
            # GLM reasoning models occasionally stream the WHOLE answer as
            # reasoning_content and leave content empty (e.g. the validator's verdict). To
            # avoid losing it WITHOUT feeding free-form thinking to the parser, recover the
            # schema-shaped JSON value embedded in the reasoning; only if none is found do
            # we fall back to the raw reasoning (preserving the prior behaviour).
            action = _recover_json(reasoning) or reasoning
        return Decision(reasoning=reasoning, action_json=_strip_fences(action))

    # ---- transport ----
    def _chat(self, messages: list[dict], on_token: TokenSink | None,
              on_reason: TokenSink | None = None,
              stop: "tuple[str, ...]" = ()) -> "tuple[str, str]":
        """Stream a chat completion, returning ``(content, reasoning)``.

        ``content`` is the visible answer; ``reasoning`` accumulates any
        ``delta.reasoning_content`` a reasoning model (GLM-4.x) streams separately — those
        tokens go to ``on_reason`` and never pollute the JSON answer. Visible content goes
        to ``on_token``. Falls back to a non-streaming call if the provider rejects
        streaming. ``stop`` is forwarded as the API ``stop`` parameter when non-empty."""
        kwargs: dict = {"model": self._model, "messages": messages,
                        "temperature": self._temperature}
        if stop:
            kwargs["stop"] = list(stop)
        try:
            stream = self._client.chat.completions.create(
                stream=True, stream_options={"include_usage": True}, **kwargs)
            parts: list[str] = []
            reasoning_parts: list[str] = []
            for chunk in stream:
                # usage chunk (last, no choices)
                if hasattr(chunk, "usage") and chunk.usage and not chunk.choices:
                    self.tokens_in += (chunk.usage.prompt_tokens or 0)
                    self.tokens_out += (chunk.usage.completion_tokens or 0)
                    continue
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                think = getattr(delta, "reasoning_content", None)
                if think:
                    reasoning_parts.append(think)
                    if on_reason:
                        on_reason(think)
                piece = getattr(delta, "content", None)
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
            return "".join(parts), "".join(reasoning_parts)
        except Exception as e:   # openai.APIError, connection errors, etc.
            # Some endpoints/models reject streaming; retry once non-streamed.
            if _is_streaming_unsupported(e):
                return self._chat_once(messages, on_token, on_reason, stop)
            raise ApiUnavailable(
                f"{self._provider.name} API call failed "
                f"(model={self._model}): {type(e).__name__}: {e}"
            ) from e

    def _chat_once(self, messages: list[dict], on_token: TokenSink | None,
                   on_reason: TokenSink | None = None,
                   stop: "tuple[str, ...]" = ()) -> "tuple[str, str]":
        kwargs: dict = {"model": self._model, "messages": messages,
                        "temperature": self._temperature, "stream": False}
        if stop:
            kwargs["stop"] = list(stop)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ApiUnavailable(
                f"{self._provider.name} API call failed "
                f"(model={self._model}): {type(e).__name__}: {e}"
            ) from e
        usage = getattr(resp, "usage", None)
        if usage:
            self.tokens_in += (getattr(usage, "prompt_tokens", 0) or 0)
            self.tokens_out += (getattr(usage, "completion_tokens", 0) or 0)
        message = resp.choices[0].message if resp.choices else None
        text = (getattr(message, "content", "") or "") if message else ""
        reasoning = (getattr(message, "reasoning_content", "") or "") if message else ""
        if reasoning and on_reason:
            on_reason(reasoning)
        if text and on_token:
            on_token(text)
        return text, reasoning


def _unpack_constraint(
    constraint: "DecodeConstraint | dict | None",
) -> "tuple[dict | None, tuple[str, ...]]":
    """Return ``(json_schema, stop)`` from a decode constraint (issue #163).

    The agent and validator pass a :class:`~vibeharness.codec.DecodeConstraint`; we read
    its ``json_schema`` (the shape to instruct, or ``None`` for an unconstrained codec) and
    its ``stop`` strings. For backward compatibility a bare schema ``dict`` is treated as
    the json_schema with no stop strings, and ``None`` means "no schema, no stop"."""
    if constraint is None:
        return None, ()
    if isinstance(constraint, DecodeConstraint):
        return constraint.json_schema, tuple(constraint.stop or ())
    if isinstance(constraint, dict):
        return constraint, ()
    # Duck-typed fallback: anything exposing the two attributes.
    return getattr(constraint, "json_schema", None), tuple(getattr(constraint, "stop", ()) or ())


def _recover_json(text: str) -> str:
    """Extract the first complete top-level JSON object/array embedded in ``text``.

    Used only as the empty-``content`` fallback (issue #179): a GLM reasoning model may put
    the whole schema-shaped answer in ``reasoning_content`` with content empty. A
    brace/bracket-balanced, string-aware scan recovers the JSON value so the constrained
    answer is not lost, while free-form thinking prose (no JSON) yields ``""`` and is NOT
    treated as a tool call. Returns the matched JSON substring, or ``""`` if none parses."""
    s = (text or "").strip()
    if not s:
        return ""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
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
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return candidate
    return ""


def _is_streaming_unsupported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "stream" in msg and ("unsupported" in msg or "not support" in msg
                                or "must be" in msg or "disabled" in msg)


def _strip_fences(text: str) -> str:
    """Strip a leading ```json / ``` fence the model may have added despite the
    instruction, so the agent's JSON parser sees a clean payload."""
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()
