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
                 temperature: float = 0.3, timeout: int = 600):
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        try:
            from openai import OpenAI
        except ImportError as e:   # pragma: no cover - exercised via providers tests
            raise ImportError(
                "the 'openai' package is required for the API LLM client; "
                "install it with `pip install openai` (or `pip install vibeharness[api]`)."
            ) from e
        self._client = OpenAI(api_key=api_key, base_url=provider.base_url,
                              timeout=timeout)

    # ---- LLMClient interface ----
    def decide(self, system: str, user: str, action_schema: dict,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        """Single-shot decision. The model is asked to emit JSON matching
        ``action_schema`` (the agent's constrained-action protocol). For the
        validator, ``action_schema`` is the verdict schema. Returns the emitted
        text as ``action_json``; reasoning is left empty (the API hides it)."""
        instructed_user = f"{user}\n\n" + _ACTION_INSTRUCTION.format(
            schema=json.dumps(action_schema, ensure_ascii=False))
        text = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": instructed_user},
        ], on_token=on_action)
        return Decision(reasoning="", action_json=_strip_fences(text))

    # ---- transport ----
    def _chat(self, messages: list[dict], on_token: TokenSink | None) -> str:
        """Stream a chat completion, emitting chunks to ``on_token``. Falls back to
        a non-streaming call if the provider does not support streaming."""
        try:
            stream = self._client.chat.completions.create(
                model=self._model, messages=messages,
                temperature=self._temperature, stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
            return "".join(parts)
        except Exception as e:   # openai.APIError, connection errors, etc.
            # Some endpoints/models reject streaming; retry once non-streamed.
            if _is_streaming_unsupported(e):
                return self._chat_once(messages, on_token)
            raise ApiUnavailable(
                f"{self._provider.name} API call failed "
                f"(model={self._model}): {type(e).__name__}: {e}"
            ) from e

    def _chat_once(self, messages: list[dict], on_token: TokenSink | None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages,
                temperature=self._temperature, stream=False,
            )
        except Exception as e:
            raise ApiUnavailable(
                f"{self._provider.name} API call failed "
                f"(model={self._model}): {type(e).__name__}: {e}"
            ) from e
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        if text and on_token:
            on_token(text)
        return text


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
