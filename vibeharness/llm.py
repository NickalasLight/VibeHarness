"""LLM client.

The agent depends on the `LLMClient` abstraction (DIP); `OllamaClient` is one
implementation. There are THREE generation paths:

  * :meth:`OllamaClient.decide_chat` — the NATIVE stateful path (issue #129/#130/#131,
    the base-agent default on ``beta_qwen3coder``). ONE /api/chat call sends the FULL
    multi-turn ``messages`` history PLUS the enveloped ``tools:`` schema, so Ollama
    applies the MODEL'S OWN trained tool template — the ``{"type":"function",...}``
    envelope and the anti-fence wording — instead of the harness hand-injecting a tool
    block. Ground truth (live runs, Ollama 0.30.8): the qwen2.5-coder 3B model still
    streams its call as TEXT in ``message.content`` and Ollama leaves ``tool_calls``
    null, so the codec's tolerant ``parse()`` of the content stays load-bearing; when
    structured ``tool_calls`` ARE returned they are captured and preferred.

  * :meth:`decide` single-phase (``two_phase=False``) — legacy system+user /api/chat,
    kept for the validator and as the ``decide`` fallback the test fakes implement.

  * :meth:`decide` two-phase (``two_phase=True``, VibeThinker) — phase 1 free reasoning
    stopped at ``</think>`` (discarded), phase 2 raw continuation prefilled past
    ``</think>``. Used by the advisor/validator; VibeThinker is NOT sent native tools
    (it gets confused by enveloped schemas — verified live).

Whether a path is CONSTRAINED depends on the active codec's ``DecodeConstraint``: only
when ``constraint.json_schema is not None`` is Ollama's ``format`` field set. The
``hermes`` codec supplies ``json_schema=None`` (unconstrained); its ``<tool_call>`` /
fenced / bare JSON output is parsed by the codec, not a decode constraint.

All paths stream token-by-token so callers can render generation live.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .codec import DecodeConstraint
from .config import Config

# A sink for streamed tokens; receives each chunk of text as it is generated.
TokenSink = Callable[[str], None]


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


@dataclass(frozen=True)
class Decision:
    reasoning: str        # phase-1 text (discarded by the agent, kept for logs)
    action_json: str      # phase-2 action payload (parsed by the codec; constrained
                          # only if the codec supplies a json_schema — see module docs)
    # Structured tool calls Ollama parsed from the response, when the native tools:
    # path is used (issue #129/#130/#131). Empty for the legacy text path AND for the
    # qwen 3B model itself, which Ollama 0.30.8 leaves as text in ``action_json`` (the
    # codec's parse() recovers it). When NON-empty the agent prefers these over parsing
    # ``action_json`` — see RalphAgent.
    tool_calls: tuple = ()


class LLMClient(ABC):
    @abstractmethod
    def decide(self, system: str, user: str, constraint: DecodeConstraint,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        ...

    def decide_chat(self, messages: list[dict], tools: list[dict] | None,
                    constraint: DecodeConstraint,
                    on_reason: TokenSink | None = None,
                    on_action: TokenSink | None = None) -> Decision:
        """Native stateful multi-turn decide (issue #129/#130/#131).

        Takes the FULL chat ``messages`` history (system / user / assistant / tool, the
        last typically the current user turn) and the enveloped ``tools`` schema list,
        and returns a :class:`Decision` whose ``action_json`` holds the model's response
        text and whose ``tool_calls`` holds any STRUCTURED calls Ollama parsed.

        The default implementation here adapts to the legacy :meth:`decide` so test
        doubles (and any client that only implements ``decide``) work unchanged: it
        flattens the messages into a system string + a single user string and delegates.
        :class:`OllamaClient` overrides this with the real native ``/api/chat`` + ``tools:``
        transport. This keeps the agent on ONE code path while preserving every existing
        ``decide``-only implementation (the fakes in the test suite)."""
        system = "\n\n".join(m["content"] for m in messages
                             if m.get("role") == "system" and m.get("content"))
        # The newest non-system message is the live turn; everything earlier is history.
        non_system = [m for m in messages if m.get("role") != "system"]
        user = non_system[-1]["content"] if non_system else ""
        return self.decide(system, user, constraint,
                           on_reason=on_reason, on_action=on_action)

    def supports_native_tools(self) -> bool:
        """Whether this client can drive Ollama-style NATIVE tool calling — i.e. the
        stateful :meth:`decide_chat` transport with an enveloped ``tools:`` field
        (issue #129/#130/#131) — as opposed to the single-shot :meth:`decide` path.

        The agent uses this as the capability gate for ``_use_chat`` (issue #163): a
        non-native client (e.g. the OpenAI-compatible :class:`~vibeharness.api_llm.ApiLLMClient`,
        which is single-shot) is automatically routed to ``_decide`` and a constrained-JSON
        codec, so the user only picks a model and the harness picks a compatible path.

        Default: a client is native-capable iff it OVERRIDES :meth:`decide_chat` (the native
        transport). A client that implements only the single-shot :meth:`decide` (the base
        contract / the API client / the test fakes that only script ``decide``) returns
        ``False`` and is driven single-shot. Concrete clients may override this explicitly."""
        return type(self).decide_chat is not LLMClient.decide_chat


def ensure_single_runner_env() -> None:
    """ISSUE #77: cap Ollama at ONE loaded model so a new runner EVICTS the old one
    instead of stacking. Set via the process environment (the only way to configure
    the ``ollama serve`` daemon's loaded-model limit) using ``setdefault`` so a value
    the user already exported is respected, not clobbered. Idempotent and side-effect
    free beyond the env var, so it is safe to call from client init and cli main.

    NOTE: this affects the daemon only if it reads the env at the time it (re)loads a
    model; an already-running ``ollama serve`` started without the var keeps its
    previous limit. The pinned num_ctx + constant keep_alive (above) hold the
    single-runner invariant regardless, so this is belt-and-braces."""
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")


class OllamaClient(LLMClient):
    def __init__(self, config: Config):
        self._cfg = config
        ensure_single_runner_env()

    def supports_native_tools(self) -> bool:
        """Ollama speaks native tool calling (the ``tools:`` /api/chat path). Stated
        explicitly per issue #163 even though the base default (decide_chat overridden)
        would already report True — it documents the capability at the concrete client."""
        return True

    def decide(self, system: str, user: str, constraint: DecodeConstraint,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        # SINGLE-phase (#125): one native /api/chat generation yields the tool call
        # directly. For non-thinking instruct models (qwen2.5-coder) the separate
        # reasoning pass only produces a discarded duplicate call. The codec parses the
        # call from the full output; there is no separate reasoning to stream/keep.
        if not self._cfg.two_phase:
            action, thinking = self._chat(system, user, constraint, on_action, on_reason)
            return Decision(reasoning=thinking, action_json=action)
        reasoning = self._reason(system, user, on_reason)
        action = self._act(system, user, reasoning, constraint, on_action)
        return Decision(reasoning=reasoning, action_json=action)

    def decide_chat(self, messages: list[dict], tools: list[dict] | None,
                    constraint: DecodeConstraint,
                    on_reason: TokenSink | None = None,
                    on_action: TokenSink | None = None) -> Decision:
        """Native /api/chat — Ollama's native THINK-THEN-ACT in a single request (#183).

        When ``config.reason_then_act`` is True (the default on ``beta_qwen3coder``,
        qwen3:4b — a REASONING model): ONE /api/chat call with ``think:True`` + ``tools:``.
        Ollama does the think-then-act internally: it routes qwen3's reasoning into the
        SEPARATE ``message.thinking`` channel and constrains ONLY the action (to the
        ``tools:`` schema), returning the tool call as a STRUCTURED ``message.tool_calls``
        entry. The thinking is captured as ``Decision.reasoning`` (streamed to
        ``on_reason``) and is NEVER parsed as the action; the call is taken from
        ``tool_calls`` (or, if the model emits it as text, the codec's tolerant
        ``parse()`` of the content, which also strips any leaked ``<think>`` block).

        Ground-truthed live (Ollama 0.30.10, qwen3:4b): with ``think:True`` + ``tools:`` a
        single call returns ``thinking`` populated, ``content`` empty, and
        ``tool_calls=[{"function":{"name":"click","arguments":{"target":"Accept all"}}}]``
        even on a heavy snapshot turn. This is exactly the scenario issue #183 needs and
        it is NATIVE: no two-phase prefill, no assistant-prefill bug. The PRIOR behaviour
        (``think:False``) did NOT stop qwen3:4b (Ollama #12917) — it reasoned anyway, the
        trace flooded ``message.content`` UNtagged, and on a non-trivial turn devoured the
        whole budget before any ``<tool_call>`` (live run 20260626_210514: 12/12 turns
        failed to parse). qwen3:4b ignores the ``"low"/"medium"`` think LEVELS (verified:
        same trace length), so there is no native thinking-token budget; ``num_predict``
        is sized to ``thinking_budget + action_tokens`` to give the trace its budget AND
        leave the action its own headroom in the single generation.

        When ``config.reason_then_act`` is False (a NON-thinking model, e.g.
        qwen2.5-coder): one native call with ``tools:`` and ``think:False`` — the call is
        emitted directly. (``think:True`` 400s on a non-thinking model, so the flag gates
        it.)

        Independent of ``config.two_phase`` (which stays False so the native path remains
        enabled — see ``RalphAgent._native``)."""
        thinking_model = self._cfg.reason_then_act
        # A reasoning model needs room for the thinking trace AND the action in the one
        # generation; a non-thinking model only emits the action.
        num_predict = (self._cfg.thinking_budget + self._cfg.action_tokens
                       if thinking_model else self._cfg.action_tokens)
        payload: dict = {
            "model": self._cfg.model,
            "messages": messages,
            "think": bool(thinking_model),
            "options": {**self._options(),
                        "temperature": self._cfg.action_temperature,
                        "num_predict": num_predict,
                        "stop": list(constraint.stop)},
        }
        if tools:
            payload["tools"] = tools
        if constraint.json_schema is not None:
            payload["format"] = constraint.json_schema
        content, tool_calls, thinking = self._stream_chat(payload, on_action, on_reason)
        return Decision(reasoning=thinking, action_json=content.strip(),
                        tool_calls=tuple(tool_calls))

    def generate(self, prompt: str, max_chars: int | None = None,
                 on_token: TokenSink | None = None) -> str:
        """One-shot raw streamed completion. If ``max_chars`` is given, inference
        is stopped early (the connection is closed) once that many characters have
        been produced. Raises :class:`OllamaUnavailable` if the server is down.
        Useful as a core-functionality / liveness check."""
        req = urllib.request.Request(
            self._cfg.ollama_url + "/api/generate",
            data=json.dumps({"model": self._cfg.model, "prompt": prompt,
                             "stream": True, "keep_alive": self._cfg.ollama_keep_alive,
                             "options": self._options()}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return self._read_stream(req, lambda obj: obj.get("response", ""),
                                 on_token, max_chars=max_chars)

    # ---- single-phase: one native chat generation (#125, two_phase=False) ----
    def _chat(self, system: str, user: str, constraint: DecodeConstraint,
              on_token: TokenSink | None,
              on_reason: TokenSink | None = None) -> "tuple[str, str]":
        """One /api/chat generation that yields the tool call directly.

        Returns ``(content, thinking)`` so callers get the thinking trace too.
        Routes through :meth:`_stream_chat` so ``message.thinking`` (qwen3 reasoning
        tokens) is captured and forwarded to ``on_reason`` instead of being silently
        discarded. The codec's tolerant parse() extracts the call from the content."""
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,  # disable Qwen3 thinking (see decide_chat for rationale)
            "options": {**self._options(),
                        "temperature": self._cfg.action_temperature,
                        "num_predict": self._cfg.reason_tokens + self._cfg.action_tokens,
                        "stop": list(constraint.stop)},
        }
        # Honour a JSON-schema constraint if a codec supplies one (hermes does not).
        if constraint.json_schema is not None:
            payload["format"] = constraint.json_schema
        content, _, thinking = self._stream_chat(payload, on_token, on_reason)
        return content.strip(), thinking

    # ---- phase 1: free reasoning, stop at </think> ----
    def _reason(self, system: str, user: str, on_token: TokenSink | None) -> str:
        return self._stream("/api/chat", {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {**self._options(), "num_predict": self._cfg.reason_tokens,
                        "stop": ["</think>"]},
        }, on_token)

    # ---- phase 2: constrained action via raw continuation ----
    def _act(self, system: str, user: str, reasoning: str, constraint: DecodeConstraint,
             on_token: TokenSink | None) -> str:
        prompt = self._render_chatml(system, user) + self._continue_after_reasoning(reasoning)
        payload = {
            "model": self._cfg.model,
            "raw": True,
            "prompt": prompt,
            "options": {**self._options(), "temperature": self._cfg.action_temperature,
                        "num_predict": self._cfg.action_tokens,
                        "stop": ["<|im_end|>", *constraint.stop]},
        }
        # Ollama only constrains via a JSON-schema `format`; a GBNF grammar needs a
        # llama.cpp backend, so it is ignored here (the codec still parses freely).
        if constraint.json_schema is not None:
            payload["format"] = constraint.json_schema
        text = self._stream("/api/generate", payload, on_token)
        return text.strip()

    # ---- two-phase over a FULL message history (advisor, issue #129/#130/#131) ----
    def _reason_chat(self, messages: list[dict], on_token: TokenSink | None) -> str:
        """Phase 1 over a multi-turn ``messages`` history: free reasoning, stop at
        ``</think>``. Used by the VibeThinker advisor so it sees the recent turns as
        real role-tagged messages instead of a flattened prose blob. NO ``tools:`` field
        is sent — VibeThinker is a reasoning model that gets confused by enveloped tool
        schemas, and its job here is free-text advice."""
        return self._stream("/api/chat", {
            "model": self._cfg.model,
            "messages": messages,
            "options": {**self._options(), "num_predict": self._cfg.reason_tokens,
                        "stop": ["</think>"]},
        }, on_token)

    def _act_chat(self, messages: list[dict], reasoning: str,
                  on_token: TokenSink | None) -> str:
        """Phase 2 over a multi-turn ``messages`` history: continue past ``</think>`` as
        free text (the advice). Appends the phase-1 reasoning (closing an open
        ``<think>``) as a prefilled assistant turn so generation resumes after it."""
        prefill = self._continue_after_reasoning(reasoning)
        convo = list(messages) + [{"role": "assistant", "content": prefill}]
        return self._stream("/api/chat", {
            "model": self._cfg.model,
            "messages": convo,
            "options": {**self._options(), "temperature": self._cfg.action_temperature,
                        "num_predict": self._cfg.action_tokens},
        }, on_token).strip()

    # ---- native /api/chat streaming that ALSO captures structured tool_calls ----
    def _stream_chat(self, payload: dict,
                     on_token: TokenSink | None,
                     on_reason: TokenSink | None = None) -> "tuple[str, list[dict], str]":
        """Stream an /api/chat generation, returning ``(content, tool_calls, thinking)``.

        Like :meth:`_stream` but additionally accumulates any ``message.tool_calls``
        Ollama emits (most small models emit none — the call comes through as content,
        which the codec then parses). Also captures ``message.thinking`` — the field
        Ollama uses for qwen3's reasoning tokens (thinking is ON by default on this
        branch; config.py §25 explains why think:false is intentionally avoided).
        The runner-shape ``keep_alive``/``num_ctx`` invariant (#77) is preserved
        exactly: this stamps the same constant keep_alive and the payload already
        carries the pinned options."""
        body = {"keep_alive": self._cfg.ollama_keep_alive, **payload, "stream": True}
        req = urllib.request.Request(
            self._cfg.ollama_url + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.request_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message", {}) or {}
                    thinking = msg.get("thinking") or ""
                    if thinking:
                        thinking_parts.append(thinking)
                        if on_reason:
                            on_reason(thinking)
                    chunk = msg.get("content") or ""
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
                    calls = msg.get("tool_calls")
                    if calls:
                        tool_calls.extend(calls)
                    if obj.get("done"):
                        break
        except urllib.error.URLError as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self._cfg.ollama_url}. "
                f"Is it running? Start it with `ollama serve`. ({e.reason})"
            ) from e
        # qwen3 emits tool calls as structured objects, not streamed content tokens —
        # emit them to on_token so they appear in the terminal action stream.
        if tool_calls and on_token:
            on_token(json.dumps(tool_calls, indent=2))
        return "".join(parts), tool_calls, "".join(thinking_parts)

    # ---- streaming transport ----
    def _stream(self, path: str, payload: dict, on_token: TokenSink | None) -> str:
        # ISSUE #77: a CONSTANT keep_alive is stamped on EVERY request here (the
        # single chokepoint for /api/chat and /api/generate via _reason/_act). Paired
        # with the pinned options.num_ctx (see _options) this means every request asks
        # for the SAME runner shape and re-pins its keep_alive, so only one
        # llama-server runner is ever spawned. An explicit per-payload keep_alive is
        # NOT overridden (none is set today), keeping this future-proof.
        body = {"keep_alive": self._cfg.ollama_keep_alive, **payload, "stream": True}
        req = urllib.request.Request(
            self._cfg.ollama_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return self._read_stream(req, self._chat_or_response, on_token)

    @staticmethod
    def _chat_or_response(obj: dict) -> str:
        # /api/chat streams text under message.content; /api/generate under response.
        return (obj.get("message", {}).get("content")
                if "message" in obj else obj.get("response", ""))

    def _read_stream(self, req: urllib.request.Request,
                     extract: Callable[[dict], str],
                     on_token: TokenSink | None,
                     max_chars: int | None = None) -> str:
        """Shared streamed-read loop: open ``req``, decode each NDJSON line, pull
        the text chunk via ``extract``, feed ``on_token``, and stop on ``done`` (or
        early once ``max_chars`` characters have been produced, closing the
        connection). Raises :class:`OllamaUnavailable` if the server is down."""
        parts: list[str] = []
        produced = 0
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.request_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunk = extract(obj)
                    if chunk:
                        parts.append(chunk)
                        produced += len(chunk)
                        if on_token:
                            on_token(chunk)
                    if (max_chars is not None and produced >= max_chars) or obj.get("done"):
                        break
        except urllib.error.URLError as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self._cfg.ollama_url}. "
                f"Is it running? Start it with `ollama serve`. ({e.reason})"
            ) from e
        return "".join(parts)

    # ---- helpers ----
    def _options(self) -> dict:
        # ISSUE #77: num_ctx (the pinned Config.num_ctx) is included here, and this
        # dict is spread into EVERY request payload (_reason/_act/generate). That
        # makes the requested runner shape identical across requests, so Ollama keys
        # them all to the SAME (model, context-size) runner — at most one is spawned.
        c = self._cfg
        return {"temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k,
                "num_ctx": c.num_ctx, "num_gpu": c.num_gpu}

    @staticmethod
    def _render_chatml(system: str, user: str) -> str:
        # Matches the Qwen2.5 ChatML template Ollama applies for this model.
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _continue_after_reasoning(reasoning: str) -> str:
        """Re-open the assistant turn right after </think> so the constrained
        JSON is generated as the post-reasoning answer."""
        if "<think>" in reasoning and "</think>" not in reasoning:
            return f"{reasoning}</think>\n"
        if not reasoning.strip():
            return ""
        return f"{reasoning}\n"
