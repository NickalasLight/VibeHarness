"""LLM client.

The agent depends on the `LLMClient` abstraction (DIP); `OllamaClient` is one
implementation. It performs the two-phase generation that converts noperator's
vLLM structural-tag idea to Ollama:
  phase 1 - free reasoning, stopped at </think>  (discarded by the caller)
  phase 2 - raw continuation prefilled past </think>, constrained by a JSON
            schema via Ollama's `format` field -> a guaranteed-valid action.

Both phases stream token-by-token so callers can render generation live.
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
    action_json: str      # phase-2 constrained action payload (parsed by the codec)


class LLMClient(ABC):
    @abstractmethod
    def decide(self, system: str, user: str, constraint: DecodeConstraint,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        ...


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

    def decide(self, system: str, user: str, constraint: DecodeConstraint,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        reasoning = self._reason(system, user, on_reason)
        action = self._act(system, user, reasoning, constraint, on_action)
        return Decision(reasoning=reasoning, action_json=action)

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
