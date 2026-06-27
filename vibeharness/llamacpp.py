"""llama.cpp (``llama-server``) LLM client.

An alternative :class:`~vibeharness.llm.LLMClient` that talks to llama.cpp's
native HTTP server instead of Ollama. The whole point of this backend is that
it HONOURS :class:`~vibeharness.codec.DecodeConstraint.gbnf`: llama.cpp can do
native GBNF grammar-constrained decoding, which Ollama cannot.

Like :class:`~vibeharness.llm.OllamaClient` it performs two-phase generation:
  phase 1 - free reasoning, stopped at ``</think>``  (discarded by the caller)
  phase 2 - raw continuation prefilled past ``</think>``, constrained by either
            a GBNF grammar (preferred) or a JSON schema -> a valid action.

Both phases stream token-by-token so callers can render generation live. Only
the stdlib (``urllib``) is used for transport, mirroring ``llm.py``.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .codec import DecodeConstraint
from .config import Config
from .llm import Decision, LLMClient, TokenSink


class LlamaCppUnavailable(RuntimeError):
    """Raised when the llama.cpp server cannot be reached."""


class LlamaCppClient(LLMClient):
    """Two-phase client backed by llama.cpp's native ``/completion`` endpoint."""

    def __init__(self, config: Config):
        self._cfg = config

    def decide(self, system: str, user: str, constraint: DecodeConstraint,
               on_reason: TokenSink | None = None,
               on_action: TokenSink | None = None) -> Decision:
        reasoning = self._reason(system, user, on_reason)
        action = self._act(system, user, reasoning, constraint, on_action)
        return Decision(reasoning=reasoning, action_json=action)

    # ---- phase 1: free reasoning, stop at </think> ----
    def _reason(self, system: str, user: str, on_token: TokenSink | None) -> str:
        prompt = self._render_chatml(system, user)
        payload = {
            "prompt": prompt,
            "n_predict": self._cfg.reason_tokens,
            "temperature": self._cfg.temperature,
            "stop": ["</think>"],
            "cache_prompt": True,
        }
        return self._stream(payload, on_token)

    # ---- phase 2: constrained action via raw continuation ----
    def _act(self, system: str, user: str, reasoning: str, constraint: DecodeConstraint,
             on_token: TokenSink | None) -> str:
        prompt = self._render_chatml(system, user) + self._continue_after_reasoning(reasoning)
        payload = {
            "prompt": prompt,
            "n_predict": self._cfg.action_tokens,
            "temperature": self._cfg.action_temperature,
            "stop": ["<|im_end|>", *constraint.stop],
            "cache_prompt": True,
        }
        # GBNF is preferred (the reason this backend exists); fall back to a JSON
        # schema, which llama.cpp also supports natively via the `json_schema` field.
        if constraint.gbnf is not None:
            payload["grammar"] = constraint.gbnf
        elif constraint.json_schema is not None:
            payload["json_schema"] = constraint.json_schema
        text = self._stream(payload, on_token)
        return text.strip()

    # ---- streaming transport ----
    def _stream(self, payload: dict, on_token: TokenSink | None) -> str:
        req = urllib.request.Request(
            self._cfg.llamacpp_url + "/completion",
            data=json.dumps({**payload, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        parts: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.request_timeout) as resp:
                for raw in resp:
                    obj = self._parse_event(raw)
                    if obj is None:
                        continue
                    chunk = obj.get("content", "")
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
                    if obj.get("stop"):
                        break
        except urllib.error.URLError as e:
            raise LlamaCppUnavailable(
                f"Could not reach llama.cpp server at {self._cfg.llamacpp_url}. "
                f"Is it running? Start it with `llama-server -m <model> --port 8080`. "
                f"({e.reason})"
            ) from e
        return "".join(parts)

    @staticmethod
    def _parse_event(raw: bytes) -> dict | None:
        """Decode one streamed line. llama.cpp emits server-sent events
        (``data: {json}\\n``) when streaming; tolerate bare JSON lines too."""
        line = raw.decode("utf-8").strip()
        if not line:
            return None
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # ---- helpers (copied from OllamaClient so this module is self-contained) ----
    @staticmethod
    def _render_chatml(system: str, user: str) -> str:
        # Matches the Qwen2.5 ChatML template applied for this model.
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _continue_after_reasoning(reasoning: str) -> str:
        """Re-open the assistant turn right after </think> so the constrained
        output is generated as the post-reasoning answer."""
        if "<think>" in reasoning and "</think>" not in reasoning:
            return f"{reasoning}</think>\n"
        if not reasoning.strip():
            return ""
        return f"{reasoning}\n"
