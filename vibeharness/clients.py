"""LLM client factory — the seam that turns a per-role :class:`ModelSpec` into a live
:class:`~vibeharness.llm.LLMClient` (issue #163).

This is the one place that knows how to BUILD a client from config: it reads the named
provider from the unified registry, derives the endpoint *kind*, and constructs the
matching client (local Ollama/llama.cpp, or an OpenAI-compatible API client). Sampling is
resolved spec-override → provider-default → :class:`Config` default. Adding a provider is
still a one-line registry entry; this factory dispatches on ``kind`` so no call site needs
to know whether a role is local or API.

Dependency direction (DIP): ``agent → factory ← {providers, config, llm, api_llm,
llamacpp}``. The factory depends on the concrete clients and the leaf modules; the agent
and CLI depend only on the :class:`LLMClient` abstraction this returns. ``providers`` and
``config`` import nothing from here, the agent, or the web layer.
"""
from __future__ import annotations

from dataclasses import replace

from . import providers as _providers
from .codec import ToolCallCodec, get_codec
from .config import Config, ModelSpec
from .llm import LLMClient, OllamaClient
from .providers import LLAMACPP, OLLAMA, OPENAI_COMPATIBLE, Provider


def build_client(spec: ModelSpec, config: Config) -> LLMClient:
    """Build the :class:`LLMClient` that drives ``spec`` (issue #163).

    The named provider's ``kind`` selects the concrete client:

      * ``ollama``            → :class:`~vibeharness.llm.OllamaClient`
      * ``llamacpp``         → :class:`~vibeharness.llamacpp.LlamaCppClient`
      * ``openai_compatible`` → :class:`~vibeharness.api_llm.ApiLLMClient`

    For local kinds the run :class:`Config` is cloned with the spec's model (and any
    sampling overrides) so the existing config-driven transport is reused untouched. For
    the API kind the key is read from the provider's ``api_key_env`` at construction time
    (never stored) — a missing key raises ``RuntimeError`` exactly as before, so callers
    that degrade gracefully (validator fallback, escalation) keep their behaviour.
    """
    provider = _providers.get_endpoint(spec.provider)
    if provider.kind == OLLAMA:
        return OllamaClient(_local_config(config, spec, provider))
    if provider.kind == LLAMACPP:
        from .llamacpp import LlamaCppClient   # local import: keep llamacpp optional
        return LlamaCppClient(_local_config(config, spec, provider))
    if provider.kind == OPENAI_COMPATIBLE:
        return _build_api_client(spec, provider)
    raise ValueError(f"provider {provider.name!r} has unknown kind {provider.kind!r}")


def _local_config(config: Config, spec: ModelSpec, provider: Provider) -> Config:
    """Clone the run Config for a local client: pin the spec's model and apply any
    sampling overrides (spec → provider-default → Config default). Only fields the spec
    actually overrides are touched, so an override-free spec yields a Config identical to
    the run's (no behaviour change for the default local path)."""
    overrides: dict = {"model": spec.model}
    temperature = _resolve(spec.temperature, provider.temperature)
    if temperature is not None:
        # The action phase is what produces the tool call; that is the temperature a
        # per-role override is meant to steer (the reasoning-phase temp is left at the
        # run default so two-phase models are unaffected unless explicitly retuned).
        overrides["action_temperature"] = temperature
    if spec.top_p is not None:
        overrides["top_p"] = spec.top_p
    if spec.top_k is not None:
        overrides["top_k"] = spec.top_k
    # A provider MAY carry a custom local base URL (e.g. Ollama on another host); honour it
    # on the matching Config field. Built-in local providers leave base_url None.
    if provider.base_url:
        if provider.kind == OLLAMA:
            overrides["ollama_url"] = provider.base_url
        elif provider.kind == LLAMACPP:
            overrides["llamacpp_url"] = provider.base_url
    return replace(config, **overrides)


def _build_api_client(spec: ModelSpec, provider: Provider) -> LLMClient:
    """Build an OpenAI-compatible API client via :func:`providers.make_api_client`.

    Routing through ``make_api_client`` keeps ONE place constructing the API client (key
    read + missing-key error), and lets the escalation path mock it. The resolved
    temperature (spec → provider-default) is forwarded only when set, so the historical
    two-argument call ``make_api_client(name, model)`` is preserved when no override
    exists."""
    temperature = _resolve(spec.temperature, provider.temperature)
    # Pass temperature ONLY when overridden, so an override-free call is exactly
    # ``make_api_client(name, model)`` — preserving the historical call shape (and the
    # escalation path's mock contract).
    extra = {"temperature": temperature} if temperature is not None else {}
    return _providers.make_api_client(spec.provider, spec.model or None, **extra)


def _resolve(*candidates):
    """First non-None candidate (spec override → provider default → …), else None."""
    for c in candidates:
        if c is not None:
            return c
    return None


def select_execution_codec(config: Config, client: LLMClient,
                           codec: ToolCallCodec, registry) -> "tuple[ToolCallCodec, bool, str | None]":
    """Pick the (codec, native_tools, note) the run should actually use (issue #163).

    Capability-driven path selection. NATIVE tool calling is used only when the run opts
    in (``native_tools``), the base agent is single-phase, the codec speaks native tools
    (``codec.tools`` non-None), AND the client can drive it
    (``client.supports_native_tools()``). When the BASE client is non-native (an API
    client), the harness AUTO-DEGRADES to the single-shot path and auto-selects a
    constrained-JSON codec: if the configured codec cannot constrain JSON
    (``constraint.json_schema is None`` — e.g. the native-only ``hermes`` codec), it is
    swapped for the ``json`` codec so the single-shot API path has a schema to instruct and
    JSON format instructions in the prompt. A codec that already constrains JSON (``json``,
    ``tagged_json``) is kept. The returned ``note`` describes any swap so the caller can log
    the auto-selection; it is ``None`` when nothing changed.
    """
    native = bool(
        getattr(config, "native_tools", False)
        and not config.two_phase
        and codec.tools(registry) is not None
        and client.supports_native_tools()
    )
    if native:
        return codec, True, None
    # Non-native execution: ensure the codec can constrain JSON for the single-shot path.
    constraint = codec.constraint(registry, config.max_actions_per_turn)
    if constraint.json_schema is None:
        fallback = get_codec("json")
        note = (f"non-native base client + codec '{codec.name}' has no JSON constraint; "
                f"auto-selected the 'json' codec for the single-shot path")
        return fallback, False, note
    return codec, False, None
