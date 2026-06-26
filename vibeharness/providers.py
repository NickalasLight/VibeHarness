"""Unified model-endpoint / provider registry.

A single, decoupled lookup of the LLM endpoints the harness can drive. Each provider
holds endpoint *coordinates* only — its ``kind``, a base URL, and (for API kinds) the
NAME of the environment variable that carries the key — **never** a secret. The key
itself is read from the environment at client-construction time, so it never lives in
this (public) repo or in a :class:`~vibeharness.config.Config` object.

Three kinds are modelled (issue #163):

  * ``ollama``            — a local Ollama daemon (base URL comes from ``Config.ollama_url``);
  * ``llamacpp``         — a local ``llama-server`` (base URL from ``Config.llamacpp_url``);
  * ``openai_compatible`` — any OpenAI-compatible HTTP API (z.ai/GLM, etc.), with a fixed
    ``base_url`` and an ``api_key_env``.

Adding a provider is a ONE-LINE entry in :data:`REGISTRY` (Open/Closed): no other file
changes. A role switches local↔API purely by naming a different provider in its
:class:`~vibeharness.config.ModelSpec` — the ``kind`` (and therefore the concrete client
the factory builds) derives from the named provider, not from the call site.

Backward compatibility: the original API-only surface (:class:`ApiProviderConfig`,
:func:`get_provider`, :func:`api_key_present`, :func:`make_api_client`) is preserved
verbatim so every existing caller and test keeps working. ``get_provider`` returns an
:class:`ApiProviderConfig` view of an ``openai_compatible`` registry entry; the unified
:func:`get_endpoint` returns the richer :class:`Provider`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# The three endpoint kinds the registry can describe. Kept as bare string constants
# (not an Enum) so a registry entry and a config value are the same trivially-(de)serialised
# token, and so adding a provider never requires touching an enum.
OLLAMA = "ollama"
LLAMACPP = "llamacpp"
OPENAI_COMPATIBLE = "openai_compatible"

# Local kinds are driven by a daemon whose base URL lives on Config (so a user who moves
# Ollama to another host changes ONE config value, not the registry). API kinds carry
# their own fixed base URL + key-env name here.
LOCAL_KINDS = frozenset({OLLAMA, LLAMACPP})


@dataclass(frozen=True)
class Provider:
    """Coordinates for ONE model endpoint. Holds no secret material.

    ``kind`` selects the concrete client the factory builds (see
    :func:`vibeharness.clients.build_client`). For local kinds (``ollama``/``llamacpp``)
    ``base_url`` is ``None`` and resolved from :class:`~vibeharness.config.Config` at build
    time; ``api_key_env``/``model`` are unused. For ``openai_compatible`` ``base_url`` is the
    fixed endpoint, ``api_key_env`` names the env var holding the key (never the key), and
    ``model`` is the provider's default model id.

    ``temperature`` is an OPTIONAL provider-level sampling default: when set it sits between
    a per-spec override and the global :class:`Config` default (spec → provider → config).
    """
    name: str
    kind: str
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    temperature: float | None = None


# The single source of truth. Add a provider by adding ONE entry here (Open/Closed).
REGISTRY: dict[str, Provider] = {
    # Local backends — base URL comes from Config at build time (None here).
    OLLAMA: Provider(name=OLLAMA, kind=OLLAMA),
    LLAMACPP: Provider(name=LLAMACPP, kind=LLAMACPP),
    # OpenAI-compatible API: z.ai / ZhipuAI GLM. Coordinates only, NO key.
    "zhipuai": Provider(
        name="zhipuai",
        kind=OPENAI_COMPATIBLE,
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZHIPUAI_API_KEY",
        model="glm-5.2",
    ),
    # OpenAI-compatible API: DeepSeek (issue #182). Coordinates only, NO key.
    # base_url is the official root the DeepSeek docs use; the OpenAI SDK appends
    # ``/chat/completions`` to it. ``/v1`` is an OPTIONAL alias DeepSeek accepts for OpenAI
    # compatibility and is unrelated to the model version — the bare root is what the
    # official examples show, so we use it. Default model is the non-thinking V3.1
    # ``deepseek-chat`` (full function-calling support). Ground truth (cited in the PR):
    #   https://api-docs.deepseek.com/                       (base_url + "Your First API Call")
    #   https://api-docs.deepseek.com/guides/function_calling (deepseek-chat tool calling)
    #   https://api-docs.deepseek.com/news/news250821         (V3.1: 128K ctx; chat=non-think)
    "deepseek": Provider(
        name="deepseek",
        kind=OPENAI_COMPATIBLE,
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
    ),
}


def get_endpoint(name: str) -> Provider:
    """Look up a provider by name in the unified registry.

    Raises ``KeyError`` (listing the known names) when the provider is not registered.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown provider {name!r}; known providers: {known}") from None


def is_local(provider: Provider) -> bool:
    """True for a local (ollama/llamacpp) endpoint — i.e. no API key is required."""
    return provider.kind in LOCAL_KINDS


# --------------------------------------------------------------------------- #
# Backward-compatible API-provider surface (kept verbatim in behaviour).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApiProviderConfig:
    """Coordinates for one OpenAI-compatible API endpoint. No secret material.

    Retained (field order unchanged) so the historical API-provider call sites and tests
    keep working; the unified :class:`Provider` is the forward-looking type. This is the
    *view* :func:`get_provider` returns for an ``openai_compatible`` registry entry.
    """
    name: str            # short id, e.g. "zhipuai"
    base_url: str        # OpenAI-compatible base URL
    api_key_env: str     # NAME of the env var holding the key (never the key itself)
    model: str           # default model id for this provider


def get_provider(name: str) -> ApiProviderConfig:
    """Look up an OpenAI-compatible API provider by name (legacy surface).

    Returns an :class:`ApiProviderConfig` view of the unified registry entry. Raises
    ``KeyError`` (with the known names) for an unknown provider, or a ``KeyError`` if the
    name resolves to a non-API (local) kind — the legacy surface only described API
    providers, so a local kind is "not an API provider" here.
    """
    p = get_endpoint(name)
    if p.kind != OPENAI_COMPATIBLE:
        known = ", ".join(sorted(n for n, e in REGISTRY.items()
                                 if e.kind == OPENAI_COMPATIBLE)) or "(none)"
        raise KeyError(f"{name!r} is not an API provider; known API providers: {known}")
    return ApiProviderConfig(name=p.name, base_url=p.base_url or "",
                             api_key_env=p.api_key_env or "", model=p.model or "")


def api_key_present(provider: "ApiProviderConfig | Provider") -> bool:
    """True if the provider's key env var is set and non-empty.

    Accepts either the legacy :class:`ApiProviderConfig` or a unified :class:`Provider`
    (both expose ``api_key_env``); a provider with no ``api_key_env`` (a local kind) is
    treated as "no key required is satisfied" → returns ``True``."""
    env = getattr(provider, "api_key_env", None)
    if not env:
        return True
    return bool(os.environ.get(env, "").strip())


def make_api_client(provider_name: str, model_override: str | None = None,
                    *, temperature: float | None = None, timeout: int | None = None,
                    user_agent: str | None = None):
    """Construct an :class:`~vibeharness.api_llm.ApiLLMClient` for ``provider_name``.

    The api key is read (at this moment, never stored here) from the provider's
    ``api_key_env`` environment variable. Raises ``RuntimeError`` if that env var is
    missing/empty, and ``KeyError`` for an unknown/non-API provider. Imported lazily so
    that simply importing this module never requires the optional ``openai`` dep.

    ``temperature`` / ``timeout`` are optional sampling/transport overrides (issue #163):
    they are forwarded to :class:`ApiLLMClient` only when given, so the historical
    two-argument call shape — ``make_api_client(name, model)`` — is byte-for-byte
    unchanged (the factory in :mod:`vibeharness.clients` adds them only on an override).
    """
    provider = get_provider(provider_name)
    key = os.environ.get(provider.api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"API key not found: set the {provider.api_key_env} environment variable "
            f"to use the {provider.name!r} provider."
        )
    from .api_llm import ApiLLMClient   # local import keeps openai an optional dep
    model = model_override or provider.model
    extra: dict = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if timeout is not None:
        extra["timeout"] = timeout
    if user_agent is not None:        # issue #198 — browser UA from Config when supplied
        extra["user_agent"] = user_agent
    return ApiLLMClient(provider=provider, api_key=key, model=model, **extra)
