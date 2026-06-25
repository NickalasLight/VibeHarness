"""API LLM provider registry.

A tiny, decoupled lookup of external LLM-API endpoints used by the escalation /
validation paths. It holds endpoint *coordinates* only (base URL, model name, and
the NAME of the environment variable that carries the key) — never a secret. The
key itself is read from the environment at client-construction time, so it never
lives in this (public) repo or in a :class:`~vibeharness.config.Config` object.

Adding a provider is a one-line entry here (Open/Closed): no other file changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiProviderConfig:
    """Coordinates for one OpenAI-compatible API endpoint. No secret material."""
    name: str            # short id, e.g. "zhipuai"
    base_url: str        # OpenAI-compatible base URL
    api_key_env: str     # NAME of the env var holding the key (never the key itself)
    model: str           # default model id for this provider


# Built-in providers — config values only, NO api keys. Extend here.
PROVIDERS: dict[str, ApiProviderConfig] = {
    "zhipuai": ApiProviderConfig(
        name="zhipuai",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZHIPUAI_API_KEY",
        model="glm-5.2",
    ),
}


def get_provider(name: str) -> ApiProviderConfig:
    """Look up a provider by name. Raises ``KeyError`` (with the known names) if
    the provider is not registered."""
    try:
        return PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS)) or "(none)"
        raise KeyError(f"unknown API provider {name!r}; known providers: {known}") from None


def api_key_present(provider: ApiProviderConfig) -> bool:
    """True if the provider's key env var is set and non-empty."""
    return bool(os.environ.get(provider.api_key_env, "").strip())


def make_api_client(provider_name: str, model_override: str | None = None):
    """Construct an :class:`~vibeharness.api_llm.ApiLLMClient` for ``provider_name``.

    The api key is read (at this moment, never stored here) from the provider's
    ``api_key_env`` environment variable. Raises ``RuntimeError`` if that env var
    is missing/empty, and ``KeyError`` for an unknown provider. Imported lazily so
    that simply importing this module never requires the optional ``openai`` dep.
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
    return ApiLLMClient(provider=provider, api_key=key, model=model)
