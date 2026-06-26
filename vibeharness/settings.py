"""Persistent user settings.

A tiny JSON store (default ``~/.vibeharness/settings.json``) that overrides the
built-in :class:`Config` defaults, so a user can change e.g. the default
temperature once and have it stick. Per-run CLI flags still take precedence.

Resolution order (low to high): Config defaults < saved settings < CLI flags.

The storage directory can be redirected with the ``VIBEHARNESS_HOME`` environment
variable, which also keeps the store trivially testable.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path

from .config import Config, ModelSpec, resolve_role_spec

# Per-role ModelSpec field -> value parser, for the nested `models.<role>.<field>` keys
# (issue #163). Only these nested fields are settable.
_MODEL_FIELD_TYPES: dict[str, type] = {
    "provider": str,
    "model": str,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    # Per-model tool-call codec + per-turn cap (issues #179/#178). Optional overrides of the
    # registry/Config resolution for this role's model.
    "codec": str,
    "max_actions_per_turn": int,
}

def _csv_set(raw: str) -> list:
    """Parse a comma-separated tool-name list into a de-duplicated, JSON-serialisable
    list (the Config field is a frozenset; the agent coerces whatever is stored with
    ``set()``). Blank entries are dropped; an empty string clears the list."""
    seen: list = []
    for token in raw.split(","):
        name = token.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


# Friendly CLI key -> (Config field name, value parser). Only these are settable.
_SETTABLE: dict[str, tuple[str, object]] = {
    "temp": ("temperature", float),
    "temperature": ("temperature", float),
    "model": ("model", str),
    "codec": ("codec", str),
    "max-steps": ("max_steps", int),
    "max_steps": ("max_steps", int),
    "max-actions-per-turn": ("max_actions_per_turn", int),
    "max_actions_per_turn": ("max_actions_per_turn", int),
    "top-p": ("top_p", float),
    "top_k": ("top_k", int),
    "num-ctx": ("num_ctx", int),
    "num_ctx": ("num_ctx", int),
    "reason-tokens": ("reason_tokens", int),
    "reason_tokens": ("reason_tokens", int),
    "action-tokens": ("action_tokens", int),
    "action_tokens": ("action_tokens", int),
    # #162: comma-separated web tool names that opt into same-turn duplicate suppression.
    "web-dedup-same-turn-tools": ("web_dedup_same_turn_tools", _csv_set),
    "web_dedup_same_turn_tools": ("web_dedup_same_turn_tools", _csv_set),
}


def settable_keys() -> list[str]:
    """Canonical, de-duplicated list of user-facing settable keys.

    Includes the nested per-role pattern ``models.<role>.<field>`` (issue #163), e.g.
    ``models.base.provider`` / ``models.validator.model`` — see :meth:`Settings.set`.
    """
    return ["temp", "model", "codec", "max-steps", "max-actions-per-turn", "top-p",
            "top_k", "num-ctx", "reason-tokens", "action-tokens",
            "web-dedup-same-turn-tools",
            "models.<role>.<field>"]


def _home() -> Path:
    override = os.environ.get("VIBEHARNESS_HOME")
    return Path(override) if override else Path.home() / ".vibeharness"


class Settings:
    """Load/save persistent overrides and merge them into a :class:`Config`."""

    @staticmethod
    def path() -> Path:
        return _home() / "settings.json"

    @staticmethod
    def load() -> dict:
        path = Settings.path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @staticmethod
    def save(data: dict) -> None:
        path = Settings.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def set(key: str, raw_value: str) -> tuple[str, object]:
        """Persist one setting. Returns ``(config_field, parsed_value)``.

        Supports both the flat friendly keys and the nested per-role dotted keys
        ``models.<role>.<field>`` (issue #163). Raises ``KeyError`` for an unknown key and
        ``ValueError`` for a value that cannot be parsed into the field's type.
        """
        if key.startswith("models."):
            return Settings._set_model_field(key, raw_value)
        if key not in _SETTABLE:
            raise KeyError(key)
        field_name, caster = _SETTABLE[key]
        value = caster(raw_value)
        data = Settings.load()
        data[field_name] = value
        Settings.save(data)
        return field_name, value

    @staticmethod
    def _set_model_field(key: str, raw_value: str) -> tuple[str, object]:
        """Persist one nested ``models.<role>.<field>`` setting (issue #163).

        Stored as ``data["models"][role][field]`` so several fields of a role accumulate
        across calls. ``Settings.apply`` later materialises each role's accumulated fields
        into a :class:`~vibeharness.config.ModelSpec`."""
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "models":
            raise KeyError(key)
        _, role, field_name = parts
        if not role or field_name not in _MODEL_FIELD_TYPES:
            raise KeyError(key)
        value = _MODEL_FIELD_TYPES[field_name](raw_value)
        data = Settings.load()
        models = data.get("models")
        if not isinstance(models, dict):
            models = {}
        role_map = models.get(role)
        if not isinstance(role_map, dict):
            role_map = {}
        role_map[field_name] = value
        models[role] = role_map
        data["models"] = models
        Settings.save(data)
        return key, value

    @staticmethod
    def reset() -> bool:
        """Delete saved settings. Returns ``True`` if a file was removed."""
        path = Settings.path()
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def apply(base: Config) -> Config:
        """Return a Config with saved overrides applied to known fields only.

        Flat keys map straight onto Config fields. The nested ``models`` store
        (``{role: {field: value}}``) is materialised into ``dict[str, ModelSpec]`` (issue
        #163): each role's saved fields override that role's current (legacy/flat)
        resolution, so a partial spec (e.g. only a provider) still yields a complete
        ModelSpec. An incomplete or unrecognised entry is skipped rather than raising."""
        data = Settings.load()
        valid = {f.name for f in fields(Config)}
        # Flat overrides only (the nested ``models`` store is handled separately below).
        flat = {k: v for k, v in data.items() if k in valid and k != "models"}
        cfg = replace(base, **flat) if flat else base

        raw_models = data.get("models")
        if isinstance(raw_models, dict) and raw_models:
            models = dict(cfg.models)
            for role, attrs in raw_models.items():
                if not isinstance(attrs, dict):
                    continue
                try:
                    current = resolve_role_spec(cfg, role)
                    cur_provider, cur_model = current.provider, current.model
                except KeyError:
                    cur_provider = cur_model = None
                provider = attrs.get("provider") or cur_provider
                model = attrs.get("model") or cur_model
                if not provider or not model:
                    continue   # incomplete spec — ignore rather than build a broken one
                models[role] = ModelSpec(
                    provider=provider, model=model,
                    temperature=attrs.get("temperature"),
                    top_p=attrs.get("top_p"), top_k=attrs.get("top_k"),
                    codec=attrs.get("codec"),
                    max_actions_per_turn=attrs.get("max_actions_per_turn"))
            if models != cfg.models:
                cfg = replace(cfg, models=models)
        return cfg
