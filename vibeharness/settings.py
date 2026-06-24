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

from .config import Config

# Friendly CLI key -> (Config field name, value parser). Only these are settable.
_SETTABLE: dict[str, tuple[str, type]] = {
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
}


def settable_keys() -> list[str]:
    """Canonical, de-duplicated list of user-facing settable keys."""
    return ["temp", "model", "codec", "max-steps", "max-actions-per-turn", "top-p",
            "top_k", "num-ctx", "reason-tokens", "action-tokens"]


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

        Raises ``KeyError`` for an unknown key and ``ValueError`` for a value that
        cannot be parsed into the field's type.
        """
        if key not in _SETTABLE:
            raise KeyError(key)
        field_name, caster = _SETTABLE[key]
        value = caster(raw_value)
        data = Settings.load()
        data[field_name] = value
        Settings.save(data)
        return field_name, value

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
        """Return a Config with saved overrides applied to known fields only."""
        valid = {f.name for f in fields(Config)}
        overrides = {k: v for k, v in Settings.load().items() if k in valid}
        return replace(base, **overrides) if overrides else base
