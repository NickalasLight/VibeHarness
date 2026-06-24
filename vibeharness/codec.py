"""The tool-call codec seam.

A :class:`ToolCallCodec` owns the *wire format* of tool calls end to end:
  1. how the model is told to emit them   (a block injected into the system prompt),
  2. how decoding is constrained to that shape (a :class:`DecodeConstraint`),
  3. how the raw output is parsed back into ``(tool, args)`` pairs.

Swapping the codec swaps the format — JSON, tagged-JSON, pure-XML, code-as-action,
GBNF — without touching the agent loop, the prompt builder, or the LLM transport,
each of which depends only on this seam. New formats are added as new, isolated
modules under ``vibeharness.codecs`` (see :func:`get_codec`), so they share no code
file with one another and merge without conflict.
"""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry

# A parsed tool call: the tool name and its argument dict.
ToolCall = tuple[str, dict]


@dataclass(frozen=True)
class DecodeConstraint:
    """How the action phase is constrained at decode time.

    A codec fills in whichever mechanism its format needs; the LLM client applies
    the one that is present and ignores the rest:

      - ``json_schema``: a JSON-schema grammar (passed as Ollama's ``format``).
      - ``gbnf``:        a raw GBNF grammar (honoured only by a llama.cpp backend).
      - ``stop``:        extra stop strings, appended to the backend's own controls.

    All three empty -> free (unconstrained) generation, parsed purely by the codec.
    """
    json_schema: dict | None = None
    gbnf: str | None = None
    stop: tuple[str, ...] = ()


class ToolCallCodec(ABC):
    """The contract every tool-call format implements. Stateless."""

    name: str = ""

    @abstractmethod
    def format_instructions(self, max_actions: int) -> str:
        """The format-specific block injected into the system prompt, telling the
        model exactly how to emit one or more tool calls in this wire format. A
        ``max_actions`` of 0 (or less) means no per-turn cap."""

    def turn_action_hint(self) -> str:
        """A brief, format-specific reminder placed at the very end of each turn
        prompt (the recency zone) of how to shape the response. Overridden per
        codec so the end-of-turn nudge matches this codec's wire format."""
        return "Respond in the exact format described above."

    def tool_definitions(self, registry: "ToolRegistry") -> str | None:
        """The tool-definition block this codec wants in the system prompt, or
        ``None`` to use the registry's default Markdown ``docs()`` (issue #105).

        Most codecs return ``None`` — the Markdown tool docs are codec-agnostic.
        A codec whose model was fine-tuned to read tool definitions in a specific
        wire shape (e.g. the Hermes ``<tools>`` function-schema block) overrides this
        and returns that block, built from the registry. The :class:`SystemPromptBuilder`
        substitutes it for the Markdown docs when present, keeping the rendering choice
        codec-local (open/closed: no edits to the prompt builder per new format)."""
        return None

    @abstractmethod
    def constraint(self, registry: "ToolRegistry", max_actions: int) -> DecodeConstraint:
        """The decode-time constraint for the action phase (may be unconstrained)."""

    @abstractmethod
    def parse(self, raw: str) -> "tuple[list[ToolCall] | None, str | None]":
        """Parse raw model output into a list of ``(tool, args)``; on malformed
        output return ``(None, <human-readable reason>)``."""


class UnknownCodec(KeyError):
    """Raised when no codec is registered under the requested name."""


def available_codecs() -> list[str]:
    """List installed codec names, discovered from the ``vibeharness.codecs`` package.

    Enumerates the package's submodules with :func:`pkgutil.iter_modules`, keeping the
    ``<name>_codec`` modules and stripping the ``_codec`` suffix, so each codec module
    contributes its name with no central registry to edit. The result is sorted.

    This uses the import system's module enumeration rather than a filesystem glob so
    discovery survives a PyInstaller ``--onefile`` freeze, where the package lives in
    the bundled archive and has no on-disk ``codecs/`` directory to scan. The build
    must still bundle the codec submodules (``--collect-submodules vibeharness.codecs``)
    so they appear here and can be imported by :func:`get_codec`.
    """
    import vibeharness.codecs as pkg

    names = [
        mod.name[: -len("_codec")]
        for mod in pkgutil.iter_modules(pkg.__path__)
        if mod.name.endswith("_codec")
    ]
    return sorted(set(names))


def get_codec(name: str) -> ToolCallCodec:
    """Resolve a codec by name.

    Each codec lives in its own isolated module ``vibeharness.codecs.<name>_codec``
    exposing a module-level ``CODEC`` instance, so a new format is added as a new
    file with no edits to any shared registry.
    """
    mod_name = f".codecs.{name}_codec"
    try:
        module = importlib.import_module(mod_name, __package__)
    except ModuleNotFoundError as e:
        # Only swallow "the codec module itself is absent"; a genuine missing
        # dependency inside an existing codec should surface, not masquerade as
        # an unknown-codec error.
        if e.name in (f"{__package__}.codecs.{name}_codec", f"vibeharness.codecs.{name}_codec"):
            raise UnknownCodec(f"no tool-call codec named '{name}'") from e
        raise
    codec = getattr(module, "CODEC", None)
    if not isinstance(codec, ToolCallCodec):
        raise UnknownCodec(f"module for codec '{name}' exposes no CODEC instance")
    return codec
