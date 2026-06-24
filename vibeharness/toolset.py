"""Toolsets: pluggable, switchable groups of tools.

A `Toolset` bundles a set of `Tool`s behind a name, plus optional prerequisite
checks and setup/teardown lifecycle hooks. The `ToolsetCatalog` lets the CLI
select one or several toolsets at runtime and merge their tools into a single
`ToolRegistry`.

Adding a new tool interface = add one `Toolset` subclass and register it in
`default_catalog()`. Nothing else changes (Open/Closed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Config
from .registry import ToolRegistry
from .tools import Tool


class Toolset(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def create_tools(self, config: Config) -> list[Tool]:
        ...

    def system_guidance(self) -> str | None:
        """Short, role-specific guidance for working with this toolset's tools.

        Returned text is assembled by :class:`~vibeharness.prompt.SystemPromptBuilder`
        into a dedicated section of the system prompt, so the *system* prompt varies
        by the active toolset(s) without any caller having to know the details. Return
        ``None`` (the default) to contribute nothing.
        """
        return None

    def check_prerequisites(self) -> list[str]:
        """Return a list of human-readable problems (empty = ready to use)."""
        return []

    def setup(self, config: Config) -> None:
        """Called once before a run (e.g. launch a browser)."""

    def teardown(self, config: Config) -> None:
        """Called once after a run (e.g. close a browser). Must not raise."""


class FilesystemToolset(Toolset):
    name = "fs"
    description = "Read, write, search, and manage files on the local filesystem."

    def create_tools(self, config: Config) -> list[Tool]:
        from .filesystem import FileSystem
        from .fs_tools import build_default_tools
        return build_default_tools(FileSystem(), config.observation_char_limit)

    def system_guidance(self) -> str | None:
        return ("Use create_file for a new file and write_file to change an existing one. "
                "After any write, read the file back to confirm it holds exactly what you intended "
                "before moving on.")


class ToolsetCatalog:
    """Named collection of available toolsets."""

    def __init__(self, toolsets: list[Toolset]):
        self._toolsets: dict[str, Toolset] = {}
        for ts in toolsets:
            if ts.name in self._toolsets:
                raise ValueError(f"duplicate toolset name: {ts.name}")
            self._toolsets[ts.name] = ts

    def names(self) -> list[str]:
        return list(self._toolsets)

    def describe(self) -> list[tuple[str, str]]:
        return [(ts.name, ts.description) for ts in self._toolsets.values()]

    def get(self, name: str) -> Toolset:
        if name not in self._toolsets:
            raise KeyError(name)
        return self._toolsets[name]

    def select(self, names: list[str]) -> list[Toolset]:
        return [self.get(n) for n in names]

    def build_registry(self, toolsets: list[Toolset], config: Config) -> ToolRegistry:
        from .validation import ValidateTool
        tools: list[Tool] = [ValidateTool()]   # core: present in every toolset
        for ts in toolsets:
            tools.extend(ts.create_tools(config))
        return ToolRegistry(tools)


def default_catalog() -> ToolsetCatalog:
    from .web import WebToolset
    return ToolsetCatalog([FilesystemToolset(), WebToolset()])
