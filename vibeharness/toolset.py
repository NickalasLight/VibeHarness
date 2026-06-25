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
        seen = {t.name for t in tools}
        for ts in toolsets:
            for tool in ts.create_tools(config):
                # De-duplicate by name: a toolset may *declare* the core `validate`
                # tool (the validator toolset does, issue #31) so its toolset is an
                # honest, self-describing unit — but the registry still holds exactly
                # one tool per name. The core injection above wins.
                if tool.name in seen:
                    continue
                seen.add(tool.name)
                tools.append(tool)
        return ToolRegistry(tools)


def default_catalog() -> ToolsetCatalog:
    from .validation import ValidatorToolset
    from .web import WebToolset
    return ToolsetCatalog([FilesystemToolset(), WebToolset(), ValidatorToolset()])


# An "agent type" is a NAMED DEFAULT TOOLSET SELECTION — nothing more. The agent's
# prompt is *derived* from the active toolsets' system_guidance (issue #19), so there
# is no parallel prompt registry here: choosing an agent only chooses which toolset(s)
# are active by default. The mapping is intentionally explicit + tightly coupled to the
# catalog: every agent name maps to the toolset(s) of the same name. Augment/override
# at the CLI with --toolset.
def agent_default_toolsets(catalog: ToolsetCatalog | None = None) -> dict[str, list[str]]:
    """Map each agent type to its default active toolset(s).

    Derived from the catalog so the set of agents stays in lock-step with the set of
    toolsets: each toolset name is itself an agent that defaults to that one toolset.
    """
    cat = catalog or default_catalog()
    return {name: [name] for name in cat.names()}


# Per-agent-type DEFAULT cap on tool calls emitted per turn (issue #52). The
# agent-type framework owns this alongside `agent_default_toolsets`, so adding an
# agent picks BOTH its default toolset(s) and its default actions-per-turn.
#
# Why these defaults:
#   - fs:        MULTIPLE actions/turn (keep the global Config default).
#   - web:       use the global Config default (was hardcoded 12 until #153/#142-iter8).
#                arXiv:2602.07359 (W&D) shows 3 calls/turn is the empirical accuracy peak
#                for web agents (68% vs 60% at 5, and 12 was originally set before this
#                research existed). The web agent now inherits `default` so Config.
#                max_actions_per_turn is the single source of truth for all agents.
#   - validator: 1 (single-shot pass/fail; never batches).
#
# Any agent NOT listed here falls back to the global Config default (so new agents
# inherit sensible behaviour without an entry). Resolution precedence lives in the
# CLI (`resolve_max_actions`): explicit flag / saved setting > this map > global default.
def agent_default_max_actions(
    default: int = Config.max_actions_per_turn,
) -> dict[str, int]:
    """Map agent type -> its default max actions per turn.

    All agent types now use the global ``default`` (Config.max_actions_per_turn = 3)
    as their starting point; only ``validator`` is pinned to 1 (single-shot)."""
    return {"fs": default, "web": default, "validator": 1}
