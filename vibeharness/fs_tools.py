"""Concrete tools. Small but powerful: behaviour is widened through optional
params (e.g. write_file.mode, search.target, manage_path.action) rather than by
adding more tools. Each tool turns its result into a past-tense sentence for the
narrative memory.
"""
from __future__ import annotations

from .filesystem import FileSystem, FileSystemError
from .tools import Param, Tool, ToolResult


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit} chars]"


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List the files and sub-folders in a directory."

    def __init__(self, fs: FileSystem, obs_limit: int):
        self._fs, self._limit = fs, obs_limit

    @property
    def parameters(self):
        return [
            Param("path", "string", "Directory to list.", required=False, default="."),
            Param("recursive", "boolean", "Also list nested contents.", required=False, default=False),
        ]

    def run(self, args: dict) -> ToolResult:
        path = args.get("path", ".")
        try:
            entries = self._fs.list_dir(path, bool(args.get("recursive", False)))
        except FileSystemError as e:
            return ToolResult(False, f"you tried to list the directory '{path}' but it returned an error: {e}.")
        listing = ", ".join(entries) if entries else "(empty)"
        return ToolResult(True, f"you listed the directory '{path}', which contained: {_truncate(listing, self._limit)}.")


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the full text content of a file."

    def __init__(self, fs: FileSystem, obs_limit: int):
        self._fs, self._limit = fs, obs_limit

    @property
    def parameters(self):
        return [Param("path", "string", "Path of the file to read.")]

    def run(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        try:
            content = self._fs.read(path, max_chars=self._limit)
        except FileSystemError as e:
            return ToolResult(False, f"you tried to read the file '{path}' but it returned an error: {e}.")
        return ToolResult(True, f"you read the file '{path}', which contained: \"{content}\".")


class WriteFileTool(Tool):
    name = "write_file"
    description = ("Write text to a file, creating it and any parent folders if needed. "
                   "Set `mode` to overwrite, append, or prepend.")

    def __init__(self, fs: FileSystem):
        self._fs = fs

    @property
    def parameters(self):
        return [
            Param("path", "string", "Path of the file to write."),
            Param("content", "string", "Text to write."),
            Param("mode", "string", "How to write the text.", required=False,
                  default="overwrite", enum=("overwrite", "append", "prepend")),
        ]

    def run(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        mode = args.get("mode", "overwrite")
        try:
            n = self._fs.write(path, args.get("content", ""), mode)
        except FileSystemError as e:
            return ToolResult(False, f"you tried to write to '{path}' but it returned an error: {e}.")
        verb = {"overwrite": "wrote", "append": "appended to", "prepend": "prepended to"}[mode]
        return ToolResult(True, f"you {verb} the file '{path}' ({n} characters).")


class SearchTool(Tool):
    name = "search"
    description = "Search a directory for text inside files, for file names, or both."

    def __init__(self, fs: FileSystem, obs_limit: int):
        self._fs, self._limit = fs, obs_limit

    @property
    def parameters(self):
        return [
            Param("query", "string", "Text or filename pattern to find."),
            Param("path", "string", "Directory to search under.", required=False, default="."),
            Param("target", "string", "What to match.", required=False, default="content",
                  enum=("content", "filename", "both")),
            Param("max_results", "integer", "Max matches to return.", required=False, default=50),
        ]

    def run(self, args: dict) -> ToolResult:
        query, path = args.get("query", ""), args.get("path", ".")
        try:
            hits = self._fs.search(query, path, args.get("target", "content"),
                                   int(args.get("max_results", 50)))
        except FileSystemError as e:
            return ToolResult(False, f"you tried to search for '{query}' but it returned an error: {e}.")
        if not hits:
            return ToolResult(True, f"you searched for '{query}' under '{path}' and found no matches.")
        joined = _truncate("; ".join(hits), self._limit)
        return ToolResult(True, f"you searched for '{query}' under '{path}' and found: {joined}.")


class ManagePathTool(Tool):
    name = "manage_path"
    description = ("Create a directory, delete a file/folder, or move/rename a path. "
                   "Pick with `action`.")

    def __init__(self, fs: FileSystem):
        self._fs = fs

    @property
    def parameters(self):
        return [
            Param("action", "string", "Operation to perform.",
                  enum=("make_directory", "delete", "move")),
            Param("path", "string", "Target path (the source for a move)."),
            Param("destination", "string", "New path. Required for 'move'.",
                  required=False),
        ]

    def run(self, args: dict) -> ToolResult:
        action, path = args.get("action", ""), args.get("path", "")
        try:
            if action == "make_directory":
                self._fs.make_directory(path)
                return ToolResult(True, f"you created the directory '{path}'.")
            if action == "delete":
                self._fs.delete(path)
                return ToolResult(True, f"you deleted '{path}'.")
            if action == "move":
                dst = args.get("destination")
                if not dst:
                    return ToolResult(False, "you tried to move a path but did not provide a destination.")
                self._fs.move(path, dst)
                return ToolResult(True, f"you moved '{path}' to '{dst}'.")
            return ToolResult(False, f"you requested an unknown action '{action}'.")
        except FileSystemError as e:
            return ToolResult(False, f"you tried to {action} '{path}' but it returned an error: {e}.")


def build_default_tools(fs: FileSystem, obs_limit: int) -> list[Tool]:
    """Factory for the filesystem toolset (keeps wiring in one place).
    The run-ending `validate` tool is injected separately as a core tool."""
    return [
        ListDirectoryTool(fs, obs_limit),
        ReadFileTool(fs, obs_limit),
        WriteFileTool(fs),
        SearchTool(fs, obs_limit),
        ManagePathTool(fs),
    ]
