"""Concrete tools. Small but powerful: behaviour is widened through optional
params (e.g. write_file.mode, search.target, manage_path.action) rather than by
adding more tools. Each tool turns its result into a past-tense sentence for the
narrative memory.
"""
from __future__ import annotations

import os

from .filesystem import FileSystem, FileSystemError
from .tools import Param, Tool, ToolResult


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit} chars]"


class ReadTracker:
    """Session-scoped record of which files have been read.

    Holds resolved (absolute) paths so the write guard can confirm a file's
    contents were seen before an overwrite discards them.
    """

    def __init__(self):
        self._seen: set[str] = set()

    def mark(self, resolved_path: str) -> None:
        self._seen.add(resolved_path)

    def has_read(self, resolved_path: str) -> bool:
        return resolved_path in self._seen


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
    description = ("Read a file one page at a time. A page is 10,000 characters. "
                  "Returns the first 10,000 characters of the requested page, reports the "
                  "total number of pages, and you can read later pages by passing `page`.")

    def __init__(self, fs: FileSystem, obs_limit: int, tracker: ReadTracker):
        self._fs, self._limit, self._tracker = fs, obs_limit, tracker

    @property
    def parameters(self):
        return [
            Param("path", "string", "Path of the file to read."),
            Param("page", "integer", "Which 10,000-character page to read.",
                  required=False, default=1),
        ]

    def run(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        page = int(args.get("page", 1))
        try:
            result = self._fs.read_page(path, page)
        except FileSystemError as e:
            return ToolResult(False, f"you tried to read the file '{path}' but it returned an error: {e}.")
        self._tracker.mark(self._fs.resolve(path))
        text = _truncate(result.text, self._limit)
        return ToolResult(
            True,
            f"you read the file '{path}' (page {result.page_number} of "
            f"{result.total_pages}, {result.total_chars} chars total): \"{text}\".")


class CreateFileTool(Tool):
    name = "create_file"
    description = ("Create a NEW file with the given content, auto-creating any parent "
                   "folders. Use this for files that do not exist yet; to change an "
                   "existing file, use write_file instead.")

    def __init__(self, fs: FileSystem):
        self._fs = fs

    @property
    def parameters(self):
        return [
            Param("path", "string", "Path of the file to create."),
            Param("content", "string", "The text to write into the new file."),
        ]

    def run(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        if os.path.exists(self._fs.resolve(path)):
            return ToolResult(False, f"you tried to create the file '{path}' but it already "
                                     f"exists — use write_file to modify an existing file.")
        try:
            n = self._fs.write(path, args.get("content", ""), "overwrite")
        except FileSystemError as e:
            return ToolResult(False, f"you tried to create the file '{path}' but it returned an error: {e}.")
        return ToolResult(True, f"you created the file '{path}' ({n} characters).")


class WriteFileTool(Tool):
    name = "write_file"
    description = ("Modify an EXISTING file. Use the mode parameter to overwrite, append, or "
                   "prepend. An overwrite is only allowed after you have read the file this "
                   "session (so you don't discard its contents); use create_file for new files.")

    def __init__(self, fs: FileSystem, tracker: ReadTracker):
        self._fs, self._tracker = fs, tracker

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
        resolved = self._fs.resolve(path)
        if not os.path.isfile(resolved):
            return ToolResult(False, f"you tried to write to '{path}' but it does not exist — "
                                     f"use create_file to create a new file.")
        if mode == "overwrite" and not self._tracker.has_read(resolved):
            return ToolResult(False, f"you tried to overwrite '{path}' but refusing to overwrite "
                                     f"'{path}' — read it first so you don't discard its contents.")
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
    description = ("Manage files and folders: create a directory, delete a file/folder, "
                   "move/rename a path, or copy a file/folder. Choose with the action parameter.")

    def __init__(self, fs: FileSystem):
        self._fs = fs

    @property
    def parameters(self):
        return [
            Param("action", "string", "Operation to perform.",
                  enum=("make_directory", "delete", "move", "copy")),
            Param("path", "string", "Target path (source path for a move or copy)."),
            Param("destination", "string",
                  "New path. Required when action is 'move' or 'copy'.",
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
            if action == "copy":
                dst = args.get("destination")
                if not dst:
                    return ToolResult(False, "you tried to copy a path but did not provide a destination.")
                self._fs.copy(path, dst)
                return ToolResult(True, f"you copied '{path}' to '{dst}'.")
            return ToolResult(False, f"you requested an unknown action '{action}'.")
        except FileSystemError as e:
            return ToolResult(False, f"you tried to {action} '{path}' but it returned an error: {e}.")


def build_default_tools(fs: FileSystem, obs_limit: int) -> list[Tool]:
    """Factory for the filesystem toolset (keeps wiring in one place).
    The run-ending `validate` tool is injected separately as a core tool.

    A single session-scoped ReadTracker is shared between read_file (which records
    each successful read) and write_file (which requires a prior read before an
    overwrite can discard a file's contents)."""
    tracker = ReadTracker()
    return [
        ListDirectoryTool(fs, obs_limit),
        ReadFileTool(fs, obs_limit, tracker),
        CreateFileTool(fs),
        WriteFileTool(fs, tracker),
        SearchTool(fs, obs_limit),
        ManagePathTool(fs),
    ]
