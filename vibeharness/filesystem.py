"""Filesystem service: the single place that touches the OS filesystem.

Kept free of any LLM/tool concerns (SRP) so it is trivially testable. Full
filesystem scope per harness config; relative paths resolve against the process
working directory. Raises FileSystemError with human-readable messages.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
from dataclasses import dataclass


class FileSystemError(Exception):
    """Raised for any filesystem operation failure, with a readable message."""


# One page of a paged read. Pages are fixed-size windows over the file's text.
PAGE_SIZE = 10000


@dataclass(frozen=True)
class Page:
    """A single window of a file's text, plus where it sits in the whole."""
    text: str
    page_number: int
    total_pages: int
    total_chars: int


# Files we won't try to read/search as text.
_SKIP_EXT = {".exe", ".dll", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
             ".zip", ".gz", ".7z", ".mp4", ".mp3", ".ico", ".so", ".pyd"}
_MAX_SEARCH_FILE_BYTES = 2_000_000


class FileSystem:
    def resolve(self, path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))

    # ---- read-only ----
    def list_dir(self, path: str = ".", recursive: bool = False) -> list[str]:
        root = self.resolve(path)
        if not os.path.isdir(root):
            raise FileSystemError(f"'{path}' is not a directory or does not exist")
        entries: list[str] = []
        if recursive:
            for base, dirs, files in os.walk(root):
                for d in sorted(dirs):
                    entries.append(os.path.relpath(os.path.join(base, d), root) + "/")
                for f in sorted(files):
                    entries.append(os.path.relpath(os.path.join(base, f), root))
        else:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                entries.append(name + ("/" if os.path.isdir(full) else ""))
        return entries

    def read(self, path: str, max_chars: int | None = None) -> str:
        full = self.resolve(path)
        if not os.path.isfile(full):
            raise FileSystemError(f"'{path}' does not exist or is not a file")
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
        except OSError as e:
            raise FileSystemError(f"could not read '{path}': {e}")
        if max_chars is not None and len(data) > max_chars:
            return data[:max_chars] + f"\n... [truncated, {len(data)} chars total]"
        return data

    def read_page(self, path: str, page: int = 1) -> Page:
        """Read one PAGE_SIZE-char window of a file.

        Pages are 1-indexed. An empty file is a single (empty) page so callers
        always get page 1 of 1. Out-of-range pages raise FileSystemError.
        """
        full = self.resolve(path)
        if not os.path.isfile(full):
            raise FileSystemError(f"'{path}' does not exist or is not a file")
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
        except OSError as e:
            raise FileSystemError(f"could not read '{path}': {e}")
        total_chars = len(data)
        total_pages = max(1, (total_chars + PAGE_SIZE - 1) // PAGE_SIZE)
        if page < 1 or page > total_pages:
            raise FileSystemError(
                f"page {page} is out of range for '{path}' "
                f"(it has {total_pages} page(s))")
        start = (page - 1) * PAGE_SIZE
        return Page(data[start:start + PAGE_SIZE], page, total_pages, total_chars)

    def search(self, query: str, path: str = ".", target: str = "content",
               max_results: int = 50) -> list[str]:
        root = self.resolve(path)
        if not os.path.exists(root):
            raise FileSystemError(f"search root '{path}' does not exist")
        q = query.lower()
        hits: list[str] = []
        want_name = target in ("filename", "both")
        want_content = target in ("content", "both")
        for base, _dirs, files in os.walk(root):
            for name in files:
                full = os.path.join(base, name)
                rel = os.path.relpath(full, root)
                if want_name and (fnmatch.fnmatch(name, query) or q in name.lower()):
                    hits.append(rel)
                    if len(hits) >= max_results:
                        return hits
                if want_content and os.path.splitext(name)[1].lower() not in _SKIP_EXT:
                    try:
                        if os.path.getsize(full) > _MAX_SEARCH_FILE_BYTES:
                            continue
                        with open(full, "r", encoding="utf-8", errors="ignore") as f:
                            for n, line in enumerate(f, 1):
                                if q in line.lower():
                                    hits.append(f"{rel}:{n}: {line.strip()[:160]}")
                                    if len(hits) >= max_results:
                                        return hits
                    except OSError:
                        continue
        return hits

    # ---- mutating ----
    def write(self, path: str, content: str, mode: str = "overwrite") -> int:
        full = self.resolve(path)
        parent = os.path.dirname(full)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if mode == "overwrite":
            new = content
        elif mode == "append":
            existing = self._read_or_empty(full)
            new = existing + content
        elif mode == "prepend":
            existing = self._read_or_empty(full)
            new = content + existing
        else:
            raise FileSystemError(f"unknown write mode '{mode}'")
        try:
            with open(full, "w", encoding="utf-8") as f:
                f.write(new)
        except OSError as e:
            raise FileSystemError(f"could not write '{path}': {e}")
        return len(content)

    def make_directory(self, path: str) -> None:
        try:
            os.makedirs(self.resolve(path), exist_ok=True)
        except OSError as e:
            raise FileSystemError(f"could not create directory '{path}': {e}")

    def delete(self, path: str) -> None:
        full = self.resolve(path)
        if not os.path.exists(full):
            raise FileSystemError(f"'{path}' does not exist")
        try:
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
        except OSError as e:
            raise FileSystemError(f"could not delete '{path}': {e}")

    def move(self, src: str, dst: str) -> None:
        src_full = self.resolve(src)
        if not os.path.exists(src_full):
            raise FileSystemError(f"source '{src}' does not exist")
        try:
            shutil.move(src_full, self.resolve(dst))
        except OSError as e:
            raise FileSystemError(f"could not move '{src}' to '{dst}': {e}")

    def copy(self, src: str, dst: str) -> None:
        """Copy a file or directory, preserving content + metadata and the
        original. Auto-creates the destination's parent dir; refuses if the
        destination already exists."""
        src_full = self.resolve(src)
        dst_full = self.resolve(dst)
        if not os.path.exists(src_full):
            raise FileSystemError(f"source '{src}' does not exist")
        if os.path.exists(dst_full):
            raise FileSystemError(f"destination '{dst}' already exists")
        parent = os.path.dirname(dst_full)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        try:
            if os.path.isdir(src_full):
                shutil.copytree(src_full, dst_full)
            else:
                shutil.copy2(src_full, dst_full)
        except OSError as e:
            raise FileSystemError(f"could not copy '{src}' to '{dst}': {e}")

    @staticmethod
    def _read_or_empty(full: str) -> str:
        if os.path.isfile(full):
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return ""
