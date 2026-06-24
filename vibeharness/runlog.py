"""Streaming per-run logging into a hidden ``.vibe/`` folder in the workspace.

A :class:`RunLogger` is bound to one run (a fixed timestamped file pair) and is
written after *every* turn, so the log reflects progress live and a killed run
still keeps its trace. Each run writes:
  - ``<stamp>.json``  full structured log INCLUDING each turn's reasoning trace
                      and the validator verdicts (for analysis)
  - ``<stamp>.md``    a human-readable transcript

Logs live alongside the work (the current workspace) so each project keeps its
own history.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .config import Config
from .agent import RunResult


def _hide(path: Path) -> None:
    """Best-effort: also set the OS 'hidden' attribute on Windows."""
    if os.name == "nt":
        try:
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
        except Exception:
            pass


class RunLogger:
    def __init__(self, workspace: Path | str, started: datetime):
        self.dir = Path(workspace) / ".vibe"
        self.started = started
        self.stamp = started.strftime("%Y%m%d_%H%M%S")

    @property
    def json_path(self) -> Path:
        return self.dir / f"{self.stamp}.json"

    def write(self, task: str, config: Config, result: RunResult) -> Path:
        """Write/overwrite the log for the current state. Safe to call each turn."""
        self.dir.mkdir(parents=True, exist_ok=True)
        _hide(self.dir)
        payload = {
            "task": task,
            "started_at": self.started.isoformat(timespec="seconds"),
            "model": config.model,
            "temperature": config.temperature,
            "max_steps": config.max_steps,
            "finished": result.finished,
            "final_summary": result.final_summary,
            "validations": result.validations,
            "turns": result.to_dict()["turns"],   # includes per-turn reasoning traces
        }
        # Encode defensively: model/browser-snapshot text can contain lone
        # surrogates or other code points that utf-8 cannot encode and that would
        # otherwise raise UnicodeEncodeError (on Windows the default cp1252 is even
        # stricter). ``errors="backslashreplace"`` guarantees the write never crashes
        # — a corrupt glyph degrades to an escape rather than losing the whole log.
        self.json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                  encoding="utf-8", errors="backslashreplace")
        (self.dir / f"{self.stamp}.md").write_text(
            result.transcript(), encoding="utf-8", errors="backslashreplace")
        return self.json_path
