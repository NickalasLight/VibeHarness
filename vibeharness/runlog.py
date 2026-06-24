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
import uuid
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

    @property
    def diagnostics_dir(self) -> Path:
        """Per-turn diagnostic dumps live in a subfolder of this run's ``.vibe/`` so
        they sit alongside the run log without cluttering the top level (issue #37)."""
        return self.dir / f"{self.stamp}-diagnostics"

    def dump_turn_diagnostics(self, turn: int, *, snapshot: str | None = None,
                              system_prompt: str | None = None) -> None:
        """Write per-turn diagnostic dumps into ``<stamp>-diagnostics/`` (issue #37).

        Two optional dumps, each only written when its text is supplied:
          - ``turn-<NNN>-snapshot-<ts>.txt`` — the COMPLETE, untruncated page
            snapshot (ground truth on its true size), prefixed with its char length.
          - ``turn-<NNN>-system-prompt-<ts>.txt`` — the EXACT system prompt string
            injected into the model that turn.

        Best-effort and exception-safe: a failure here must NEVER abort the agent
        loop, so every step is guarded and errors are swallowed. Files are stamped
        with the turn number (zero-padded) plus a sub-second timestamp so repeated
        turns never collide and ordering is obvious.
        """
        try:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            _hide(self.dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = f"turn-{turn:03d}"
            if snapshot is not None:
                body = (f"# turn {turn} raw page snapshot (untruncated)\n"
                        f"# char length: {len(snapshot)}\n"
                        f"# captured at: {ts}\n\n{snapshot}")
                (self.diagnostics_dir / f"{tag}-snapshot-{ts}.txt").write_text(
                    body, encoding="utf-8", errors="backslashreplace")
            if system_prompt is not None:
                body = (f"# turn {turn} injected system prompt\n"
                        f"# char length: {len(system_prompt)}\n"
                        f"# captured at: {ts}\n\n{system_prompt}")
                (self.diagnostics_dir / f"{tag}-system-prompt-{ts}.txt").write_text(
                    body, encoding="utf-8", errors="backslashreplace")
        except Exception:
            # Diagnostics are a best-effort aid; never let a dump failure break the run.
            pass

    def log_validator(self, *, task: str, history: str, claim: str,
                      reasoning: str, passed: bool, reason: str,
                      model: str | None = None, config: Config | None = None) -> None:
        """Persist ONE validator invocation to its own file in this run's ``.vibe/``.

        The validator subagent (``LLMValidator.validate``) runs a single-shot model
        call each time the agent calls ``validate``; without this its reasoning and
        verdict are lost from the on-disk record (issue #47). Each call writes its own
        ``validator_<guid>.json`` (``validator_`` marks the producer; the uuid4 hex
        guid guarantees no clobber when ``validate`` is called multiple times).

        Best-effort and exception-safe — exactly the contract of the #37 diagnostics
        dump: a logging failure must NEVER throw into the run, so every step is guarded
        and errors are swallowed; the validator's verdict is unaffected either way.
        """
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            _hide(self.dir)
            now = datetime.now()
            payload = {
                "type": "validator",
                "run_stamp": self.stamp,
                "timestamp": now.isoformat(timespec="seconds"),
                "model": (config.model if config is not None else model),
                "config": {
                    "model": config.model,
                    "temperature": config.temperature,
                    "action_temperature": config.action_temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                } if config is not None else None,
                "inputs": {
                    "task": task,
                    "history": history,
                    "claim": claim,
                },
                "reasoning": reasoning,
                "verdict": {
                    "passed": passed,
                    "reason": reason,
                },
            }
            guid = uuid.uuid4().hex
            path = self.dir / f"validator_{guid}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8", errors="backslashreplace")
        except Exception:
            # Logging is a best-effort record; never let it break the validation run.
            pass

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
