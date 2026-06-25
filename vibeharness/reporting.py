"""Reporters: observers that render a run to some output.

The agent depends on the `Reporter` interface (DIP), not on the console. The
console implementation streams each turn live — reasoning, the action JSON, and
the result — so `vibe` feels like a basic coding agent.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Action, RunResult
    from .config import Config


class Reporter(ABC):
    @abstractmethod
    def run_start(self, task: str, workdir: str, config: "Config") -> None: ...
    @abstractmethod
    def turn_start(self, index: int) -> None: ...
    @abstractmethod
    def reasoning_token(self, text: str) -> None: ...
    @abstractmethod
    def action_token(self, text: str) -> None: ...
    @abstractmethod
    def action_result(self, action: "Action") -> None: ...
    @abstractmethod
    def advisor_start(self) -> None: ...
    @abstractmethod
    def advisor_token(self, text: str) -> None: ...
    @abstractmethod
    def advisor_end(self) -> None: ...
    @abstractmethod
    def note(self, text: str) -> None: ...
    @abstractmethod
    def run_end(self, result: "RunResult") -> None: ...

    # ---- validator subagent stream ----
    # The main agent hands control to a separate validator subagent. These hooks
    # let a reporter render that subagent's live generation distinctly from the
    # main agent's turn.
    @abstractmethod
    def validator_start(self) -> None: ...
    @abstractmethod
    def validator_reasoning_token(self, text: str) -> None: ...
    @abstractmethod
    def validator_verdict_token(self, text: str) -> None: ...


class NullReporter(Reporter):
    def run_start(self, task, workdir, config): pass
    def turn_start(self, index): pass
    def reasoning_token(self, text): pass
    def action_token(self, text): pass
    def action_result(self, action): pass
    def note(self, text): pass
    def run_end(self, result): pass
    def validator_start(self): pass
    def validator_reasoning_token(self, text): pass
    def validator_verdict_token(self, text): pass
    def advisor_start(self): pass
    def advisor_thinking_token(self, text): pass
    def advisor_token(self, text): pass
    def advisor_end(self): pass


# ANSI styling (enabled on Windows 10+ consoles).
_C = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
      "green": "\033[32m", "red": "\033[31m", "cyan": "\033[36m", "yellow": "\033[33m",
      "magenta": "\033[35m", "blue": "\033[34m"}


def _enable_ansi() -> None:
    """Enable Windows VT / ANSI processing on stdout and stderr.

    ``os.system("")`` (the old approach) launches a cmd.exe subprocess whose
    console handle gets VT mode set — but that does NOT help when Python's own
    stdout/stderr are pipes (e.g. ``Tee-Object`` in PS7).  The ctypes approach
    calls ``SetConsoleMode`` directly on the handles Python actually holds.

    When a handle IS a real console, we set ENABLE_VIRTUAL_TERMINAL_PROCESSING
    so conhost renders the escape sequences.  When the handle is a pipe (Tee-Object),
    ``GetConsoleMode`` fails and we leave the handle alone — PowerShell 7 with
    ``$PSStyle.OutputRendering = 'Ansi'`` (or Windows Terminal) will render the
    raw escape sequences on its end, so no conversion is needed.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        import ctypes.wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12

        for std_handle in (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE):
            handle = kernel32.GetStdHandle(std_handle)
            if handle and handle != ctypes.wintypes.HANDLE(-1).value:
                mode = ctypes.wintypes.DWORD(0)
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    # Only a real console responds; set VT mode on it.
                    kernel32.SetConsoleMode(
                        handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                    )
                # If GetConsoleMode fails the handle is a pipe — leave it; ANSI
                # passes through to PS7 / Windows Terminal automatically.
    except Exception:
        pass  # non-Windows or ctypes unavailable — never raise from a display helper


class ConsoleReporter(Reporter):
    """Streams a live, color-coded view of each turn to the terminal."""

    def __init__(self, color: bool = True, result_limit: int = 2000):
        self._color = color
        self._result_limit = result_limit   # console-only preview cap; agent gets the full result
        if color:
            _enable_ansi()
        self._reason_open = False
        self._action_open = False
        self._val_reason_open = False
        self._val_verdict_open = False

    def _c(self, code: str, text: str) -> str:
        return f"{_C[code]}{text}{_C['reset']}" if self._color else text

    def _w(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def run_start(self, task: str, workdir: str, config) -> None:
        self._w(self._c("bold", f"\n vibe ") + self._c("dim", f"({config.model}, temp {config.temperature})\n"))
        self._w(self._c("dim", f" workspace: {workdir}\n"))
        self._w(f" task: {task}\n")

    def turn_start(self, index: int) -> None:
        self._reason_open = self._action_open = False
        self._w(self._c("cyan", f"\n┌─ turn {index} " + "─" * 40 + "\n"))

    def reasoning_token(self, text: str) -> None:
        if not self._reason_open:
            self._w(self._c("dim", "│ thinking: "))
            self._reason_open = True
        self._w(self._c("dim", text))

    def action_token(self, text: str) -> None:
        if not self._action_open:
            self._w(self._c("yellow", "\n│ action: "))
            self._action_open = True
        self._w(self._c("yellow", text))

    def note(self, text: str) -> None:
        self._w(self._c("dim", f"│ {text}\n"))

    # ---- validator subagent stream (rendered in magenta, clearly labeled) ----
    def validator_start(self) -> None:
        self._val_reason_open = self._val_verdict_open = False
        self._w(self._c("magenta", "\n│ ╭─ validator subagent ─────\n"))

    def validator_reasoning_token(self, text: str) -> None:
        if not self._val_reason_open:
            self._w(self._c("magenta", "│ ╎ thinking: "))
            self._val_reason_open = True
        self._w(self._c("magenta", text))

    def validator_verdict_token(self, text: str) -> None:
        if not self._val_verdict_open:
            self._w(self._c("magenta", "\n│ ╎ verdict: "))
            self._val_verdict_open = True
        self._w(self._c("magenta", text))

    def action_result(self, action) -> None:
        color = "green" if action.ok else "red"
        mark = "✓" if action.ok else "✗"
        # Collapse to one line and cap length for readability. This is display-only:
        # the agent's memory and the .vibe log keep the full, untruncated result.
        preview = " ".join(action.observation.split())
        if len(preview) > self._result_limit:
            preview = preview[:self._result_limit] + f" …(+{len(preview) - self._result_limit} more chars)"
        self._w("\n" + self._c(color, f"└ {mark} {preview}") + "\n")

    # ---- advisor stream (rendered in blue, clearly labeled) ----
    def advisor_start(self) -> None:
        self._adv_think_open = False
        self._adv_advice_open = False
        self._w(self._c("blue", "\n│ ╭─ advisor ──────────────────\n"))

    def advisor_thinking_token(self, text: str) -> None:
        if not self._adv_think_open:
            self._w(self._c("blue", "│ ╎ thinking: "))
            self._adv_think_open = True
        self._w(self._c("dim", text))

    def advisor_token(self, text: str) -> None:
        if not self._adv_advice_open:
            self._w(self._c("blue", "\n│ ╎ advice: "))
            self._adv_advice_open = True
        self._w(self._c("blue", text))

    def advisor_end(self) -> None:
        self._w(self._c("blue", "\n│ ╰────────────────────────────\n"))

    def run_end(self, result) -> None:
        n = len(result.turns)
        if result.finished:
            self._w(self._c("green", f"\n done in {n} turns — {result.final_summary}\n"))
        else:
            self._w(self._c("red", f"\n stopped after {n} turns without finishing.\n"))
