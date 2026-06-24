"""Web toolset: a stateful browser exposed through one `browse` tool.

Wraps the Playwright **Agent CLI** (`playwright-cli`, from `@playwright/cli`),
which keeps a browser alive between calls within a named session, so navigation,
clicks, and content extraction all share state. Following the same minimal-but-
powerful principle as the filesystem toolset, a single `browse` tool covers the
whole browser via an `action` parameter.

`snapshot` is the agent's eyes: it is the only way to observe the page.

Install the backend with:  npm install -g @playwright/cli@latest
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Callable

from .config import Config
from .toolset import Toolset
from .tools import Param, Tool, ToolResult

BINARY = "playwright-cli"


class PlaywrightCli:
    """Thin, injectable wrapper around the stateful `playwright-cli` binary."""

    def __init__(self, session: str, timeout: int):
        self._session = session
        self._timeout = timeout
        self._binary = shutil.which(BINARY)

    @property
    def available(self) -> bool:
        return self._binary is not None

    def run(self, *args: str) -> tuple[bool, str]:
        """Run one CLI command in this session. Returns (ok, combined_output).

        Every command is HARD-BOUNDED by ``self._timeout``. On timeout the whole
        process tree is killed and a clear error is returned — the call never
        hangs (issue #4).

        We drive the child with Popen + ``communicate(timeout=...)`` rather than
        ``subprocess.run(timeout=...)`` on purpose: ``playwright-cli`` is a Node
        process that spawns a *browser grandchild* which inherits the captured
        stdout/stderr pipe handles. ``subprocess.run``'s timeout path kills only
        the direct child and then re-reads the pipes to drain them; while the
        browser grandchild is still alive holding the write-end, those pipes
        never reach EOF and that post-kill read blocks *forever* — so the
        ``TimeoutExpired`` is never delivered and the agent turn wedges. Killing
        the whole tree first lets the drain complete (or be skipped) promptly.
        """
        if not self._binary:
            return False, f"{BINARY} is not installed"
        cmd = self._command(*args)
        # Put the child in its own process group/session so that, on timeout, we
        # can signal the WHOLE tree (its browser grandchildren too) without
        # touching the harness's own group.
        popen_kw = {}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kw["start_new_session"] = True
        # Force UTF-8 decoding: page snapshots contain emoji/unicode that the
        # default Windows codec (cp1252) cannot decode, which would otherwise
        # crash the reader thread and return empty output.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", **popen_kw)
        try:
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire tree (child + browser grandchildren) so the pipe
            # write-ends close; only then can we drain without re-blocking.
            self._kill_tree(proc)
            try:
                proc.communicate(timeout=self._kill_grace)
            except subprocess.TimeoutExpired:
                pass  # drained best-effort; never block the agent turn on it
            return False, f"command timed out after {self._timeout}s"
        out = ((stdout or "") + (stderr or "")).strip()
        return proc.returncode == 0, out

    def _command(self, *args: str) -> list[str]:
        """Build the argv for one CLI invocation. Isolated as a seam so the
        bounded-execution path in ``run`` can be exercised against a stand-in
        command (e.g. a deliberately slow process) without a live browser."""
        return [self._binary, f"-s={self._session}", *args]

    # Grace period to reap a killed process tree before giving up the drain.
    _kill_grace = 5

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen") -> None:
        """Best-effort kill of ``proc`` and every descendant it spawned.

        ``proc.kill()`` only terminates the direct child; the Node-launched
        browser would survive and keep the captured pipes open. We use the OS
        process-tree killers (``taskkill /T`` on Windows, process-group signal
        on POSIX) and always fall back to a plain ``proc.kill()``.
        """
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                import os
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass


# action -> (CLI arg builder, required params, past-tense verb)
_ACTIONS = {
    "goto":       (lambda p: ["goto", p["url"]],                ["url"],            "navigated to"),
    "snapshot":   (lambda p: ["snapshot"],                      [],                "read"),
    "click":      (lambda p: ["click", p["target"]],            ["target"],        "clicked"),
    "fill":       (lambda p: ["fill", p["target"], p["text"]],  ["target", "text"], "filled"),
    "type":       (lambda p: ["type", p["text"]],               ["text"],          "typed into"),
    "select":     (lambda p: ["select", p["target"], p["value"]], ["target", "value"], "selected an option in"),
    "check":      (lambda p: ["check", p["target"]],            ["target"],        "checked"),
    "uncheck":    (lambda p: ["uncheck", p["target"]],          ["target"],        "unchecked"),
    "upload":     (lambda p: ["upload", p["file"]],             ["file"],          "uploaded a file to"),
    "hover":      (lambda p: ["hover", p["target"]],            ["target"],        "hovered over"),
    "press":      (lambda p: ["press", p["key"]],               ["key"],           "pressed a key on"),
    "drag":       (lambda p: ["drag", p["target"], p["end"]],   ["target", "end"], "dragged on"),
    "eval":       (lambda p: ["eval", p["expression"]],         ["expression"],    "evaluated JavaScript on"),
    "screenshot": (lambda p: ["screenshot"] + ([p["target"]] if p.get("target") else []), [], "screenshotted"),
    "back":       (lambda p: ["go-back"],                       [],                "went back on"),
    "forward":    (lambda p: ["go-forward"],                    [],                "went forward on"),
    "reload":     (lambda p: ["reload"],                        [],                "reloaded"),
}


class BrowseTool(Tool):
    name = "browse"
    description = (
        "Drive one stateful browser — page, cookies, and history persist between calls. "
        "Pick what to do with `action`.\n"
        "SEEING THE PAGE: `snapshot` is your eyes — the ONLY way to observe a page. It returns "
        "the visible text, every link with its URL, every form field, and a stable ref per "
        "element (like `e6`). Snapshot after you navigate or change the page, read it, then act "
        "on an element by passing its ref (or a CSS selector) as `target`.\n"
        "FLOW: goto -> snapshot -> interact (click/fill/...) using refs -> snapshot again -> repeat.\n"
        "ACTIONS: goto (open `url`); snapshot (read the page); click (`target`); fill (set "
        "`target` to `text`, clearing it first); type (`text` into the focused element); select "
        "(option `value` in `target`); check / uncheck (`target`); upload (`file` to the active "
        "file input); hover (`target`); press (`key`, e.g. 'Enter'); drag (`target` -> `end`); "
        "eval (run JS `expression`); screenshot (save a PNG); back / forward / reload."
    )

    def __init__(self, cli: PlaywrightCli, observation_limit: int):
        self._cli = cli
        self._limit = observation_limit

    @property
    def parameters(self):
        return [
            Param("action", "string", "What to do in the browser.", enum=tuple(_ACTIONS.keys())),
            Param("url", "string", "URL to open. Required for goto.", required=False),
            Param("target", "string", "Element ref from a snapshot (e.g. 'e6') or a CSS selector. "
                  "Required for click/fill/select/check/uncheck/hover/drag.", required=False),
            Param("text", "string", "Text to enter. Required for fill and type.", required=False),
            Param("value", "string", "Option to choose. Required for select.", required=False),
            Param("file", "string", "Absolute path of the file to upload. Required for upload.",
                  required=False),
            Param("key", "string", "Keyboard key, e.g. 'Enter' or 'Tab'. Required for press.",
                  required=False),
            Param("end", "string", "Destination element ref/selector. Required for drag.",
                  required=False),
            Param("expression", "string", "JavaScript function to evaluate, e.g. "
                  "\"() => document.title\". Required for eval.", required=False),
        ]

    def run(self, args: dict) -> ToolResult:
        action = args.get("action")
        spec = _ACTIONS.get(action)
        if spec is None:
            return ToolResult(False, f"you requested an unknown browser action '{action}'.")
        build_args, required, verb = spec
        missing = [p for p in required if not args.get(p)]
        if missing:
            return ToolResult(False, f"you tried to '{action}' but did not provide: "
                              f"{', '.join(missing)}.")

        ok, output = self._cli.run(*build_args(args))
        subject = args.get("url") or args.get("target") or "the page"
        if not ok:
            return ToolResult(False, f"you tried to {verb} {subject} but it failed: "
                              f"{self._trim(output)}")
        return ToolResult(True, f"you {verb} {subject}. Result:\n{self._trim(output)}")

    def _trim(self, text: str) -> str:
        if len(text) <= self._limit:
            return text
        return text[:self._limit] + f"\n…[+{len(text) - self._limit} chars truncated]"


def capture_page_snapshot(cli: PlaywrightCli, char_limit: int) -> str:
    """Capture a fresh `snapshot` of the live page from an EXISTING session and
    return its text, truncated to ``char_limit`` (issue #24).

    Reuses the supplied :class:`PlaywrightCli` — i.e. the SAME named session the
    agent's `browse` tool drives — so the captured snapshot reflects the actual
    page the model is acting on; it never launches a second browser. On any failure
    (no session open yet, CLI error, timeout) it returns "" so the caller simply
    renders no page section that turn rather than crashing the run.

    ``cli`` is the injectable seam: tests pass a stand-in whose ``run`` returns
    canned snapshot text, so the per-turn injection can be exercised with no browser.
    """
    try:
        ok, output = cli.run("snapshot")
    except Exception:
        return ""
    if not ok:
        return ""
    text = (output or "").strip()
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + f"\n…[+{len(text) - char_limit} chars truncated]"


def make_snapshot_provider(config: Config) -> Callable[[], str]:
    """Build a per-turn page-snapshot provider bound to the run's web session.

    Mirrors cli.py's ``render_workspace`` seam: returns a zero-arg callable that,
    each time it is called (once per turn, at prompt-build time), captures a FRESH
    snapshot from the run's existing Playwright session and returns it truncated to
    ``config.web_snapshot_char_limit``. The session name and timeout come from
    ``config`` so the snapshot CLI shares the exact session the `browse` tool uses.
    """
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
    return lambda: capture_page_snapshot(cli, config.web_snapshot_char_limit)


class WebToolset(Toolset):
    name = "web"
    description = ("Browse the web with a stateful browser: navigate, read page content and "
                   "links via snapshot, click, fill forms, upload files, and screenshot.")

    def system_guidance(self) -> str | None:
        return (
            "A snapshot is your only view of the page, so take a fresh one after every "
            "navigation or click and read it before deciding what to do next. "
            "Act only on elements present in the current snapshot, referencing them by their "
            "ref — never guess a selector or element id. "
            "If a cookie/consent banner or modal dialog blocks what you need, clear it first: "
            "locate its Accept / Agree / Reject / Dismiss / Continue control in the snapshot and "
            "click that ref before doing anything else."
        )

    def create_tools(self, config: Config) -> list[Tool]:
        cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
        return [BrowseTool(cli, config.web_observation_char_limit)]

    def check_prerequisites(self) -> list[str]:
        if shutil.which(BINARY) is None:
            return [f"'{BINARY}' not found on PATH. Install it with: "
                    f"npm install -g @playwright/cli@latest"]
        return []

    def setup(self, config: Config) -> None:
        # Open the browser once for the run. Headed by default so a human can watch.
        flags: list[str] = []
        if not config.web_headless:
            flags.append("--headed")
        if config.web_browser:
            flags += ["--browser", config.web_browser]
        PlaywrightCli(config.web_session, config.web_cli_timeout).run("open", *flags)

    def teardown(self, config: Config) -> None:
        PlaywrightCli(config.web_session, config.web_cli_timeout).run("close")
