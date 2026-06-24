"""Web toolset: a stateful browser exposed through discrete, first-class tools.

Wraps the Playwright **Agent CLI** (`playwright-cli`, from `@playwright/cli`),
which keeps a browser alive between calls within a named session, so navigation,
clicks, and content extraction all share state.

Each basic browser operation the CLI supports is its own ``Tool`` subclass
(``goto``, ``click``, ``fill``, ``type``, ``press_key``, ``select_option``,
``hover``, ``navigate_back``, ``evaluate``, …) — see :class:`WebToolset`. This
replaces the old monolithic ``browse(action=...)`` dispatcher: every operation is
now its own named tool with its own typed parameters and description, so the model
chooses a tool the same way it chooses ``read_file`` vs ``write_file``.

The agent does NOT request the page contents. There is no agent-callable
``snapshot`` tool: the live page is captured automatically every turn and injected
into the system prompt under '# Current page (live snapshot — provided
automatically)'. The internal capture helpers (:func:`capture_page_snapshot`,
:func:`capture_page_snapshot_raw`, :func:`make_snapshot_provider`,
:func:`make_raw_snapshot_provider`) drive that auto-injection and are NOT tools.

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
        # Handle to the most recent child this wrapper spawned. Retained so that a
        # later ``close()`` can tree-kill any browser daemon/grandchildren that the
        # session left alive (issue #15) — not just the direct CLI child.
        self._last_proc: "subprocess.Popen | None" = None

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
        # Remember this child so teardown/close can reap its whole tree (#15).
        self._last_proc = proc
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

    def close(self) -> None:
        """Tear the session down for good — gracefully if possible, forcibly always.

        Two layers, both best-effort and exception-safe (issue #15):

        1. Ask ``playwright-cli`` to ``close`` the session so the browser shuts down
           cleanly. This itself goes through bounded ``run`` (kills its own tree on
           timeout), so a wedged close can never hang teardown.
        2. Regardless of whether ``close`` succeeded, tree-kill the LAST child this
           wrapper spawned (the ``open``/``close`` invocation and, on POSIX, its whole
           process group — the Node daemon + chrome grandchildren). A crashed run may
           have skipped a clean close, or ``close`` may have orphaned the daemon; this
           guarantees no ``chrome``/``node`` tree is left leaking.

        Idempotent: calling it again after the tree is already reaped is a harmless
        no-op. Never raises — teardown must always complete.
        """
        try:
            self.run("close")
        except Exception:
            pass
        proc = self._last_proc
        if proc is not None:
            self._kill_tree(proc)
            self._last_proc = None


# Shared element-targeting guidance, woven into every tool that takes a `target`.
# NOTE: deliberately does NOT embed the page-section heading text verbatim — that
# heading is a structural marker tests use to detect the auto-injected page section,
# and repeating it inside a tool description would create false matches.
_REF_NOTE = (
    "Target an element by the stable ref shown for it in the live page view "
    "(e.g. 'e6'), or by a CSS selector. That page view is provided to you "
    "automatically each turn — never guess a ref."
)


class _WebTool(Tool):
    """Base for every discrete browser tool.

    Each concrete subtool declares its ``name``, ``description``, ``parameters``,
    a past-tense ``_verb`` for the narrative observation, and a ``_build`` that maps
    validated args to the ``playwright-cli`` argv. The base owns the shared run
    machinery: missing-param guard, bounded CLI invocation, error/success
    observation phrasing, and output truncation. New operations = one tiny subclass.
    """

    _verb: str = "acted on"          # past-tense narrative verb
    _required: tuple[str, ...] = ()  # arg names that must be present + non-empty

    def __init__(self, cli: PlaywrightCli, observation_limit: int):
        self._cli = cli
        self._limit = observation_limit

    # ---- subclasses implement these two ----
    def _build(self, args: dict) -> list[str]:
        """Map validated args to the playwright-cli argv for this operation."""
        raise NotImplementedError

    def _subject(self, args: dict) -> str:
        """The thing acted on, for the narrative observation. Override as needed."""
        return args.get("url") or args.get("target") or "the page"

    def run(self, args: dict) -> ToolResult:
        missing = [p for p in self._required if not args.get(p)]
        if missing:
            return ToolResult(False, f"you called `{self.name}` but did not provide: "
                              f"{', '.join(missing)}.")
        ok, output = self._cli.run(*self._build(args))
        subject = self._subject(args)
        if not ok:
            return ToolResult(False, f"you tried to {self._verb} {subject} but it failed: "
                              f"{self._trim(output)}")
        return ToolResult(True, f"you {self._verb} {subject}. Result:\n{self._trim(output)}")

    def _trim(self, text: str) -> str:
        if len(text) <= self._limit:
            return text
        return text[:self._limit] + f"\n…[+{len(text) - self._limit} chars truncated]"


# ---------------------------------------------------------------------------
# Discrete, first-class browser tools — one per basic playwright-cli operation.
# Each is registered by WebToolset.create_tools and appears in the prompt + the
# codec call-schema in its own right. There is NO `snapshot` tool: the page is
# auto-injected every turn (see the module docstring + capture_* helpers).
# ---------------------------------------------------------------------------


class GotoTool(_WebTool):
    name = "goto"
    description = "Navigate the browser to a URL. The page, cookies and history persist."
    _verb = "navigated to"
    _required = ("url",)

    @property
    def parameters(self):
        return [Param("url", "string", "The URL to open, e.g. 'https://example.com'.")]

    def _build(self, args):
        return ["goto", args["url"]]


class ClickTool(_WebTool):
    name = "click"
    description = "Click an element (a link, button, checkbox, …). " + _REF_NOTE
    _verb = "clicked"
    _required = ("target",)

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref (e.g. 'e6') or CSS selector to click.")]

    def _build(self, args):
        return ["click", args["target"]]


class FillTool(_WebTool):
    name = "fill"
    description = ("Set a text input / textarea to an exact value, clearing it first. " + _REF_NOTE)
    _verb = "filled"
    _required = ("target", "text")

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref or CSS selector of the field to fill."),
            Param("text", "string", "The exact text to put in the field."),
        ]

    def _build(self, args):
        return ["fill", args["target"], args["text"]]


class TypeTool(_WebTool):
    name = "type"
    description = ("Type text into the currently focused element, keystroke by keystroke "
                  "(use `fill` to set a field's value directly).")
    _verb = "typed into"
    _required = ("text",)

    def _subject(self, args):
        return "the focused element"

    @property
    def parameters(self):
        return [Param("text", "string", "The text to type into the focused element.")]

    def _build(self, args):
        return ["type", args["text"]]


class PressKeyTool(_WebTool):
    name = "press_key"
    description = "Press a single keyboard key, e.g. 'Enter', 'Tab', 'ArrowLeft', 'Escape'."
    _verb = "pressed a key on"
    _required = ("key",)

    @property
    def parameters(self):
        return [Param("key", "string", "The key to press, e.g. 'Enter' or 'Tab'.")]

    def _build(self, args):
        return ["press", args["key"]]


class SelectOptionTool(_WebTool):
    name = "select_option"
    description = "Choose an option in a <select> dropdown. " + _REF_NOTE
    _verb = "selected an option in"
    _required = ("target", "value")

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref or CSS selector of the dropdown."),
            Param("value", "string", "The option value or visible label to select."),
        ]

    def _build(self, args):
        return ["select", args["target"], args["value"]]


class CheckTool(_WebTool):
    name = "check"
    description = "Check a checkbox or radio button (no-op if already checked). " + _REF_NOTE
    _verb = "checked"
    _required = ("target",)

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref or CSS selector of the checkbox/radio.")]

    def _build(self, args):
        return ["check", args["target"]]


class UncheckTool(_WebTool):
    name = "uncheck"
    description = "Uncheck a checkbox (no-op if already unchecked). " + _REF_NOTE
    _verb = "unchecked"
    _required = ("target",)

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref or CSS selector of the checkbox.")]

    def _build(self, args):
        return ["uncheck", args["target"]]


class HoverTool(_WebTool):
    name = "hover"
    description = "Move the mouse over an element (e.g. to reveal a menu). " + _REF_NOTE
    _verb = "hovered over"
    _required = ("target",)

    @property
    def parameters(self):
        return [Param("target", "string", "Element ref or CSS selector to hover over.")]

    def _build(self, args):
        return ["hover", args["target"]]


class DragTool(_WebTool):
    name = "drag"
    description = "Drag one element and drop it onto another. " + _REF_NOTE
    _verb = "dragged"
    _required = ("target", "end")

    @property
    def parameters(self):
        return [
            Param("target", "string", "Element ref or CSS selector to drag FROM."),
            Param("end", "string", "Element ref or CSS selector to drop ONTO."),
        ]

    def _build(self, args):
        return ["drag", args["target"], args["end"]]


class UploadTool(_WebTool):
    name = "upload"
    description = "Upload one or more files to the active file input."
    _verb = "uploaded a file to"
    _required = ("file",)

    def _subject(self, args):
        return "the file input"

    @property
    def parameters(self):
        return [Param("file", "string", "Absolute path of the file to upload.")]

    def _build(self, args):
        return ["upload", args["file"]]


class EvaluateTool(_WebTool):
    name = "evaluate"
    description = ("Run a JavaScript function on the page and return its result, e.g. "
                  "\"() => document.title\".")
    _verb = "evaluated JavaScript on"
    _required = ("expression",)

    @property
    def parameters(self):
        return [Param("expression", "string", "A JS function to evaluate, e.g. "
                      "\"() => document.title\".")]

    def _build(self, args):
        return ["eval", args["expression"]]


class ScreenshotTool(_WebTool):
    name = "screenshot"
    description = "Save a PNG screenshot of the current page (or of one element if `target` is given)."
    _verb = "screenshotted"

    @property
    def parameters(self):
        return [Param("target", "string", "Optional element ref/selector to screenshot just that "
                      "element instead of the whole page.", required=False)]

    def _build(self, args):
        return ["screenshot"] + ([args["target"]] if args.get("target") else [])


class NavigateBackTool(_WebTool):
    name = "navigate_back"
    description = "Go back to the previous page in the browser history."
    _verb = "went back on"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["go-back"]


class NavigateForwardTool(_WebTool):
    name = "navigate_forward"
    description = "Go forward to the next page in the browser history."
    _verb = "went forward on"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["go-forward"]


class ReloadTool(_WebTool):
    name = "reload"
    description = "Reload the current page."
    _verb = "reloaded"

    @property
    def parameters(self):
        return []

    def _build(self, args):
        return ["reload"]


# The full, ordered set of discrete web tools the toolset exposes. One per basic
# playwright-cli operation; NO snapshot (page is auto-injected every turn).
_WEB_TOOL_CLASSES: tuple[type[_WebTool], ...] = (
    GotoTool, ClickTool, FillTool, TypeTool, PressKeyTool, SelectOptionTool,
    CheckTool, UncheckTool, HoverTool, DragTool, UploadTool, EvaluateTool,
    ScreenshotTool, NavigateBackTool, NavigateForwardTool, ReloadTool,
)


def capture_page_snapshot(cli: PlaywrightCli, char_limit: int) -> str:
    """Capture a fresh `snapshot` of the live page from an EXISTING session and
    return its text, truncated to ``char_limit`` (issue #24).

    Reuses the supplied :class:`PlaywrightCli` — i.e. the SAME named session the
    agent's discrete browser tools (goto/click/fill/…) drive — so the captured
    snapshot reflects the actual page the model is acting on; it never launches a
    second browser. This is internal auto-injection, NOT an agent tool. On any failure
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


def capture_page_snapshot_raw(cli: PlaywrightCli) -> str:
    """Capture the COMPLETE, UNTRUNCATED `snapshot` of the live page (issues #37, #43).

    Same session-sharing and never-raises contract as :func:`capture_page_snapshot`,
    but applies NO char cap. #37 uses it for ground-truth diagnostic logging; #43
    uses it so the per-turn injection can apply the DYNAMIC context-budget truncation
    (truncate only as much as the context window requires). Returns "" on any failure
    (no session, CLI error, timeout) so callers record nothing rather than crash.
    """
    try:
        ok, output = cli.run("snapshot")
    except Exception:
        return ""
    if not ok:
        return ""
    return (output or "").strip()


def make_raw_snapshot_provider(config: Config) -> Callable[[], str]:
    """Build a per-turn provider of the RAW, untruncated page snapshot (issues #37, #43).

    Mirrors :func:`make_snapshot_provider` but returns the full snapshot with no char
    cap — for #37 diagnostic ground-truth sizing and #43's dynamic-budget truncation.
    Uses the run's existing session (same name/timeout from ``config``) so it reflects
    the page the model acts on.
    """
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
    return lambda: capture_page_snapshot_raw(cli)


def make_snapshot_provider(config: Config) -> Callable[[], str]:
    """Build a per-turn page-snapshot provider bound to the run's web session.

    Mirrors cli.py's ``render_workspace`` seam: returns a zero-arg callable that,
    each time it is called (once per turn, at prompt-build time), captures a FRESH
    snapshot from the run's existing Playwright session and returns it truncated to
    ``config.web_snapshot_char_limit``. The session name and timeout come from
    ``config`` so the snapshot CLI shares the exact session the discrete browser
    tools use.

    NOTE (#43): this fixed-cap provider is retained for backward compatibility and
    tests; the live run now uses :func:`make_raw_snapshot_provider` plus the dynamic
    budget so the snapshot is sized against the full message each turn.
    """
    cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
    return lambda: capture_page_snapshot(cli, config.web_snapshot_char_limit)


class WebToolset(Toolset):
    name = "web"
    description = ("Browse the web with a stateful browser: navigate, click, fill forms, "
                   "select options, hover, press keys, upload files, run JS, and screenshot. "
                   "The page is shown to you automatically each turn.")

    def system_guidance(self) -> str | None:
        return (
            "Each turn the current page is shown to you automatically under "
            "'# Current page (live snapshot — provided automatically)' — you do NOT and CANNOT "
            "request it; just read it before deciding what to do next. "
            "There is no tool to fetch the page; use the discrete browser tools (goto, click, "
            "fill, type, select_option, hover, press_key, navigate_back, evaluate, …) to ACT. "
            "Act only on elements present in that snapshot, referencing them by their "
            "ref — never guess a selector or element id. "
            "If a cookie/consent banner or modal dialog blocks what you need, clear it first: "
            "locate its Accept / Agree / Reject / Dismiss / Continue control in the snapshot and "
            "click that ref before doing anything else."
        )

    def create_tools(self, config: Config) -> list[Tool]:
        cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
        limit = config.web_observation_char_limit
        return [cls(cli, limit) for cls in _WEB_TOOL_CLASSES]

    def check_prerequisites(self) -> list[str]:
        if shutil.which(BINARY) is None:
            return [f"'{BINARY}' not found on PATH. Install it with: "
                    f"npm install -g @playwright/cli@latest"]
        return []

    def __init__(self) -> None:
        # One CLI wrapper for the whole run so the ``open`` child handle survives to
        # teardown and its tree can be reaped (issue #15). Created lazily in setup().
        self._cli: PlaywrightCli | None = None
        self._atexit_hook: Callable[[], None] | None = None

    def setup(self, config: Config) -> None:
        # Open the browser once for the run. Headed by default so a human can watch.
        self._cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
        # Defensive last-resort reaper: if the run is hard-killed (Ctrl-C escaping the
        # cli finally, os._exit, an unhandled signal) the toolset's teardown may never
        # run. Registering close() with atexit ensures the browser tree is still reaped
        # on interpreter shutdown. teardown() unregisters it once it has run cleanly so
        # we never double-reap on a normal exit (close() is idempotent regardless).
        import atexit
        self._atexit_hook = self._cli.close
        atexit.register(self._atexit_hook)
        flags: list[str] = []
        if not config.web_headless:
            flags.append("--headed")
        if config.web_browser:
            flags += ["--browser", config.web_browser]
        self._cli.run("open", *flags)

    def teardown(self, config: Config) -> None:
        """Close the session AND reap its whole browser process tree. Must never raise
        (the cli run path swallows teardown errors, but we are defensive here too)."""
        try:
            if self._atexit_hook is not None:
                import atexit
                atexit.unregister(self._atexit_hook)
                self._atexit_hook = None
            cli = self._cli
            if cli is None:
                # teardown without a prior setup (e.g. setup failed early): still make
                # a best-effort close of any session left over by name.
                cli = PlaywrightCli(config.web_session, config.web_cli_timeout)
            cli.close()
        except Exception:
            pass
        finally:
            self._cli = None
