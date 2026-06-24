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
        """Run one CLI command in this session. Returns (ok, combined_output)."""
        if not self._binary:
            return False, f"{BINARY} is not installed"
        cmd = [self._binary, f"-s={self._session}", *args]
        try:
            # Force UTF-8 decoding: page snapshots contain emoji/unicode that the
            # default Windows codec (cp1252) cannot decode, which would otherwise
            # crash the reader thread and return empty output.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self._timeout)
        except subprocess.TimeoutExpired:
            return False, f"command timed out after {self._timeout}s"
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return proc.returncode == 0, out


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


class WebToolset(Toolset):
    name = "web"
    description = ("Browse the web with a stateful browser: navigate, read page content and "
                   "links via snapshot, click, fill forms, upload files, and screenshot.")

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
