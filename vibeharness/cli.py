"""Command-line interface for vibeharness.

Exposed as the ``vibe`` console script (see pyproject.toml) and runnable via
``python -m vibeharness`` or ``python run.py``. The workspace is the current
terminal directory unless ``--workdir`` is given; each turn streams live.

Run ``vibe --help`` to see every command and parameter.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .agent import RalphAgent, RunResult
from .codec import UnknownCodec, available_codecs, get_codec
from .config import Config
from .filesystem import FileSystem, FileSystemError
from .llm import OllamaClient, OllamaUnavailable
from .lock import SingleInstanceLock, VibeAlreadyRunning
from .prompt import SystemPromptBuilder
from .reporting import ConsoleReporter
from .runlog import RunLogger
from .settings import Settings, settable_keys
from .snapshot_budget import compute_snapshot_budget, render_budgeted_snapshot
from .toolset import ToolsetCatalog, agent_default_toolsets, default_catalog
from .validation import LLMValidator
from .web import make_raw_snapshot_provider


def build_parser() -> argparse.ArgumentParser:
    saved_temp = Settings.apply(Config()).temperature
    epilog = f"""\
examples:
  vibe "create a README and fill in a project overview"
  vibe --temp 1.0 "draft notes.txt"        run once at a different temperature
  vibe --max-steps 30 "refactor this dir"  allow more steps for a big task
  vibe "task" --codec tagged_json          use a different tool-call wire format

manage persistent defaults (saved to ~/.vibeharness/settings.json):
  vibe --set temp 0.5                      change the default temperature
  vibe --set max-steps 25                  change the default step budget
  vibe --show-config                       show current settings
  vibe --reset-config                      restore built-in defaults

settable keys: {', '.join(settable_keys())}
current default temperature: {saved_temp}
"""
    p = argparse.ArgumentParser(
        prog="vibe",
        description="vibe - a tiny local coding agent (VibeThinker via Ollama)",
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="*", help="the task for the agent to perform (quote it)")
    p.add_argument("--task-file", default=None, metavar="PATH",
                   help="read the task text from a file instead of the command line")
    p.add_argument("--temp", type=float, default=None, metavar="T",
                   help="sampling temperature for this run only")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model name for this run only")
    p.add_argument("--codec", default=None, metavar="CODEC",
                   help="tool-call wire format for this run only; one of: "
                        f"{', '.join(available_codecs())} (default: {Config.codec})")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="max turns for this run only (0 = unlimited, until finish)")
    p.add_argument("--max-actions-per-turn", type=int, default=None, metavar="N",
                   help="max tool calls the model may emit per turn for this run only")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="working directory (default: current terminal directory)")
    p.add_argument("--agent", default=None, metavar="TYPE",
                   help="agent type for this run; selects a default toolset of the same name "
                        f"({', '.join(agent_default_toolsets())}). e.g. --agent web. "
                        "--toolset overrides/augments which toolset(s) are active.")
    p.add_argument("--toolset", action="append", metavar="NAME",
                   help="toolset(s) to load; repeatable or comma-separated. Overrides/augments "
                        "--agent's default. (default: fs). e.g. --toolset web,fs")
    p.add_argument("--headless", action="store_true",
                   help="run the web browser headless (default: headed so you can watch)")
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    p.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                   help="persist a default, e.g. --set temp 0.5")
    p.add_argument("--show-config", action="store_true", help="print current settings and exit")
    p.add_argument("--reset-config", action="store_true", help="clear saved settings and exit")
    p.add_argument("--list-toolsets", action="store_true", help="list available toolsets and exit")
    p.add_argument("--list-agents", action="store_true", help="list available agent types and exit")
    p.add_argument("--print-system", action="store_true", help="print the system prompt and exit")
    return p


def selected_toolset_names(args: argparse.Namespace) -> list[str]:
    """Resolve the active toolset names: --toolset wins, else --agent's default,
    else today's default (fs).

    This is the precedence rule from issue #22: an --agent type is a *named default
    toolset selection*, and --toolset overrides/augments it. So
    ``--agent web --toolset web,fs`` → [web, fs]; ``--agent web`` → [web]; neither
    given → [fs].
    """
    names: list[str] = []
    for item in (args.toolset or []):
        names += [n.strip() for n in item.split(",") if n.strip()]
    if names:
        return names
    agent = getattr(args, "agent", None)
    if agent:
        return agent_default_toolsets().get(agent, [agent])
    return ["fs"]


def agent_error(name: str) -> str | None:
    """Return a user-facing error if ``name`` is not a known agent type, else None.

    Centralised (mirrors :func:`codec_error`) so the run path and tests share one rule.
    """
    agents = agent_default_toolsets()
    if name in agents:
        return None
    return f"error: unknown agent '{name}'. Available: {', '.join(agents)}"


def codec_error(name: str) -> str | None:
    """Return a user-facing error if ``name`` is not an installed codec, else None.

    Centralised so the per-run flag and tests share one validation rule.
    """
    if name in available_codecs():
        return None
    return f"error: unknown codec '{name}'. Available: {', '.join(available_codecs())}"


def resolve_config(args: argparse.Namespace) -> Config:
    """Config defaults < saved settings < CLI flags (only those provided)."""
    cfg = Settings.apply(Config())
    overrides: dict[str, object] = {}
    if args.temp is not None:
        overrides["temperature"] = args.temp
    if args.model is not None:
        overrides["model"] = args.model
    if getattr(args, "codec", None) is not None:
        overrides["codec"] = args.codec
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if getattr(args, "max_actions_per_turn", None) is not None:
        overrides["max_actions_per_turn"] = args.max_actions_per_turn
    if getattr(args, "headless", False):
        overrides["web_headless"] = True
    return replace(cfg, **overrides) if overrides else cfg


def cmd_list_toolsets() -> int:
    print("available toolsets:")
    for name, description in default_catalog().describe():
        print(f"  {name:<6} {description}")
    print("\nselect with --toolset (repeatable or comma-separated), default: fs")
    return 0


def cmd_list_agents() -> int:
    catalog = default_catalog()
    descriptions = dict(catalog.describe())
    print("available agents:")
    for name, toolsets in agent_default_toolsets(catalog).items():
        desc = descriptions.get(name, "")
        print(f"  {name:<6} toolset(s): {', '.join(toolsets):<10} {desc}")
    print("\nselect with --agent; --toolset overrides/augments the default toolset(s).")
    return 0


def cmd_show_config() -> int:
    saved = Settings.load()
    effective = Settings.apply(Config())
    print(f"settings file: {Settings.path()}")
    print(f"saved overrides: {saved or '(none)'}")
    print("effective defaults:")
    print(f"  model       = {effective.model}")
    print(f"  temperature = {effective.temperature}")
    print(f"  max_steps   = {effective.max_steps}")
    print(f"  max_actions_per_turn = {effective.max_actions_per_turn}")
    print(f"  top_p       = {effective.top_p}")
    print(f"  top_k       = {effective.top_k}")
    return 0


def cmd_set(key: str, value: str) -> int:
    try:
        field, parsed = Settings.set(key, value)
    except KeyError:
        print(f"error: '{key}' is not settable. Settable keys: {', '.join(settable_keys())}")
        return 2
    except ValueError:
        print(f"error: '{value}' is not a valid value for '{key}'.")
        return 2
    print(f"saved: {field} = {parsed}")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    else:
        task = " ".join(args.task)
    config = resolve_config(args)
    # Validate the agent type (if given) before any model work — mirrors --codec.
    if getattr(args, "agent", None) is not None:
        err = agent_error(args.agent)
        if err is not None:
            print(err)
            return 2
    # Validate the (possibly overridden) codec before any model work.
    err = codec_error(config.codec)
    if err is not None:
        print(err)
        return 2
    catalog = default_catalog()
    names = selected_toolset_names(args)

    try:
        toolsets = catalog.select(names)
    except KeyError as e:
        print(f"error: unknown toolset {e}. Available: {', '.join(catalog.names())}")
        return 2
    problems = [p for ts in toolsets for p in ts.check_prerequisites()]
    if problems:
        print("error: missing prerequisites for the selected toolset(s):")
        for p in problems:
            print(f"  - {p}")
        return 2

    registry = catalog.build_registry(toolsets, config)
    try:
        codec = get_codec(config.codec)
    except UnknownCodec as e:
        print(f"error: {e}")
        return 2
    # Vary the system prompt by the SELECTED toolset(s): each advertises its own short
    # guidance, assembled into one "# Working with your tools" section.
    guidance = SystemPromptBuilder.assemble_guidance(toolsets)
    system_prompt = SystemPromptBuilder(
        registry, config.max_actions_per_turn, codec,
        guidance=guidance).build(task)   # task anchored at the front

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)
    workdir = Path.cwd()

    # Single-instance lock: only one model stream is supported at a time.
    # Acquire now that workdir/log_path are known but before any model work;
    # a crashed prior run leaves a stale lock that we auto-reclaim.
    started = datetime.now()
    logger = RunLogger(workdir, started)
    lock = SingleInstanceLock()
    try:
        lock.acquire(str(workdir), str(logger.json_path))
    except VibeAlreadyRunning as e:
        print("error: another vibe run is already active "
              "(only one stream is supported at a time).", file=sys.stderr)
        print(f"  workdir: {e.workdir}", file=sys.stderr)
        print(f"  log:     {e.log_path}", file=sys.stderr)
        print(f"  started: {e.started}  (pid {e.pid})", file=sys.stderr)
        print("Wait for it to finish, then try again.", file=sys.stderr)
        return 3
    try:
        return _run_locked(args, task, config, registry, codec, names, workdir,
                           logger, system_prompt, toolsets)
    finally:
        lock.release()


def make_system_prompt_provider(builder, config, task, render_workspace,
                                raw_snapshot_provider, logger=None):
    """Build the per-turn system-prompt provider with the DYNAMIC snapshot budget (#43).

    Returns a callable taking the per-turn ``user`` message (RalphAgent passes it so
    the snapshot can be sized against the FULL model message). Each turn:

      1. Render the workspace and capture the UNTRUNCATED live page snapshot.
      2. Build ``rest`` = the system prompt WITHOUT the page section + the user
         message. This is everything that will share the context window besides the
         snapshot.
      3. Size the snapshot to the remaining token budget and truncate it to fit.
      4. Rebuild the system prompt with the budgeted snapshot injected.

    When no snapshot provider is wired (web inactive), this degrades to the original
    workspace-only refresh — no page section, no budgeting work.

    Per-turn diagnostic logging (issue #37): when a ``logger`` is given, each turn we
    dump the COMPLETE raw snapshot (ground-truth size, BEFORE budgeting) and the EXACT
    system prompt injected into the model into the run's .vibe diagnostics folder. This
    is the single per-turn seam, so it captures exactly what the model saw. All logging
    is guarded so a dump failure can never break the turn.
    """
    turn_counter = itertools.count(1)

    def _dump(prompt: str, raw: str | None) -> None:
        if logger is None:
            return
        try:
            turn = next(turn_counter)
            logger.dump_turn_diagnostics(turn, snapshot=raw, system_prompt=prompt)
        except Exception:
            pass   # diagnostics must never abort the turn

    def provider(user: str = "") -> str:
        workspace = render_workspace()
        raw = raw_snapshot_provider() if raw_snapshot_provider is not None else None
        if not raw:
            # No live page this turn: render no page section (issue #24 behaviour).
            prompt = builder.build(task, workspace=workspace, page="")
            _dump(prompt, raw or None)
            return prompt
        # The "rest" of the message: system prompt with NO page section + the user
        # turn message. We measure this to learn how much room the snapshot has.
        rest_system = builder.build(task, workspace=workspace, page="")
        rest_text = rest_system + user
        budget = compute_snapshot_budget(config, rest_text)
        if budget.overflow:
            # The rest of the message already fills the window: inject NO snapshot and
            # warn, so we never overflow num_ctx (issue #43).
            print(
                f"\nwarning: the message without the page snapshot "
                f"(~{budget.rest_tokens} tokens) already meets the input budget of "
                f"~{budget.input_budget_tokens} tokens; injecting no page snapshot this "
                f"turn to stay within num_ctx ({config.num_ctx}).",
                file=sys.stderr,
            )
            prompt = builder.build(task, workspace=workspace, page="")
            _dump(prompt, raw)
            return prompt
        page = render_budgeted_snapshot(raw, budget.budget_chars)
        prompt = builder.build(task, workspace=workspace, page=page)
        _dump(prompt, raw)
        return prompt
    return provider


def _run_locked(args, task, config, registry, codec, names, workdir, logger,
                system_prompt, toolsets) -> int:
    reporter = ConsoleReporter(color=not args.no_color)
    reporter.run_start(task, str(workdir), config)
    print(f" toolsets: {', '.join(names)} (+ validate)")

    client = OllamaClient(config)
    validator = LLMValidator(client, logger=logger, config=config)

    # Refresh the system prompt every turn so its "# Workspace" section reflects
    # files the agent creates as it goes. Scanning Path.cwd() each call (rather
    # than a cached tree) is what makes newly written files appear next turn.
    builder = SystemPromptBuilder(
        registry, config.max_actions_per_turn, codec,
        guidance=SystemPromptBuilder.assemble_guidance(toolsets))
    fs = FileSystem()

    def render_workspace() -> str:
        cwd = Path.cwd()
        try:
            tree = fs.tree(str(cwd))
        except FileSystemError as e:
            tree = f"(could not list: {e})"
        return f"Working directory: {cwd}\n{tree}"

    # When the web toolset is active, auto-inject a FRESH page snapshot into the
    # per-turn system prompt (issue #24). The provider captures from the run's
    # existing Playwright session each turn; because the whole prompt is rebuilt per
    # turn, only the latest snapshot is ever present (stale-dropping by regeneration —
    # never written to narrative memory). When web is not active, no page section.
    #
    # Issue #43: the snapshot is sized DYNAMICALLY. We capture it UNTRUNCATED, then
    # truncate it only as much as the context window requires once the rest of the
    # message (system prompt minus the page section + the per-turn user message) is
    # known. So the provider takes the per-turn ``user`` message (see RalphAgent).
    raw_snapshot_provider = (make_raw_snapshot_provider(config) if "web" in names else None)

    # The per-turn provider applies #43's dynamic snapshot budget AND, given the
    # logger, performs #37's per-turn diagnostic dump (raw snapshot + injected prompt).
    system_prompt_provider = make_system_prompt_provider(
        builder, config, task, render_workspace, raw_snapshot_provider, logger)
    agent = RalphAgent(client, registry, system_prompt, config, validator,
                       reporter=reporter, system_prompt_provider=system_prompt_provider,
                       codec=codec)

    checkpoint = lambda res: _safe_log(logger, task, config, res)   # stream log each turn
    # Write an initial log at run START (before turn 1). on_turn fires only AFTER a
    # turn completes, so without this a turn that hangs or raises mid-way would leave
    # NO log at all. Creating the .vibe/ folder + a seed log up front guarantees a
    # run always leaves a trace, however early it dies.
    _safe_log(logger, task, config, RunResult(task=task))
    print(f" log: {logger.json_path}")

    # Setup is INSIDE the try so its teardown ALWAYS runs in the finally — even if a
    # later toolset's setup raises after the web browser is already open, or the run
    # body dies with an uncaught exception or Ctrl-C. Otherwise a crash between an
    # opened browser and the agent run would leak the whole chrome/node tree (#15).
    try:
        for ts in toolsets:
            ts.setup(config)
        result = agent.run(task, on_turn=checkpoint)
    except OllamaUnavailable as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1   # the start/streamed log on disk already holds the last known state
    finally:
        # Reap every toolset, swallowing per-toolset errors, so one failing teardown
        # never prevents the others (e.g. the web browser) from being torn down.
        for ts in reversed(toolsets):
            try:
                ts.teardown(config)
            except Exception:
                pass

    reporter.run_end(result)
    _safe_log(logger, task, config, result)   # final write
    return 0 if result.finished else 2


def _safe_log(logger: RunLogger, task: str, config: Config, result) -> None:
    """Write the run log without ever aborting the run — but never *silently*: a
    write failure is surfaced as a warning so a missing .vibe log is diagnosable
    instead of vanishing (the old bare ``except: pass`` hid real UnicodeEncodeErrors)."""
    try:
        logger.write(task, config, result)
    except Exception as e:
        print(f"\nwarning: could not write run log to {logger.json_path} "
              f"({type(e).__name__}: {e})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # robust unicode on Windows consoles
    except AttributeError:
        pass

    parser = build_parser()
    # Friendly help: bare `vibe`, `vibe help`, or `vibe -help` all show help.
    if not argv or argv[0] in ("help", "-help"):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.show_config:
        return cmd_show_config()
    if args.reset_config:
        print("settings cleared." if Settings.reset() else "no saved settings to clear.")
        return 0
    if args.set:
        return cmd_set(args.set[0], args.set[1])
    if args.list_toolsets:
        return cmd_list_toolsets()
    if args.list_agents:
        return cmd_list_agents()
    if args.print_system:
        catalog = default_catalog()
        try:
            toolsets = catalog.select(selected_toolset_names(args))
        except KeyError as e:
            print(f"error: unknown toolset {e}. Available: {', '.join(catalog.names())}")
            return 2
        print(SystemPromptBuilder(
            catalog.build_registry(toolsets, Config()),
            guidance=SystemPromptBuilder.assemble_guidance(toolsets)).build())
        return 0
    if not args.task and not args.task_file:
        print("error: no task given.\n")
        parser.print_help()
        return 2

    return run_agent(args)
