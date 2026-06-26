"""Command-line interface for vibeharness.

Exposed as the ``vibe`` console script (see pyproject.toml) and runnable via
``python -m vibeharness`` or ``python run.py``. The workspace is the current
terminal directory unless ``--workdir`` is given; each turn streams live.

Run ``vibe --help`` to see every command and parameter.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .agent import RalphAgent, RunResult
from .codec import UnknownCodec, available_codecs, get_codec
from .config import Config
from .filesystem import FileSystem, FileSystemError
from .llm import OllamaClient, OllamaUnavailable, ensure_single_runner_env
from .lock import SingleInstanceLock, VibeAlreadyRunning
from .prompt import SystemPromptBuilder, render_page_section
from .reporting import ConsoleReporter
from .runlog import RunLogger
from .settings import Settings, settable_keys
from .snapshot_budget import compute_snapshot_budget, render_budgeted_snapshot
from .toolset import (
    ToolsetCatalog,
    agent_default_max_actions,
    agent_default_toolsets,
    default_catalog,
)
from .validation import LLMValidator
from .web import annotate_filled_snapshot, make_raw_snapshot_provider, resolve_web_session
from .snapshot_prose import aria_yaml_to_prose
from .advisor import VibeThinkerAdvisor


def build_parser() -> argparse.ArgumentParser:
    effective = Settings.apply(Config())
    saved_temp = effective.temperature
    epilog = f"""\
examples:
  vibe "create a README and fill in a project overview"
  vibe --temp 1.0 "draft notes.txt"          run once at a different temperature
  vibe --max-steps 30 "refactor this dir"    allow more steps for a big task
  vibe --agent web "find the top 5 HN posts" drive a real browser (web toolset)
  vibe --toolset web,fs "scrape & save"      compose toolsets for one run
  vibe --task-file task.txt                  read a long task from a file
  vibe "task" --codec tagged_json            use a different tool-call wire format
  vibe --model qwen3:4b "..."                pick a different Ollama model
  vibe --num-ctx 16384 --top-p 0.9 "..."     tune the context window / sampling

inspect without running a task:
  vibe --version                             package version + the source path it runs from
  vibe --list-agents                         agent types and their default toolset(s)
  vibe --list-toolsets                       toolsets and the tools in each
  vibe --print-system                        the exact system prompt the model receives
  vibe --show-config                         effective defaults + saved overrides

per-role / API endpoints (local <-> hosted; mix and match):
  # run the base agent locally but VALIDATE with a hosted GLM model (needs the API key):
  set ZHIPUAI_API_KEY=...                     (PowerShell: $env:ZHIPUAI_API_KEY="...")
  vibe --set codec hermes "fill the signup form"
  # (per-role --base-provider/--base-model/--validator-provider/--validator-model flags
  #  land with #163/PR #168; until then the API validator is configured in config.py:
  #  validation_provider=zhipuai, validation_model=glm-5.2.)

manage persistent defaults (saved to {Settings.path()}):
  vibe --set temp 0.5                        change the default temperature
  vibe --set max-steps 25                    change the default step budget
  vibe --set codec hermes                    change the default tool-call format
  vibe --set num-ctx 16384                   change the default context window
  vibe --show-config                         show current settings
  vibe --reset-config                        restore built-in defaults

settable keys (use with --set KEY VALUE): {', '.join(settable_keys())}
resolution order: built-in defaults < saved settings < per-run flags
current effective defaults: model={effective.model}, codec={effective.codec}, """ \
        f"""temp={saved_temp}, top_p={effective.top_p}, top_k={effective.top_k}, """ \
        f"""num_ctx={effective.num_ctx}, max_steps={effective.max_steps}
"""
    p = argparse.ArgumentParser(
        prog="vibe",
        description="vibe - a tiny local coding/web agent: a small model (Ollama) works a "
                    "task one step at a time (read/write/search files or drive a real "
                    "browser), streaming its reasoning and actions, gated by a validator.",
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- task input ---
    p.add_argument("task", nargs="*",
                   help="the task for the agent to perform; quote it as one argument")
    p.add_argument("--task-file", default=None, metavar="PATH",
                   help="read the task text from a file instead of the command line "
                        "(useful for long or multi-line tasks)")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="directory the agent operates in / treats as its workspace "
                        "(default: the current terminal directory); created if missing")

    # --- model / sampling (per-run overrides; persist with --set) ---
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model tag to run this task with "
                        f"(default: {Config.model}); overrides the saved/default model")
    p.add_argument("--temp", type=float, default=None, metavar="T",
                   help=f"sampling temperature for this run (default: {Config.temperature}); "
                        "higher = more diverse/creative, lower = more deterministic")
    p.add_argument("--top-p", dest="top_p", type=float, default=None, metavar="P",
                   help=f"nucleus sampling top-p for this run (default: {Config.top_p})")
    p.add_argument("--top_k", type=int, default=None, metavar="K",
                   help=f"top-k sampling cutoff for this run (default: {Config.top_k}; "
                        "0 disables the top-k filter)")

    # --- tool-call format / agent / toolsets ---
    p.add_argument("--codec", default=None, metavar="CODEC",
                   help="tool-call wire format for this run; one of: "
                        f"{', '.join(available_codecs())} (default: {Config.codec}). "
                        "Controls how calls are described, decode-constrained, and parsed.")
    p.add_argument("--agent", default=None, metavar="TYPE",
                   help="agent type for this run; selects a default toolset of the same name "
                        f"({', '.join(agent_default_toolsets())}), e.g. --agent web. "
                        "--toolset overrides/augments which toolset(s) are active.")
    p.add_argument("--toolset", action="append", metavar="NAME",
                   help="toolset(s) to load; repeatable or comma-separated, overriding/"
                        "augmenting --agent's default (default: fs). e.g. --toolset web,fs")

    # --- loop / turn budgets ---
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help=f"max turns for this run (default: {Config.max_steps}; "
                        "0 = unlimited, run until the validator passes)")
    p.add_argument("--max-actions-per-turn", type=int, default=None, metavar="N",
                   help="max tool calls the model may emit per turn for this run "
                        "(default: the selected agent type's cap, else "
                        f"{Config.max_actions_per_turn}; 0 = unlimited)")

    # --- context window / token budgets ---
    p.add_argument("--num-ctx", dest="num_ctx", type=int, default=None, metavar="N",
                   help=f"Ollama context window in tokens for this run (default: "
                        f"{Config.num_ctx}). The whole window is shared by the system "
                        "prompt, chat history, the live page snapshot, and generation. "
                        "Output is reserved up front (reason-tokens + action-tokens, "
                        "settable via --set); the live snapshot is then sized to fit the "
                        "remaining input budget.")

    # --- web agent ---
    p.add_argument("--headless", action="store_true",
                   help="run the web browser headless (default: headed so you can watch it)")
    p.add_argument("--web-snapshot-prose", action="store_true",
                   help="render the live page snapshot as pruned, ref-keyed WebArena-style "
                        "prose instead of raw ARIA-YAML (issue #64; A/B seam)")

    # --- advisor ---
    p.add_argument("--advisor", action="store_true",
                   help="enable the advisor: after every N accumulated tool calls (counted "
                        "across turns, injected at end-of-turn) an advisor model emits "
                        "free-text advice injected into the next agent turn (see config "
                        "advisor_model / advisor_interval)")

    # --- output ---
    p.add_argument("--no-color", action="store_true",
                   help="disable ANSI colored terminal output")

    # --- settings management (each exits immediately) ---
    p.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                   help="persist a default setting and exit, e.g. --set temp 0.5 "
                        f"(settable keys: {', '.join(settable_keys())})")
    p.add_argument("--show-config", action="store_true",
                   help="print effective defaults + saved overrides, then exit")
    p.add_argument("--reset-config", action="store_true",
                   help="clear all saved settings (restore built-in defaults) and exit")

    # --- introspection (each exits immediately, no model required) ---
    p.add_argument("--list-toolsets", action="store_true",
                   help="list available toolsets and the tools in each, then exit")
    p.add_argument("--list-agents", action="store_true",
                   help="list available agent types and their default toolset(s), then exit")
    p.add_argument("--print-system", action="store_true",
                   help="print the generated system prompt (honours --agent/--toolset/"
                        "--codec) and exit")
    p.add_argument("--version", action="store_true",
                   help="print the package version, git build identity, and the absolute "
                        "source path the running vibeharness is loaded from, then exit")
    return p


def build_identity() -> str:
    """A one-line build identity: package version + git short-sha of the source tree.

    The sha is read at RUNTIME from the git checkout this package lives in, so an
    EDITABLE install (``pip install -e .``) reports the exact commit you are running
    — the whole point of issue #51's "am I on a stale build?" check. Falls back to
    'unknown' when git or the repo is unavailable (e.g. a wheel install), and never
    raises.
    """
    from . import __version__

    sha = "unknown"
    try:
        import subprocess
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            sha = out.stdout.strip()
    except Exception:
        pass
    return f"vibe {__version__} (build {sha})"


def build_source_report() -> str:
    """A multi-line report: build identity + the ABSOLUTE source path the running
    ``vibeharness`` package is imported from, plus a warning when that path is not a
    git checkout of this repo.

    The whole point (issue #175): a ``vibe`` installed editable from a STALE, unrelated
    clone runs that clone's code while ``import vibeharness`` from the cwd may resolve
    elsewhere — so the only reliable way to know what you are actually running is to
    print the on-disk location of the loaded package. Show it, and flag it loudly if it
    is not a git working tree (i.e. a wheel install or a detached copy that cannot report
    a commit).
    """
    import vibeharness as _pkg

    pkg_dir = Path(_pkg.__file__).resolve().parent       # .../vibeharness
    repo_dir = pkg_dir.parent                            # repo root (editable checkout)
    lines = [
        build_identity(),
        f"source:  {pkg_dir}",
        f"repo:    {repo_dir}",
    ]
    if not (repo_dir / ".git").exists():
        lines.append(
            "warning: this source is NOT a git checkout (wheel install or detached copy) "
            "— you cannot confirm which commit you are running. For development, install "
            "editable from the repo root: `pip install -e .`"
        )
    return "\n".join(lines)


def cmd_version() -> int:
    print(build_source_report())
    return 0


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


def resolve_max_actions(args: argparse.Namespace) -> int:
    """Resolve the per-turn tool-call cap for the selected agent (issue #52).

    Precedence (high to low):
      1. an EXPLICIT ``--max-actions-per-turn`` flag, or a saved setting for it,
      2. the selected ``--agent`` type's default (``agent_default_max_actions``),
      3. the global ``Config.max_actions_per_turn`` default.

    The resolved value is the SINGLE source of truth: it is written into the run's
    Config so BOTH the prompt builder ("you may emit up to N actions") and the agent
    loop (which executes at most N) read the same number and cannot drift.
    """
    # An explicit per-run flag wins outright.
    if getattr(args, "max_actions_per_turn", None) is not None:
        return args.max_actions_per_turn
    # A saved setting is an explicit user choice too — honour it over the agent default.
    saved = Settings.load()
    if "max_actions_per_turn" in saved:
        return int(saved["max_actions_per_turn"])
    # Otherwise fall back to the selected agent's default, then the global default.
    global_default = Config.max_actions_per_turn
    agent = getattr(args, "agent", None)
    if agent:
        return agent_default_max_actions(global_default).get(agent, global_default)
    return global_default


def resolve_config(args: argparse.Namespace) -> Config:
    """Config defaults < saved settings < CLI flags (only those provided).

    ``max_actions_per_turn`` additionally folds in the per-agent-type default (#52):
    see :func:`resolve_max_actions` for the full precedence.
    """
    cfg = Settings.apply(Config())
    overrides: dict[str, object] = {}
    if args.temp is not None:
        overrides["temperature"] = args.temp
    if getattr(args, "top_p", None) is not None:
        overrides["top_p"] = args.top_p
    if getattr(args, "top_k", None) is not None:
        overrides["top_k"] = args.top_k
    if getattr(args, "num_ctx", None) is not None:
        overrides["num_ctx"] = args.num_ctx
    if args.model is not None:
        overrides["model"] = args.model
    if getattr(args, "codec", None) is not None:
        overrides["codec"] = args.codec
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    # Resolve the per-turn cap once for the selected agent (explicit flag / saved
    # setting > agent-type default > global default) and thread it through Config so
    # the prompt builder and the agent loop share one source of truth.
    overrides["max_actions_per_turn"] = resolve_max_actions(args)
    if getattr(args, "headless", False):
        overrides["web_headless"] = True
    if getattr(args, "web_snapshot_prose", False):
        overrides["web_snapshot_prose"] = True
    if getattr(args, "advisor", False):
        overrides["advisor_enabled"] = True
    cfg = replace(cfg, **overrides) if overrides else cfg
    # Mint a UNIQUE per-run Playwright session name (issues #111/#112) so concurrent
    # runs never share — and tear down — one another's browser daemon. An explicitly
    # set name (settings) is honoured as an override; only the default is replaced.
    # Resolved ONCE here so every web tool and both snapshot providers (all read
    # ``config.web_session``) share the exact same name for this run.
    return replace(cfg, web_session=resolve_web_session(cfg))


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
    print(f"  num_ctx     = {effective.num_ctx}")
    print(f"  codec       = {effective.codec}")
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
    _native = bool(
        getattr(config, "native_tools", False)
        and not config.two_phase
        and codec.tools(registry) is not None
    )
    system_prompt = SystemPromptBuilder(
        registry, config.max_actions_per_turn, codec,
        guidance=guidance).build(task, native_tools=_native)   # task anchored at the front

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


def make_render_workspace(fs, names):
    """Build the per-turn ``# Workspace`` tree renderer, gated on the fs toolset.

    The workspace directory tree is only meaningful to an agent that can touch the
    filesystem. A web-only worker (no ``fs`` tools) gets an EMPTY string, so the
    builder omits the ``# Workspace`` section entirely (mirroring the page/guidance
    empty-section behaviour) — the file tree is irrelevant noise competing with the
    worker's live page snapshot + tool guidance. When ``fs`` IS active (``fs``, or
    ``fs``+``web``, …) it renders the cwd tree as before (now minus hidden folders).

    Scanning ``Path.cwd()`` each call (rather than a cached tree) is what makes newly
    written files appear next turn.
    """
    fs_active = "fs" in names

    def render_workspace() -> str:
        if not fs_active:
            return ""
        cwd = Path.cwd()
        try:
            tree = fs.tree(str(cwd))
        except FileSystemError as e:
            tree = f"(could not list: {e})"
        return f"Working directory: {cwd}\n{tree}"

    return render_workspace


def _budget_snapshot_for_turn(builder, config, task, render_workspace,
                              raw_snapshot_provider, user, include_tool_guidance,
                              native_tools, cache=None):
    """Shared per-turn budgeting core (#43). Returns ``(rest_system, page, raw)``.

    ``rest_system`` is the system prompt WITHOUT the page section (issue #146: the page
    no longer lives in the system prompt at all). ``page`` is the live snapshot trimmed
    to the dynamic budget — sized so ``rest_system + user + page`` fits the input window
    — or "" when there is no live page this turn or the rest already fills the window.
    ``raw`` is the untruncated capture (for #37 diagnostics) or None.

    The MAIN agent calls this twice per turn with the SAME ``user`` — once for the system
    prompt, once for the user-turn snapshot (#146). A live browser capture is comparatively
    expensive and could drift between two reads, so an optional one-entry ``cache`` dict
    (keyed by ``user`` + the two flags) memoises the result, guaranteeing ONE capture per
    (turn, user) shared by both providers. The validator path uses its own cache so its
    later, independent capture is fresh."""
    if cache is not None:
        key = (user, include_tool_guidance, native_tools)
        if cache.get("key") == key:
            return cache["value"]
    workspace = render_workspace()
    rest_system = builder.build(task, workspace=workspace, page="",
                                include_tool_guidance=include_tool_guidance,
                                native_tools=native_tools)
    raw = raw_snapshot_provider() if raw_snapshot_provider is not None else None

    def _result(rest, page, raw_capture):
        out = (rest, page, raw_capture)
        if cache is not None:
            cache["key"] = (user, include_tool_guidance, native_tools)
            cache["value"] = out
        return out

    if not raw:
        return _result(rest_system, "", raw or None)
    # The "rest" of the message: system prompt with NO page section + the user turn
    # (which itself does not yet carry the snapshot). We measure this to learn how much
    # room the snapshot has, exactly as before — only the snapshot's final RESTING PLACE
    # changed (user turn, not system prompt), not the total it is budgeted against.
    rest_text = rest_system + user
    budget = compute_snapshot_budget(config, rest_text)
    if budget.overflow:
        print(
            f"\nwarning: the message without the page snapshot "
            f"(~{budget.rest_tokens} tokens) already meets the input budget of "
            f"~{budget.input_budget_tokens} tokens; injecting no page snapshot this "
            f"turn to stay within num_ctx ({config.num_ctx}).",
            file=sys.stderr,
        )
        return _result(rest_system, "", raw)
    page = render_budgeted_snapshot(raw, budget.budget_chars)
    return _result(rest_system, page, raw)


def make_system_prompt_provider(builder, config, task, render_workspace,
                                raw_snapshot_provider, logger=None,
                                include_tool_guidance=True, native_tools=False,
                                cache=None):
    """Build the per-turn system-prompt provider with the DYNAMIC snapshot budget (#43).

    Returns a callable taking the per-turn ``user`` message (RalphAgent passes it so
    the snapshot can be sized against the FULL model message). Each turn it renders the
    workspace, captures + budgets the snapshot (see ``_budget_snapshot_for_turn``), and
    returns the system prompt.

    Issue #146: the budgeted snapshot is NO LONGER injected into the system prompt for
    the MAIN agent — it is appended to the END of the user turn instead (see
    ``RalphAgent.run``). To preserve the validator
    path (issue #57), the TOOL-LESS variant (``include_tool_guidance=False``) STILL
    renders the page as a `# Current page` section appended to the returned context, so
    the validator sees the same budgeted snapshot the agent saw. The full-tool variant
    (the agent's system prompt) renders NO page section.

    Issue #57: ``include_tool_guidance`` is threaded through to every ``builder.build``
    call so the SAME function can produce the TOOL-LESS variant fed to the validator.

    Per-turn diagnostic logging (issue #37): when a ``logger`` is given, each turn we
    dump the COMPLETE raw snapshot (ground-truth size, BEFORE budgeting) and the EXACT
    system prompt injected into the model into the run's .vibe diagnostics folder. The
    tool-less validator variant does NOT dump (logger left None) so it never
    double-counts a turn.
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
        rest_system, page, raw = _budget_snapshot_for_turn(
            builder, config, task, render_workspace, raw_snapshot_provider, user,
            include_tool_guidance, native_tools, cache=cache)
        # The MAIN agent's system prompt (full tools) carries NO page section — the
        # snapshot rides the user turn (#146). The validator's TOOL-LESS context keeps a
        # `# Current page` section appended so it still sees the budgeted snapshot (#57).
        prompt = rest_system
        if not include_tool_guidance:
            prompt = rest_system + render_page_section(page)
        _dump(prompt, raw)
        return prompt
    return provider




def _run_locked(args, task, config, registry, codec, names, workdir, logger,
                system_prompt, toolsets) -> int:
    reporter = ConsoleReporter(color=not args.no_color)
    reporter.run_start(task, str(workdir), config)
    print(f" toolsets: {', '.join(names)} (+ validate)")

    # Advisor setup.
    # When advisor_model is "" (default), Qwen self-advises — same model, one VRAM slot.
    # When advisor_model names a different model (e.g. "vibethinker:latest"), use model-swap
    # mode (OLLAMA_MAX_LOADED_MODELS=1): only one model in VRAM at a time; swapping costs
    # ~15-30s per advisor call but avoids the 9.4 GB OOM on the 8 GB card.
    if config.advisor_enabled:
        resolved_advisor = config.advisor_model or config.model
        if resolved_advisor != config.model:
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
        advisor = VibeThinkerAdvisor(config)
    else:
        advisor = None

    client = OllamaClient(config)

    # Validator uses the API model by default (stronger, independent verdict).
    # Falls back to the local Ollama client when the provider/key is unavailable.
    _validator_client = client
    if config.validation_provider:
        try:
            from .providers import api_key_present, get_provider, make_api_client
            _vp = get_provider(config.validation_provider)
            if api_key_present(_vp):
                _validator_client = make_api_client(
                    config.validation_provider, config.validation_model or None)
        except Exception:
            pass  # provider unknown or openai missing — use local client silently
    validator = LLMValidator(_validator_client, logger=logger, config=config)

    # Refresh the system prompt every turn so its "# Workspace" section reflects
    # files the agent creates as it goes. Scanning Path.cwd() each call (rather
    # than a cached tree) is what makes newly written files appear next turn.
    builder = SystemPromptBuilder(
        registry, config.max_actions_per_turn, codec,
        guidance=SystemPromptBuilder.assemble_guidance(toolsets))
    fs = FileSystem()

    # Per-turn "# Workspace" renderer, gated on whether the fs toolset is active:
    # a web-only worker gets "" (no section); fs / fs+web render the cwd tree
    # (now minus hidden folders). See make_render_workspace.
    render_workspace = make_render_workspace(fs, names)

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
    # Issue #64: behind a config seam, render the captured ARIA-YAML snapshot into
    # WebArena-style prose BEFORE it enters the dynamic-budget / diagnostics pipeline.
    # This is an A/B toggle, not a removal: with web_snapshot_prose False (default) the
    # raw ARIA snapshot flows through exactly as before. The transform preserves the
    # native [ref=eN] inline so the discrete web subtools keep resolving targets, and
    # falls back to the raw text on any parse surprise (never blanks the page section).
    if raw_snapshot_provider is not None and config.web_snapshot_prose:
        _inner_snapshot_provider = raw_snapshot_provider
        raw_snapshot_provider = lambda: aria_yaml_to_prose(_inner_snapshot_provider())

    # Filled-control annotation: track which refs the agent has successfully filled this
    # run and annotate those lines in the snapshot with "ALREADY FILLED WITH '...'".
    # Updated by the checkpoint after each turn; read by the snapshot provider each turn.
    filled_controls: dict[str, str] = {}
    if raw_snapshot_provider is not None:
        import json as _json
        _inner_for_fill = raw_snapshot_provider
        raw_snapshot_provider = lambda: annotate_filled_snapshot(
            _inner_for_fill(), filled_controls)

    # NATIVE tool calling active? (issue #129/#130/#131) — same predicate the agent uses:
    # opted in, single-phase, and the codec speaks native tools. When active, the live
    # system prompt must OMIT the `# Tools` block + format instructions (Ollama injects
    # them from the model's template via the tools: field).
    native_tools = bool(
        getattr(config, "native_tools", False)
        and not config.two_phase
        and codec.tools(registry) is not None
    )

    # Per-turn snapshot budget cache: shared by the system-prompt provider (diagnostics +
    # workspace) and the validator context provider. The post-turn snapshot (appended as
    # a role:tool observation by the agent after tools execute) is a SEPARATE capture
    # with its own timing, so it does not share this cache.
    _turn_snapshot_cache: dict = {}
    # The per-turn provider rebuilds the system prompt each turn (workspace tree refresh)
    # and runs #37 diagnostics. It does NOT inject the page snapshot into the system
    # prompt — the snapshot now rides the final tool observation instead (see agent.py).
    system_prompt_provider = make_system_prompt_provider(
        builder, config, task, render_workspace, raw_snapshot_provider, logger,
        native_tools=native_tools, cache=_turn_snapshot_cache)
    # Issue #57: a TOOL-LESS twin of the same per-turn prompt to feed the validator —
    # task + workspace + the same #43-budgeted page snapshot (appended as a `# Current
    # page` section), with the tool descriptions / format-instruction block stripped.
    # A SEPARATE cache so its capture happens fresh at validate-time.
    validator_context_provider = make_system_prompt_provider(
        builder, config, task, render_workspace, raw_snapshot_provider,
        logger=None, include_tool_guidance=False, cache={})
    agent = RalphAgent(client, registry, system_prompt, config, validator,
                       reporter=reporter, system_prompt_provider=system_prompt_provider,
                       codec=codec, validator_context_provider=validator_context_provider,
                       raw_snapshot_provider=raw_snapshot_provider,
                       turn_input_logger=logger.dump_turn_input,
                       turn_output_logger=logger.dump_turn_output)

    # Advisor advice buffer: the checkpoint populates it after N accumulated tool calls;
    # advice_provider drains it once at the start of the next turn.
    _advice_buffer: list[str] = []
    _tool_calls_since_advisor = [0]  # mutable int via list so closure can write it

    # cli keeps a live handle on the agent's RunResult (the SAME object the agent mutates
    # and hands to on_turn each turn) so ANY abnormal exit can still persist the real
    # partial state with a clear reason — never a silent exit-0 (#170).
    _live: dict = {"result": RunResult(task=task)}

    def checkpoint(res: RunResult) -> None:
        _live["result"] = res
        # Update filled_controls from the latest turn's successful fill/select actions.
        if res.turns:
            latest = res.turns[-1]
            for a in latest.actions:
                if not a.ok or not a.args:
                    continue
                tgt = a.args.get("target")
                if a.tool in ("fill", "type") and tgt and isinstance(a.args.get("text"), str):
                    filled_controls[tgt] = a.args["text"]
                elif a.tool == "select_option" and tgt and isinstance(a.args.get("value"), str):
                    filled_controls[tgt] = a.args["value"]
                elif a.tool == "check" and tgt:
                    filled_controls[tgt] = "(checked)"
            # Count tool calls this turn (real calls only, not error placeholders).
            turn_tool_calls = sum(1 for a in latest.actions if a.tool is not None)
            _tool_calls_since_advisor[0] += turn_tool_calls
        _safe_log(logger, task, config, res)
        # Trigger advisor when accumulated tool calls reach the threshold (checked at
        # end-of-turn so the base agent is never paused mid-turn for the advisor).
        if (advisor is not None and res.turns
                and _tool_calls_since_advisor[0] >= config.advisor_interval):
            advice = advisor.advise(task, res.turns, reporter=reporter)
            _advice_buffer.clear()
            _advice_buffer.append(advice)
            _tool_calls_since_advisor[0] = 0  # reset counter after advisor call

    def advice_provider(turn_idx: int) -> str | None:
        return _advice_buffer.pop() if _advice_buffer else None

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
        result = agent.run(task, on_turn=checkpoint,
                           advice_provider=advice_provider if advisor else None)
        _live["result"] = result
    except OllamaUnavailable as e:
        result = _live["result"]
        result.stop_reason = result.stop_reason or f"Ollama unavailable: {e}"
        print(f"\nerror: {e}", file=sys.stderr)
        sys.stderr.flush()
        _safe_log(logger, task, config, result)
        _save_run_score(logger, task, config, result)
        return 1   # the start/streamed log on disk already holds the last known state
    except BaseException as e:
        # NO SILENT DEATH (#170): surface ANY abnormal end loudly (full traceback +
        # a one-line reason), persist the agent's partial result with that reason, and
        # exit non-zero. The agent already checkpointed the last completed turn; here we
        # record WHY the run ended so a `.vibe` log never shows a bare, unexplained stop.
        result = _live["result"]
        reason = f"run aborted mid-turn: {type(e).__name__}: {e}"
        result.stop_reason = result.stop_reason or reason
        print(f"\n!!! RUN ABORTED — {reason}", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        _safe_log(logger, task, config, result)
        _save_run_score(logger, task, config, result)
        # Ctrl-C should still surface as an interrupt; ordinary errors exit with a clear
        # non-zero code rather than propagating a bare traceback past the caller.
        if isinstance(e, KeyboardInterrupt):
            raise
        return 3
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
    _save_run_score(logger, task, config, result)  # always write score row
    return 0 if result.finished else 2


def _save_run_score(logger: RunLogger, task: str, config: Config, result) -> None:
    """Compute and append a score entry to .vibe/scores.jsonl after every run."""
    try:
        fields_set: dict = {}
        page_advances = 0
        ok_count = 0
        fail_count = 0
        steps_seen: set = set()
        for t in result.turns:
            for a in t.actions:
                if a.ok:
                    ok_count += 1
                else:
                    fail_count += 1
                obs = a.observation or ""
                for m in re.finditer(r"Step (\d+) of 8", obs):
                    steps_seen.add(int(m.group(1)))
                if not a.ok:
                    continue
                args = a.args or {}
                tgt = args.get("target", "")
                if a.tool in ("fill", "type"):
                    val = args.get("text") or args.get("value") or ""
                    if val:
                        fields_set[(tgt, val)] = val
                elif a.tool == "select_option":
                    val = args.get("value", "")
                    if "selected" in obs and "combobox" in obs and val:
                        fields_set[(tgt, val)] = val
                elif a.tool == "check":
                    fields_set[(tgt, "checked")] = "checked"
                elif a.tool == "click":
                    if any(kw in obs for kw in ("PAGE CHANGED", "Continue", "Next", "Submit", "Review")):
                        page_advances += 1
        max_step = max(steps_seen) if steps_seen else 1
        page_advances = max(page_advances, max_step - 1)
        unique_fields = len(fields_set)
        score = unique_fields + 10 * page_advances
        entry = {
            "stamp": logger.stamp,
            "log": str(logger.json_path),
            "finished": result.finished,
            "turns": len(result.turns),
            "ok_actions": ok_count,
            "fail_actions": fail_count,
            "unique_fields": unique_fields,
            "page_advances": page_advances,
            "max_step": max_step,
            "score": score,
            "model": config.model,
        }
        scores_path = logger.dir / "scores.jsonl"
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\nscore: {unique_fields} fields + {10*page_advances} page-bonus = {score}"
              f"  (step {max_step}/8, {'FINISHED' if result.finished else 'incomplete'})")
    except Exception as e:
        print(f"\nwarning: could not save run score ({type(e).__name__}: {e})", file=sys.stderr)


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
    # Force UTF-8 on both stdout and stderr so box-drawing characters and other
    # non-ASCII output aren't mis-encoded as Cp1252/Latin-1 in log files or
    # pipe-captured streams (e.g. Tee-Object).  reconfigure() is Python 3.7+
    # on TextIOWrapper; the AttributeError guard keeps us safe on exotic runtimes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # ISSUE #77: cap Ollama at one loaded model on startup so a new runner evicts the
    # old one rather than stacking (the OOM/CUDA root cause). setdefault inside, so a
    # user-set value is preserved. Done here too (not only in OllamaClient.__init__)
    # so the var is present before any backend object is constructed.
    ensure_single_runner_env()

    parser = build_parser()
    # Friendly help: bare `vibe`, `vibe help`, or `vibe -help` all show help.
    if not argv or argv[0] in ("help", "-help"):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.version:
        return cmd_version()
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
        # Render with the CONFIGURED codec (issue #123): on this branch the default codec
        # is `hermes`, so --print-system must show the actual <tools>/<tool_call> prompt the
        # model receives, not the json-codec default the SystemPromptBuilder would otherwise
        # fall back to. Honour an explicit --codec override too. (On `beta`, where the
        # default is `json`, this is a no-op.)
        cfg = Config()
        codec_name = args.codec or cfg.codec
        try:
            codec = get_codec(codec_name)
        except UnknownCodec as e:
            print(f"error: {e}")
            return 2
        # Show the prompt the model ACTUALLY receives: when native tools are active
        # (issue #129/#130/#131) Ollama injects the # Tools block + format instructions
        # from the model's template, so the harness prompt omits them — render that.
        _registry = catalog.build_registry(toolsets, cfg)
        _native = bool(
            getattr(cfg, "native_tools", False)
            and not cfg.two_phase
            and codec.tools(_registry) is not None
        )
        print(SystemPromptBuilder(
            _registry, cfg.max_actions_per_turn, codec,
            guidance=SystemPromptBuilder.assemble_guidance(toolsets)).build(
                native_tools=_native))
        return 0
    if not args.task and not args.task_file:
        print("error: no task given.\n")
        parser.print_help()
        return 2

    return run_agent(args)
