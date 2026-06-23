"""CLI entrypoint for the vibeharness Ralph-loop agent.

`vibe` runs this. The workspace is the current terminal directory unless
--workdir is given. Each turn streams live to the console.

Examples:
  vibe "Create notes.txt containing 'hello hello hello', then read it back."
  vibe --temp 0.3 --max-steps 12 "List this folder and summarize what's here."
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import OllamaClient
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.reporting import ConsoleReporter

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vibe", description="vibe — a tiny local coding agent")
    p.add_argument("task", nargs="+", help="the task for the agent (quote it)")
    p.add_argument("--model", default=Config.model)
    p.add_argument("--temp", type=float, default=Config.temperature)
    p.add_argument("--max-steps", type=int, default=Config.max_steps)
    p.add_argument("--workdir", default=None,
                   help="working directory (default: current terminal directory)")
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    p.add_argument("--print-system", action="store_true", help="print the system prompt and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    task = " ".join(args.task)
    config = Config(model=args.model, temperature=args.temp, max_steps=args.max_steps)

    fs = FileSystem()
    registry = ToolRegistry(build_default_tools(fs, config.observation_char_limit))
    system_prompt = SystemPromptBuilder(registry).build()

    if args.print_system:
        print(system_prompt)
        return 0

    # Workspace = current terminal directory unless overridden.
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)
    workdir = Path.cwd()

    reporter = ConsoleReporter(color=not args.no_color)
    reporter.run_start(task, str(workdir), config)

    agent = RalphAgent(OllamaClient(config), registry, system_prompt, config, reporter=reporter)
    result = agent.run(task)
    reporter.run_end(result)

    # Save transcript (timestamped; never overwrites).
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = runs / f"run_t{config.temperature}_{stamp}.txt"
    out.write_text(result.transcript(), encoding="utf-8")
    print(f" transcript: {out}")
    return 0 if result.finished else 2


if __name__ == "__main__":
    raise SystemExit(main())
