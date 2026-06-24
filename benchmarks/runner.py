"""The benchmark runner.

Runs the VibeHarness agent over the :mod:`benchmarks.tasks` ladder under one or
more tool-call codecs and prints a per-codec scorecard plus a cross-codec
comparison table.

For each (codec, task) cell the runner:
  1. makes a fresh temporary working directory and ``chdir``s into it (the agent's
     FileSystem resolves relative paths against the process cwd, exactly like the
     CLI does),
  2. runs the task's ``setup`` to seed any pre-existing files,
  3. builds a ToolRegistry (fs toolset + validate), the codec, the system prompt
     and a :class:`~vibeharness.agent.RalphAgent`,
  4. runs the task to completion (bounded by ``config.max_steps``),
  5. calls the task's deterministic ``check`` and records pass/fail, turns used and
     wall-clock time,
  6. restores the original cwd and cleans up the temp dir.

CI-SAFE BY DESIGN. The real model is reached only through two injectable factories:

  - ``client_factory(config) -> LLMClient``    (default: a real ``OllamaClient``)
  - ``validator_factory(client) -> Validator`` (default: a real ``LLMValidator``)

Passing scripted fakes (see ``tests/test_benchmark.py``) lets the entire harness be
driven end-to-end with no Ollama server and no network. The CLI uses the real
defaults; the tests inject fakes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Optional

from vibeharness.agent import RalphAgent
from vibeharness.codec import UnknownCodec, get_codec
from vibeharness.codec import available_codecs as _discover_codecs
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem, FileSystemError
from vibeharness.llm import LLMClient, OllamaClient
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.toolset import default_catalog
from vibeharness.validation import LLMValidator, Validator

from .tasks import Task, get_tasks

# Factory seams. These default to the real, live implementations; tests override
# them with scripted fakes so nothing here ever needs a model or network in CI.
ClientFactory = Callable[[Config], LLMClient]
ValidatorFactory = Callable[[LLMClient], Validator]


def _default_client_factory(config: Config) -> LLMClient:
    return OllamaClient(config)


def _default_validator_factory(client: LLMClient) -> Validator:
    return LLMValidator(client)


# --------------------------------------------------------------------------- #
# Codec discovery.
# --------------------------------------------------------------------------- #
def available_codecs() -> list[str]:
    """Names of every codec the harness exposes, with the baseline ``json`` first
    when present.

    Discovery is delegated to :func:`vibeharness.codec.available_codecs` (the single,
    frozen-safe canonical implementation); this wrapper only applies the benchmark's
    "surface json first" ordering on top of that sorted result."""
    names = _discover_codecs()
    if "json" in names:  # surface the baseline first
        names = ["json"] + [n for n in names if n != "json"]
    return names


# --------------------------------------------------------------------------- #
# Result records.
# --------------------------------------------------------------------------- #
@dataclass
class TaskResult:
    """The outcome of one (codec, task) cell."""
    task_id: str
    task_number: int
    passed: bool
    detail: str
    turns: int
    seconds: float
    finished: bool          # did the agent itself call validate->pass?
    error: Optional[str] = None  # set if the run raised (e.g. model unavailable)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "task_number": self.task_number,
            "passed": self.passed, "detail": self.detail, "turns": self.turns,
            "seconds": round(self.seconds, 3), "finished": self.finished,
            "error": self.error,
        }


@dataclass
class CodecScorecard:
    """Aggregate results for one codec across the task ladder."""
    codec: str
    results: list[TaskResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_turns(self) -> int:
        return sum(r.turns for r in self.results)

    @property
    def total_seconds(self) -> float:
        return sum(r.seconds for r in self.results)

    def to_dict(self) -> dict:
        return {
            "codec": self.codec,
            "passed": self.passed,
            "total": self.total,
            "total_turns": self.total_turns,
            "total_seconds": round(self.total_seconds, 3),
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------- #
# The runner.
# --------------------------------------------------------------------------- #
@contextmanager
def _in_temp_workdir() -> Iterator[Path]:
    """A fresh temp directory that is also the process cwd for its lifetime, then
    restored and removed. The agent's FileSystem resolves relative paths against
    the process cwd, so chdir-ing here scopes every task run to its own sandbox."""
    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="vh_bench_")
    try:
        os.chdir(tmp)
        yield Path(tmp)
    finally:
        os.chdir(prev)
        # Best-effort cleanup; never let teardown fail a benchmark.
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


class BenchmarkRunner:
    def __init__(self, config: Config | None = None,
                 client_factory: ClientFactory = _default_client_factory,
                 validator_factory: ValidatorFactory = _default_validator_factory,
                 verbose: bool = True,
                 transcript_dir: Path | str | None = None):
        self._cfg = config or Config()
        self._client_factory = client_factory
        self._validator_factory = validator_factory
        self._verbose = verbose
        # When set, the full per-run transcript (reasoning, raw actions, every
        # observation) is saved per (codec, task) for later analysis. Resolved to an
        # ABSOLUTE path now because each task runs inside a chdir'd temp sandbox.
        self._transcript_dir = Path(transcript_dir).resolve() if transcript_dir else None

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg, flush=True)

    def _save_transcript(self, codec_name: str, task: Task, result) -> None:
        """Write the run's full transcript + structured dump under
        ``<transcript_dir>/<codec>/<NN_taskid>.{txt,json}``. Best-effort: a save
        failure must never fail the benchmark."""
        if self._transcript_dir is None:
            return
        try:
            out = self._transcript_dir / codec_name
            out.mkdir(parents=True, exist_ok=True)
            stem = f"{task.number:02d}_{task.id}"
            (out / f"{stem}.txt").write_text(result.transcript(), encoding="utf-8")
            (out / f"{stem}.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"  (warning: could not save transcript for {task.id}: {e})")

    def run_task(self, codec_name: str, task: Task) -> TaskResult:
        """Run a single task under a single codec in its own temp sandbox."""
        # Resolve the codec up front so an unknown codec surfaces clearly.
        codec = get_codec(codec_name)
        cfg = replace(self._cfg, codec=codec_name)

        with _in_temp_workdir() as workdir:
            task.run_setup(workdir)

            registry = default_catalog().build_registry(
                default_catalog().select(["fs"]), cfg)
            system_prompt = SystemPromptBuilder(
                registry, cfg.max_actions_per_turn, codec).build(task.prompt)

            # Refresh the workspace tree into the system prompt each turn, exactly
            # like the CLI, so newly created files become visible to the agent.
            fs = FileSystem()

            def render_workspace() -> str:
                cwd = Path.cwd()
                try:
                    tree = fs.tree(str(cwd))
                except FileSystemError as e:
                    tree = f"(could not list: {e})"
                return f"Working directory: {cwd}\n{tree}"

            builder = SystemPromptBuilder(registry, cfg.max_actions_per_turn, codec)
            provider = lambda: builder.build(task.prompt, workspace=render_workspace())

            client = self._client_factory(cfg)
            validator = self._validator_factory(client)
            agent = RalphAgent(client, registry, system_prompt, cfg, validator,
                               system_prompt_provider=provider, codec=codec)

            start = time.perf_counter()
            error: Optional[str] = None
            finished = False
            turns = 0
            try:
                result = agent.run(task.prompt)
                turns = len(result.turns)
                finished = result.finished
            except Exception as e:  # e.g. OllamaUnavailable — record, don't crash
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - start

            if error is not None:
                return TaskResult(task.id, task.number, False,
                                  f"run error: {error}", turns, elapsed, finished, error)

            self._save_transcript(codec_name, task, result)
            passed, detail = task.run_check(workdir)
            return TaskResult(task.id, task.number, passed, detail,
                              turns, elapsed, finished)

    def run_codec(self, codec_name: str, tasks: list[Task]) -> CodecScorecard:
        card = CodecScorecard(codec=codec_name)
        self._log(f"\n=== codec: {codec_name} ===")
        for task in tasks:
            res = self.run_task(codec_name, task)
            mark = "PASS" if res.passed else "FAIL"
            self._log(f"  [{mark}] {res.task_number:>2}. {res.task_id:<16} "
                      f"turns={res.turns:<3} {res.seconds:6.2f}s  {res.detail}")
            card.results.append(res)
        self._log(f"  -> {card.passed}/{card.total} passed, "
                  f"{card.total_turns} turns, {card.total_seconds:.2f}s")
        return card

    def run(self, codec_names: list[str], tasks: list[Task]) -> list[CodecScorecard]:
        return [self.run_codec(name, tasks) for name in codec_names]


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def comparison_table(cards: list[CodecScorecard]) -> str:
    """A plain-text cross-codec comparison table for stdout."""
    if not cards:
        return "(no results)"
    rows = [("codec", "passed", "turns", "time(s)")]
    for c in cards:
        rows.append((c.codec, f"{c.passed}/{c.total}",
                     str(c.total_turns), f"{c.total_seconds:.2f}"))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    out = []
    for i, r in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(r))
        out.append(line)
        if i == 0:
            out.append("  ".join("-" * widths[j] for j in range(4)))
    return "\n".join(out)


def markdown_table(cards: list[CodecScorecard]) -> str:
    lines = ["| codec | passed | turns | time(s) |", "| --- | --- | --- | --- |"]
    for c in cards:
        lines.append(f"| {c.codec} | {c.passed}/{c.total} | "
                     f"{c.total_turns} | {c.total_seconds:.2f} |")
    return "\n".join(lines)


def results_to_dict(cards: list[CodecScorecard]) -> dict:
    return {"codecs": [c.to_dict() for c in cards]}


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _parse_task_numbers(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    nums: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            nums.append(int(part))
    return nums or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.runner",
        description="Benchmark VibeHarness tool-call codecs on file-operation tasks.",
    )
    p.add_argument("--codec", default="json",
                   help="codec to benchmark, or 'all' for every available codec "
                        "(default: json)")
    p.add_argument("--tasks", default=None, metavar="N,N,...",
                   help="subset of 1-based task numbers to run (default: all 10)")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model name (default: Config().model)")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="max turns per task (default: Config().max_steps)")
    p.add_argument("--json-out", default=None, metavar="PATH",
                   help="also write the full results as JSON to this path")
    p.add_argument("--md-out", default=None, metavar="PATH",
                   help="also write the comparison table as markdown to this path")
    p.add_argument("--transcript-dir", default=None, metavar="DIR",
                   help="save each run's full transcript (+JSON dump) under "
                        "DIR/<codec>/<task>.{txt,json} for later analysis")
    p.add_argument("--list-codecs", action="store_true",
                   help="list available codecs and exit")
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    overrides: dict[str, object] = {}
    if args.model is not None:
        overrides["model"] = args.model
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    return replace(cfg, **overrides) if overrides else cfg


def resolve_codecs(spec: str) -> list[str]:
    if spec == "all":
        return available_codecs()
    # Validate eagerly so a typo fails fast with a clear message.
    get_codec(spec)
    return [spec]


def main(argv: list[str] | None = None,
         client_factory: ClientFactory = _default_client_factory,
         validator_factory: ValidatorFactory = _default_validator_factory) -> int:
    """CLI entry point. ``client_factory``/``validator_factory`` keep this callable
    from tests with scripted fakes, while the console invocation uses live defaults."""
    args = build_parser().parse_args(argv)

    if args.list_codecs:
        print("available codecs:", ", ".join(available_codecs()))
        return 0

    try:
        codec_names = resolve_codecs(args.codec)
    except UnknownCodec as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        tasks = get_tasks(_parse_task_numbers(args.tasks))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    config = resolve_config(args)
    runner = BenchmarkRunner(config, client_factory, validator_factory,
                             transcript_dir=args.transcript_dir)
    cards = runner.run(codec_names, tasks)

    print("\n=== comparison ===")
    print(comparison_table(cards))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results_to_dict(cards), indent=2), encoding="utf-8")
        print(f"\nwrote JSON results to {args.json_out}")
    if args.md_out:
        Path(args.md_out).write_text(markdown_table(cards), encoding="utf-8")
        print(f"wrote markdown table to {args.md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
