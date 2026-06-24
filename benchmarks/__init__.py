"""File-operation benchmark harness for VibeHarness tool-call codecs.

This package benchmarks the different :class:`~vibeharness.codec.ToolCallCodec`
wire formats (json, tagged_json, xml, codeact, gbnf, …) against one another on a
graded ladder of file-operation tasks. It is deliberately self-contained and
imports only from the public ``vibeharness`` surface — it adds no behaviour to,
and edits none of, the production modules.

Two pieces:
  - :mod:`benchmarks.tasks`   — ten increasing-difficulty file-op tasks, each with
                                a deterministic ``check(workdir)`` (and optional
                                ``setup(workdir)``).
  - :mod:`benchmarks.runner`  — runs the agent (real Ollama by default, or any
                                injected client factory) per task per codec and
                                prints a comparison scorecard.

The runner accepts a pluggable ``client_factory`` and ``validator_factory`` so the
whole harness can be exercised in CI with a scripted fake LLM — no live model or
network required (see ``tests/test_benchmark.py``).
"""
from __future__ import annotations

from .tasks import TASKS, Task, get_tasks

__all__ = ["TASKS", "Task", "get_tasks"]
