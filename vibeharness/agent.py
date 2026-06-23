"""The Ralph loop.

Each turn: render the task + narrative -> ask the model for one constrained
action -> parse -> execute via the registry -> append a natural-language
observation. Repeat until `finish` or the step budget is exhausted.

The agent depends only on abstractions: an LLMClient, a ToolRegistry, a
NarrativeMemory and a system prompt string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Config
from .llm import LLMClient
from .memory import NarrativeMemory
from .prompt import build_turn_prompt
from .registry import ToolRegistry
from .reporting import NullReporter, Reporter


@dataclass
class Step:
    index: int
    reasoning: str
    action_json: str
    tool: str | None
    args: dict
    observation: str
    ok: bool


@dataclass
class RunResult:
    task: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = False
    final_summary: str = ""

    def transcript(self) -> str:
        out = [f"TASK: {self.task}", ""]
        for s in self.steps:
            out.append(f"--- Step {s.index} ---")
            out.append(f"action: {s.action_json}")
            out.append(f"result: {s.observation}")
            out.append("")
        out.append(f"FINISHED: {self.finished}")
        if self.final_summary:
            out.append(f"SUMMARY: {self.final_summary}")
        return "\n".join(out)


class RalphAgent:
    def __init__(self, client: LLMClient, registry: ToolRegistry,
                 system_prompt: str, config: Config, reporter: Reporter | None = None):
        self._client = client
        self._registry = registry
        self._system = system_prompt
        self._cfg = config
        self._reporter = reporter or NullReporter()

    def run(self, task: str) -> RunResult:
        memory = NarrativeMemory()
        result = RunResult(task=task)
        schema = self._registry.action_schema()

        for i in range(1, self._cfg.max_steps + 1):
            self._reporter.turn_start(i)
            user = build_turn_prompt(task, memory.render())
            decision = self._client.decide(
                self._system, user, schema,
                on_reason=self._reporter.reasoning_token,
                on_action=self._reporter.action_token,
            )
            tool_name, args, parse_error = self._parse(decision.action_json)

            if parse_error:
                observation = f"your last action was invalid and could not be run: {parse_error}."
                step = Step(i, decision.reasoning, decision.action_json, None, {}, observation, False)
                memory.record(observation)
                self._emit(step, result)
                continue

            tool = self._registry.get(tool_name)
            if tool is None:
                observation = f"you tried to use '{tool_name}', which is not a real tool."
                step = Step(i, decision.reasoning, decision.action_json, tool_name, args, observation, False)
                memory.record(observation)
                self._emit(step, result)
                continue

            tool_result = tool.run(args)
            step = Step(i, decision.reasoning, decision.action_json, tool_name, args,
                        tool_result.observation, tool_result.ok)
            memory.record(tool_result.observation)
            self._emit(step, result)

            if tool_result.is_final:
                result.finished = True
                result.final_summary = args.get("summary", "")
                break

        return result

    # ---- helpers ----
    @staticmethod
    def _parse(action_json: str):
        try:
            obj = json.loads(action_json)
        except json.JSONDecodeError as e:
            return None, {}, f"not valid JSON ({e})"
        if not isinstance(obj, dict) or "tool" not in obj:
            return None, {}, "missing 'tool' field"
        args = obj.get("args", {})
        if not isinstance(args, dict):
            return obj.get("tool"), {}, "'args' must be an object"
        return obj.get("tool"), args, None

    def _emit(self, step: Step, result: RunResult) -> None:
        result.steps.append(step)
        self._reporter.step_result(step)
