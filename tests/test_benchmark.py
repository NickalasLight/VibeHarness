"""End-to-end tests for the file-operation benchmark harness — WITHOUT a model.

The benchmark runner reaches a live model only through two injectable factories
(``client_factory`` and ``validator_factory``). Here we inject a scripted fake LLM
client (modelled on ``tests/test_agent.py``'s ``FakeLLMClient``) and a stub
validator that always passes, so the whole harness — fresh temp workdir, task
setup, agent loop, and the deterministic ``check`` — is exercised with no Ollama
server and no network.
"""
import unittest

from vibeharness.config import Config

from benchmarks.tasks import TASKS, get_tasks
from benchmarks.runner import BenchmarkRunner, available_codecs, main

# Shared scripted fakes (no model / no network) — one canonical copy in
# tests/_fakes.py. FakeLLMClient streams nothing by default and repeats its last
# scripted action once exhausted; FakeValidator(passed=True) always approves.
from tests._fakes import FakeLLMClient as ScriptedClient
from tests._fakes import FakeValidator as PassValidator


VALIDATE = {"tool": "validate", "args": {}}


def client_factory_for(actions):
    """Build a ``client_factory`` that ignores config and returns a fresh scripted
    client each call (so a multi-codec run gets an independent script per cell)."""
    return lambda config: ScriptedClient(list(actions))


def pass_validator_factory(client):
    return PassValidator()


# --------------------------------------------------------------------------- #
# Task ladder invariants.
# --------------------------------------------------------------------------- #
class TaskLadderTest(unittest.TestCase):
    def test_exactly_ten_tasks(self):
        self.assertEqual(len(TASKS), 10)

    def test_ids_unique(self):
        ids = [t.id for t in TASKS]
        self.assertEqual(len(ids), len(set(ids)), "task ids must be unique")

    def test_numbers_are_one_through_ten_in_order(self):
        self.assertEqual([t.number for t in TASKS], list(range(1, 11)))

    def test_every_task_has_prompt_and_check(self):
        for t in TASKS:
            self.assertTrue(t.prompt.strip(), f"{t.id} has an empty prompt")
            self.assertTrue(callable(t.check), f"{t.id} check is not callable")

    def test_get_tasks_subset(self):
        subset = get_tasks([1, 3])
        self.assertEqual([t.number for t in subset], [1, 3])

    def test_get_tasks_rejects_unknown(self):
        with self.assertRaises(ValueError):
            get_tasks([99])


# --------------------------------------------------------------------------- #
# Runner end-to-end with the scripted fake.
# --------------------------------------------------------------------------- #
class RunnerTest(unittest.TestCase):
    def _runner(self, actions, config=None):
        return BenchmarkRunner(
            config or Config(max_steps=5),
            client_factory=client_factory_for(actions),
            validator_factory=pass_validator_factory,
            verbose=False,
        )

    def test_task_one_passes_with_correct_script(self):
        task = get_tasks([1])[0]            # create greeting.txt = "Hello, world!"
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "Hello, world!"}},
            VALIDATE,
        ]
        result = self._runner(actions).run_task("json", task)
        self.assertTrue(result.passed, result.detail)
        self.assertTrue(result.finished)         # the stub validator approved
        self.assertGreaterEqual(result.turns, 1)
        self.assertIsNone(result.error)

    def test_wrong_content_is_scored_failed(self):
        task = get_tasks([1])[0]
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "WRONG CONTENT"}},
            VALIDATE,
        ]
        result = self._runner(actions).run_task("json", task)
        # The (stub) validator still approves, but the deterministic check fails:
        # the harness scores correctness from check(), not from the agent's claim.
        self.assertFalse(result.passed)
        self.assertIn("WRONG CONTENT", result.detail)

    def test_missing_file_is_scored_failed(self):
        task = get_tasks([1])[0]
        # Agent does nothing useful, just validates.
        result = self._runner([VALIDATE]).run_task("json", task)
        self.assertFalse(result.passed)
        self.assertIn("not created", result.detail)

    def test_setup_seeds_preexisting_file(self):
        # Task 2 reads source.txt (seeded by setup) and writes line_count.txt.
        task = get_tasks([2])[0]
        actions = [
            {"tool": "read_file", "args": {"path": "source.txt"}},
            {"tool": "create_file", "args": {"path": "line_count.txt", "content": "3"}},
            VALIDATE,
        ]
        result = self._runner(actions).run_task("json", task)
        self.assertTrue(result.passed, result.detail)

    def test_scorecard_aggregates(self):
        task1 = get_tasks([1])[0]
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "Hello, world!"}},
            VALIDATE,
        ]
        card = self._runner(actions).run_codec("json", [task1])
        self.assertEqual(card.codec, "json")
        self.assertEqual(card.total, 1)
        self.assertEqual(card.passed, 1)
        self.assertGreaterEqual(card.total_turns, 1)
        self.assertGreaterEqual(card.total_seconds, 0.0)
        d = card.to_dict()
        self.assertEqual(d["passed"], 1)
        self.assertEqual(len(d["results"]), 1)

    def test_transcript_dir_captures_run_artifacts(self):
        import json
        import tempfile
        from pathlib import Path
        task = get_tasks([1])[0]
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "Hello, world!"}},
            VALIDATE,
        ]
        with tempfile.TemporaryDirectory() as td:
            runner = BenchmarkRunner(
                Config(max_steps=5),
                client_factory=client_factory_for(actions),
                validator_factory=pass_validator_factory,
                verbose=False, transcript_dir=td)
            runner.run_task("json", task)
            txt = Path(td) / "json" / f"{task.number:02d}_{task.id}.txt"
            js = Path(td) / "json" / f"{task.number:02d}_{task.id}.json"
            self.assertTrue(txt.exists(), "transcript .txt was not written")
            self.assertTrue(js.exists(), "transcript .json was not written")
            self.assertIn("TASK:", txt.read_text(encoding="utf-8"))
            self.assertIn("greeting.txt", txt.read_text(encoding="utf-8"))
            json.loads(js.read_text(encoding="utf-8"))  # valid JSON dump

    def test_each_task_runs_in_isolated_workdir(self):
        # Two tasks in one codec run must not see each other's files: task 1 creates
        # greeting.txt; task 3 (dir tree) should pass independently in its own sandbox.
        t1, t3 = get_tasks([1])[0], get_tasks([3])[0]
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "Hello, world!"}},
            {"tool": "manage_path",
             "args": {"action": "make_directory", "path": "project/src"}},
            {"tool": "create_file", "args": {"path": "project/README.txt", "content": "x"}},
            {"tool": "create_file", "args": {"path": "project/src/main.txt", "content": "y"}},
            VALIDATE,
        ]
        card = self._runner(actions, Config(max_steps=8)).run_codec("json", [t1, t3])
        # Task 1 passes (greeting created); task 3 passes (tree created); the create
        # of greeting.txt in task 3's script is harmless in its own fresh sandbox.
        self.assertEqual(card.passed, 2, [r.detail for r in card.results])


class CodecDiscoveryTest(unittest.TestCase):
    def test_json_codec_is_available(self):
        codecs = available_codecs()
        self.assertIn("json", codecs)
        self.assertEqual(codecs[0], "json")  # baseline surfaced first


class CliTest(unittest.TestCase):
    def test_main_runs_task_one_with_injected_fakes(self):
        actions = [
            {"tool": "create_file",
             "args": {"path": "greeting.txt", "content": "Hello, world!"}},
            VALIDATE,
        ]
        rc = main(["--codec", "json", "--tasks", "1", "--max-steps", "4"],
                  client_factory=client_factory_for(actions),
                  validator_factory=pass_validator_factory)
        self.assertEqual(rc, 0)

    def test_main_rejects_unknown_codec(self):
        rc = main(["--codec", "no_such_codec", "--tasks", "1"],
                  client_factory=client_factory_for([VALIDATE]),
                  validator_factory=pass_validator_factory)
        self.assertEqual(rc, 2)

    def test_list_codecs(self):
        rc = main(["--list-codecs"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
