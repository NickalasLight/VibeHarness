# BUG: benchmark runs leave no `.vibe/` chat logs (reasoning/actions lost by default)

- **Branch / work item:** `fix/vibe-log-missing` (worktree `vh-vibe-log-bug`)
- **Severity:** medium (no data loss to user files, but **run evidence is silently lost** —
  blocks analysis/debugging of agent behaviour, which is the whole point of this project)
- **Status:** confirmed, root-caused. Fix not yet implemented.
- **Reported from:** a live run pointed at `C:\git\vibetest_latest1` that finished with an
  **empty** workspace and **no `.vibe/` logs anywhere**.

---

## Summary

The user expects every run's chat log (per-turn reasoning trace + actions + observations) to
land in a `.vibe/` folder in the workspace, the way the `vibe` CLI does it. A run executed via
the **benchmark runner** (`python -m benchmarks.runner`) produced **no `.vibe/` logs and no
workspace files at all**. This is not a crash — it is a behavioural gap: the benchmark path
**never writes `.vibe/`**, runs in a throwaway temp sandbox that is deleted, and persists
transcripts **only if `--transcript-dir` is explicitly passed**. With no flag, the run's entire
trace exists only in memory and is discarded after the scorecard prints to stdout.

**Desired outcome (per request): chat logs should ALSO be saved in the `.vibe/` folder for
benchmark runs**, consistent with the CLI, so runs can be spot-checked after the fact.

---

## Repro

```
python -m benchmarks.runner --codec json --tasks 1
# (no --transcript-dir)
```
**Observed:** scorecard prints to stdout; afterwards there is no `.vibe/` folder, no transcript,
and no workspace files anywhere. The target workspace (`vibetest_latest1`) stays empty.
**Expected:** a `.vibe/<stamp>.json` (+`.md`) per run, including the reasoning trace, in a
persistent location.

---

## Root cause (evidence, `benchmarks/runner.py` on `beta` @ `472ad9d`)

1. **Runs happen in a throwaway temp sandbox that is deleted.**
   `_in_temp_workdir()` does `tempfile.mkdtemp(prefix="vh_bench_")`, `chdir`s in, then
   `shutil.rmtree`s it on exit (`runner.py:146-163`). So the agent's files are created under
   `…\AppData\Local\Temp\vh_bench_*` — **not** in the workspace the user named — and are removed
   when the task ends. (Verified: no `vh_bench_*` dirs survive; `vibetest_latest1` is empty.)

2. **The benchmark path never writes `.vibe/`.**
   `run_task` drives `RalphAgent` directly (`runner.py:233-241`) and **never instantiates
   `RunLogger`**. `RunLogger` (which writes `<workspace>/.vibe/<stamp>.{json,md}`, see
   `vibeharness/runlog.py:35-63`) is only wired into the **`vibe` CLI**. The benchmark path has
   no equivalent.

3. **Transcripts persist only with `--transcript-dir`.**
   `_save_transcript` returns immediately when `self._transcript_dir is None`
   (`runner.py:189-190`); it is only populated from the `--transcript-dir` flag
   (`runner.py:339`). Without the flag, nothing is written — the `result` (which *does* hold the
   full per-turn reasoning + actions; see `RunResult.transcript()` / `to_dict()`) is printed as a
   scorecard and dropped.

**Net:** three independent reasons the run left no trace, none of which is a logging *crash*.

---

## Why this matters (and the deeper smell)

This is a **silent-loss** failure: the only signal that capture was skipped is the *absence* of
files. A run can complete "successfully," print a green scorecard, and leave zero evidence behind
— exactly the anti-pattern where missing output reads as "nothing happened" when in fact a full
run occurred. For a project whose purpose is *studying where a small model goes wrong from its
traces*, losing traces by default is a primary-function defect, not a cosmetic one.

---

## Key design constraint for the fixer (read before implementing)

**You cannot simply write `.vibe/` inside the benchmark's working directory** — that directory is
the temp sandbox and is `rmtree`d on exit (`runner.py:160-161`). A `.vibe/` written there is
deleted with it. **The log must be written to a persistent location outside the sandbox.** Note
`_save_transcript` already resolves `--transcript-dir` to an **absolute** path up front *because
of exactly this* (`runner.py:178-179`) — follow the same pattern for `.vibe/` output.

---

## Proposed fix (options; implementer's choice)

Goal: after any benchmark run, a `.vibe/`-style chat log (per-turn reasoning + actions +
observations, same schema as the CLI) exists in a **persistent** path.

- **Option A — reuse `RunLogger`, persistent path (recommended).** In `run_task`, after
  `agent.run(...)`, write the run via `RunLogger` into a persistent, per-cell directory, e.g.
  `<out>/<codec>/<NN_taskid>/.vibe/<stamp>.{json,md}` where `<out>` defaults to the invocation
  cwd (or a new `--vibe-dir`/reuse `--transcript-dir`). Reusing `RunLogger` keeps the **schema
  identical to CLI logs**, so the same analysis tooling reads both. (To capture *per-turn* live
  logs rather than only the final state, pass an `on_turn` checkpoint to `agent.run(...)` exactly
  as the CLI does — `agent.run(task, on_turn=...)`.)
- **Option B — make capture on by default.** Default `transcript_dir` to a sensible persistent
  location (e.g. `./benchmark_runs/<timestamp>/`) instead of `None`, so a bare
  `python -m benchmarks.runner` always persists something. Keep `--transcript-dir`/a `--no-save`
  to override.
- **Option C — both:** default-on persistence (B) **and** emit `RunLogger`-format `.vibe/` logs
  (A) so benchmark output is drop-in compatible with CLI `.vibe/` logs.

Whichever is chosen, also **emit a one-line warning when a run's trace is NOT being persisted**
(e.g. "transcripts disabled; this run will leave no `.vibe/` log") so silent loss can never
recur.

---

## Acceptance criteria

- [ ] A bare `python -m benchmarks.runner --codec json --tasks 1` (no extra flags) leaves a
      persistent `.vibe/` chat log containing the **per-turn reasoning trace, actions, and
      observations**, in the same JSON schema `RunLogger` produces for the CLI.
- [ ] The log survives the temp-sandbox cleanup (written to an absolute/persistent path, not
      inside `vh_bench_*`).
- [ ] If trace persistence is ever disabled, the runner prints a visible warning (no silent loss).
- [ ] CLI `.vibe/` logging is unchanged; benchmark and CLI logs are schema-compatible.
- [ ] A test asserts a `.vibe/` log is written for a benchmark run (the runner is already
      CI-driveable via injected fake `client_factory`/`validator_factory` — see
      `runner.py:19-26` / `tests/test_benchmark.py`).

---

## Affected files

- `benchmarks/runner.py` — `_in_temp_workdir` (`:146-163`), `run_task` (`:202-255`),
  `_save_transcript` (`:185-200`), `--transcript-dir` (`:339`), `BenchmarkRunner.__init__`
  (`:166-179`).
- `vibeharness/runlog.py` — `RunLogger` (`:35-63`) — the writer to reuse.
- (reference, do not change) `vibeharness/cli.py` — how the CLI wires `RunLogger` + the per-turn
  `on_turn` checkpoint; mirror this in the runner.

## Notes

- Not a regression in `RunLogger` itself — that class works; the benchmark path simply never calls
  it. Investigation confirmed: no `.vibe/` written anywhere in the run window, no surviving temp
  sandboxes, no `--json-out`/`--md-out`/`--transcript-dir` output — i.e. the run was genuinely
  capture-less, consistent with this root cause.
- Workaround until fixed: pass `--transcript-dir <abs path>` to the benchmark runner, **or** use
  the `vibe` CLI directly (which already writes `.vibe/<stamp>.{json,md}`).
