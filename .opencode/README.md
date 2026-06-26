# VibeHarness ↔ OpenCode control-plane integration

This `.opencode/` folder exposes the **intact** VibeHarness Python harness to
[OpenCode](https://opencode.ai) as a first-class **toolset + subagent**. OpenCode
never manages the harness's models or touches its agent loop / snapshot system —
it only **orchestrates background runs** and **surfaces their output**.

The harness is invoked exclusively through its existing CLI:

```
python -m vibeharness "<task>" --agent web --model <m> --codec hermes \
       --max-steps N [--headless] --workdir <ws> --set <key> <val> ...
```

## Layout

| Path | What it is |
|---|---|
| `vibeharness.json` | **The one file a human edits** to change default models/codec. |
| `lib/vibe.ts` | Shared logic: config load/save, run-state read/write, runId resolution, CLI-arg assembly, detached spawn, runlog parsing, status computation. |
| `tools/vibe_set_models.ts` | Update `vibeharness.json` defaults for future runs. |
| `tools/vibe_start.ts` | Start a run (`task` REQUIRED); spawns **detached**; returns a small `{runId,status,logPath}`. |
| `tools/vibe_status.ts` | Compact status of a run (latest by default). |
| `tools/vibe_stop.ts` | Kill a run's process tree (Windows `taskkill /F /T /PID`). |
| `tools/vibe_restart.ts` | Re-spawn a run from its saved task (new runId). |
| `tools/vibe_info.ts` | Paths + optional log tail (`which: summary\|runlog\|screenshots\|workspace\|tail`). |
| `plugins/vibe-stream.ts` | Tails each active run's `.vibe/` runlog and streams per-action progress into the live session (+ milestone toasts). |
| `agent/vibe-apply.md` | `mode: subagent` that delegates a web job-application/browse task to the harness via these tools. |
| `vibe-runs/<runId>/` | Per-run state: `meta.json` (pid, task, resolved config, sessionID, status, paths) + `stdout.log` + an isolated `workspace/` (whose `.vibe/` holds the harness runlog). |

## Defaults (`vibeharness.json`)

```json
{
  "baseModel": "qwen3:4b",
  "validatorModel": "glm-5.2",
  "validatorProvider": "zhipuai",
  "escalationModel": "glm-5.2",
  "codec": "hermes",
  "headless": false,
  "maxSteps": 15
}
```

Mapped to CLI flags at spawn time:

| JSON key | CLI |
|---|---|
| `baseModel` | `--model` |
| `codec` | `--codec` |
| `headless` | `--headless` (only when `true`) |
| `maxSteps` | `--max-steps` |
| `validatorProvider` / `validatorModel` | `--set validation_provider` / `--set validation_model` |
| `escalationModel` (+provider) | `--set escalation_model` / `--set escalation_provider` |

> **Validator note.** The GLM/`zhipuai` validator path requires `ZHIPUAI_API_KEY`
> in the environment. If the key is absent the harness silently falls back to the
> local Ollama model for validation. (See the run with `validatorProvider:zhipuai`
> and the local fallback both exercised in the PR's integration test.)

## Output contract

- **Tool return value = a TINY summary** (runId / status / verdict / paths) — never
  the transcript.
- **Live progress for the human** is streamed into the session out-of-band by
  `plugins/vibe-stream.ts` (per-action lines via `client.session.prompt({noReply:true})`
  + milestone `client.tui.showToast`).
- The **full transcript** is reachable only via `vibe_info` (the runlog `.json`/`.md`
  under the run's `workspace/.vibe/`).

## Concurrency & runId

Each run has a `runId`. All tools default to the **latest** run when `runId` is
omitted; pass an explicit `runId` to address an older/concurrent run.

## Usage

From an OpenCode session (or `opencode run`):

```
use vibe_start to run: "Go to https://example.com and report the page heading"
```

or delegate to the subagent: `@vibe-apply apply to <job-url> using my default profile`.

The repo-root `opencode.json` wires a local Ollama provider so the primary
OpenCode agent itself has a model (`ollama/qwen3:4b`); edit it to taste.
