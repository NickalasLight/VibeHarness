# VibeHarness

[![CI](https://github.com/NickalasLight/VibeHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/NickalasLight/VibeHarness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**A tiny, dependency-free Ralph-loop agent harness for small local models.**

VibeHarness turns a 3B local model ([VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B), served by [Ollama](https://ollama.com)) into a basic command-line agent. You type a task; it works one step at a time — reading, writing, and searching files, or driving a real web browser — and streams its reasoning and actions live to the terminal. A second instance of the model acts as a **validator** that checks the work before the run is allowed to finish.

```powershell
cd C:\my\project
vibe "Create a CHANGELOG.md and seed it with an Unreleased section."
```

```
 vibe (vibethinker, temp 0.3)
 workspace: C:\my\project
 task: Create a CHANGELOG.md and seed it with an Unreleased section.

┌─ step 1 ────────────────────────────────────────
│ thinking: <think>No changelog exists yet, so I'll create one…</think>
│ action: {"tool":"create_file","args":{"path":"CHANGELOG.md","content":"# Changelog\n\n## [Unreleased]\n"}}
└ ✓ you created the file 'CHANGELOG.md' (38 characters).

┌─ step 2 ────────────────────────────────────────
│ action: {"tool":"validate","args":{}}
└ ✓ validator agreed the task is complete.

 done — Created CHANGELOG.md with an Unreleased section.
```

> **Status:** working prototype. VibeThinker-3B is a math/reasoning specialist and is *not* tuned for tool use, so treat this as a research toy for studying small-model agentic behaviour — not a production agent.

> **⚙️ Branch `beta_qwen3coder` (issue #123).** This branch replaces VibeThinker-3B with a
> **~3B Qwen-Coder** model end to end and defaults to the **`hermes`** tool-call codec
> (the native Qwen2.5 `<tool_call>{"name","arguments"}` + bare `<tools>` schema format,
> ground-truthed from the model's chat template — see
> [`QWEN3CODER_ANALYSIS.md`](./QWEN3CODER_ANALYSIS.md)).
>
> **⚠️ 3B-parity caveat — there is no dense ~3B *Qwen3-Coder*.** The Qwen3-Coder line is
> MoE-only (smallest is 30B-A3B = 30B total / 3B *active*, then 80B-A3B "Next", then
> 480B-A35B), which breaks apples-to-apples parity with the 3B-**dense** VibeThinker-3B
> and won't fit an 8 GB GPU. So this branch uses the closest true ~3B dense coder,
> **`qwen2.5-coder:3b-instruct`** (VibeThinker itself derives from Qwen2.5-(Coder-)3B), as
> a documented, flagged substitute — never a silent swap. Pull it with:
> ```bash
> ollama pull qwen2.5-coder:3b-instruct   # ~1.9 GB (Q4_K_M), fits 8 GB GPU
> ```
> Governance + protected divergences: [`QWEN3CODER_DIVERGENCE.md`](./QWEN3CODER_DIVERGENCE.md).
> Sync is one-way `beta → beta_qwen3coder` only (CLAUDE.md §6). Live end-to-end validation
> is tracked in **#125**.

---

## Quickstart

Install everything (requires [Ollama](https://ollama.com/download) and Python ≥ 3.10):

```bash
# 1. Install Ollama from https://ollama.com/download, then pull the model.
#    Branch beta_qwen3coder default (#123): a ~3B Qwen2.5-Coder (used directly from the
#    Ollama library — no `ollama create` needed; config.py points the harness at this tag).
ollama pull qwen2.5-coder:3b-instruct
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness

# 2. Install the harness (creates the `vibe` command)
pip install -e .

# 3. Confirm you're running the source you think you are, then run a task
vibe --version           # version + build sha + the ABSOLUTE source path it loaded from
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
```

No `pip install`? Use `python run.py "<task>"` instead (Windows users can also run `bin\vibe.cmd`). See [Prerequisites](#prerequisites) and [Install](#install) for details and platform notes.

---

## Why it's interesting

- **Constrained actions.** Each turn the model emits a JSON array of one or more tool calls, validated against a JSON schema *at decode time* (Ollama's `format` grammar), so it can't produce malformed actions — even at high temperature, where small models otherwise drift into garbage.
- **Two-phase turns.** The model first reasons freely, then emits the action(s) under the schema constraint. (This is an Ollama adaptation of [noperator's vLLM structural-tag trick](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).) The reasoning is dropped from the running context — but kept on disk (see below).
- **Natural-language memory.** Instead of a JSON/ChatML transcript, the agent's past is a plain-English narrative ("First, you… Then, you…"), which is what a small model follows most reliably.
- **A validator gate, not a self-claim.** The agent never declares itself done. It calls `validate`, and a *separate* validator agent — the same model, given the original task plus the work history — returns a strict pass/fail. On a fail it returns a concrete, ref-specific next step that the working agent reads verbatim. See [Agent types](#agent-types).
- **Swappable tool-call wire format.** A `ToolCallCodec` seam owns how tool calls are described, decode-constrained, and parsed, so you can A/B different formats (`--codec json|tagged_json|xml|codeact|gbnf`) without touching the agent loop. See [Tool-call codecs](#tool-call-codecs).
- **Discrete, self-describing tools.** Each toolset is a handful of small tools; the system prompt and the enforced grammar are both generated from one `ToolRegistry`, so docs and the grammar can never drift apart.
- **Pluggable toolsets and agent types.** Toolsets are swappable and composable (`--toolset web,fs`); an agent type couples a default toolset with a tuned system prompt and a per-turn action cap (`--agent web`).
- **Full reasoning + diagnostic logs.** Every run is written in full — including each turn's reasoning trace, plus optional per-turn diagnostic dumps and every validator invocation — to a hidden `.vibe/` folder in the workspace.
- **Zero runtime dependencies.** Pure Python standard library (the web toolset shells out to the Playwright CLI).

---

## Prerequisites

You need three things: Python, Ollama, and a VibeThinker model registered in Ollama.

### 1. Python ≥ 3.10
Check with `python --version`. (Windows, macOS, and Linux are all supported; the harness is pure Python. Developed and tested on Windows 10.)

### 2. Ollama
Install from [ollama.com/download](https://ollama.com/download) and make sure the server is running:
```bash
ollama --version
ollama serve        # usually already running as a background service / tray app
```

### 3. The model — `qwen2.5-coder:3b-instruct` (branch `beta_qwen3coder`, #123)
This branch swaps VibeThinker-3B for a **~3B Qwen2.5-Coder**, available directly as an
official Ollama library tag (no GGUF registration needed — `config.py` points the harness
straight at the tag):
```bash
ollama pull qwen2.5-coder:3b-instruct      # ~1.9 GB (Q4_K_M), 32K ctx, fits 8 GB GPU
ollama run qwen2.5-coder:3b-instruct "hi"  # quick sanity check
```
**Why not literally "Qwen3-Coder"?** There is **no dense ~3B Qwen3-Coder** — that line is
MoE-only (smallest 30B-A3B; then 80B-A3B "Next"; then 480B-A35B), which breaks
apples-to-apples parity with the 3B-dense VibeThinker-3B and won't fit an 8 GB card.
`qwen2.5-coder:3b-instruct` is the closest true ~3B **dense** coder (VibeThinker itself
derives from Qwen2.5-(Coder-)3B). This is a documented, flagged substitute — see
[`QWEN3CODER_ANALYSIS.md`](./QWEN3CODER_ANALYSIS.md) §A. The optional
[`Modelfile`](./Modelfile) only points at the weights — the harness sets all sampling
parameters per request.

### Hardware
The Q8_0 quant of this 3B model needs **~3.5 GB of VRAM** (or runs on CPU, just slower). Any modern GPU with ≥4 GB, or a CPU with ≥8 GB RAM, is fine. The default context window is large (`num_ctx = 131072`); on an 8 GB card the KV overflow spills to system RAM, so it fills more slowly but does not OOM.

### 4. (Optional) Web toolset
For `--agent web` / `--toolset web` you also need [Node.js](https://nodejs.org) and the Playwright Agent CLI:
```bash
npm install -g @playwright/cli@latest
```
See [The web agent](#the-web-agent) for details.

---

## Install

### Option A — install the package (recommended; cross-platform `vibe` command)
```bash
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
pip install -e .          # EDITABLE install: creates the `vibe` console command
vibe --version           # version + build sha + the source path it loaded from
vibe "list this folder and tell me what's here"
```

> **Run the source you think you're running.** Always install **editable** (`pip install -e .`)
> from the checkout you are editing — then the `vibe` command imports *this* working tree, so
> source changes take effect immediately with no reinstall. Confirm it any time with:
> ```bash
> vibe --version            # prints version, git short-sha, AND the absolute source path
> pip show vibeharness      # "Editable project location" must point at your checkout
> ```
> `vibe --version` prints, for example:
> ```
> vibe 0.1.0 (build 8b8cc48)
> source:  C:\path\to\VibeHarness\vibeharness
> repo:    C:\path\to\VibeHarness
> ```
> The **source path** is the on-disk location of the package actually loaded — the reliable
> tell when a `vibe` was installed editable from a *different* clone (the exact issue #175
> guards against). If that path is not the checkout you are editing, or `--version` warns the
> source is not a git checkout, re-run `pip install -e .` from the right directory. On Windows,
> if the console script can't be written to a system `Scripts\` folder (Access denied), install
> into your user site with `pip install -e . --user`.

### Option B — no install
```bash
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
python run.py "list this folder and tell me what's here"
# or:  python -m vibeharness "..."
```

### Windows convenience launcher
`bin\vibe.cmd` is a launcher that works in both PowerShell and CMD without `pip install`. Add the `bin` folder to your PATH (open a **new** terminal afterwards):
```powershell
[Environment]::SetEnvironmentVariable('Path',
  [Environment]::GetEnvironmentVariable('Path','User') + ';' + (Resolve-Path .\bin),
  'User')
```

> **One run at a time.** VibeHarness takes a machine-global single-instance lock at startup
> (`~/.vibeharness/vibe.lock`), because Ollama serves one stream at a time. A second `vibe`
> while one is running is refused with a message pointing at the active run; a crashed run's
> stale lock is reclaimed automatically.

---

## Usage

`vibe` runs **in your current terminal directory** — that directory is the agent's workspace.

```powershell
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
vibe --temp 1.0 "draft a haiku into poem.txt"        # override temperature for one run
vibe --max-steps 30 "tidy up this folder"            # raise the step budget
vibe --workdir C:\some\other\dir "summarise it"      # operate elsewhere
vibe --agent web "find the top 5 Hacker News titles" # use the web agent
vibe --codec tagged_json "..."                       # try a different tool-call format
```

### Commands & settings
```powershell
vibe --help                 # list every command and parameter
vibe --version              # package version + git build identity
vibe --list-agents          # show agent types and their default toolset
vibe --list-toolsets        # show available toolsets and their tools
vibe --print-system         # print the generated system prompt and exit
vibe --show-config          # show effective defaults + saved overrides
vibe --set temp 0.5         # persist a new default temperature
vibe --set max-steps 25     # persist a new default step budget
vibe --reset-config         # restore built-in defaults
```

`--print-system` honours `--agent` / `--toolset`, so `vibe --agent web --print-system`
prints the exact web-agent system prompt the model would receive (no model required).

### Tuning sampling and the context window per run
The model, sampling, and context window can all be overridden for a single run (or persisted
with `--set`):

```powershell
vibe --model qwen3:4b "..."                  # different Ollama model for one run
vibe --temp 0.7 --top-p 0.9 --top_k 40 "..." # sampling knobs for one run
vibe --num-ctx 16384 "..."                   # smaller context window for one run
```

`--num-ctx` is the whole token window shared by the system prompt, chat history, the live
page snapshot, and generation. The output reservation (`reason-tokens` + `action-tokens`,
both settable via `--set`) is subtracted first, and the live page snapshot is then sized to
fit whatever input budget remains.

### Local ⇄ API model endpoints (per-role)
The harness can mix a **local** Ollama model with a **hosted API** model for specific roles.
Today the **validator** (and the stuck-run **escalation** path) can run on a hosted
OpenAI-compatible endpoint — by default ZhipuAI's **GLM** (`glm-5.2`) — while the base agent
stays on the local Ollama model. The provider holds endpoint coordinates only; the API key is
read from an environment variable at call time and is never stored in the repo or config:

```powershell
# Run the base agent locally, but VALIDATE with a hosted GLM model:
$env:ZHIPUAI_API_KEY = "your-key"            # PowerShell  (bash: export ZHIPUAI_API_KEY=...)
vibe --agent web "fill in the signup form and submit it"
# The local model drives the browser; when it calls `validate`, GLM-5.2 renders the verdict.
```

If `ZHIPUAI_API_KEY` is unset, the validator silently falls back to the local Ollama model,
so nothing breaks offline. The provider/model used for each role lives in
[`config.py`](./vibeharness/config.py) (`validation_provider` / `validation_model`,
`escalation_provider` / `escalation_model`); providers are registered one line at a time in
[`vibeharness/providers.py`](./vibeharness/providers.py).

> **Coming with #163 (PR #168):** dedicated per-role CLI flags
> `--base-provider` / `--base-model` / `--validator-provider` / `--validator-model` to set each
> role's endpoint directly on the command line. This README and `vibe --help` will document
> them once that change lands on this branch.

Persistent defaults live in `~/.vibeharness/settings.json` (override the location with the
`VIBEHARNESS_HOME` env var). Resolution order is **built-in defaults < saved settings <
per-run flags**. Settable keys: `temp`, `model`, `codec`, `max-steps`, `max-actions-per-turn`,
`top-p`, `top_k`, `num-ctx`, `reason-tokens`, `action-tokens`. The built-in default
temperature is `0.3`.

Each run is logged (with reasoning traces) to a hidden `.vibe/` folder in the workspace —
see [Run logs](#run-logs-vibe).

---

## Agent types

An **agent type** bundles a default toolset, a tuned system prompt, and a per-turn
tool-call cap. Pick one with `--agent`:

```bash
vibe --list-agents          # list agent types and their default toolset(s)
vibe --agent fs "..."       # filesystem agent (this is the default)
vibe --agent web "..."      # web-browsing agent
vibe --agent validator …    # the reviewer (normally invoked internally, not by you)
```

| agent | default toolset | per-turn action cap | what it's for |
|-------|-----------------|---------------------|---------------|
| `fs` (default) | `fs` | 4 | Read, write, search, and manage files on the local filesystem. |
| `web` | `web` | 4 | Drive a stateful real browser: navigate, click, fill forms, select, hover, press keys, upload, screenshot. The page is shown to the agent automatically each turn. |
| `validator` | `validator` | 1 | Review another agent's completed work and return a strict single-shot pass/fail verdict (the `validate` tool). |

### `--agent` vs `--toolset` precedence
An `--agent` type is a *named default toolset selection*. `--toolset` overrides or augments it:

- `--agent web` → toolset `[web]`
- `--agent web --toolset web,fs` → toolset `[web, fs]`
- `--toolset web` (no `--agent`) → toolset `[web]`
- neither → toolset `[fs]` (the default)

`--toolset` is repeatable or comma-separated (`--toolset web --toolset fs` == `--toolset web,fs`).

The per-turn action cap follows its own precedence: an explicit `--max-actions-per-turn`
flag (or saved setting) wins; otherwise the selected agent's default applies (`web`=4,
`validator`=1, `fs`=the global default of 4); otherwise the global default. The resolved
number is the single source of truth — both the prompt ("you may emit up to N actions") and
the loop (which executes at most N) read it, so they can't drift.

### The validator
When the working agent calls `validate`, a second invocation of the model reviews the run.
It is given the **exact context the working agent had** — the original task, a snapshot of
the workspace (and, for the web agent, the live page snapshot), and the plain-English account
of every action and its result — but **its tools are stripped out** of that context so it
judges work rather than planning more of it. There is no self-claim to take on faith. On a
fail, the validator must return a *concrete, ref-specific next step* (e.g. "click the Accept
control [e88] to clear the consent dialog, then fill the search box [e78]") that the working
agent reads verbatim and acts on. Every validator invocation is logged (see [Run logs](#run-logs-vibe)).

---

## Toolsets

Toolsets are the discrete tools an agent can call. Select and compose them directly with
`--toolset` (which overrides/augments `--agent`'s default):

```bash
vibe --list-toolsets                 # show available toolsets and their tools
vibe --toolset web "..."             # use the web toolset
vibe --toolset web,fs "..."          # compose web + filesystem
```

| toolset | tools | needs |
|---------|-------|-------|
| `fs` (default) | `validate`, `list_directory`, `read_file`, `create_file`, `write_file`, `search`, `manage_path` | nothing |
| `web` | `validate`, `goto`, `click`, `fill`, `type`, `press_key`, `select_option`, `check`, `uncheck`, `hover`, `drag`, `upload`, `screenshot`, `navigate_back`, `navigate_forward`, `reload` | Node + `@playwright/cli` |
| `validator` | `validate` | nothing |

Every toolset includes `validate` (end the turn for review) instead of a self-asserted
`finish`. Adding a new toolset is one class: implement `Toolset` and register it in
`default_catalog()`; its tools merge into the action schema automatically, and a same-named
agent type is derived for it.

### Filesystem tools (`fs`)
| tool | purpose | key params |
|------|---------|-----------|
| `list_directory` | list a folder | `path?`, `recursive?` |
| `read_file` | read a file **one 10,000-char page at a time** | `path`, `page?` (default 1) |
| `create_file` | create a **new** file (auto-creates parent folders) | `path`, `content` |
| `write_file` | modify an **existing** file; an overwrite is only allowed after the file was read this session | `path`, `content`, `mode?` (overwrite/append/prepend) |
| `search` | find text or filenames | `query`, `path?`, `target?` (content/filename/both), `max_results?` |
| `manage_path` | make_directory / delete / move / copy | `action`, `path`, `destination?` |
| `validate` | end the turn and request validation | none |

`create_file` and `write_file` are deliberately split: `create_file` refuses to clobber an
existing file, and `write_file`'s overwrite mode refuses to run until you've read the file
this session, so the model can't silently discard contents.

### The web agent
The `web` toolset drives a single, **stateful** browser through discrete subtools, backed by
the [Playwright Agent CLI](https://playwright.dev/docs/getting-started-cli) (`playwright-cli`
from `@playwright/cli`). There is **no monolithic `browse` tool, and no `snapshot` or
`evaluate` tool** — the agent cannot fetch the page on demand and cannot run arbitrary
JavaScript. Instead:

- **The page is auto-injected each turn.** A live snapshot of the current page (its text,
  links with URLs, form fields, and element **refs**) is rendered into the per-turn system
  prompt automatically under "# Current page (live snapshot)". The agent reads it; it does
  not (and cannot) request it.
- **Acting uses snapshot refs only.** Tools that target an element (`click`, `fill`,
  `select_option`, `check`, `uncheck`, `hover`, `drag`, …) take an element **ref** (e.g.
  `e163`) exactly as it appears in the current snapshot. Guessed CSS selectors / classes /
  ids / tags are **rejected before the browser is touched**, and the rejection lists the refs
  that actually exist. A "no-match" target is reported as a failure (not a silent success).
- **Dynamic snapshot budget.** The injected snapshot is truncated only when including it
  whole would push the full model message past the usable context window — so on a large
  window the agent can see a very large page, and the old fixed cap is now just an absolute
  safety ceiling. (Details: `vibeharness/snapshot_budget.py`, `Config` fields.)
- **Browser keeps state.** Page, cookies, and history persist across actions
  (`goto`, `navigate_back`, `navigate_forward`, `reload`, …).

Install the backend once:
```bash
npm install -g @playwright/cli@latest
```
The browser runs **headed by default** so you can watch; add `--headless` to hide it. Example:
```bash
vibe --agent web "Go to https://news.ycombinator.com and list the top 5 story titles."
vibe --agent web --task-file task.txt     # read a long task from a file
```

#### WebArena-style prose snapshots (`--web-snapshot-prose`)
By default the auto-injected page is raw Playwright ARIA-YAML. `--web-snapshot-prose`
(experimental; issue #64; off by default) instead runs it through a deterministic
ARIA→WebArena-style **prose** transform: it prunes generic/image noise and emits one
ref-keyed line per interesting control, which small models reason over more reliably.
Refs are preserved inline, so the discrete subtools keep working unchanged. This is an A/B
seam — budgeting and diagnostics are unchanged; only the text fed into the page section
differs. A committed sample of the transform on a real YouTube consent page lives at
[`samples/youtube_consent_prose.txt`](./samples/youtube_consent_prose.txt).

---

## Tool-call codecs

The **wire format** of tool calls is owned by a swappable `ToolCallCodec` seam: it controls
(1) how the model is told to emit calls, (2) how decoding is constrained to that shape, and
(3) how raw output is parsed back into `(tool, args)`. Swapping the codec swaps the format
without touching the agent loop, prompt builder, or LLM transport. Pick one per run with
`--codec` (or persist it with `--set codec ...`):

| codec | format | decode constraint |
|-------|--------|-------------------|
| `json` (default) | JSON array of `{"tool":…, "args":{…}}` | JSON-schema grammar (Ollama `format`) |
| `tagged_json` | JSON wrapped in tags | stop strings |
| `xml` | pure XML | stop strings |
| `codeact` | code-as-action | stop strings |
| `gbnf` | JSON shaped by a raw GBNF grammar | GBNF (honoured by a llama.cpp backend) |

```bash
vibe --codec tagged_json "..."     # one run in a different format
vibe --set codec xml               # persist it as the default
```

New codecs are added as isolated modules under `vibeharness/codecs/` (`<name>_codec.py`
exposing a `CODEC` instance) and are auto-discovered — no central registry to edit, and the
discovery is frozen-build-safe. The `gbnf` codec is honoured only by the optional `llamacpp`
backend (`Config.backend = "llamacpp"`); under Ollama, codecs that need a grammar fall back
to stop-string parsing.

---

## How it works

Each turn is two model calls — reason, then act under a constraint:

```
 task + natural-language narrative of past actions (+ live page snapshot for web)
        │
        ▼
 ┌─ phase 1: free reasoning ──────────┐   /api/chat, stop at </think>
 │  <think> … </think>                │   (streamed live, then discarded)
 └────────────────────────────────────┘
        │
        ▼
 ┌─ phase 2: constrained action(s) ───┐   /api/generate, raw continuation,
 │  [{"tool":"...","args":{...}}, …]   │   constrained per the active codec
 └────────────────────────────────────┘
        │
        ▼
 parse → execute each action in order via ToolRegistry → append a
 plain-English observation per action
        │
        └──────────► repeat until `validate` passes or the step budget
```

A turn may **batch several actions** (up to the per-turn cap) when the model is confident of
the outcomes, or emit a single action when it needs the result before deciding the next move.
When the agent calls `validate`, the [validator](#the-validator) reviews the run; a pass ends
the run, a fail feeds concrete next-step guidance back into the loop.

### Run logs (`.vibe/`)
Each run writes into a hidden `.vibe/` folder in the workspace:
- `<stamp>.json` — the complete structured log, **including every turn's reasoning trace**,
  the actions, results, and the config used.
- `<stamp>.md` — a readable transcript.
- `<stamp>-diagnostics/` — optional per-turn diagnostic dumps (the raw page snapshot and the
  fully-rendered system prompt that was injected that turn) for the web agent (issue #37).
- `validator_<guid>.json` — one file per validator invocation, capturing exactly what the
  validator saw and the verdict it returned (issue #47).

A partial `RunResult` is flushed even if a turn crashes mid-run, so logs are not lost.
These are intended for analysis — diffing reasoning across runs, spotting where a small model
goes wrong, and tuning the prompt.

### Architecture
```
vibeharness/
  tools.py          Tool interface + Param/ToolResult (docs & schema derived from params)
  filesystem.py     FileSystem service — the only code that touches the OS (SRP)
  fs_tools.py       concrete filesystem tools, each self-describing
  web.py            stateful browser toolset over the Playwright Agent CLI
  validation.py     the validator agent + validate tool
  toolset.py        Toolset catalog; agent types + per-agent action caps
  registry.py       ToolRegistry — builds docs + action schema (OCP: add tools freely)
  codec.py          ToolCallCodec seam (format + decode constraint + parse)
  codecs/           one module per wire format (json, tagged_json, xml, codeact, gbnf)
  snapshot_budget.py  dynamic per-turn page-snapshot sizing
  snapshot_prose.py   deterministic ARIA -> WebArena-style prose transform
  prompt.py         system prompt + per-turn prompt builders
  memory.py         NarrativeMemory — the English account of past actions
  llm.py            LLMClient interface + OllamaClient two-phase streaming (DIP)
  llamacpp.py       optional llama.cpp backend (honours GBNF-constrained decoding)
  reporting.py      Reporter interface + ConsoleReporter (live, colored output)
  runlog.py         streaming per-run logging + diagnostics into .vibe/
  lock.py           machine-global single-instance lock
  agent.py          RalphAgent — the loop orchestrator
  config.py         immutable Config value object
  settings.py       persistent user settings
  cli.py            argument parsing and command dispatch
run.py              no-install entrypoint   |   bin/vibe.cmd  Windows launcher
```
The design leans on small interfaces: the agent depends on `LLMClient` and `Reporter`
abstractions, so the whole loop is testable with a fake client and a null reporter — no model
required.

---

## Testing

```bash
python -m unittest discover -s tests -v     # standard library, no install needed
# or, with pytest:
pip install -e ".[dev]" && pytest -q
```
Two tiers:
- **Unit tests** (fast, zero dependencies) cover the filesystem service, every tool,
  schema/toolset/agent building, the codec seam, the settings store, narrative memory, prompt
  building, the dynamic snapshot budget, the prose transform, run logging, and the full agent
  loop (single- and multi-action turns, the validator) via a fake LLM client.
- **Live integration tests** (`tests/integration/`) talk to the *real* dependencies, so a
  crashed Ollama or a broken generation/tool path is actually caught (the fast unit tests mock
  these and can't):
  - `test_model_live.py` — hits Ollama, generates text from the model and stops inference
    early, and verifies a clear error when the server is down.
  - `test_web_live.py` — drives the real web tools through `playwright-cli` against a demo app
    at `http://localhost:3000`.

  They **auto-skip** when Ollama / the CLI / the demo server aren't present, so CI stays
  green; run them locally to confirm core functionality.

---

## Troubleshooting

- **`Could not reach Ollama …`** — the server isn't running. Start it with `ollama serve` (or launch the Ollama app).
- **`vibe` refuses to start, says another run is active** — only one run is allowed at a time (single-instance lock). Wait for the active run, or if it crashed, the stale lock is reclaimed automatically on the next attempt.
- **`vibe --version` shows an unexpected sha** — you're on a stale build; re-run `pip install -e .` from the checkout you're editing.
- **It's slow / not using my GPU** — confirm with `ollama ps` (`PROCESSOR` should say `100% GPU`) and `nvidia-smi` (VRAM should be in use). On laptops with both an NVIDIA dGPU *and* an integrated GPU, Ollama's Vulkan backend may pick the iGPU; force CUDA with `setx OLLAMA_VULKAN 0` and restart the Ollama server.
- **Garbled / non-English tokens in output** — small models drift at high temperature; the *action* is always valid (codec-constrained), but lower `--temp` (e.g. 0.3) for cleaner reasoning and content.
- **Web agent rejects my element target** — only refs (e.g. `e163`) from the current page snapshot are accepted; CSS selectors/classes/ids are rejected by design. Use the ref shown in the live snapshot; the rejection message lists the available refs.
- **No colors on Windows** — colors use ANSI; pass `--no-color` if your console doesn't render them.
- **Running several agents at once** — by default Ollama serves one request at a time, and VibeHarness itself enforces a single run per machine. To experiment with parallel generation you must lift both limits and watch VRAM: Ollama allocates `num_ctx × parallel` of KV cache, so the large default `num_ctx` can OOM with parallelism. Use `OLLAMA_NUM_PARALLEL=1` (the recommended default) so a single instance gets the whole window.

---

## Acknowledgements

- [VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) by WeiboAI (MIT).
- The constrained-decoding-after-reasoning idea is adapted from [noperator's structural-tag gist](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).
- The prose snapshot format is adapted from [WebArena](https://webarena.dev/)'s accessibility-tree linearization.
- [Ollama](https://ollama.com) for painless local model serving.

## License

[MIT](./LICENSE) © 2026 Nickalas Light
