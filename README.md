# VibeHarness

[![CI](https://github.com/NickalasLight/VibeHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/NickalasLight/VibeHarness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**A tiny, dependency-free Ralph-loop agent harness for small local models.**

VibeHarness turns a 3B local model ([VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B), served by [Ollama](https://ollama.com)) into a basic command-line coding agent. You type a task; it works on your current directory one step at a time — reading, writing, searching, and managing files — and streams its reasoning and actions live to the terminal.

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
│ action: {"tool":"write_file","args":{"path":"CHANGELOG.md","content":"# Changelog\n\n## [Unreleased]\n"}}
└ ✓ you wrote the file 'CHANGELOG.md' (38 characters).

┌─ step 2 ────────────────────────────────────────
│ action: {"tool":"finish","args":{"summary":"Created CHANGELOG.md with an Unreleased section."}}
└ ✓ you finished the task: Created CHANGELOG.md with an Unreleased section.

 done in 2 steps — Created CHANGELOG.md with an Unreleased section.
```

> **Status:** working prototype. VibeThinker-3B is a math/reasoning specialist and is *not* tuned for tool use, so treat this as a research toy for studying small-model agentic behaviour — not a production agent.

---

## Quickstart

Install everything (requires [Ollama](https://ollama.com/download) and Python ≥ 3.10):

```bash
# 1. Install Ollama from https://ollama.com/download, then pull + register the model
ollama pull hf.co/mradermacher/VibeThinker-3B-GGUF:Q8_0
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
ollama create vibethinker -f Modelfile

# 2. Install the harness (creates the `vibe` command)
pip install -e .

# 3. Run it — the agent works in your current directory
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
```

No `pip install`? Use `python run.py "<task>"` instead (Windows users can also run `bin\vibe.cmd`). See [Prerequisites](#prerequisites) and [Install](#install) for details and platform notes.

---

## Why it's interesting

- **Constrained actions.** Each turn the model emits a JSON array of one or more tool calls, validated against a JSON schema *at decode time* (Ollama's `format` grammar), so it can't produce malformed actions — even at high temperature, where small models otherwise drift into garbage.
- **Two-phase turns.** The model first reasons freely, then emits the action(s) under the schema constraint. (This is an Ollama adaptation of [noperator's vLLM structural-tag trick](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).) The reasoning is dropped from the running context — but kept on disk (see below).
- **Natural-language memory.** Instead of a JSON/ChatML transcript, the agent's past is a plain-English narrative ("First, you… Then, you…"), which is what a small model follows most reliably.
- **Full reasoning logs.** Every run is written in full — *including* each turn's reasoning trace — to a hidden `.vibe/` folder in the workspace, so you can mine the traces to improve the prompt and model.
- **Small but powerful tools.** Each toolset is a handful of tools; behaviour is widened with optional parameters (e.g. `write_file.mode` = overwrite/append/prepend, `browse.action` = goto/snapshot/click/fill/…) rather than by adding more tools.
- **Pluggable toolsets.** Tool interfaces are swappable and composable (`--toolset web,fs`); a filesystem toolset and a stateful-browser toolset ship in the box.
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

### 3. The `vibethinker` model
VibeThinker ships as safetensors; Ollama needs GGUF. Pull a community GGUF and register it under the name `vibethinker` using the included [`Modelfile`](./Modelfile):
```bash
ollama pull hf.co/mradermacher/VibeThinker-3B-GGUF:Q8_0
ollama create vibethinker -f Modelfile
ollama run vibethinker "hi"      # quick sanity check
```
The `Modelfile` only points at the weights — the harness sets all sampling parameters per request.

### Hardware
The Q8_0 quant of this 3B model needs **~3.5 GB of VRAM** (or runs on CPU, just slower). Any modern GPU with ≥4 GB, or a CPU with ≥8 GB RAM, is fine.

### 4. (Optional) Web toolset
For `--toolset web` you also need [Node.js](https://nodejs.org) and the Playwright Agent CLI:
```bash
npm install -g @playwright/cli@latest
```
See [Toolsets](#toolsets) for details.

---

## Install

### Option A — install the package (recommended; cross-platform `vibe` command)
```bash
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
pip install -e .          # creates the `vibe` console command
vibe "list this folder and tell me what's here"
```

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

---

## Usage

`vibe` runs **in your current terminal directory** — that directory is the agent's workspace.

```powershell
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
vibe --temp 1.0 "draft a haiku into poem.txt"     # override temperature for one run
vibe --max-steps 30 "tidy up this folder"         # raise the step budget
vibe --workdir C:\some\other\dir "summarise it"   # operate elsewhere
```

### Commands & settings
```powershell
vibe --help                 # list every command and parameter
vibe --show-config          # show effective defaults + saved overrides
vibe --set temp 0.5         # persist a new default temperature
vibe --set max-steps 25     # persist a new default step budget
vibe --reset-config         # restore built-in defaults
vibe --print-system "x"     # print the generated system prompt
```
Persistent defaults live in `~/.vibeharness/settings.json` (override the location with the `VIBEHARNESS_HOME` env var). Resolution order is **built-in defaults < saved settings < per-run flags**. Settable keys: `temp`, `model`, `max-steps`, `top-p`, `top_k`. The built-in default temperature is `0.3`.

Each run is logged (with reasoning traces) to a hidden `.vibe/` folder in the workspace — see [Run logs](#run-logs-vibe).

---

## How it works

Each turn is two model calls — reason, then act under a schema constraint:

```
 task + natural-language narrative of past actions
        │
        ▼
 ┌─ phase 1: free reasoning ──────────┐   /api/chat, stop at </think>
 │  <think> … </think>                │   (streamed live, then discarded)
 └────────────────────────────────────┘
        │
        ▼
 ┌─ phase 2: constrained action(s) ───┐   /api/generate, raw continuation,
 │  [{"tool":"...","args":{...}}, …]   │   format = tools JSON schema (an array)
 └────────────────────────────────────┘
        │
        ▼
 parse → execute each action in order via ToolRegistry → append a
 plain-English observation per action
        │
        └──────────► repeat until `finish` or the step budget
```

A turn may **batch several actions** (e.g. write a file and read it back) when the
model is confident of the outcomes, or emit a single action when it needs the
result before deciding the next move.

### Run logs (`.vibe/`)
Each run writes two timestamped files into a hidden `.vibe/` folder in the
workspace:
- `<stamp>.json` — the complete structured log, **including every turn's reasoning
  trace**, the actions, results, and the config used.
- `<stamp>.md` — a readable transcript.

These are intended for analysis — diffing reasoning across runs, spotting where a
small model goes wrong, and tuning the prompt.

### Tools
| tool | purpose | key params |
|------|---------|-----------|
| `list_directory` | list a folder | `path?`, `recursive?` |
| `read_file` | read a file | `path` |
| `write_file` | create/modify a file | `path`, `content`, `mode?` (overwrite/append/prepend) |
| `search` | find text or filenames | `query`, `path?`, `target?` (content/filename/both), `max_results?` |
| `manage_path` | mkdir / delete / move | `action`, `path`, `destination?` |
| `finish` | end the task | `summary` |

The system prompt documents each tool in plain English **and** embeds the formal JSON schema — both generated from the single `ToolRegistry` source of truth, so docs and the enforced grammar can never drift apart.

### Architecture
```
vibeharness/
  tools.py       Tool interface + Param/ToolResult (docs & schema derived from params)
  filesystem.py  FileSystem service — the only code that touches the OS (SRP)
  fs_tools.py    concrete tools wrapping FileSystem, each self-describing
  registry.py    ToolRegistry — builds docs + action schema (OCP: add tools freely)
  prompt.py      system prompt + per-turn prompt builders
  memory.py      NarrativeMemory — the English account of past actions
  llm.py         LLMClient interface + OllamaClient two-phase streaming (DIP)
  reporting.py   Reporter interface + ConsoleReporter (live, colored output)
  agent.py       RalphAgent — the loop orchestrator
  settings.py    persistent user settings
  cli.py         argument parsing and command dispatch
run.py           no-install entrypoint   |   bin/vibe.cmd  Windows launcher
```
The design leans on small interfaces: the agent depends on `LLMClient` and `Reporter` abstractions, so the whole loop is testable with a fake client and a null reporter — no model required.

---

## Toolsets

Tools are grouped into **toolsets** you select — and compose — at runtime:

```bash
vibe --list-toolsets                 # show available toolsets
vibe --toolset web "..."             # use the web toolset
vibe --toolset web,fs "..."          # compose web + filesystem
```

| toolset | tools | needs |
|---------|-------|-------|
| `fs` (default) | `list_directory`, `read_file`, `write_file`, `search`, `manage_path`, `finish` | nothing |
| `web` | `browse`, `finish` | Node + `@playwright/cli` |

Adding a new tool interface is one class: implement `Toolset` and register it in
`default_catalog()`. Its tools merge into the agent's action schema automatically.

### Web toolset
The `web` toolset drives a single, **stateful** browser through one `browse` tool,
backed by the [Playwright Agent CLI](https://playwright.dev/docs/getting-started-cli).
`snapshot` is the agent's eyes — it returns the page's text, links (with URLs),
form fields, and element refs the agent then uses as `click`/`fill` targets. The
browser keeps its page and cookies across actions (navigate, click, type, select,
check, upload, screenshot, …).

Install the backend once:
```bash
npm install -g @playwright/cli@latest
```
The browser runs **headed by default** so you can watch; add `--headless` to hide
it. Example:
```bash
vibe --toolset web "Go to https://news.ycombinator.com and list the top 5 story titles."
vibe --toolset web --task-file task.txt     # read a long task from a file
```

---

## Testing

```bash
python -m unittest discover -s tests -v     # standard library, no install needed
# or, with pytest:
pip install -e ".[dev]" && pytest -q
```
Two tiers:
- **Unit tests** (fast, zero dependencies) cover the filesystem service, every tool, schema/toolset building, the settings store, narrative memory, prompt building, the LLM helpers, run logging, and the full agent loop (single- and multi-action turns) via a fake LLM client.
- **Live integration tests** (`tests/integration/`) drive the *real* `browse` tool through `playwright-cli` against a demo web app at `http://localhost:3000`, exercising the agent's actual tool-calling surface — navigate, snapshot, click, fill. They **auto-skip** when the CLI or server isn't present (so CI stays green); start the demo app to run them. These are what catch integration bugs the fake-CLI unit tests can't (e.g. a snapshot returning empty due to a codec crash).

---

## Troubleshooting

- **`Could not reach Ollama …`** — the server isn't running. Start it with `ollama serve` (or launch the Ollama app).
- **It's slow / not using my GPU** — confirm with `ollama ps` (`PROCESSOR` should say `100% GPU`) and `nvidia-smi` (VRAM should be in use). On laptops with both an NVIDIA dGPU *and* an integrated GPU, Ollama's Vulkan backend may pick the iGPU; force CUDA with `setx OLLAMA_VULKAN 0` and restart the Ollama server.
- **Garbled / non-English tokens in output** — small models drift at high temperature; the *action* is always valid (schema-constrained), but lower `--temp` (e.g. 0.3) for cleaner reasoning and content.
- **No colors on Windows** — colors use ANSI; pass `--no-color` if your console doesn't render them.

---

## Acknowledgements

- [VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) by WeiboAI (MIT).
- The constrained-decoding-after-reasoning idea is adapted from [noperator's structural-tag gist](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).
- [Ollama](https://ollama.com) for painless local model serving.

## License

[MIT](./LICENSE) © 2026 Nickalas Light
