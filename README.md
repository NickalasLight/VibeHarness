# vibethinkharnessProto1

A minimal **Ralph-loop** agent harness for small local models (built for
**VibeThinker-3B** via Ollama). It gives the model a tiny, high-level toolset for
operating on the Windows filesystem and drives it one action at a time until the
task is done.

## Design

- **Ralph loop:** each turn the model picks exactly one tool call; the harness
  executes it and appends the result. Repeat until `finish` or the step budget.
- **Constrained actions:** the model emits a single JSON object
  `{"tool": ..., "args": {...}}`, enforced by Ollama's `format` (JSON-schema)
  grammar so the action is always structurally valid — even at high temperature.
- **Two-phase per turn** (converted from noperator's vLLM structural-tag trick):
  1. *reason freely* (stopped at `</think>`), then
  2. *act* under the schema constraint. The reasoning is discarded.
- **Natural-language memory:** the model's past is a plain-English narrative
  ("First, you … Then, you …"), not a JSON/ChatML transcript. Thinking is never
  carried across turns.
- **Small but powerful tools:** behaviour is widened with optional params instead
  of more tools (e.g. `write_file.mode` = overwrite/append/prepend,
  `manage_path.action` = make_directory/delete/move).

### Tools
| tool | purpose | key params |
|------|---------|-----------|
| `list_directory` | list a folder | `path?`, `recursive?` |
| `read_file` | read a file | `path` |
| `write_file` | create/modify a file | `path`, `content`, `mode?` |
| `search` | find text or filenames | `query`, `path?`, `target?`, `max_results?` |
| `manage_path` | mkdir / delete / move | `action`, `path`, `destination?` |
| `finish` | end the task | `summary` |

The system prompt documents each tool in plain English **and** embeds the formal
JSON schema — both generated from the single `ToolRegistry` source of truth.

## Architecture (SOLID)
```
vibeharness/
  tools.py       Tool interface + Param/ToolResult (docs & schema derived from params)
  filesystem.py  FileSystem service — the only code that touches the OS (SRP)
  fs_tools.py    concrete tools wrapping FileSystem, each self-describing
  registry.py    ToolRegistry — builds docs + action schema (OCP: add tools freely)
  prompt.py      system prompt + per-turn prompt builders
  memory.py      NarrativeMemory — the English account of past actions
  llm.py         LLMClient interface + OllamaClient two-phase impl (DIP)
  agent.py       RalphAgent — the loop orchestrator
run.py           CLI
```

## Usage

`vibe` runs the agent **in your current terminal directory** (like a basic coding
agent) and streams each turn live. Requires Ollama running with a `vibethinker`
model (Q8_0 GGUF); GPU is forced via `OLLAMA_VULKAN=0`.

```powershell
cd C:\some\project
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
vibe --temp 1.0 --max-steps 12 "List this folder and summarize what's here."
vibe --print-system "x"          # inspect the generated system prompt
```

`bin\vibe.cmd` is added to the User PATH, so `vibe` works in **both PowerShell and
CMD** (open a new terminal after install so the PATH is picked up). You can also
run it directly: `python run.py "<task>"`.

Transcripts are saved to `runs/` (timestamped, never overwritten).

No third-party dependencies — standard library only.
