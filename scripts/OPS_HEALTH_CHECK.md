# Ops health check — for heavy Claude Code sessions on this machine

Long, high-concurrency sessions exhausted host RAM and OOM-killed the Claude Code node
process — which takes every in-flight background subagent with it (root cause of the
session crashes; see the crash analysis). `scripts/leak_check.ps1` warns before that.

## Mitigations
1. **Commit frequently** — every subagent commits incrementally, so a crash loses at most the last step. (This is the primary mitigation; it replaces an earlier "cap concurrent subagents at 3-4" rule.)
2. **Memory headroom** — keep ≥ 8 GB free / commit < ~75%. The background watcher alerts when free RAM drops below 8 GB.
3. **Leak watch (known bug #32)** — a standalone `llama-server.exe` leaked/faulted. Do NOT kill Ollama or its bundled llama.cpp server (VibeThinker + output constraints depend on it). Tracked for a later sprint.
4. **Orphan reap (#15)** — after any Playwright/browser run, kill orphaned `chrome`/`node` trees.

## Run the watcher
```
# one-shot
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/leak_check.ps1
# recurring (as a Claude Code background Monitor, bash):
while true; do powershell -NoProfile -ExecutionPolicy Bypass -File scripts/leak_check.ps1; sleep 300; done
```
It prints an `ALERT LOW-RAM: ...` line (with the llama-server/chrome/node/claude breakdown) only when free RAM is low; silent otherwise.
