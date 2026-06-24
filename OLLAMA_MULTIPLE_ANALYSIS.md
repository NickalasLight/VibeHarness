# Ollama Multiple `llama-server` Accumulation — Root-Cause Analysis (#77 spec)

**Status:** Analysis only. This document defines the fix for issue #77.
**Branch:** `research/ollama-multiple-analysis` (off `beta` e561c12)
**Date:** 2026-06-24
**Environment:** Windows 10, RTX 3080 Laptop (8 GB VRAM), Ollama 0.30.8, model `vibethinker` (VibeThinker-3B Q8_0, ~3.3 GB).

---

## TL;DR

Multiple concurrent `llama-server` runner processes accumulate because:

1. **The harness never pins the context size.** Ollama is asked to load the
   *same model* with **different effective context lengths from request to
   request** (observed `-c 4096`, `-c 16384`, `-c 32768`). Ollama treats
   *(model, context-size, …)* as a **distinct runner configuration**, so each
   new context size spawns a **brand-new `llama-server` process** instead of
   reusing the existing one.
2. **Nothing tells Ollama to keep only one runner.** `OLLAMA_MAX_LOADED_MODELS`
   is **unset (=0 → auto, allows up to ~3 concurrent runners)** and
   `OLLAMA_KEEP_ALIVE` is **unset (=default 5m)**. So the old runner from the
   previous context size is **not evicted** when the new one loads — it lingers
   for its independent 5-minute keep-alive, holding GPU + host memory.
3. The two together mean **2–3+ runners coexist on an 8 GB card**, each holding
   model weights (3.1 GB) + KV cache, which **exhausts VRAM and triggers the
   OOM / CUDA crashes** that killed the harness.

The runners are **LEAKED in effect** (unintentional, redundant, never reused),
not deliberately kept alive — the harness only ever uses one model and would be
perfectly served by a single persistent runner.

---

## Ollama's process / config model (how runners spawn)

`ollama serve` is a scheduler (`ollama.exe`, Go). It does **not** run inference
itself; for each *loaded model configuration* it spawns a child
`llama-server.exe` ("runner") on its own localhost port and proxies requests to
it. Whether a request reuses an existing runner or spawns a new one is decided
by the scheduler (`sched.go`).

A runner is keyed by, among other things:

- the model (blob),
- **the context length `-c` (num_ctx)**,
- parallelism `-np` (`OLLAMA_NUM_PARALLEL`),
- GPU offload / kv-cache / flash-attn flags.

**If any of these differ from a loaded runner, the scheduler starts a new
`llama-server` rather than reusing the existing one.** Context length is the
one that varies in our workload.

Relevant server config (from the live `server.log`, `routes.go:1919 "server config"`):

```
OLLAMA_NUM_PARALLEL:1            <- SET (good; from User env var)
OLLAMA_MAX_LOADED_MODELS:0       <- UNSET → auto (allows multiple runners)
OLLAMA_KEEP_ALIVE:5m0s           <- UNSET → default 5 min (old runner lingers)
OLLAMA_CONTEXT_LENGTH:262144     <- server ceiling, not a per-request pin
OLLAMA_LOAD_TIMEOUT:5m0s
OLLAMA_MAX_QUEUE:512
```

- `OLLAMA_NUM_PARALLEL` — concurrent request *slots within one runner*. Already
  pinned to `1` (the config.py comment recommended it). **Not the leak.**
- `OLLAMA_MAX_LOADED_MODELS` — how many *distinct runners* may be resident at
  once. `0` = auto (Ollama picks based on VRAM, up to ~3× GPU count). **This is
  why old runners are not evicted when a new context size loads.**
- `OLLAMA_KEEP_ALIVE` — how long an idle runner stays resident. Default `5m`,
  so a stale runner survives long after its last use — long enough to overlap
  the next run, and (across back-to-back runs) to appear with start times from
  **prior days**.

---

## Live + log evidence

### 1. Same model, different context → a second runner (smoking gun)

From the **current** session `server.log`, two consecutive `/api/chat`
requests each spawned their own runner, differing only in `-c`:

```
02:36:53  starting llama-server ... --port 54243 ... -c 16384 -np 1 ... -ngl 99
02:36:55  loaded runners count=1
...
02:37:25  starting llama-server ... --port 54266 ... -c 32768 -np 1 ... -ngl 99
02:37:27  loaded runners count=1
```

Two `llama-server` processes, same model blob, same `-np 1`, **different `-c`
(16384 vs 32768)**, on different ports — both resident.

### 2. The harness emits a *range* of context sizes

Across all server logs, the runners launched for this one model used:

```
   9  -c 16384
   3  -c 4096
   1  -c 32768
```

Three different context sizes → three distinct runner configurations → up to
three distinct runners that never coalesce.

### 3. Accumulation is real and persists across days

```
13 total "starting llama-server" spawn events across server*.log
spawn dates:  61 events on 2026-06-23, 17 on 2026-06-24
server-5.log alone: 8 spawns in a single serve session
```

A single `ollama serve` session spawned the runner **8 times** — repeated
load/leak churn, not one stable runner.

### 4. The crash signature is GPU/VRAM-related

`server-5.log` shows the loads failing in CUDA during warm-up:

```
CUDA error: device kernel image is invalid
llama-server terminated ... exit status 0xc0000409 (stack buffer overrun)
[GIN] 500 ... POST "/api/chat"
```

…after which Ollama retried with a smaller `-c 4096`, then fell back to
`-ngl 0` (CPU, `reason=cpu`). The pattern — load a runner, fail/redundant-load,
shrink context, spawn again — is exactly the runner-churn-under-memory-pressure
the leak produces. With 8 GB VRAM and each runner needing ~3.1 GB model + KV,
two coexisting runners is enough to push the card over.

### 5. Why `-c` ≠ `num_ctx` (131072)

`config.py` sets `num_ctx = 131072`, but the runners launch with `-c 16384` /
`32768` / `4096`. Ollama 0.30.8 **auto-fits** the requested context to free
VRAM (`common_params_fit_impl: projected to use … MiB … will leave … no changes
needed`). Because the *fitted* size depends on transient free-VRAM and the
per-turn request, the **effective `-c` varies between requests** — which is the
very thing that makes the scheduler spin up new runners. So even though the
harness *thinks* it always asks for 131072, Ollama lands on different real
context sizes and treats each as a new config.

---

## How the harness calls Ollama (code review)

- `vibeharness/config.py`
  - `num_ctx: int = 131072`, `num_gpu: int = 99`.
  - **No `keep_alive` field.** The comment at line 39 even says *"Use
    OLLAMA_NUM_PARALLEL=1 so a single instance gets the whole window"* — but
    that only governs slots, not runner count, and the more important
    keep-alive / max-loaded knobs are never addressed.
- `vibeharness/llm.py` (`OllamaClient`)
  - `_options()` (line 151) sends `temperature, top_p, top_k, num_ctx,
    num_gpu` on **every** request. **`keep_alive` is never sent** → server
    default 5m applies.
  - Two endpoints are used: `/api/chat` (phase-1 reasoning) and `/api/generate`
    (phase-2 constrained action). Both carry the same `_options()`, so both
    route to the same runner *for a given context size* — good.
  - **But** `num_ctx` is taken straight from `config.num_ctx`, and the dynamic
    snapshot budget (`snapshot_budget.py`, issue #43) means the *effective*
    window the model is asked to use shifts per turn; combined with Ollama's
    auto-fit (evidence #5) the runner config is not stable.
- `vibeharness/cli.py:467-468`
  - **The agent and the validator share ONE `OllamaClient` instance**
    (`client = OllamaClient(config)` then `LLMValidator(client, …)`). So the
    validator does **not** spawn a second runner by itself — it reuses the same
    client, model, and options. **The validator is NOT a separate cause.** The
    duplication comes from context-size variance, not from agent-vs-validator.
- **No process management anywhere.** `grep` for `subprocess|Popen|taskkill|
  kill|reap|ollama serve` in `vibeharness/` finds process handling **only for
  the Playwright web CLI** (`web.py`) and a PID liveness check in `lock.py`.
  The harness **never starts, inspects, or reaps `ollama` / `llama-server`
  processes**. Nothing cleans up stale runners on startup or teardown.

---

## Confirmed root cause

> The harness lets Ollama choose the runner context size per request (no pinned,
> stable `num_ctx`, aggravated by Ollama 0.30.8 auto-fit), while leaving
> `OLLAMA_MAX_LOADED_MODELS` at auto and `OLLAMA_KEEP_ALIVE` at the 5-minute
> default. Each distinct effective context size spawns a **new `llama-server`
> runner that is not evicted**, so 2–3+ runners for the *same* model coexist on
> an 8 GB GPU and exhaust VRAM → the OOM / CUDA crashes.

The extra runners are **leaked (stale, redundant, never reused)**, not an
intentional keep-alive of distinct models.

---

## Recommended fix (#77 spec)

Aim: **exactly one resident `llama-server` for the harness model, reused across
every request and every run, with no stale leftovers.** Implement all of the
following (defense in depth):

### A. Pin a single, stable runner configuration (primary fix)
1. **Send a constant `keep_alive`** on every Ollama request from
   `OllamaClient` (add a `keep_alive` to the `/api/chat` and `/api/generate`
   payloads, e.g. `"keep_alive": "30m"` or `-1` to keep resident for the whole
   run). Add a `Config.ollama_keep_alive` field (default e.g. `"30m"`).
2. **Stop varying the effective context size.** Either:
   - pin `num_ctx` to a *single value that actually fits 8 GB* (e.g. 16384 or
     32768 — the sizes that already load successfully), **and**
   - ensure the same `num_ctx` is sent on **every** request (phase 1, phase 2,
     liveness `generate`), so Ollama always matches the one existing runner.
   This removes the per-turn context drift that creates new runner configs.
   (Setting `num_ctx` realistically also stops the auto-fit from landing on
   different sizes.)

### B. Constrain Ollama to one runner (belt-and-braces, env)
3. Set, in the Ollama server environment (User env vars / service config):
   - `OLLAMA_MAX_LOADED_MODELS=1` — only one runner may be resident; loading a
     new config **evicts** the old one instead of stacking.
   - `OLLAMA_NUM_PARALLEL=1` — already set; keep it.
   - Optionally `OLLAMA_KEEP_ALIVE=-1` (or a generous value) as a system-level
     backstop to (A.1).
   The harness should **verify these on startup** (read `/api/ps` or the server
   config) and warn loudly if `OLLAMA_MAX_LOADED_MODELS != 1`.

### C. Reap stale runners on startup/teardown (cleanup safety net)
4. On harness **startup**: query `GET /api/ps`; if any runner for our model is
   already loaded with a *different* config, unload it (a `keep_alive: 0`
   request, or `ollama stop <model>`), so the run starts from a clean single
   runner.
5. On harness **teardown** (normal exit and crash/finally): send a
   `keep_alive: 0` request (or `ollama stop vibethinker`) to **unload the model
   immediately**, so nothing lingers into the next run with a prior-day start
   time.

### Minimum viable fix
If only one thing ships: **(A) pin `num_ctx` to a fits-in-8GB value and send a
constant `keep_alive` on every request, plus (B) `OLLAMA_MAX_LOADED_MODELS=1`.**
That alone collapses the workload to a single reused runner and eliminates the
accumulation and the OOM.

### Do NOT
- Do not rely on `OLLAMA_NUM_PARALLEL` to fix this — it controls slots inside a
  runner, not runner count (it is already `1` and the leak persists).
- Do not keep `num_ctx=131072` on an 8 GB card; auto-fit will keep landing on
  varying sizes and re-spawning runners (and the KV for 131072 cannot fit
  anyway).

---

## Evidence appendix (paths)

- Harness source (worktree `C:\git\vh-ollama-analysis`):
  - `vibeharness/config.py` — `num_ctx=131072`, no `keep_alive` field.
  - `vibeharness/llm.py` — `_options()` (no `keep_alive`), `/api/chat` +
    `/api/generate` payloads.
  - `vibeharness/cli.py:467-468` — agent and validator share one client.
  - `vibeharness/snapshot_budget.py` — per-turn dynamic window sizing.
- Ollama logs / state: `C:\Users\NickL\AppData\Local\Ollama\server*.log`,
  `app.log`, `ollama.pid`.
- Live checks: `ollama --version` (0.30.8), `ollama ps`, `ollama list`,
  `[Environment]::GetEnvironmentVariable('OLLAMA_NUM_PARALLEL','User')` → `1`;
  `OLLAMA_MAX_LOADED_MODELS` / `OLLAMA_KEEP_ALIVE` → unset.
