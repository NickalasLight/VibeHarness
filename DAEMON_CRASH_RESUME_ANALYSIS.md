# Daemon Crash & Clean-Resume Analysis (issue #102)

Research output for the `research/daemon-crash-resume` worktree. **ANALYSIS + DESIGN
ONLY** — this document does not modify `vibeharness/web.py`. The immediate cause is
being fixed in parallel by #101/#75; this is the broader ground-truth + the
clean-resume design that informs and composes with those fixes.

> **STATUS (implemented in `fix/browser-daemon-dies`, closing #101 + #75):** the
> Part D clean-resume design has been built. `PlaywrightCli.run` is now self-healing
> (detect daemon-death signature → reap/reopen with the run's `open_flags` →
> re-navigate to the tracked `last_url` → retry once, bounded by `SessionState`),
> snapshots are death-tolerant (resume-free `snapshot()` seam), and an agent-callable
> `open_browser` tool plus updated `system_guidance` give the model an explicit
> recovery lever. See `vibeharness/web.py` (`SessionState`, `PlaywrightCli.run/
> _resume/_run_once`, `OpenBrowserTool`) and `tests/test_web.py`. NOTE on root cause:
> live experiments on Windows showed the per-command `taskkill /F /T` does NOT in
> fact reach the detached/unref'd daemon (it survives every per-command tree-kill),
> so the dominant real-world trigger is the SHARED `vibe` session being stopped by a
> concurrent run's teardown/atexit (or an external reaper) — the self-healing resume
> makes the session survive ANY of these causes regardless.

**Goal:** explain WHY the `@playwright/cli` daemon dies during sustained automation,
and design a CLEAN RESUME mechanism so a run that loses its daemon re-opens and
re-navigates instead of stalling forever in the dead-end "browser 'vibe' is not
open" loop.

Measured against `@playwright/cli` (`playwright-cli`), Chrome, headed, session
`vibe`, `web_cli_timeout = 90` (`vibeharness/config.py:97-98`).

---

## TL;DR

- **The error reproduces in BOTH real runs.** `C:\git\vibetestruns\jobapp-vibe3b\run.out`
  (line **658-662**) and `...\jobapp-vibe3b-v2\run.out` (line **301-307**) both end
  in the same dead-end: a burst of `fill`/`click` actions that every one fails with
  `The browser 'vibe' is not open, please run open first`. Crucially the daemon was
  **alive earlier in the same run** — in v1 a `click e81` (Continue →) **succeeded**
  at line **441**, then the next action batch (line 658+) all fail "not open". The
  browser died *mid-run*, between two action batches.

- **#1 cause is self-inflicted, not an external Chrome crash.** Our own per-command
  timeout reaper (`PlaywrightCli._kill_tree`, `web.py:177-202`) runs
  `taskkill /F /T /PID <cli-pid>` on the process **tree** of a timed-out CLI command.
  In `@playwright/cli`'s **client-daemon architecture** (confirmed below) the
  persistent daemon *is* the browser owner; if it sits in the tree of the killed
  client (true for the first `open`, whose handle we deliberately retain in
  `_last_proc`), the tree-kill takes the daemon **and the shared browser** down with
  it, leaving every later command to hit "not open". This is the #101 hypothesis and
  the evidence supports it as the dominant mode.

- **Recovery is impossible today by construction.** Nothing in `web.py` ever calls
  `open` again after `WebToolset.setup` (`web.py:717-733`). Once the daemon is gone,
  `goto`/`click`/`fill`/`snapshot` all fail identically forever; the validator just
  keeps telling the 3B agent to "fill e41", the agent keeps trying, and the run
  burns its whole budget in a loop (visible in both `run.out` validator sections).

- **Recommended fix: a self-healing `run()` wrapper.** Detect daemon death from the
  "not open" / target-closed signature, transparently re-`open` the session,
  re-`goto` the last known URL, then **retry the original command once**. Hook it in
  `PlaywrightCli.run` (so snapshots, guards and every discrete tool inherit it) with
  the last-URL state tracked on the wrapper. This composes cleanly with #75's
  `open_browser` tool and #101's "don't co-kill the daemon" fix.

---

## PART A — How the daemon actually works (ground-truth)

`@playwright/cli` uses a **client–daemon architecture**. Per the official docs and
multiple third-party deep-dives, when you run `playwright-cli -s=vibe click e21`:

1. the CLI process parses the command and picks the target session;
2. the command travels **over a socket to a background daemon process**;
3. the **daemon** resolves the ref and drives the browser.

> "The command travels over a Unix socket to a background daemon process… The daemon
> stays running between commands, so the browser does not relaunch every time."
> — [TestDino: Playwright CLI](https://testdino.com/blog/playwright-cli)

> "Daemon architecture — persistent browser process means no startup cost per command."
> — [Playwright CLI — Introduction](https://playwright.dev/agent-cli/introduction)

Key consequences for us:

- **The browser is owned by ONE long-lived daemon per session name** (`-s=vibe`), NOT
  by the short-lived per-command CLI client. Each of our tool calls spawns a *new*
  CLI client (`PlaywrightCli.run` → `subprocess.Popen`, `web.py:150`). They are all
  thin clients onto the same daemon-held browser. (Our code reflects this: every
  `_WebTool` and both snapshot providers construct their **own** `PlaywrightCli`
  bound to the same `config.web_session`, `web.py:656,674,701`.)

- **Management/health commands exist and matter for our design:** `playwright-cli list`
  (enumerate live sessions — a real health check), `close-all` (graceful), and
  `kill-all` (force-kill zombies). — [Playwright — Sessions & Dashboard](https://playwright.dev/agent-cli/sessions),
  [microsoft/playwright-cli session-management.md](https://github.com/microsoft/playwright-cli/blob/main/skills/playwright-cli/references/session-management.md)

- **`--persistent` saves the profile to disk** so cookies/storage survive a browser
  restart; without it (our default) a restart loses in-memory profile state. Relevant
  to what "clean resume" can and cannot restore (Part D).
  — [Playwright CLI session management](https://knightli.com/en/2026/04/15/playwright-cli-session-management/)

The docs do **not** publish a precise idle/keep-alive timeout or the exact
daemon↔client parent relationship; those gaps are called out as risks in Part B.

---

## PART B — Crash causes, ranked, with evidence

### Rank 1 — SELF-INFLICTED: our timeout reaper co-kills the shared daemon (the #101 cause)

**This is the dominant cause in our evidence.** Mechanism:

- Each CLI command is hard-bounded by `web_cli_timeout = 90s`. On `TimeoutExpired`,
  `run()` calls `self._kill_tree(proc)` (`web.py:156-159`), which on Windows runs
  `taskkill /F /T /PID <pid>` — a **tree** kill (`web.py:187-189`).
- The persistent daemon (browser owner) is **not** logically part of any single
  command; but on Windows it can sit in the process tree descended from the first
  `open` invocation, whose `Popen` handle we *intentionally keep alive* in
  `_last_proc` for exactly this tree-kill (`web.py:152-153,225-227`). A `/T` kill of
  that tree — or a snapshot/command spawned under the same group — drags the daemon
  and the shared Chrome down with it.
- The snapshot is captured **every single turn** via `cli.run("snapshot")`
  (`web.py:619,640`), so the surface area for "a CLI command times out and gets
  tree-killed" is large: it is not just the agent's explicit actions, it's also two
  auto-snapshot calls per turn, each able to trip the 90 s reaper on a heavy page.

**Why the evidence points here and not at Chrome:**

- In `jobapp-vibe3b\run.out` the daemon demonstrably **worked**: `goto` (line 155),
  `click e81` **succeeded** (line **441**), file reads ran (443-445) — then the very
  next action batch (`fill e41/e44/e48`, line **658-662**) all fail "not open". A
  genuine renderer/OOM crash typically surfaces as `Target page, context or browser
  has been closed` from Playwright; our log shows the **CLI's own** "not open, please
  run `open` first" message, i.e. the *daemon/session is gone entirely*, which is the
  signature of the **session being killed**, not a page crashing under a live daemon.
- The failure is **clustered and total** (every subsequent command fails identically
  and permanently), consistent with the single shared daemon being terminated once —
  exactly what a `/T` tree-kill of the daemon produces.

> On Windows, when the daemon process is killed externally it terminates the
> associated browser instances; killing the daemon affects all sessions it manages.
> — synthesis from [Playwright — Sessions & Dashboard](https://playwright.dev/agent-cli/sessions)
> and [session-management.md](https://github.com/microsoft/playwright-cli/blob/main/skills/playwright-cli/references/session-management.md)

> Compare Playwright's own debate about whether a `TimeoutError` should kill the
> browser process — it is a known footgun to conflate "this command timed out" with
> "tear down the browser." — [microsoft/playwright #2705](https://github.com/microsoft/playwright/issues/2705)

**Self-inflicted sub-variants also in scope:**
- `WebToolset.teardown` / the `atexit` reaper (`web.py:726-727,735-752`) firing early
  if the harness sees a transient `Ctrl-C`/signal — `close()` runs `kill-tree` and
  ends the session for the rest of the run.
- A small code-smell worth flagging to #101: `run()` references `self._kill_grace`
  (`web.py:161`) but it is defined only as a **class attribute** (`web.py:175`); fine
  today, but any instance-level confusion around the reaper path is exactly where a
  premature kill hides.

### Rank 2 — EXTERNAL: browser/renderer crash or OOM under sustained load

Genuine Chrome death is real for long automation and would *also* yield "not open"
(once the daemon notices its browser vanished) or `Target page, context or browser
has been closed`:

- **`/dev/shm` exhaustion / OOM-killed renderer** is the classic long-run killer on
  Linux/containers; mitigated by `--disable-dev-shm-usage`.
  — [WebCrawlerAPI: Target closed after crashes](https://webcrawlerapi.com/glossary/playwright/how-to-fix-playwright-target-closed-crash)
- **Memory pressure from many tabs / heavy pages** — the browser may kill the active
  memory-hungry tab. — [Apify: How to fix 'Target closed'](https://docs.apify.com/academy/node-js/how_to_fix_target-closed)
- Microsoft tracks recurring `Target page, context or browser has been closed`
  reports during automation: [#33515](https://github.com/microsoft/playwright/issues/33515),
  [#36360](https://github.com/microsoft/playwright/issues/36360),
  [#13038](https://github.com/microsoft/playwright/issues/13038);
  the standard remedy is **attach a `disconnected` handler and auto-restart the browser**
  ([Render: headless crash](https://render.discourse.group/t/playwright-headless-crash/36172)).

For our job-app benchmark (a single small local form on `localhost:3000`, headed
Chrome) OOM is **unlikely** to be the primary cause — which is why Rank 1 leads — but
the resume design must handle it anyway because the *symptom is identical*.

### Rank 3 — EXTERNAL: daemon idle / keep-alive timeout

The docs confirm a persistent daemon but do **not** publish an idle TTL. A long
between-command gap (e.g. the 3B model thinking for a long time, or a slow
`read_file` of a multi-page resume — see lines 443-445 right before the failure) could
plausibly let an undocumented idle timeout close the browser. Unverified from docs;
listed as a candidate because the v1 failure occurs **immediately after** a sequence
of non-browser file reads, i.e. a quiet period for the daemon. A health-check-on-resume
(Part C) covers this regardless of the exact TTL.

### Rank 4 — EXTERNAL: version-specific known issues / zombie daemons

`kill-all` exists precisely because stale/zombie daemon processes are a known failure
mode; a wedged daemon answers the socket but no longer drives a live browser, again
surfacing as "not open". — [session-management.md](https://github.com/microsoft/playwright-cli/blob/main/skills/playwright-cli/references/session-management.md)

---

## PART C — Mitigations: keep the daemon healthy + isolated

1. **Don't co-kill the daemon (the core #101 fix).** A per-command timeout must kill
   only *that command's client*, never the shared daemon/browser. Two robust options:
   - Stop retaining/tree-killing the `open` child for normal command timeouts. Kill
     only the specific timed-out client `proc` (and on Windows scope `/T` so it cannot
     ascend to the daemon). Reserve daemon teardown for `teardown()` only.
   - Or replace ad-hoc tree-kills with the CLI's own lifecycle verbs: on timeout do
     **nothing** to the daemon; on teardown use `playwright-cli -s=vibe close`, and
     only fall back to `kill-all` for confirmed zombies.

2. **One daemon per run, not per command (keep current model, isolate it).** The
   shared long-lived daemon is the right design (it is what makes state persist across
   tools). The bug is not "shared daemon"; it is "command-scoped kill reaching a
   run-scoped daemon." Keep one daemon; protect it.

3. **Health-check before declaring death.** Use `playwright-cli list` (or a cheap
   `snapshot`/`list` probe) to distinguish "daemon alive, command failed" from
   "daemon gone." This is the trigger for resume (Part D).

4. **Restart-on-death + `disconnected`-style auto-restart**, the documented remedy for
   genuine crashes (Rank 2/3). Our `run()` wrapper is where this lives.

5. **Reduce crash surface:** consider `--persistent` so a restart can restore profile
   state; for any future headless/container runs add `--disable-dev-shm-usage`.

6. **Lower the auto-snapshot timeout-kill exposure:** snapshots run every turn; if a
   snapshot times out it should NOT be allowed to reap the session (it currently goes
   through the same `run()`/`_kill_tree` path). The Part D wrapper handles this by
   making snapshot capture death-tolerant rather than session-fatal.

---

## PART D — Clean-resume design

**Objective:** when the daemon dies, transparently bring the session back and continue
the agent run, instead of looping on "not open".

### D.1 Where it hooks in

Wrap **`PlaywrightCli.run`** (`web.py:119`). Every path — discrete tools
(`_WebTool.run`, `web.py:284`), target-guard snapshots (`_guard_target`,
`web.py:308`), and both per-turn snapshot providers (`web.py:619,640`) — already funnel
through `PlaywrightCli.run`. Wrapping there gives **one** resume implementation that
the whole toolset inherits, with no change to any tool subclass. This is deliberately
the same seam #75 (`open_browser`) and #101 (timeout fix) touch, so the three compose:
#101 stops the self-kill, #75 gives an explicit re-open verb, and #102 makes `run`
self-heal automatically.

### D.2 State to track on the wrapper

Add to `PlaywrightCli` (run-scoped, since `WebToolset` holds one `self._cli` for the
run — `web.py:719`; note the *tool* instances each hold their own `PlaywrightCli`, so
last-URL state should live on a **shared, run-scoped object**, e.g. the `WebToolset`'s
`_cli`, or a small shared `SessionState` injected into every `PlaywrightCli` — see
D.5):

- `last_url`: updated on every successful `goto` (and observed from snapshot
  `Page URL:` lines, which the CLI already prints — see `run.out` line 441).
- `open_flags`: the `--headed`/`--browser` flags `setup()` used (`web.py:728-733`), so
  a re-open matches the original.
- `resumes`: a counter to bound resume attempts (avoid an infinite re-open loop if the
  environment itself is broken).

### D.3 Detection

A command result indicates **daemon death** (vs an ordinary no-match) when its output
matches a death signature, distinct from the existing `_NO_MATCH_MARKERS`
(`web.py:48-57`):

```
DAEMON_DEAD_MARKERS = (
    "is not open",                                  # CLI: "The browser 'vibe' is not open"
    "please run open first",
    "target page, context or browser has been closed",
    "browser has been closed",
    "session not found",
)
```

Optionally confirm with a `list` probe before resuming, so a flaky single command does
not trigger an unnecessary restart.

### D.4 Resume sequence (inside the `run` wrapper)

On a non-recovery command (i.e. not itself an `open`/`close`), when the result matches a
death marker and `resumes < max_resumes`:

1. **Reap any zombie**, best-effort: `playwright-cli -s=vibe close` (ignore failure);
   escalate to `kill-all` only if a follow-up `list` still shows a wedged session.
2. **Re-open** the session with the saved `open_flags`: `open [--headed] [--browser chrome]`.
3. **Re-navigate** to `last_url` if known: `goto <last_url>`. (Re-establishes the page
   the agent was on; refs are re-derived from the fresh snapshot next turn, which is
   correct because refs are snapshot-scoped anyway — `parse_snapshot_refs`,
   `web.py:60`.)
4. **Restore relevant state** (best-effort, bounded): with `--persistent`, cookies/
   storage return automatically; without it, re-navigation alone restores the page but
   not multi-step in-progress form input. For the job-app benchmark the agent simply
   re-fills on the next turn (the validator already re-issues "fill e41…"), so
   re-`goto` is sufficient to break the dead-end. Document this limit explicitly.
5. **Retry the original command once** and return *its* result to the caller. The tool
   layer above (`_WebTool.run`) then reports a normal success/failure, and the agent
   never sees "not open" — the resume is transparent.
6. Increment `resumes`. If `resumes` is exhausted, fall through and return the original
   "not open" failure (so a genuinely unrecoverable environment still terminates rather
   than spinning).

### D.5 Sketch (illustrative — for #101/#75 to implement, not committed code)

```python
def run(self, *args):
    ok, out = self._run_once(*args)              # current Popen+communicate body
    if args[:1] == ("goto",) and ok:
        self._state.last_url = _parse_page_url(out) or args[1]
    if ok or not self._is_daemon_dead(out):
        return ok, out
    if not self._state.allow_resume():           # bounded, and skip when args is open/close
        return ok, out
    self._resume()                               # close-if-zombie -> open(flags) -> goto(last_url)
    return self._run_once(*args)                 # retry original command once

def _resume(self):
    self._run_once("close")                      # best-effort reap
    self._run_once("open", *self._state.open_flags)
    if self._state.last_url:
        self._run_once("goto", self._state.last_url)
    self._state.resumes += 1
```

`self._state` is a single run-scoped `SessionState` (last_url, open_flags, resumes)
**shared** by every `PlaywrightCli` the run creates (tools + snapshot providers), so a
resume triggered by any tool — or by the per-turn snapshot — heals the session for all
of them. `WebToolset.setup` constructs it and threads it into
`create_tools`/`make_*_snapshot_provider`.

### D.6 Why this composes with #101 and #75

- **#101 (don't co-kill the daemon):** removes the *primary* trigger, so resume becomes
  the rare safety net for genuine Rank-2/3 crashes rather than firing on every run.
- **#75 (`open_browser` tool):** gives the model an explicit re-open affordance;
  `_resume()` reuses the exact same `open` path, and the saved `open_flags` keep
  headed/browser parity. If #75 lands first, `_resume` can call its helper directly.
- **Idempotent + bounded:** capped `resumes` and best-effort `close` mean resume never
  loops and never double-reaps (mirrors the existing idempotent `close()`,
  `web.py:204-228`).

---

## Sources

- [Playwright CLI — Introduction](https://playwright.dev/agent-cli/introduction)
- [Playwright — Sessions & Dashboard](https://playwright.dev/agent-cli/sessions)
- [microsoft/playwright-cli — session-management.md](https://github.com/microsoft/playwright-cli/blob/main/skills/playwright-cli/references/session-management.md)
- [TestDino — Playwright CLI: every command, real benchmarks](https://testdino.com/blog/playwright-cli)
- [Playwright CLI Session Management deep-dive](https://knightli.com/en/2026/04/15/playwright-cli-session-management/)
- [microsoft/playwright #2705 — should Playwright kill the browser on TimeoutError?](https://github.com/microsoft/playwright/issues/2705)
- [microsoft/playwright #33515 / #36360 / #13038 — Target page/context/browser has been closed](https://github.com/microsoft/playwright/issues/33515)
- [WebCrawlerAPI — Target closed after crashes (OOM / dev-shm)](https://webcrawlerapi.com/glossary/playwright/how-to-fix-playwright-target-closed-crash)
- [Apify — How to fix 'Target closed' in Puppeteer/Playwright](https://docs.apify.com/academy/node-js/how_to_fix_target-closed)
- [Render — Playwright headless crash (disconnected handler + restart)](https://render.discourse.group/t/playwright-headless-crash/36172)

Evidence files (local): `C:\git\vibetestruns\jobapp-vibe3b\run.out` (lines 441, 658-662),
`C:\git\vibetestruns\jobapp-vibe3b-v2\run.out` (lines 301-307);
`vibeharness/web.py` (`PlaywrightCli.run` 119-166, `_kill_tree` 177-202, `close` 204-228,
`WebToolset.setup/teardown` 717-752).
