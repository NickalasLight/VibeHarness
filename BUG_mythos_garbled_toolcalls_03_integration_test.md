# BUG (mythos #3/3 — INTEGRATION TEST): real-world web run on mythos_fast

- **Branch:** `beta_mythos_fast` (ISOLATED — never merges back to `beta`).
- **Type:** integration / real-world end-to-end verification.
- **Status:** BLOCKED — do not start until `BUG_mythos_garbled_toolcalls_02_fix.md`
  is marked **COMPLETE**.
- **Depends on:** `BUG_mythos_garbled_toolcalls_02_fix.md` (the bug-fix implementation).

---

## Goal

Prove the fix works against a real browser task end-to-end with the actual fine-tune.

## The test

Run the harness with the **web worker (web toolset / `--agent web`)** and this exact task:

```
go to youtube and search for cat videos and choose the best one and play it
```

Suggested invocation (adjust per the fix's final defaults):
```
vibe --agent web "go to youtube and search for cat videos and choose the best one and play it"
```

## Required conditions (all must hold)
1. **Web worker is used** — the run drives a real browser via the web toolset (Playwright
   CLI / the run-scoped browser session), not a filesystem-only run.
2. **Prompts updated & in proper syntax** — the system prompt + tool definitions are the
   native-aligned ones from the fix (`<tools>` bare schemas, `<tool_call>{"name","arguments"}`
   instructions). Verify via `vibe --print-system --agent web` before the run.
3. **No constraint / encoding secondary step** — IF the fix item determined the bolted-on
   constraint/encoding pass to be the cause, confirm it is OFF for this run (the action
   phase is unconstrained; the model speaks its native dialect end-to-end).

## Acceptance
- [ ] The model emits **clean, parseable `<tool_call>` blocks** throughout the run
      (no garbling) — capture the `.vibe/` run log as evidence.
- [ ] The agent actually navigates YouTube, searches "cat videos", selects a result, and
      initiates playback (or fails for a non-format reason, documented honestly).
- [ ] Tool-call format adherence and parse-success recorded from the transcript.
- [ ] Result (pass/fail + evidence paths) recorded in this file; status flipped to DONE.

## Environment prerequisites (record actual state honestly)
- Ollama running with `hf.co/Shadow0482/mythos_fast:Q6_K` pulled (`ollama ps`).
- `playwright-cli` installed (`npm install -g @playwright/cli@latest`).
- If the environment cannot run a live model/browser, record that explicitly rather than
  fabricating a result, and provide the exact reproduction command for a later run.
