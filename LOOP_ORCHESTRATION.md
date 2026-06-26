# Autonomous Optimisation Loop — Orchestration Brief

**Status file:** `loop_state.json` (same dir) — read and update every iteration.
**Branch:** `beta_qwen3coder`
**Goal:** Perfect score on the job-application flow (26 form fields + page advances).
**Run command:** `vibe --agent web --advisor --max-steps 0 --task-file task_iter4.txt`
**No user available** — operate fully autonomously, no clarification requests.

---

## Prerequisites before first run

### In-flight agents to wait for

| Agent | PR | What it does |
|---|---|---|
| a9f23524 | Check GitHub for PR against beta_qwen3coder | Full native Ollama rewrite + stateful chat history (#129/#130/#131) |
| a5bb01e2 | Check GitHub for PR against beta_qwen3coder | HF chat-template compliance test (#133) |

Check: `GH_TOKEN=$GH_TOKEN gh pr list --base beta_qwen3coder --repo NickalasLight/VibeHarness --state open`

If PRs are open and passing: merge them (review first), then rebuild.  
If PRs are still not open: wait another 5 minutes.

### Merge + rebuild sequence (once PRs are ready)

```bash
cd /c/git/vibethinkharnessProto1

# Merge each PR
GH_TOKEN=... gh pr merge <N> --merge --repo NickalasLight/VibeHarness
git pull origin beta_qwen3coder

# Rebuild
pip install -e . -q

# Verify
python -c "from vibeharness.web import _WEB_TOOL_CLASSES; print([t.name for t in _WEB_TOOL_CLASSES])"
```

---

## Every iteration — strict sequence

### 1. VRAM clear (BEFORE every run)
```bash
ollama stop vibethinker:latest 2>/dev/null || true
ollama stop qwen2.5-coder:3b-instruct 2>/dev/null || true
# Wait for eviction
sleep 5
# Verify ~5 MiB idle target
```

### 2. Run vibe (background so we can schedule wakeup)
```bash
cd /c/git/vibethinkharnessProto1
vibe --agent web --advisor --max-steps 0 --task-file task_iter4.txt 2>&1 | tee /tmp/vibe_run_$(date +%s).log
```
Run in FOREGROUND within the Opus subagent (subagent blocks until done — this is fine, main loop uses ScheduleWakeup timer separately).

### 3. Score the run
Parse the most recent `.vibe/*.json` log:
- Count `"ok": true` actions where tool is `fill`/`type`/`select_option`/`check` with correct values matching task_iter4.txt expected values
- Each correct unique field: +1 point (max 26)
- Each page successfully advanced (Next/Continue clicked successfully): +10 points
- Read `"finished"` field: if true, maximum bonus

### 4. VRAM clear AFTER run
```bash
ollama stop vibethinker:latest 2>/dev/null || true
ollama stop qwen2.5-coder:3b-instruct 2>/dev/null || true
```

### 5. Analyse failures
Read the run log carefully. Identify:
- Which fields were NOT filled or filled incorrectly
- Which tools failed and why (error message)
- Loops detected (same tool+target appearing 3+ times)
- DOM issues (hidden elements, dropdowns not activated)
- VibeThinker advisor quality (did advice help or hurt?)

### 6. VibeThinker performance check
If VibeThinker advisor calls are slow (>60s each) OR the advice appears to cause loops/confusion:
- Set `advisor_enabled: bool = False` in `vibeharness/config.py` as the DEFAULT
- Or pass `--no-advisor` flag (if implemented)
- Update `loop_state.json` → `"vibethinker_enabled": false`
- Note this in the commit message

### 7. Research optimisations (Opus should do this)
Search online for:
- Qwen2.5-Coder agentic tool-calling best practices (English + Mandarin sources: 知乎, CSDN, Qwen blog at qwenlm.github.io)
- Specific Ollama + Qwen2.5 configuration tips
- Multi-turn tool-calling prompt engineering for small 3B models
- Any known issues with qwen2.5-coder:3b-instruct tool calling in Ollama

### 8. Implement optimisations
Edit relevant files (hermes_codec.py, config.py, advisor.py, agent.py, web.py, prompt.py).
Focus on whatever caused the most failures in the run.

### 9. Commit with score + run analysis (REQUIRED)

Every commit MUST include a structured run analysis block in the commit body. This is mandatory policy — do not skip it.

```bash
git add -A
git commit -m "opt(iter-N): score=XX/26+Yp — <one-line description of what changed>

## Run analysis
Score: XX fields correct, Y page advances (pages A→B→C reached)
Previous score: ZZ. Delta: +NN.

### Flow summary
Turn 1-3: [what happened — fields filled, tools called]
Turn 4-N: [where it stalled — element ref, error message, loop pattern]
Page advance blocked by: [specific validation error / missing field / widget type]
Final state: [page N, stuck on element eXX, widget type]

### Failures discovered
- eXX (FieldName): [what went wrong — fill on combobox / calendar not navigated / loop / error]
- eYY (FieldName): [root cause]

### Loops detected
- [tool]+[target] repeated N times: [why — anti-loop guard state / validation block]

### What was fixed this iteration
- [file:line] — [what changed and expected impact]

### Remaining blockers
- [widget/field]: [why not yet fixed]

VibeThinker: enabled/disabled.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### 10. File bug issues for systematic problems
For any non-trivial bug found (not a config tweak, but an architectural gap):
```bash
GH_TOKEN=$GH_TOKEN gh issue create \
  --title "Bug: <description>" --label "bug" --body "..."
```

---

## Stopping condition

Stop iterating when:
- `loop_state.json` → `"finished": true` (the run completed the full application)
- OR score = 26+ (all fields correctly filled + form submitted)

---

## Key file locations

- Task file: `task_iter4.txt` (contains all candidate info)
- Run logs: `.vibe/*.json` (most recent = last run)
- Config: `vibeharness/config.py`
- Hermes codec: `vibeharness/codecs/hermes_codec.py`
- Web tools: `vibeharness/web.py`
- Advisor: `vibeharness/advisor.py`
- Agent loop: `vibeharness/agent.py`

## GH credentials
Set `GH_TOKEN` in your environment (do not commit the token).

## HF token (for research)
Set `HF_TOKEN` in your environment (do not commit the token).
