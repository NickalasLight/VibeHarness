# CLAUDE.md — How we track and execute work in this repo

This document is the **standard operating procedure** for issue tracking and agent
orchestration in VibeHarness. A new Claude agent should be able to follow it with no prior
context. Read it before creating issues, dispatching subagents, or opening PRs.

> **Process doc is repo-wide.** This file currently lives on `beta_mythos_fast`. It SHOULD
> also be landed on `beta` and `main` so every branch carries the same methodology. If you
> are an agent on `beta`/`main` and don't see this file, port it over (orchestrator's call).

---

## TL;DR

- **GitHub issues are the source of truth** for every unit of work — features *and* bugs.
- **One branch/worktree per issue.** Branch name encodes the kind of work; commits cite the
  issue: `area(#NN): summary`.
- **Work is stage-separated and dependency-chained:** *analysis → development → testing* are
  OFTEN three separate, gated issues. The later stage does not start until the earlier
  stage's issue/PR is **complete and reviewed**.
- **Subagents do the implementation; the main agent orchestrates.** The orchestrator creates
  and chains issues, dispatches subagents, enforces gating, and **reviews every PR** before
  merge.
- **Every subagent dispatch MUST carry its issue/branch/dependency/PR context** (see the
  copy-pasteable briefing checklist below).
- **Tooling:** `gh` (GitHub Issues + PRs) is the primary workflow. When `gh`/`git-bug` are
  absent, the on-repo fallback is committed `BUG_*.md` files with `Depends on:` / `Gates:`
  headers that encode the same chain.

---

## 1. Issue-tracking model

| Principle | Detail |
|---|---|
| **Source of truth** | A GitHub issue per unit of work (feature or bug). Issue numbers (`#105`, `#107`, `#118`, …) are referenced throughout commits, branches, and `BUG_*.md` files. |
| **One branch/worktree per issue** | Each issue gets a dedicated branch (and often a `git worktree`) so units of work stay isolated and parallelizable. |
| **Commit style** | `area(#NN): summary` — e.g. `docs(#102): mark clean-resume design as implemented`. Cite the issue number in the commit so history links back to the tracker. |

### Branch naming conventions

Inferred from the live branch list — pick the prefix by the *kind* of work:

| Prefix | Use for | Example |
|---|---|---|
| `feat/*`   | new features | `feat/snapshot-prose`, `feat/raw-chatturn-log` |
| `fix/*`    | bug fixes | `fix/vibe-log-missing`, `fix/cli-vibe-log` |
| `research/*` | investigation / analysis spikes | `research/cookie-consent`, `research/snapshot-to-nl` |
| `chore/*`  | maintenance, audits | `chore/completion-audit`, `chore/snapshot-size-analysis` |
| `test/*`   | test / benchmark runs | `test/ashley-benchmark`, `test/cookie-evidence` |
| `task/*`   | scoped one-off tasks | `task/hf-token-login` |
| `bug/*`    | bug-tracking branches (alt to `fix/*`) | `bug/<slug>` |
| `sync/*`   | cross-branch reconciliation | `sync/beta-into-mythos` |

> Worktrees are referenced by a short directory slug (e.g. worktree `vh-vibe-log-bug` for
> branch `fix/vibe-log-missing`). Keep the worktree slug discoverable from the issue/`BUG_*`
> file's **Branch / work item** header.

---

## 2. Stage-separated, dependency-chained issues

Non-trivial work is split into **separate, gated issues** by stage:

```
ANALYSIS  ──gates──▶  DEVELOPMENT (fix/feature)  ──gates──▶  TESTING
(root cause,          (implement per the              (real-world / integration
 recommendations)      analysis's findings)            verification)
```

**Gating rule:** a stage's issue MUST NOT start until the prerequisite stage's issue/PR is
marked **COMPLETE** and reviewed. Analysis gates development; development gates testing.

### How dependencies are expressed

- **On GitHub:** use issue links / "Depends on #NN" / "Blocked by #NN" in the issue body,
  and keep the dependent issue in a *blocked* state until the prerequisite closes.
- **In `BUG_*.md` fallback files:** explicit headers at the top of each file:
  - `Depends on:` — the prerequisite item (must be COMPLETE first).
  - `Gates:` — the item this one unblocks.
  - `Status:` — e.g. `COMPLETE`, `BLOCKED — do not start until <prereq> is COMPLETE`.

### Worked example — the mythos garbled-tool-calls trio (canonical)

Three chained files, one stage each, all on `beta_mythos_fast`:

| File | Stage | Status header | Chain |
|---|---|---|---|
| `BUG_mythos_garbled_toolcalls_01_analysis.md` | Analysis | `COMPLETE` | `Depends on: nothing` · `Gates: …_02_fix.md` |
| `BUG_mythos_garbled_toolcalls_02_fix.md` | Development | `BLOCKED — do not start until …_01 is COMPLETE` | `Depends on: …_01_analysis.md` · `Gates: …_03_integration_test.md` |
| `BUG_mythos_garbled_toolcalls_03_integration_test.md` | Testing | `BLOCKED — do not start until …_02 is COMPLETE` | `Depends on: …_02_fix.md` |

The analysis item produced a `## FINDINGS` + `## RECOMMENDATIONS` section and was flipped to
`COMPLETE` (analysis-only — **no production code changes**); only then could the fix item
begin, and only then the integration test. Each file states **Deliverable / Acceptance**
checkboxes that the assigned subagent must satisfy before the gate opens.

---

## 3. Orchestrator ↔ subagent contract

### The main (orchestrator) agent does:

- **Creates and chains the issues** (analysis → development → testing) with explicit
  dependency/gating headers.
- **Dispatches subagents** to issues — one subagent analyzes, a *separate* subagent
  implements, a *separate* subagent tests, each on the issue's tied branch/worktree.
- **Enforces gating** — does NOT dispatch a dependent stage until the prerequisite stage's
  issue/PR is complete and reviewed.
- **Reviews every PR** tied to an issue **before merge**. The orchestrator is the reviewer.
- **Ensures every subagent prompt carries full tracking context** (issue #, tied branch,
  dependency/gate, PR expectation, protected-file rules) — this is mandatory, not optional.
- Does **little implementation directly** — orchestration is the job.

### A subagent does:

- Works **only** on its assigned issue, on the **tied branch/worktree** — does not wander.
- Respects its **dependency**: if the prerequisite is not COMPLETE, it stops and reports.
- Produces the **deliverable** in the issue's acceptance criteria; updates the issue/`BUG_*`
  file status when done.
- **Opens or updates a PR linked to the issue** (`Fixes #NN` / `Refs #NN`) and signals that
  it is ready for the orchestrator's review.
- Respects **protected files** and branch governance (Section 5).

### The gating rule (one line)

> Do not start a dependent stage until the prerequisite stage's issue **and** PR are complete
> **and** the orchestrator has reviewed it.

---

## 4. Copy-pasteable subagent-briefing checklist

The orchestrator MUST include this block (filled in) in **every** subagent dispatch:

```
SUBAGENT BRIEFING — issue tracking in this repo
- Issue: #NN — <one-line title>
- Stage: <analysis | development | testing>
- Tied branch / worktree: <feat|fix|research|chore|test|task|bug>/<slug>  (worktree: <slug>)
- Depends on: #MM — DO NOT START unless it is COMPLETE & reviewed. If not, stop and report.
- Gates: #PP — your completion unblocks this; flip status when your acceptance criteria pass.
- Deliverable: <exact acceptance criteria / checkboxes from the issue or BUG_*.md file>
- Open/link a PR: open or update a PR for this work; reference the issue (Fixes/Refs #NN).
- Orchestrator reviews: the main agent reviews your PR before merge — do NOT self-merge.
- Tracking fallback: if gh/git-bug are unavailable, update the matching BUG_*.md file
  (Status / Depends on / Gates headers) and commit it alongside your change.
- Protected-file rules: read MYTHOS_DIVERGENCE.md. If this is a beta_mythos_fast PR, do not
  clobber protected files and include the token DIVERGENCE-REVIEWED in the PR body.
- Commit style: area(#NN): summary
```

---

## 5. Tooling & on-repo fallback

| Workflow | Primary (preferred) | Fallback (when tooling absent) |
|---|---|---|
| Issue tracking | **GitHub Issues** | committed **`BUG_*.md`** files at repo root |
| PRs / review | **`gh` CLI** (`gh pr create`, `gh pr view`) | PR opened later via web; track intent in the `BUG_*.md` file |
| Dependency chain | issue links / "Depends on #NN" | `Depends on:` / `Gates:` headers in `BUG_*.md` |

> **Environment reality:** `gh` and `git-bug` are **not currently installed** in this
> environment. The repo therefore uses committed `BUG_*.md` markdown files as the on-repo
> fallback issue tracker. `gh` (GitHub CLI) is a **prerequisite** for the full
> issue+PR workflow — install it to use the primary path.

**Fallback file examples (real, in this repo):**

- `BUG_vibe_log_missing.md` — single-item bug doc with `Branch / work item`, `Severity`,
  `Status`, root-cause evidence, `Acceptance criteria`, `Affected files`.
- `BUG_mythos_garbled_toolcalls_0{1,2,3}_*.md` — the stage-separated, dependency-chained
  trio (Section 2), each with `Depends on:` / `Gates:` / `Status:` headers.

Each `BUG_*.md` should be updated **in place** as the work progresses (flip `Status`, fill in
`FINDINGS`/`RECOMMENDATIONS`, check acceptance boxes) and committed with the change.

---

## 6. Branch governance

### Code-owner review / branch protection
- `.github/CODEOWNERS` makes `@NickalasLight` the owner of `*`, so **every change requires a
  code-owner review** (branch protection). The orchestrator's PR review is part of this gate;
  human owner approval is still required to merge.

### CI gate
- `.github/workflows/ci.yml` runs the test suite (`python -m unittest discover -s tests`) on
  PRs into `main` across Python 3.10–3.12. Keep the suite green; subagents run
  `python -m pytest -q --ignore=tests/integration` (or unittest) locally before requesting
  review.

### `beta` ↔ `beta_mythos_fast` one-way divergence
`beta_mythos_fast` is a **long-term isolated branch** for the `mythos_fast` fine-tune. Its
divergences from `beta` are **intentional and load-bearing**; a blind merge would defeat the
fine-tune. Governance:

- **One-way sync only:** `beta → beta_mythos_fast`. **Never** merge/cherry-pick/push
  `beta_mythos_fast → beta`.
- **Never blind-merge** (`git merge beta`). Apply each beta fix feature-by-feature,
  preserving protected files. Prefer the **`mythos-sync` agent**
  (`.claude/agents/mythos-sync.md`).
- **Protected files** (keep the mythos version; integrate logic, don't overwrite) are listed
  in **`MYTHOS_DIVERGENCE.md`** — primarily `vibeharness/codecs/hermes_codec.py`, the
  `<tools>` rendering seam in `registry.py` / `prompt.py` / `codec.py`, and the `model` /
  `codec` defaults in `config.py`.
- **Every PR into `beta_mythos_fast`** must confirm divergence review by including the token
  **`DIVERGENCE-REVIEWED`** in the PR body — enforced by
  `.github/workflows/mythos-divergence-check.yml`.

> **Subagent mandate:** any subagent touching `beta_mythos_fast` must read
> `MYTHOS_DIVERGENCE.md`, leave protected files' mythos format intact, and put
> `DIVERGENCE-REVIEWED` in the PR body. If a beta change conflicts with a protected file,
> STOP and escalate to the orchestrator — do not silently alter the mythos format.

---

## Quick reference — the loop

1. Orchestrator creates the issue(s); for non-trivial work, split into
   **analysis → development → testing** with `Depends on:` / `Gates:` chains.
2. Create the tied branch/worktree (`feat|fix|research|chore|test|task|bug/<slug>`).
3. Dispatch a subagent with the **full briefing block** (Section 4).
4. Subagent works on the tied branch, satisfies acceptance criteria, opens a PR
   (`Fixes #NN`), updates the `BUG_*.md` fallback if `gh` is unavailable.
5. **Orchestrator reviews the PR.** Code-owner approval + green CI required.
6. Merge; only then unblock the next stage in the chain.
