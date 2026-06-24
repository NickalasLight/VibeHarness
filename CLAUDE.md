# CLAUDE.md — How we track and execute work in this repo

This tells any Claude agent (and humans) how work is planned, tracked, and shipped in
**VibeHarness**. Read it before starting a task.

> Repo-wide process doc — kept identical on `main`, `beta`, and `beta_mythos_fast`.

## TL;DR
- **GitHub Issues are the single source of truth.** Use the `gh` CLI. Every feature, bug,
  analysis, and test is a GitHub issue.
- **Work is staged across dependent issues:** Analysis → Development (fix/feature) →
  Testing / Re-verification — each its own issue, chained by explicit dependency references.
- **One branch (or git worktree) per issue.** A PR links to and closes its issue and is
  **reviewed by the main (orchestrator) agent** before merge.
- **The main agent orchestrates; subagents execute.** The orchestrator files & chains
  issues, dispatches subagents, enforces stage gating (don't start a stage until its
  prerequisite issue/PR is merged), and reviews PRs. It implements little directly.

## 1. Issue tracking (GitHub Issues + `gh`)
- Labels in use: `bug`, `enhancement`, `documentation`, `question`, `duplicate`, `wontfix`,
  `invalid`, `help wanted`, `good first issue`.
- Browse / create:
  - `gh issue list --state all` (or `--state open|closed`)
  - `gh issue view <n>`
  - `gh issue create --title "…" --label <label> --body "…"`
- Body convention (match existing issues): open with `## <Type>: <Title>`, then context, a
  spec / ground-truth section, and an **acceptance checklist** (`- [ ]`). Cross-reference
  related work inline as `#NN` / `(PR #MM)`.

## 2. Staged, dependency-chained issues
Stages are tracked as **separate** issues that reference each other. Title patterns actually
used in this repo:

| Stage | Title pattern | Examples |
|---|---|---|
| Analysis | `Analysis: …` / `Analysis FIRST: …` | #76, #71, #61 |
| Fix / Development | `Fix: … (Dependent on #NN)` | #77 (dependent on #76) |
| Feature | `Feature: …` | #22, #24 |
| Test / Re-verify | `Dependent test: …`, `Re-verify (live): …`, `Test: …` | #89, #58–#60, #21 |
| Follow-up | `Follow-up (#NN): …` | #92, #99, #100, #113 |

- Express the dependency in the body: **`Dependent on #NN`** / `gated on #NN` /
  `(depends on the analysis)`. A dependent stage does NOT start until the prerequisite
  issue (and its PR) is complete and merged.
- Worked chains: **#76** (analysis: leaked `llama-server` runners) → **#77** (fix,
  *Dependent on #76*). And **#105** (align prompts/schema to mythos_fast) → **#106**
  (cat-search YouTube test, *gated on #105*) → **#116** (validate & verify alignment).

## 3. Branches, worktrees & commits
- **One branch / worktree per issue.** Naming (from the live branch set):
  `feat/<slug>`, `fix/<slug>`, `research/<slug>`, `chore/<slug>`, `test/<slug>`,
  `task/<slug>`, `sync/<slug>`, `bug/<slug>`.
- Commit style: `area(#NN): summary` — e.g. `fix(config): rebalance reason/action token
  reservation (#92)`; merges recorded as `Merge #NN (PR #MM): …`.
- End commit messages with the required `Co-Authored-By` trailer.

## 4. PRs (tied to issues, orchestrator-reviewed)
- Open with `gh pr create --base <branch> --title "…" --body "…"`. Put `Closes #NN`
  (or `Fixes #NN`) in the body so the issue auto-closes on merge.
- The **main / orchestrator agent reviews every PR** before merge
  (`gh pr view`, `gh pr diff`, `gh pr review`).
- For `beta_mythos_fast` PRs, include the `DIVERGENCE-REVIEWED` token (see §6).

## 5. Orchestrator ↔ subagent contract
**Orchestrator (main agent):** files & chains issues; dispatches subagents; enforces stage
gating; reviews PRs; merges.
**Subagent:** works ONE issue on its tied branch; opens a PR that links the issue; reports
back for review.
**Every subagent dispatch MUST state:**
- the **issue number** it serves and the **stage** (analysis / fix / test);
- the **tied branch / worktree**, and that it must not switch branches;
- its **dependencies / what gates it**;
- the **deliverable** + acceptance criteria from the issue;
- "**open/link a PR with `Closes #NN`**; the orchestrator will review";
- **protected-file rules** (§6) and the commit / PR conventions above.

## 6. Branch governance
- `main` ← `beta` ← feature/fix branches. CODEOWNERS enforces review on protected branches;
  CI runs via `.github/workflows/ci.yml`.
- **`beta_mythos_fast` is an isolated line for the mythos_fast fine-tune.** Sync is
  **one-way: `beta → beta_mythos_fast` only — never the reverse.** Preserve the protected
  files listed in `MYTHOS_DIVERGENCE.md` (codec, the `<tools>` seam, config defaults, chat
  template). Prefer the `mythos-sync` agent (`.claude/agents/mythos-sync.md`). PRs into this
  branch must include `DIVERGENCE-REVIEWED` (enforced by
  `.github/workflows/mythos-divergence-check.yml`).

## 7. `gh` setup
- Install: `scoop install gh` (or winget / choco).
- Auth: prefer a dedicated token — a classic PAT with `repo` + `read:org`, or a fine-grained
  PAT with Issues / Pull-requests / Contents read-write — and run `gh auth login` once. If
  only the Git Credential Manager token is available, it works for issue/PR API calls via
  `GH_TOKEN` but lacks `read:org` for a full `gh auth login`. Setup is tracked in **#121**.

## The loop (end to end)
1. Orchestrator files the issue(s), staged & chained.
2. Create each issue's branch / worktree.
3. Dispatch a subagent with the full briefing (§5); gate dependent stages.
4. Subagent implements + opens a PR (`Closes #NN`).
5. Orchestrator reviews the PR; iterate; merge.
6. The dependent stage (test / re-verify) proceeds once its prerequisite is merged.
