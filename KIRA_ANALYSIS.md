# Kira (Terminus-KIRA) Prompt & Tool-Definition Analysis

Research target: the **"Kira"** agent in the **"Terminus 2"** harness family.

## What it is, and where it lives

"Kira" is **Terminus-KIRA**, an agent harness from **KRAFTON AI** that reached 74.8% on
Terminal-Bench. It is a thin subclass of **Terminus 2** (Harbor's reference terminal agent)
that swaps Terminus 2's in-context-learning (ICL) JSON/XML response *parsing* for **native LLM
tool calling**.

Source files read (verbatim, via `raw.githubusercontent.com` / GitHub API):

- System prompt: `krafton-ai/KIRA` → `prompt-templates/terminus-kira.txt`
  https://raw.githubusercontent.com/krafton-ai/KIRA/main/prompt-templates/terminus-kira.txt
- Agent + tool definitions: `krafton-ai/KIRA` → `terminus_kira/terminus_kira.py`
  https://raw.githubusercontent.com/krafton-ai/KIRA/main/terminus_kira/terminus_kira.py
- Baseline for comparison — Terminus base prompt:
  `harbor-framework/terminal-bench` → `terminal_bench/agents/prompt-templates/terminus.txt`
  https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/terminal_bench/agents/prompt-templates/terminus.txt
- Terminus 2 itself now lives in the `harbor` package (`harbor.agents.terminus_2`), which
  `terminus_kira.py` imports (`from harbor.agents.terminus_2 import Terminus2`).

Background references (corroborating, not quoted as prompt text):
- KRAFTON blog "How We Reached 74.8% on terminal-bench with Terminus-KIRA"
  https://krafton-ai.github.io/blog/terminus_kira_en/
- Harbor docs, Terminus 2: https://www.harborframework.com/docs/agents/terminus-2

> Note: `krafton-ai/KIRA` is a multi-product repo; "Terminus-KIRA" is the
> `terminus_kira/` + `prompt-templates/terminus-kira.txt` part (distinct from the unrelated
> `KIRA-Slack/` product in the same repo).

---

## The entire Kira system prompt (verbatim — it really is this short)

```
You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Your plan MUST account that you as an AI agent must complete the entire task without any human intervention, and you should NOT expect any human interventions. Also, you do NOT have eyes or ears, so you MUST resort to various programmatic/AI tools to understand multimedia files.

Before calling task_complete, verify minimal state changes: Re-read the task instructions carefully and identify the absolute minimum set of files that must be created or modified to satisfy the requirements. List these files explicitly. Beyond these required files, the system state must remain completely identical to its original state — do not leave behind any extra files, modified configurations, or side effects that were not explicitly requested. Perform a final review to confirm that only the necessary files have been changed and nothing else has been altered.

Task Description:
{instruction}

Current terminal state:
{terminal_state}
```

That is the **complete** prompt — four short paragraphs and two template slots. Compare with
the base Terminus prompt (`terminus.txt`), which inlines `{response_schema}` and spends a long
paragraph teaching tmux keystroke mechanics. Kira's prompt is dramatically leaner.

## Design decisions — what makes it token-efficient

1. **The action schema is NOT in the prompt.** The base Terminus prompt says
   *"Your response must be a JSON object that matches this schema: {response_schema}"* and
   pastes the schema in. Kira **deletes that entirely** and instead passes the schema as the
   API `tools` parameter (native tool calling). The structure is enforced by the
   decoder/provider, not by re-printing it as prose. This is the single biggest token win and
   the core architectural change of Kira over Terminus 2.

2. **Tool docs live in the tool definitions, not the system prompt.** All "how to use a tool"
   text sits in each tool's `description` field (`_KEYSTROKES_DESC`, `_DURATION_DESC`, etc.),
   which travels with the `tools` parameter — not duplicated in the system prompt. One source
   of truth per tool; the system prompt stays about *task posture*, not tool mechanics.

3. **Three tools only, behaviour widened by parameters.** `execute_commands`, `task_complete`,
   `image_read`. `execute_commands` carries `analysis` + `plan` + a `commands[]` array, each
   command being `{keystrokes, duration}`. The agent batches many shell commands per turn (one
   round-trip), and the model is *forced* to think first because `analysis` and `plan` are
   `required` fields of the call — structured chain-of-thought without a prose instruction.

4. **Parameter descriptions are terse, imperative, and example-driven.** e.g. duration:
   *"On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds … Never wait
   longer than 60 seconds; prefer to poll …"*. Concrete examples replace abstract rules; rules
   are stated as commands ("Most bash commands should end with a newline").

5. **Negative scoping to prevent misuse.** `image_read` says
   *"Use this ONLY for image files … Do NOT use this for text files — use shell commands
   (cat, head, etc.) instead."* Cheap one-liners steer the model away from expensive wrong tools.

6. **Completion gating is a separate runtime message, not prompt bloat.** The "are you sure"
   double-confirmation checklist (requirements / robustness to changed values / test-eng + QA +
   user perspectives) is injected by `_get_completion_confirmation_message()` **only after**
   the model calls `task_complete` — so those tokens are spent once at the finish line, never on
   every turn. The system prompt only carries the short *"verify minimal state changes"* rule.

7. **No-eyes/no-ears framing.** *"you do NOT have eyes or ears, so you MUST resort to … AI
   tools to understand multimedia files."* One sentence sets the autonomy + modality model.

8. **Minimal-diff discipline as the dominant standing instruction.** A whole paragraph is spent
   only on "change the minimum, leave no side effects" — the highest-leverage correctness rule
   for graded terminal tasks — while everything else is pushed out of the prompt.

## Notable conventions

- Prompt = **role + autonomy posture + completion discipline + two `{slots}`**. Nothing else.
- Tool descriptions are plain prose strings assigned to named constants
  (`_EXECUTE_COMMANDS_DESC` …) then referenced inside the JSON tool spec — readable and DRY.
- `task_complete` takes **no parameters** (`"properties": {}`) — the finish signal is a bare
  call, not a summary blob. The expensive verification lives in the post-call confirmation turn.
- Task instruction sits in the prompt body via `{instruction}`; current terminal state is the
  last thing before generation via `{terminal_state}` — i.e. **task near the top, fresh
  observation at the very end** (primacy + recency).

---

## Techniques VibeHarness should adopt

- **Drop the embedded JSON action-schema from the system prompt.** VibeHarness already
  constrains decoding via Ollama's `format` field (`OllamaClient._act` passes
  `action_schema`), so re-printing the full `oneOf` schema in the prompt is redundant — exactly
  the Terminus→Kira change. Replace it with a one-line shape reminder
  (`{"tool": ..., "args": {...}}`) at most.
- **Keep tool docs in the tool/param descriptions only** (VibeHarness already derives
  `docs()` from `Tool.description` + `Param.description`) and stop duplicating mechanics in the
  loop rules.
- **Tighten parameter descriptions to terse, imperative, example-bearing one-liners.** Prefer
  "do X (e.g. Y)" over multi-clause prose.
- **Use negative scoping** on overlapping tools ("use this, NOT that") to cut wrong-tool calls.
- **Spend verification tokens at the finish line, not every turn.** VibeHarness's separate
  `validate` subagent already follows this pattern — keep the per-turn loop rules about it short
  and let the validator carry the strict checking.
- **Keep the system prompt to: role + loop posture + minimal-diff/verify discipline + tools.**
  Push everything situational into the per-turn message.
- **Preserve task primacy + recency** (task at front of system prompt, reminder at end of turn
  prompt) — Kira does the analogous thing with `{instruction}` early and `{terminal_state}` last.
