# Tool-Call Decoding Robustness — Improvements & Recommendations Analysis

**Audience:** the VibeHarness agent swarm + the main orchestrator.
**Branch this lands on:** `feat/toolcall-codec-seam` (the `ToolCallCodec` seam is the
right home for most of this).
**Scope:** making tool calls from a 3B reasoning model (VibeThinker-3B via Ollama)
resistant to *bad, hallucinated, or corrupted* calls — without losing the model's
reasoning strength or the harness's token efficiency.

> Status: analysis / recommendation only. **No production code is modified by this
> document.** Every recommendation is mapped to a concrete seam so it can be picked
> up as an isolated change (mostly new `codecs/*` modules + small, additive seam
> extensions).

---

## 0. TL;DR for the orchestrator

1. **The structural problem is already solved; the *content* problem is not.** The
   JSON grammar guarantees a call is *shaped* right (valid JSON, real tool name,
   required fields). It cannot constrain the *characters inside a free-form string*
   (`path`, `content`, `summary`). That is where the corruption lives
   (`persist_proof.txt` → `persist UA proof.txt`). See §2.
2. **This is not a hard limit of the 3B *system*, only of the 3B *model*.** The cure
   is to move the fragile work off the model: bind referential string args to the
   real workspace (decode-time enums), carry the reasoning's *conclusion* (not its
   garbled transcript) into the action phase, and add deterministic post-decode
   guards. See §3, §4.
3. **Highest-leverage, lowest-risk wins** (details in §4, ranked):
   - R1 Clean the reasoning→action **seam** (stop feeding the garbled reasoning tail
     next to the JSON).
   - R2 **Reference-binding** for args that must point at something real (enum of
     actual paths), expressed as a `json_schema` enum so **Ollama honours it today**.
   - R3 **Truncation handling** in the codec/loop (Ollama will *not* close cut-off
     JSON — confirmed upstream).
   - R4 **Discriminated decode** (tool-name enum → that tool's arg sub-schema) to
     replace the heavy `oneOf`.
   - R5 **Teach the tools** (few-shot in the codec's `format_instructions`) — we
     can't assume latent tool knowledge the way Kira assumes latent bash knowledge.
   - R6 Feed the **validator real bytes**, not just the narrative self-report.
4. **A second LLM "validation agent" is the *wrong* tool for the corruption bug.** A
   second 3B has correlated failure modes and (today) sees only the narrative, so it
   can't catch a mis-spelled filename. Use deterministic checks for fidelity; reserve
   the LLM validator for semantic completeness. See §3.4, §4-R6.

---

## 1. What the harness already does well (do **not** redo this)

Grounded in the code on this branch:

- **Two-phase turn: reason free, then act under constraint** (`llm.py:50-55`,
  `_reason` `:94`, `_act` `:106`). Phase 1 streams free reasoning stopped at
  `</think>`; phase 2 is a constrained continuation. **This is exactly the design the
  research recommends** — see *Let Me Speak Freely?* (§5): format constraints cost
  10–30% on reasoning, and the fix is to separate thinking from formatting. Keep this.
- **The codec seam itself** (`codec.py`): `ToolCallCodec` owns the wire format end to
  end — `format_instructions` (prompt block), `constraint` (decode-time
  `DecodeConstraint`), and `parse` (raw → `(tool, args)`). Codecs are isolated modules
  under `vibeharness/codecs/<name>_codec.py` exposing `CODEC` (`get_codec` `:71`), so
  new formats merge without touching shared files. **This is the correct abstraction
  for everything below.**
- **Greedy action phase** (`action_temperature: float = 0.0`, `config.py:11`; applied
  `llm.py:113`) — right call for verbatim-string fidelity.
- **Structural guarantees** via the JSON schema as Ollama's `format` (`json_codec.py:31-33`,
  `registry.action_schema` `:30`): valid JSON, `tool` pinned to a `const`, required
  fields, `maxItems` per-turn cap. Hallucinated *tool names* and malformed *shapes* are
  already prevented.
- **Deterministic state guards already exist** in the fs toolset (`fs_tools.py`):
  `create_file` refuses if the path exists (`:109`); `write_file` refuses if it does
  *not* exist (`:141`) and refuses an overwrite unless the file was read this session
  (`ReadTracker`, `:144`). These are good post-decode guards — we extend the pattern,
  not reinvent it.
- **Workspace grounding is wired and live.** `cli.render_workspace()` scans `Path.cwd()`
  every turn via `fs.tree(...)` and feeds it into the system prompt through
  `system_prompt_provider` (`cli.py:190-201`), placed right after the task block
  (`prompt.py:69-72`). So the model *is* shown the real tree each turn — the plumbing
  for grounding exists; §4-R2 builds on it.
- **Task anchored at both high-attention ends**: front of the system prompt
  (`prompt.py:63-68`) and end of the turn prompt (`prompt.py:86-92`). This matches the
  *Lost in the Middle* U-curve (§5) — primacy + recency, nothing wasted in the middle.

---

## 2. The failure taxonomy (why "drift" is really 3+ distinct bugs)

Treating all bad calls as one phenomenon is the trap; each row has a *different* cure.

| # | Failure | Example | Solved today? | Real cure |
|---|---------|---------|---------------|-----------|
| A | **Malformed shape** (bad JSON, missing field) | `{ not json` | ✅ `format` grammar + `parse` | — |
| B | **Hallucinated tool name** | `{"tool":"frobnicate"}` | ✅ `tool: {const}` / `oneOf` | — |
| C | **Corrupted string *argument*** | `persist_proof.txt` → `persist UA proof.txt` | ❌ grammar can't see inside strings | R1 + R2 (decode) + R6 (guard) |
| D | **Truncated call** (hit token cap mid-string) | `…"content":"abc` then stop | ❌ Ollama won't close it | R3 |
| E | **Wrong-but-valid tool/arg choice** | deletes the wrong real file | partial (state guards) | R2 + R5 + LLM validator |
| F | **Knowledge gap** (doesn't know how a tool behaves) | mis-uses `write_file` vs `create_file` | partial (docs) | R5 (teach + few-shot) |

**The evidence (Type C).** From `runs/run_t1.0_…` the task *"Create persist_proof.txt"*
produced, across steps, `"persist UA proof.txt"` and a summary saying
`"persistpillot proof.txt"`. Every one of those is **schema-valid** — valid JSON, real
tool, required args present. The grammar did its job perfectly; the corruption is
*inside a free-form `string` node*, which a JSON-schema→GBNF grammar accepts as "any
valid string." **No amount of structural constraint, prompt placement, or voting fixes
a `string` node.** That is the crux of this whole document.

### Why even *greedy* decoding corrupts the string (root-cause hypothesis)

Phase 2 is greedy (temp 0), so determinism is not the issue — *conditioning* is. In
`_act` (`llm.py:106-108`) the phase-2 prompt is:

```
_render_chatml(system, user) + _continue_after_reasoning(reasoning)
```

i.e. the **entire phase-1 reasoning is re-appended immediately before the JSON is
generated** (`_continue_after_reasoning` `:169-177`). The README itself notes the 3B
"drifts into garbage" at high temperature. When the model conditions on a garbled,
*off-distribution* tail, its whole next-token distribution degrades; the grammar keeps
the output *structurally* valid while the *content* rides the degraded distribution.
**The garble enters at the seam, not in the grammar.** (Validation note for the swarm:
confirm by correlating corrupted args in `.vibe/*.json` with garbled phase-1 tails — if
they co-occur, R1 alone removes most of Type C.)

A second, independent risk at the same seam: `_render_chatml` (`:160-167`)
**hand-rebuilds** the Qwen2.5 ChatML, and the `Modelfile` has **no `TEMPLATE`** (only
`FROM …GGUF`). So phase 2 (`raw: True`) depends on the hand-rolled string *exactly*
matching the GGUF's real template. Any mismatch (BOS, a system wrapper, an auto
`<think>` prefill) makes every phase-2 generation subtly off-distribution — itself a
plausible *cause* of the string drift. **Verify** with `ollama show --template vibethinker`
and byte-compare.

---

## 3. The principle: keep the model where it's strong, remove it where it's weak

VibeThinker-3B is (claimed) strong at reasoning for its size but is **not** tuned for
tool use and has weak verbatim-string fidelity and weak latent tool knowledge. So:

1. **Let it reason** (decide *what* to do) — its strength; keep phase 1 unconstrained.
2. **Don't let it spell** referential strings — bind them to the real set at decode time
   (R2). "Pick which file" is reasoning; "reproduce the exact filename" is fidelity.
3. **Carry the conclusion, not the transcript** — phase 2 needs the reasoning's *verdict*
   (a few clean tokens), not 2,000 tokens of possibly-garbled prose sitting at the seam
   (R1).
4. **Deterministic checks beat a second model** for fidelity (R3, R6 guards). A second
   3B validator has *correlated* errors and currently sees only the narrative
   (`validation.py` → `memory.render()`), so it cannot see that a filename is wrong. It
   is the right tool for *semantic completeness* (Type E/F), the wrong tool for Type C/D.

This reframes the user's question — *"is this a hard limit?"* — precisely: it is a hard
limit of the **model's** fidelity, but **not of the system**, because the system can move
that burden off the model.

---

## 4. Recommendations (prioritized, each mapped to the seam)

Ranked by (leverage on Types C/D/E) × (low risk) ÷ (effort).

### R1 — Clean the reasoning→action seam *(highest priority, smallest change)*
**Problem:** garbled reasoning tail is re-appended right before the constrained JSON
(`_act`/`_continue_after_reasoning`), degrading the distribution the grammar samples
under.
**Do (cheapest → most robust):**
1. **Trim + scaffold the seam.** Keep only the last clean segment of the reasoning, then
   insert a fixed, in-distribution scaffold before generation:
   ```
   …<clean tail>…
   </think>

   Action (tool calls only):
   ```
   The fixed scaffold gives a clean left-context at the exact generation point.
2. **Garble gate (deterministic).** Detect a bad trace (non-ASCII ratio, n-gram
   repetition, langdetect). On trip: regenerate phase 1 (only `reason_tokens=2048`) or
   fall back to a clean re-ask that drops the reasoning entirely.
3. **Conclusion handoff (most robust).** Have phase 1 end in a short structured decision
   ("DECISION: create_file path=… content=…"), and condition phase 2 only on that line.
   If the 3B won't self-summarize reliably, insert a tiny *constrained* intermediate step
   whose output is grammar-clean by construction.
**Where:** `OllamaClient._act` / `_continue_after_reasoning` (`llm.py`). This is
transport-level, *below* the codec, so it improves every codec at once.
**Risk:** low. **Effort:** S. **Validates:** *Let Me Speak Freely?* (separate thinking
from formatting) and the truncation/seam findings.

### R2 — Reference-binding for referential args *(highest leverage on Type C/E)*
**Idea:** when an argument must refer to something that **already exists**, don't decode a
free string — decode a **choice over the real set**. `read_file.path`, `write_file.path`
(must exist, `fs_tools.py:141`), `manage_path.path` (delete/move/copy source), and the
web toolset's element `ref`s are all referential. Build the domain each turn from the
live workspace tree the harness already computes (`cli.render_workspace`).
**Crucial implementation fact:** express it as a **JSON-schema `enum`**, e.g.
`{"type":"string","enum":[<real paths>]}`. **Ollama honours `enum` via `format` today** —
no GBNF, no new backend. `"persist UA proof.txt"` becomes *structurally impossible*
because it isn't in the enum.
**Seam change required (small, additive):** `ToolCallCodec.constraint(registry, max_actions)`
(`codec.py:57-59`) has **no access to per-turn runtime state**, so it can't build a
dynamic enum today. Extend the signature to pass a turn-context (e.g.
`constraint(registry, max_actions, context)` where `context` exposes the current path
set), and have the agent build it each turn from the workspace. Tools that own a
referential param advertise its dynamic domain (e.g. a `Param.domain` hook the registry
resolves against `context`).
**Caveats / honesty:**
- New-file params (`create_file.path`, `write_file` for a not-yet-existing target) **can't**
  be enumerated. For those, fall back to (a) char-class constraint *iff* using a GBNF
  backend (see R7 — Ollama's `format` does **not** reliably honour JSON-schema `pattern`),
  or (b) bind to the value the model just produced in its R1 conclusion, or (c) the R6
  guard. Don't claim R2 covers creation; it covers *reference*.
- Large enums add grammar-masking cost (upstream notes token masking isn't parallelized);
  cap the domain (e.g. workspace files only, not absolute FS).
**Risk:** low-med (seam signature touch). **Effort:** M. **Leverage:** very high.

### R3 — Truncation handling *(correctness, confirmed-real)*
**Confirmed upstream:** *"Ollama does not validate the full response against the schema,
so if the model stops producing tokens mid-JSON without closing braces, it won't be valid
JSON despite the grammar."* With `action_tokens=16384` (`config.py:31`) a runaway string
fills the budget and `parse` (`json_codec.py:35`) then fails → a wasted turn
(`agent.py:124-127`).
**Do:** (a) lower `action_tokens` to a tool-appropriate cap; (b) in the codec/client,
detect "stopped mid-structure" (unbalanced braces / hit `num_predict`) and either
continue-generate or surgically close, instead of discarding the turn; (c) surface a
*specific* observation ("call was cut off") rather than generic "invalid JSON".
**Where:** `OllamaClient._act` (detect truncation) + codec `parse` (repair). **Risk:** low.
**Effort:** S–M.

### R4 — Discriminated decode (replace `oneOf`)
**Problem:** `items: {oneOf: [<every tool>]}` (`registry.py:38`) is the weakest part of the
grammar for a small model and the most expensive to mask (one branch per tool, evaluated
together).
**Do:** decode in two bites — constrain `tool` to a **name enum**, read it, then splice in
*that tool's* arg sub-schema as the next constraint. Guarantees args-match-the-chosen-tool,
shrinks the grammar, removes `oneOf` ambiguity. Naturally a new codec
(`codecs/discriminated_codec.py`) or an option on the JSON codec. Also add
`additionalProperties: false` to `args`/call schemas so the model can't smuggle extra keys.
**Risk:** med (changes decode flow / may need per-call constraint application). **Effort:** M.

### R5 — Teach the tools (few-shot), because we can't assume latent knowledge
**Premise:** Kira's lean prompt works because a frontier model *already knows* bash, files,
and tool-call conventions. VibeThinker-3B does **not** — it's a reasoning specialist, not a
tool-use model. So a *justified* divergence from Kira: spend a few tokens Kira doesn't.
**Do:** add **1–2 concrete worked examples** of a correct call sequence to the codec's
`format_instructions` (`json_codec.py:19`) — a 3B learns the pattern far better from an
example than from rules. Tighten `create_file` vs `write_file` disambiguation in the docs
(`fs_tools.py:91-123`) since that's a live Type-F confusion. Keep examples *in the codec* so
they travel with the format and never drift.
**Where:** codec `format_instructions`. **Risk:** very low. **Effort:** S.
**Note:** this *adds* a little prompt; it's the one place where "more tokens" is correct,
and it's still cheaper than re-printing the action schema (which we correctly do **not** do).

### R6 — Give the validator ground truth (not the self-report)
**Problem:** `LLMValidator` judges from `memory.render()` — the agent's own narrative
(`agent.py:159`, `validation.py`). It can't see real bytes, so it can't catch a wrong
filename or empty file, and its errors correlate with the agent's.
**Do:** before/inside validation, run a **deterministic** check (does the claimed file exist?
do its bytes match what the task/plan implied? — the harness already nudges "read a file
back after writing", `prompt.py:42`, but nothing *enforces* the comparison). Feed the
validator the **actual listing/contents**, and demote the LLM validator to what it's good
at: semantic completeness ("were all parts done"), Type E/F. **Risk:** low. **Effort:** S–M.

### R7 — (Strategic) a GBNF-capable backend, only if R2 isn't enough
`DecodeConstraint.gbnf` already exists (`codec.py:41-43`) but is **dead on Ollama** —
`_act` applies only `json_schema` as `format` (`llm.py:117-120`), and Ollama has **no raw
GBNF passthrough** (upstream issue ollama/ollama#11911). True char-class constraints on
*new* paths/strings, or dynamic grammars beyond enums, need a **llama.cpp-server**
`LLMClient` that honours `constraint.gbnf`. The seam is already shaped for this (the `gbnf`
field is the hook). **Do this only if** R2's enum approach proves insufficient for creation
args — it's the largest change (a new backend) for the narrowest remaining gap.
**Risk:** high (new backend). **Effort:** L.

### Minor seam cleanups (low effort, worth folding in)
- `build_turn_prompt` hardcodes *"Respond with a JSON array of one or more actions"*
  (`prompt.py:91-92`) — a **format leak** in codec-agnostic code. Source it from the active
  codec, or an XML/code codec will contradict the turn prompt.
- Consider routing **heavy reasoning only when needed**: for obvious follow-ups ("read back
  the file you just wrote") 2k reasoning tokens is pure downside (more surface to garble at
  the seam). A cheap router → straight-to-action on easy turns shrinks Type-C exposure and
  speeds runs.

---

## 5. External grounding (best practice)

- **Separate reasoning from formatting** — *Let Me Speak Freely? A Study on the Impact of
  Format Restrictions on the Performance of LLMs* (Tam et al., EMNLP 2024 Industry):
  constrained/JSON-mode decoding degrades reasoning **10–30%**; the recommended remedy is to
  reason freely, then convert to the structured format under constraint. This validates the
  harness's two-phase design **and** R1's "carry the conclusion" handoff. See also **CRANE**
  (reasoning with constrained generation) for interleaving reasoning and constrained spans.
- **Ollama structured outputs = JSON-schema → GBNF in llama.cpp**: tokens illegal under the
  grammar are masked at sampling. Confirmed limitations we rely on: (a) **no full-response
  validation — truncated JSON stays invalid** (→ R3); (b) **grammar masking can slow
  generation and isn't parallelized** (→ keep enums/`oneOf` small, R2/R4); (c) **no raw GBNF
  passthrough in Ollama** (ollama/ollama#11911) (→ R7 needs a llama.cpp backend; R2 must use
  `enum`, which Ollama *does* honour).
- **Lost in the Middle** (Liu et al., 2023/TACL): U-shaped accuracy — strongest at the
  **start (primacy)** and **end (recency)**, weakest in the **middle**. Validates the existing
  task anchoring (`prompt.py`) and argues for keeping the injected workspace tree **compact**
  so it doesn't push tool docs into the dead middle.

Sources:
- https://arxiv.org/html/2408.02442v1 — *Let Me Speak Freely?* (format restrictions vs reasoning)
- https://aclanthology.org/2024.emnlp-industry.91.pdf — same, EMNLP proceedings
- https://par.nsf.gov/servlets/purl/10633770 — *CRANE: Reasoning with constrained LLM generation*
- https://docs.ollama.com/capabilities/structured-outputs — Ollama structured outputs
- https://blog.danielclayton.co.uk/posts/ollama-structured-outputs/ — Ollama→GBNF internals + truncation caveat
- https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output — GBNF grammar & sampling
- https://github.com/ollama/ollama/issues/11911 — no raw GBNF passthrough in Ollama
- https://arxiv.org/pdf/2510.10276 / https://github.com/nelson-liu/lost-in-the-middle — *Lost in the Middle*

---

## 6. Suggested sequencing for the swarm

1. **R1** (seam clean) + the `Modelfile`/ChatML template verification — smallest change,
   likely removes most of Type C. Confirm against `.vibe/` logs first.
2. **R3** (truncation) + the `build_turn_prompt` format-leak cleanup — cheap correctness.
3. **R2** (reference-binding enums) — the big leverage; requires the `constraint(...,
   context)` seam extension. Land the signature change first, then a
   `codecs/grounded_json_codec.py`.
4. **R5** (few-shot teaching) + **R6** (validator ground truth) — independent, parallelizable.
5. **R4** (discriminated decode) as its own codec for A/B vs the JSON baseline.
6. **R7** (llama.cpp/GBNF backend) only if R2 leaves a real gap on creation args.

## 7. Open decisions for the orchestrator

- **Backend strategy:** stay Ollama-only and live within `enum`-expressible constraints
  (R2), or invest in a llama.cpp-server backend to unlock `gbnf` (R7)? This gates how far
  reference-binding can go for *new* strings.
- **Seam signature:** is extending `ToolCallCodec.constraint(...)` with a turn-context
  acceptable, or should dynamic domains be resolved by the registry/tools instead? (Either
  works; it's an API-taste call that affects every future codec.)
- **Reasoning router:** worth adding per-turn (skip heavy reasoning on trivial follow-ups),
  or keep every turn uniform for simplicity?

---

## Appendix — what was read to ground this (this branch, `feat/toolcall-codec-seam`)

`codec.py` (seam + `DecodeConstraint` + `get_codec`), `codecs/json_codec.py`,
`codecs/__init__.py`, `llm.py` (`decide`/`_reason`/`_act`/`_render_chatml`/
`_continue_after_reasoning`), `agent.py` (loop, `parse`, `_validate`, `_execute`),
`registry.py` (`action_schema`/`oneOf`), `tools.py` (`Param`/`Tool`/`call_schema`),
`fs_tools.py` (`create_file`/`write_file`/`ReadTracker`/`manage_path`), `prompt.py`
(system template + `build_turn_prompt`), `config.py`, `validation.py`, `memory.py`,
`cli.py` (`render_workspace`/`system_prompt_provider` wiring), `Modelfile`, `README.md`,
`tests/test_codec.py`, and `runs/run_t1.0_*.txt` (the Type-C evidence). External claims
verified via web search (§5).
