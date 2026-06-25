# Alumnium Harness Analysis (Issue #159)

> Repo: https://github.com/alumnium-hq/alumnium (branch `main`). All line numbers below
> are from `main` at time of research (2026-06). Paths are repo-relative.

## TL;DR architecture (read this first)
Alumnium is **not** a monolithic ReAct agent. It is a **thin multi-language client**
(`packages/python`, `packages/typescript`, `packages/java`) that drives a **local HTTP
"brain" server** (`packages/typescript/src/server`). The server runs **six small, single-shot,
*stateless* specialist agents** over a compacted ARIA/accessibility tree:

| Agent | Endpoint | Job |
|---|---|---|
| `PlannerAgent` | `POST /plans` | goal → ordered list of natural-language step strings |
| `ActorAgent` | `POST /steps` | one step → tool calls (bindTools) |
| `RetrieverAgent` | `POST /statements` | `get()` / `check()` data extraction |
| `LocatorAgent` | `POST /elements` | `find()` single element id |
| `AreaAgent` | `POST /areas` | scope tree to a sub-region |
| `ChangesAnalyzerAgent` | `POST /changes` | diff of before/after tree → 1-2 sentence summary |

The headline WebVoyager run uses this as an **MCP browser-subagent**: Claude Code (Sonnet 4.6)
is the planner/orchestrator, and Alumnium MCP (GPT-5 Nano) is the cheap browser executor that
"compresses how to browse" into `do/get/check` and **returns only a concise description of what
changed** rather than dumping raw trees back up. This is the core engineering bet.

## Benchmark claim
- **Not WebArena — it is WebVoyager.** Score: **98.5%** (610 tasks), beating prior SOTA
  Surfer 2 at 97.1%. Cost ≈ **$5** total API spend.
- Source: `websites/docs/src/content/blog/2026-03-26-webvoyager-benchmark.md`
  (live: https://alumnium.ai/blog/webvoyager-benchmark/).
- Stack used for the run (from the blog): Claude Code + Sonnet 4.6 as primary agent;
  **Alumnium MCP 0.18 with GPT-5 Nano** as browser subagent; Selenium 4.29 +
  undetected-chromedriver for anti-bot sites. Minimal top prompt:
  `"You are a browser agent for %URL%. %Task%"`. Accessibility-tree-first, not vision-first.

---

## a. Webpage Representation Compaction

Two-stage pipeline: **(1) client** turns the browser's raw CDP accessibility tree into raw XML
(with stable `raw_id`s); **(2) server** aggressively prunes/flattens that XML and renumbers nodes
with small sequential `id`s before it ever reaches the model.

**Stage 1 — client raw XML build (Chromium/CDP):**
`packages/python/src/alumnium/accessibility/chromium_accessibility_tree.py`
- `to_str()` (lines 24-63): consumes `Accessibility.getFullAXTree` CDP `nodes`, builds a
  `nodeId → node` lookup, separates true roots from iframe child roots.
- `_node_to_xml()` (lines 65-122): role becomes the **XML tag name** (line 67-69); attaches a
  sequential **`raw_id`** attribute (lines 72-73) — this is the durable handle used later to map
  model output back to a real DOM node. Emits `backendDOMNodeId`, `nodeId`, `ignored`, `name`,
  and all ARIA `properties` as attributes (lines 84-106).
- **iframe inlining** (lines 115-121): iframe subtrees are spliced inline under their `<Iframe>`
  parent so the model sees one flat tree across frames.
- `scope_to_area()` (lines 170-203) and `element_by_id()` (130-168): subtree extraction by
  `raw_id` (powers `al.area()` context narrowing).

**Stage 2 — server compaction + renumber (the real token win):**
`packages/typescript/src/server/accessibility/ServerChromiumAccessibilityTree.ts`
- `SKIPPED_PROPERTIES` (lines 9-19): drops `backendDOMNodeId`, `ignored`, `name`, `nodeId`,
  `raw_id`, and **`expanded`** (comment lines 14-17: skipping `expanded` stops the LLM from
  wastefully clicking comboboxes to expand them, since `SelectTool` handles that).
- `#xmlToNode()` (lines 40-97): re-keys every element with a **small sequential `id`** via
  `getNextId()` and stores `simplifiedId → rawId` in `simplifiedToRawId` (lines 45-51). The model
  only ever sees the tiny ids, never the large CDP backend ids → big token savings + less ambiguity.
- `toXml()` (lines 104-177) — node-level pruning rules:
  - `StaticText` nodes collapse to bare text content (lines 117-118).
  - `role === "none"` **or** `ignored` nodes are **unwrapped** — children promoted to parent
    (lines 119-124).
  - `role === "generic"` with no children is **dropped entirely** (lines 125-127).
  - `name`/`id`/property attributes emitted only if not in `excludeAttrs` (lines 131-145).
- `#pruneRedundantName()` (lines 183-238): removes a parent node's `name`/`label` text when it is
  already present in descendant text (dedup of repeated labels), sorting descendant content
  longest-first to handle overlapping substrings (lines 213-228). Root `RootWebArea` is preserved.

**Attribute budget controls:**
- Global env knob `ALUMNIUM_EXCLUDE_ATTRIBUTES` → `EXCLUDE_ATTRIBUTES`
  (`packages/python/src/alumnium/__init__.py` line 11), passed per-session and merged into
  `excludeAttrs` at every `toXml()` call site in `serverApp.ts`.
- Per-agent extra exclusions: Retriever and ChangesAnalyzer additionally strip `id`
  (`RetrieverAgent.ts` line 61; `ChangesAnalyzerAgent.ts` line 20) because those agents don't act
  on elements, so ids are pure noise — see merge in `serverApp.ts` lines 238-243 and 335-338.

---

## b. Agent Loop Flow + Multi-Agent Architecture

The orchestration loop lives **client-side** in `do()`; each iteration calls a stateless server
agent. There is **no internal agent self-loop / no max-iteration retry inside an agent** — the
planner produces all steps up front, then the actor executes them one by one.

**`do()` loop —** `packages/python/src/alumnium/alumni.py` lines 86-133:
1. Capture `app` + `initial_accessibility_tree` (line 96-97); snapshot before-tree/url only if
   change-analysis enabled (98-99).
2. `client.plan_actions(goal, tree)` → `(explanation, steps[])` (line 100) → `POST /plans`.
3. **For each step** (lines 103-117):
   - Re-read the **live** tree each step (`idx==0` reuses initial, else fresh
     `driver.accessibility_tree`) — line 105. This is the only cross-step "memory".
   - `client.execute_action(goal, step, tree)` → `(reasoning, actions[])` (line 106) → `POST /steps`.
   - Execute each returned tool call locally via `BaseTool.execute_tool_call` (lines 113-115).
4. Optional `analyze_changes(before, after)` → human-readable change summary (lines 119-132) →
   `POST /changes`. This summary is what gets surfaced back to an outer orchestrator (e.g. Claude
   Code in the MCP setup).

**Planner toggle / degenerate mode:** if `planner` is false the server returns the goal itself as
a single step (`serverApp.ts` lines 116-121), and the client substitutes the actor's reasoning as
the explanation (`alumni.py` lines 108-109).

**Server agent wiring —** `packages/typescript/src/server/session/Session.ts`:
- One `Session` per client; instantiates all six agents sharing one `LlmContext` + one `llm`
  (lines 90-100). `processTree()` picks the platform tree class (lines 154-168).
- HTTP routing for the whole agent surface: `packages/typescript/src/server/serverApp.ts`
  — `/plans` (110-151), `/steps` (157-180), `/statements` (228-261), `/areas` (265-289),
  `/elements` (295-316), `/changes` (322-365), `/examples` add/clear (186-222), `/caches`
  save/discard (369-401).

**Planner agent —** `packages/typescript/src/server/agents/PlannerAgent.ts`:
- Structured output via zod `Plan = {explanation, actions: string[]}` (lines 41-51), forced with
  `llm.withStructuredOutput(Plan, {includeRaw:true})` (line 133).
- `invoke()` lines 175-205: single LLM call, returns explanation + filtered action strings.
- Actions are **natural-language strings** ("click button \"Foobar\""), *not* tool calls — the
  Actor later grounds each string into an actual tool call. This decouples planning from grounding.

**Actor agent —** `packages/typescript/src/server/agents/ActorAgent.ts`:
- `invoke(goal, step, treeXml)` lines 62-102; returns `[reasoning, toolCalls]`. Empty/whitespace
  step short-circuits to no-op (lines 68-70).

**Termination / retry:** agents are single-shot. Retry happens only on transient LLM errors
(`BaseAgent.invokeChain` lines 173-254; `shouldRetry` lines 106-169 — rate-limit/timeout only,
provider-specific). Client wraps `do/check/get/find` with `@retry(tries=RETRIES=2)`
(`alumni.py` lines 85, 135, 161, 183; `RETRIES` from `__init__.py` line 14). If the planner finds
no path it returns an **empty action list** (planner prompt instruction) rather than looping.

---

## c. Tool Interface Optimizations

**Tool set (15 tools) —** `packages/python/src/alumnium/tools/__init__.py` lines 1-33:
`ClickTool, DragAndDropTool, DragSliderTool, ExecuteJavascriptTool, HoverTool, NavigateBackTool,
NavigateToUrlTool, PressKeyTool, PrintToPdfTool, ScrollTool, SwitchToNextTabTool,
SwitchToPreviousTabTool, TypeTool, UploadTool`. Tools are filtered per driver capability
(`driver.supported_tools`) plus user `extra_tools` (`alumni.py` lines 57-59).

**Schema design — Pydantic class → JSON tool schema:**
`packages/python/src/alumnium/tools/tool_to_schema_converter.py`
- `convert_tool_to_schema()` lines 49-69: tool **class name** = function name; **docstring** =
  description; pydantic `model_fields` → JSON `properties`/`required`; `_pydantic_to_json_type`
  (lines 10-46) maps Python types incl. `Enum → {type:string, enum:[...]}` and `list[T] → array`.
- Tools are intentionally **tiny**. `ClickTool` (`tools/click_tool.py` lines 8-11) is a one-field
  `id:int` model whose docstring embeds a **negative instruction**:
  `"Click an element. NEVER use ClickTool to upload files - use UploadTool instead."` — i.e. tool
  disambiguation is baked into the description, not the system prompt.
- `BaseTool.execute_tool_call` (`tools/base_tool.py` lines 9-27) instantiates `tool(**args)` and
  `.invoke(driver)`, returning a printable form like `ClickTool(id=42)` for logging/results.

**Element-id indirection (key optimization):** the model selects elements by the small sequential
`id`, and the server **remaps `id`/`from_id`/`to_id` back to real `raw_id`s** before returning tool
calls to the client:
- `BaseServerAccessibilityTree.mapToolCallsToRawId()` lines 27-48 and `getRawId`/`#extractId`
  lines 17-67 (the `#extractId` handles model quirks: Gemini float ids, Llama string/nested-dict
  ids).
- Applied in `serverApp.ts` `/steps` line 173 (`mapToolCallsToRawId(actions)`).
Benefit: model reasons over compact ids; client acts on stable DOM handles; no large ids in context.

**Actor tool binding:** `ActorAgent` constructor `prompt.pipe(llm.bindTools(toolSchemas))`
(`ActorAgent.ts` line 59) — native function-calling, schemas passed at session creation
(`Session.ts` line 90, schemas from client `convert_tools_to_schemas`, `http_client.py` line 36).

**Structured-output result decoding (Retriever) —** `RetrieverAgent.ts`:
- Returns zod `{explanation, value}` (lines 15-28); lists are encoded as a single string joined by
  a sentinel `<SEP>` (`#LIST_SEPARATOR` line 62) and split client-side (lines 163-173). Robustness
  hacks for model misbehavior: trims stray separators (147-153), fixes GPT-5 Nano brace
  replacement (154-158), Grok escaped-tag `&lt;SEP&gt;` (line 160). `"NOOP"` sentinel = not found.

---

## d. System Prompt Optimizations

Prompts are **per-agent × per-provider** markdown files under
`packages/typescript/src/server/agents/prompts/<agent>/<dev>/{system,user}.md`, bundled at build
time (`prompts/prompts.ts` `loadAgentPrompts` lines 55-124; bundled via `bundledPrompts.ts`).
Provider→prompt-dialect mapping `PROVIDER_TO_PROMPTS_DEV` (`prompts.ts` lines 35-49); selection in
`BaseAgent` constructor lines 95-104 (falls back to `openai` dialect). Templated with a Python-style
`pythonicFormat` `{var}` substitution.

**Actor — minimal, grounding-focused** (`prompts/actor/openai/system.md`, 5 lines):
> "You are a helpful assistant that performs actions to achieve a task on a webpage based on the
> given step and final goal. You can reason about the accessibility tree of the page given as XML,
> locate elements by their identifier (ID), and interact with them. **Use goal only for context,
> focus on executing individual step.** Think through which element to interact with and why before
> making your tool call. Your reasoning will help ensure accurate element selection."

User template (`actor/openai/user.md`): `Goal: {goal}` / `Step: {step}` / ```` ```xml {accessibility_tree} ``` ````.
Note the explicit **CoT-before-tool-call** instruction and the **step-not-goal focus** to prevent
the actor from over-reaching.

**Planner — numbered constraint list + few-shot** (`prompts/planner/openai/system.md`, 51 lines):
- Analysis checklist (lines 5-11) then **formulation rules** (lines 12-23), notably:
  - "Use only the following action types: `{tools}`." (tool names injected, line 14).
  - "Do not include element IDs in the actions." (line 17) — keeps plan model-agnostic/groundable.
  - "Ground the actions in the accessibility tree provided." (line 19).
  - "Always aim to **minimize the number of actions**… do not break it down further." (line 21).
  - Empty action list if goal not achievable (line 23).
- Two canned few-shot examples (lines 25-50) + `{extra_examples}` slot (line 51) for learned ones.
- **Provider divergence:** `prompts/planner/anthropic/system.md` adds rule 9
  (line 22): "When planning to type into textbox, skip clicking it regardless of its focused
  status," and a matching example tweak (line 49). Demonstrates per-model prompt tuning.
- `PlannerAgent.ts` injects built-in examples for navigate (lines 59-70) and upload (lines 72-84)
  only when those tools are present, and supports runtime few-shot via `al.learn()` →
  `addExample()` / `#formatExample()` (lines 143-165) and `clearExamples()` (137-141).

**Retriever — strict grounding / anti-hallucination** (`prompts/retriever/openai/system.md`, 13 lines):
> "You are a precise robot… CRITICAL INSTRUCTIONS: Think through the problem first; **ONLY retrieve
> information directly present**…; If NOT present, **RESPOND ONLY WITH: \"NOOP\"**; Do NOT use
> external/common knowledge; Avoid duplicates…; Preserve order; If a list, separate items with
> `{separator}`. ANY VIOLATION… IS NOT PERMITTED." (`{separator}` = `<SEP>`).
User template adds `title`/`url` context (`retriever/openai/user.md`).

**ChangesAnalyzer — output-shape-constrained summarizer** (`prompts/changes-analyzer/openai/system.md`, 72 lines):
- Explains the `+/-` diff format (lines 4-8); **hard output rules** (lines 9-16): "Output 1-2 short
  paragraphs (3-5 sentences)", first sentence = high-level navigation/action, focus on user-visible
  content, **ignore footer/nav chrome**, group related changes. Four worked input→output examples
  (lines 18-72, e.g. login→dashboard, Airbnb search). This is what keeps the returned-to-orchestrator
  payload tiny.

**Locator — single-answer, id-only output** (`prompts/locator/openai/system.md`, 12 lines):
"identify the **single most specific element**", prefer interactive/semantic matches, **fallback to
root id** if unsure, and "Your final response must be **only the numerical `id`**… no other words."

**Area** (`prompts/area/openai/system.md`): same single-id selection pattern for region scoping.

---

## e. Live Context Management

**Core principle: agents are stateless and history-free.** Each agent call builds a **fresh**
2-message prompt (`system` + `human`) from the *current* inputs — there is **no accumulated
conversation history, no scratchpad, no message buffer** carried between turns. Verify:
- `ActorAgent.ts` lines 50-60 — `ChatPromptTemplate.fromMessages([[system],[human]])` rebuilt per call.
- `PlannerAgent.#generateChain()` lines 120-135; `RetrieverAgent.invoke` lines 126-138;
  `ChangesAnalyzerAgent.invoke` lines 37-44 — all construct messages from scratch each invocation.
- `LlmContext` (`LlmContext.ts` lines 13-41) holds **only** a `prompt-string → meta` map used for
  cache keying (`assignPromptsMeta`/`clearPromptsMeta`, set/cleared around each chain call in
  `BaseAgent.invokeChain` lines 213-232) — it is **not** a chat memory.
- `SessionContext` (`SessionContext.ts` lines 22-43) persists **only** the `app` id across calls.

**What carries information across turns (the substitutes for history):**
1. **The live DOM itself** — re-fetched every step (`alumni.py` line 105). State lives in the page,
   not in a transcript; the model always sees current truth, eliminating stale-context drift.
2. **The ChangesAnalyzer summary** — `do()` returns a compact NL `changes` string (`alumni.py`
   lines 119-133) that an *outer* orchestrator (Claude Code / the MCP host) uses as its memory of
   what happened. URL-change is prepended deterministically without an LLM call
   (`serverApp.ts` lines 344-351) and `\n\n`→space collapsed (`ChangesAnalyzerAgent.ts` line 46).
3. **Few-shot examples** — the only durable per-session learned state, injected into the Planner
   system prompt (`PlannerAgent` lines 143-165), persisting across `do()` calls within a session.

**Snapshot scoping / budgeting:**
- `al.area(description)` narrows the tree to a sub-region for subsequent calls
  (`alumni.py` lines 197-220; `scope_to_area` in `chromium_accessibility_tree.py` lines 170-203) —
  explicit context-window reduction primitive.
- Per-agent attribute stripping (see §a) reduces tokens differently per agent role.
- Vision is **opt-in** and tree-replacing: when `screenshot` is passed to the Retriever it sends the
  image **instead of** the tree text (`RetrieverAgent.ts` lines 95-115; gated by `vision=` in
  `check`/`get`, `alumni.py` lines 155, 178).

**Caching (cross-call token/cost reduction):**
- `ServerCache` + `ElementsCache` family (`packages/typescript/src/server/cache/…`,
  e.g. `ResponseCache.ts`, `ElementsCache/ActorAgentElementsCache.ts`,
  `PlannerAgentElementsCache.ts`) keyed via `LlmContext` meta; `CacheFactory` wires it into the LLM
  (`Session.ts` lines 81-88). Save/discard exposed as `/caches` endpoints (`serverApp.ts` 369-401;
  client `save_cache`/`discard_cache` in `http_client.py` 201-213).

---

## Key files index

| File (repo-relative) | What it contains |
|---|---|
| `websites/docs/src/content/blog/2026-03-26-webvoyager-benchmark.md` | The 98.5% WebVoyager claim, run stack, cost |
| `packages/python/src/alumnium/alumni.py` | **Client `do/check/get/find/area/learn` loop** (lines 86-220) — the real orchestration |
| `packages/python/src/alumnium/clients/http_client.py` | REST client to the brain server; per-endpoint payloads; auto-spawns local server |
| `packages/python/src/alumnium/__init__.py` | Env config: `EXCLUDE_ATTRIBUTES`, `PLANNER`, `RETRIES`, `CHANGE_ANALYSIS` (lines 9-14) |
| `packages/python/src/alumnium/accessibility/chromium_accessibility_tree.py` | Stage-1 CDP→XML build, `raw_id`, iframe inlining, `scope_to_area` |
| `packages/python/src/alumnium/tools/__init__.py` | The 15-tool registry |
| `packages/python/src/alumnium/tools/base_tool.py` | `execute_tool_call` (lines 9-27) |
| `packages/python/src/alumnium/tools/click_tool.py` | Tiny pydantic tool + negative-instruction docstring |
| `packages/python/src/alumnium/tools/tool_to_schema_converter.py` | Pydantic class → JSON function schema (lines 49-69) |
| `packages/typescript/src/server/serverApp.ts` | **All agent HTTP endpoints / server-side glue** (plans/steps/statements/areas/elements/changes/caches) |
| `packages/typescript/src/server/session/Session.ts` | Per-session agent instantiation + `processTree` |
| `packages/typescript/src/server/session/SessionContext.ts` / `LlmContext.ts` | Proof of **no chat history**; only app-id + cache-meta state |
| `packages/typescript/src/server/agents/BaseAgent.ts` | invoke/retry, per-provider prompt selection, response normalization |
| `packages/typescript/src/server/agents/PlannerAgent.ts` | Plan zod schema, few-shot examples, navigate/upload examples, `learn` |
| `packages/typescript/src/server/agents/ActorAgent.ts` | Step→toolcalls, `bindTools` |
| `packages/typescript/src/server/agents/RetrieverAgent.ts` | get/check extraction, `<SEP>` list codec, model-quirk fixes |
| `packages/typescript/src/server/agents/ChangesAnalyzerAgent.ts` | Diff→short NL summary |
| `packages/typescript/src/server/agents/LocatorAgent.ts` / `AreaAgent.ts` | Single-id element / region selection |
| `packages/typescript/src/server/accessibility/ServerChromiumAccessibilityTree.ts` | **Stage-2 compaction**: SKIPPED_PROPERTIES, prune/flatten, id renumber, redundant-name dedup |
| `packages/typescript/src/server/accessibility/BaseServerAccessibilityTree.ts` | `simplifiedToRawId` map, `mapToolCallsToRawId`, id-extraction quirk handling |
| `packages/typescript/src/server/accessibility/AccessibilityTreeDiff.ts` | git-style before/after tree diff fed to ChangesAnalyzer |
| `packages/typescript/src/server/agents/prompts/<agent>/<provider>/{system,user}.md` | **All prompt text**; per-provider variants (openai/anthropic/meta/mistralai/deepseek/xai) |
| `packages/typescript/src/server/agents/prompts/prompts.ts` | `PROVIDER_TO_PROMPTS_DEV` map, prompt loader/bundler |
| `packages/typescript/src/server/cache/…` | Response + per-agent elements caching |

### Notes for the synthesis agent (#160)
- The brain is the **TypeScript server**, not the Python package; Python/Java are thin clients.
  Java/Python accessibility trees mirror the TS ones (`packages/java/.../accessibility/*`).
- Biggest transferable wins for our harness: (1) **two-stage tree compaction + small sequential id
  ↔ raw_id indirection**; (2) **split planner (NL steps) / actor (grounding+tools)**; (3) **stateless
  agents that rely on the live DOM instead of chat history**; (4) **ChangesAnalyzer that returns a
  3-5 sentence summary instead of raw trees** (the key to cheap MCP-subagent operation); (5)
  **per-provider prompt dialects** + **anti-hallucination NOOP** retriever; (6) tool **descriptions
  carry disambiguation** (negative instructions) rather than bloating the system prompt.
