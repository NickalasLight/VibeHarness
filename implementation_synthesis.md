# Implementation Synthesis — Alumnium → VibeHarness (Issue #160)

> Drives real implementation work on `beta_qwen3coder`. Every takeaway is grounded in
> ground-truth: `alum_research.md` (Alumnium reverse-engineering with file:line citations,
> issue #159), `SNAPSHOT_SCOPING_ANALYSIS.md` (our snapshot prune/cap research), and
> `improvements_recommendations_analysis.md` (our tool-call codec robustness analysis). The
> "Implementation in VibeHarness" blocks target the ACTUAL files read for this synthesis:
> `vibeharness/web.py`, `agent.py`, `codec.py`, `llm.py`, `config.py`, `snapshot_prose.py`,
> `snapshot_budget.py`, `prompt.py`, `registry.py`, `cli.py`, `codecs/hermes_codec.py`.

## How VibeHarness already maps onto Alumnium (read first)

| Alumnium concept | VibeHarness equivalent today | Gap this doc closes |
|---|---|---|
| Stage-1 CDP→XML with `raw_id` | Playwright ARIA-YAML with `[ref=e1300]` | refs are large + non-sequential |
| Stage-2 prune/flatten | `snapshot_prose.aria_yaml_to_prose` (WebArena prune) | no renumber, no name-dedup |
| Small sequential `id` ↔ `raw_id` map | **none** — model sees raw `e1300` | Takeaways 1, 7, 11 |
| `area()` subtree scoping | **none** — whole page every turn | Takeaway 2 |
| 6 stateless specialist agents | 1 `RalphAgent` + validator + advisor | Takeaways 3, 4 |
| Plan-all-steps-up-front | per-turn reason→act, no global plan | Takeaway 4 |
| ChangesAnalyzer 3-5 sentence summary | full snapshot re-injected as `page_snapshot` | Takeaways 5, 8 |
| Negative-instruction docstrings | partial (a few tools) | Takeaway 6 |
| `<SEP>` list codec | **none** — JSON only | Takeaway 9 |
| Per-provider prompt dialects | per-codec `format_instructions` only | Takeaway 10 |
| Decode-time grounding | structural only (`oneOf` schema) | Takeaway 12 |

---

## Takeaway 1: Two-stage ARIA compaction with sequential-ID renumber

### What
Replace the current "prune-only" snapshot pipeline with Alumnium's full two-stage pipeline:
**(stage 1)** parse the raw ARIA tree keeping the durable native ref as `raw_id`; **(stage 2)**
prune noise (already done), **then renumber every kept node to a small sequential `id`
(1, 2, 3…)**, deduplicate parent names already present in descendants, and store a
`simplified_id → raw_ref` map so tool calls can be translated back. VibeHarness already does the
prune (stage 2a) in `snapshot_prose.py` but stops short of the renumber/dedup that produces the
real token win.

### Why
Alumnium's server re-keys every element with a small sequential `id` via `getNextId()` and stores
`simplifiedId → rawId` (`ServerChromiumAccessibilityTree.ts:45-51`). "The model only ever sees the
tiny ids, never the large CDP backend ids → big token savings + less ambiguity"
(`alum_research.md` §a). For VibeHarness this is doubly valuable: our refs are 4-digit
(`e1300`, `e1275`) and the 3B model **hallucinates and increments them** — the single most
guarded-against failure across `web.py`/`agent.py` (the entire `_guard_target`, `target_block_counts`
HARD STOP, and "NEVER guess or increment refs" guidance exist to fight this). A dense `1..N`
namespace tokenizes to a single token each (vs `e`,`1300` multi-token) and makes "increment e1300
→ e1301" land on a *real adjacent element* instead of a phantom, collapsing a whole class of
wasted turns. The `#pruneRedundantName()` dedup (`ServerChromiumAccessibilityTree.ts:183-238`)
removes repeated label text, further shrinking the YouTube/BBC snapshots measured in
`SNAPSHOT_SCOPING_ANALYSIS.md` (Measurement table).

### Source
`alum_research.md` §a "Webpage Representation Compaction", Stage 2:
`ServerChromiumAccessibilityTree.ts` `#xmlToNode()` (40-97), `toXml()` pruning (104-177),
`#pruneRedundantName()` (183-238); `BaseServerAccessibilityTree.mapToolCallsToRawId()` (27-48).
Cross-ref `SNAPSHOT_SCOPING_ANALYSIS.md` PART A (prune is lossless on actionable elements).

### Implementation in VibeHarness
```python
# Target: vibeharness/snapshot_prose.py
# Current state: aria_yaml_to_prose() prunes + emits one line per kept node, keeping the
#   native ref inline as "[e1300]". No renumber, no name dedup, no reverse map. The
#   identifier-decision docstring explicitly chose to keep the native ref to avoid a mapping
#   table — Alumnium shows the mapping table is cheap and worth it for a small model.
# Proposed change: add a renumber pass that returns BOTH the compacted prose and a
#   {simplified_id: raw_ref} map, plus a name-dedup pass. Keep the old function as a thin
#   wrapper so the A/B seam (config.web_snapshot_prose) is unchanged.

from dataclasses import dataclass

@dataclass
class CompactedSnapshot:
    """Prose with small sequential ids, plus the map back to playwright refs."""
    prose: str                 # "[3] button \"Accept all\" — click"
    id_to_ref: dict[str, str]  # {"3": "e1275"}

def _iter_all(node: "_Node"):
    for c in node.children:
        yield c
        yield from _iter_all(c)

def _dedup_descendant_names(root: "_Node") -> None:
    """Alumnium #pruneRedundantName: drop a parent name already in a descendant's text.

    Mirrors ServerChromiumAccessibilityTree.ts:183-238 — gather descendant content, sort
    longest-first, and strip the parent name when it is a substring of any descendant. The
    synthetic root is never stripped."""
    def descendant_text(n: "_Node") -> list[str]:
        out: list[str] = []
        for c in n.children:
            if c.name:
                out.append(c.name)
            if c.text:
                out.append(c.text)
            out.extend(descendant_text(c))
        return out
    for n in _iter_all(root):
        if n.role == "__root__" or not n.name:
            continue
        haystack = sorted(descendant_text(n), key=len, reverse=True)
        if any(n.name in d for d in haystack):
            n.name = ""

def aria_yaml_to_compacted(raw: str) -> CompactedSnapshot:
    """Stage-2 compaction: prune (existing) + name-dedup + sequential renumber.

    Renumbers kept refs to a dense 1..N namespace and returns the reverse map so the agent
    can translate a model-chosen small id back to the real playwright ref before driving the
    browser (Alumnium mapToolCallsToRawId, BaseServerAccessibilityTree.ts:27-48)."""
    if not raw or not raw.strip():
        return CompactedSnapshot(prose=raw, id_to_ref={})
    try:
        page_url, page_title, body = _split_header_and_yaml(raw)
        root = _build_tree(body)
        if not root.children:
            return CompactedSnapshot(prose=raw, id_to_ref={})
        _dedup_descendant_names(root)
        # Assign small sequential ids to every node carrying a native ref, in the SAME
        # depth-first order the prose walk emits, so id N is the Nth actionable line.
        id_to_ref: dict[str, str] = {}
        ref_to_id: dict[str, str] = {}
        counter = iter(range(1, 10_000))
        for n in _iter_all(root):
            if n.ref and n.ref not in ref_to_id:
                sid = str(next(counter))
                ref_to_id[n.ref] = sid
                id_to_ref[sid] = n.ref
                n.ref = sid                    # prose now renders "[3]" not "[e1275]"
        out: list[str] = []
        recent: list[str] = []
        for child in root.children:
            _walk(child, 0, out, recent)
        # (preamble assembly identical to aria_yaml_to_prose; omitted for brevity)
        return CompactedSnapshot(prose="\n".join(out), id_to_ref=id_to_ref)
    except Exception:
        return CompactedSnapshot(prose=raw, id_to_ref={})
```
The reverse translation is wired in Takeaway 7 (tool-call remap at the `_WebTool` seam).

---

## Takeaway 2: `area()` subtree scoping with per-turn caching

### What
Add an `area`-style scoping primitive: before an action that targets a known region (a consent
dialog, the active wizard step, an open listbox), inject ONLY that subtree's compacted snapshot
instead of the whole page, and cache the scoped view so repeated reads in the same turn are free.
VibeHarness currently injects the whole-page snapshot every turn (`make_raw_snapshot_provider` →
`capture_page_snapshot_raw`), budgeted only by size (`compute_snapshot_budget`).

### Why
Alumnium's `al.area(description)` narrows the tree to a sub-region for subsequent calls
(`alumni.py:197-220`, `scope_to_area` in `chromium_accessibility_tree.py:170-203`) — an explicit
context-window reduction primitive, cached for budget control (the `ElementsCache` family). For
VibeHarness the page snapshot is the single largest token consumer (the entire `snapshot_budget`
module exists to keep it from crowding out history; `config.py` `num_ctx=32768` leaves only
~23.5k input tokens). `SNAPSHOT_SCOPING_ANALYSIS.md` already proves a `role=dialog` subtree is the
highest-value scope ("hoist `role=dialog` subtrees to the top", recommendation 3). Scoping to the
active wizard step also directly attacks the iter-1/2/3 failure mode documented throughout
`web.py` — the model acting on stale refs from a previous step — because a scoped snapshot
*cannot contain* the gone refs.

### Source
`alum_research.md` §e "Live Context Management": `al.area(description)` (`alumni.py:197-220`),
`scope_to_area` (`chromium_accessibility_tree.py:170-203`), `ElementsCache`/`CacheFactory`
(`Session.ts:81-88`). Cross-ref `SNAPSHOT_SCOPING_ANALYSIS.md` "Combined recommendation" #3
(dialog-first / region scope).

### Implementation in VibeHarness
```python
# Target: vibeharness/snapshot_prose.py (scoping helper) + vibeharness/cli.py (provider wiring)
# Current state: _find_active_dialog(root) already locates an [active] dialog subtree, but the
#   prose renders the WHOLE tree. cli.py wraps the raw provider with aria_yaml_to_prose.
# Proposed change: a scope_to_region() that, when a foreground dialog (or, optionally, the
#   active form/step) is present, renders ONLY that subtree. Cache keyed by the raw snapshot
#   so a second call in the same turn (the DOM-delta path captures snapshots repeatedly)
#   reuses the parse.

_SCOPE_CACHE: dict[int, str] = {}   # hash(raw) -> scoped prose; cleared each turn

def scope_to_region(raw: str) -> str:
    """Return the compacted prose of the highest-priority foreground subtree, or the whole
    page when no such region is present. Mirrors Alumnium scope_to_area: focus the tree
    before acting. Falls back to the full page so we never blank the snapshot."""
    key = hash(raw)
    if key in _SCOPE_CACHE:
        return _SCOPE_CACHE[key]
    try:
        _url, _title, body = _split_header_and_yaml(raw)
        root = _build_tree(body)
        region = _find_active_dialog(root)        # existing helper, dialog-first
        out: list[str] = []
        recent: list[str] = []
        targets = [region] if region is not None else root.children
        for child in targets:
            _walk(child, 0, out, recent)
        prose = "\n".join(out) if out else aria_yaml_to_prose(raw)
    except Exception:
        prose = aria_yaml_to_prose(raw)
    _SCOPE_CACHE[key] = prose
    return prose

def clear_scope_cache() -> None:
    """Called by RalphAgent at turn start so a stale page never leaks across turns."""
    _SCOPE_CACHE.clear()
```
```python
# Target: vibeharness/cli.py  (around line 578-580, the prose A/B wrap)
# Current: raw_snapshot_provider = lambda: aria_yaml_to_prose(_inner_snapshot_provider())
# Proposed: scope to the foreground region first when one is present.
if raw_snapshot_provider is not None and config.web_snapshot_prose:
    _inner = raw_snapshot_provider
    raw_snapshot_provider = lambda: scope_to_region(_inner())
```

---

## Takeaway 3: Stateless specialist sub-agents (planner / actor / locator split)

### What
Decompose the monolithic decide() turn into stateless specialists, each fed ONLY what it needs:
a **Planner** (goal + compacted tree → ordered natural-language steps), an **Actor** (one step +
live tree → tool calls), and a **Locator** (description + tree → a single id) used by guards that
today scan the snapshot with regex. Each is a fresh, history-free LLM call. VibeHarness has one
`RalphAgent` that does everything in `decide()`/`decide_chat()`, plus a separate `Validator` and an
optional `advisor`.

### Why
Alumnium runs six small single-shot stateless agents over the compacted tree (`alum_research.md`
TL;DR table); each builds a fresh 2-message prompt from current inputs with no shared history
(`alum_research.md` §e). The transferable win called out explicitly is "split planner (NL steps) /
actor (grounding+tools)" (`alum_research.md` Notes-for-#160). For a 3B model this matters more than
for a frontier model: `improvements_recommendations_analysis.md` §3 is built on the same principle —
"keep the model where it's strong, remove it where it's weak". Planning (decide *what*) is the
model's strength; grounding a description to a ref is fidelity work where the 3B is weak. A
dedicated Locator also lets us replace several brittle regex guards in `web.py`
(`_find_date_combobox_ref`, `find_option_ref_by_text`) with a uniform resolver returning a single
id — Alumnium's LocatorAgent pattern ("Your final response must be only the numerical id",
`alum_research.md` §d).

### Source
`alum_research.md` TL;DR agent table; §b (PlannerAgent NL strings vs ActorAgent grounding,
`PlannerAgent.ts:175-205`, `ActorAgent.ts:62-102`); §d LocatorAgent
(`prompts/locator/openai/system.md`). Cross-ref `improvements_recommendations_analysis.md` §3.

### Implementation in VibeHarness
```python
# Target: vibeharness/agent.py (new lightweight specialist seam alongside RalphAgent)
# Current state: RalphAgent.run does build_turn_prompt -> _decide/_decide_chat -> codec.parse,
#   all in one role. No planner; the only other LLM roles are Validator + advisor.
# Proposed change: a stateless Specialist that each turn receives ONLY (current step,
#   compacted page) — no narrative history — matching Alumnium's per-call freshness. The
#   Planner (Takeaway 4) supplies the step. Additive: gated behind a config flag.

class Specialist:
    """A stateless single-shot LLM role: fresh 2-message prompt, no history carried."""
    def __init__(self, client: "LLMClient", system: str, codec: "ToolCallCodec"):
        self._client, self._system, self._codec = client, system, codec

    def act(self, goal: str, step: str, page: str,
            constraint: "DecodeConstraint") -> "list[ToolCall]":
        # Alumnium ActorAgent: "Use goal only for context, focus on executing individual
        # step." No narrative, no chat_history — just goal + step + the live compacted tree.
        user = (f"Goal (context only): {goal}\n"
                f"Step to execute now: {step}\n\n"
                f"Current page:\n{page}")
        decision = self._client.decide(self._system, user, constraint)
        actions, _err = self._codec.parse(decision.action_json)
        return actions or []
```
```python
# Target: vibeharness/web.py — a Locator that unifies the ad-hoc ref-finding regexes.
# Current state: _find_date_combobox_ref / find_option_ref_by_text / find_nav_button_ref each
#   re-implement "scan snapshot for a line matching X -> return its ref".
# Proposed change: one resolver; deterministic substring match first (cheap), LLM Locator
#   only on a miss.
def locate_ref(snapshot: str, description: str) -> str | None:
    """Single-id element resolution (Alumnium LocatorAgent). Deterministic substring match
    first; an LLM Locator call can be added on ambiguity. Returns the bare ref or None (we
    fall back to None so the caller surfaces the real refs, vs Alumnium's root-id fallback)."""
    desc = description.lower()
    for line in snapshot.splitlines():
        if desc in line.lower():
            m = _REF_RE.search(line)
            if m:
                return f"e{m.group(1)}"
    return None
```

---

## Takeaway 4: Plan-all-steps-up-front

### What
Make ONE planning LLM call at run start that turns the goal into an ordered list of
natural-language steps, then execute them one-by-one with a fresh live-DOM read per step — instead
of re-deriving "what to do next" via full reasoning every single turn. VibeHarness today plans
implicitly inside every turn's reasoning.

### Why
Alumnium's `do()` calls the planner once (`alumni.py:100`, `POST /plans`), then loops the steps
executing each against a freshly re-read tree (`alumni.py:103-117`); there is "no internal agent
self-loop / no max-iteration retry inside an agent" (`alum_research.md` §b). The planner emits
natural-language strings ("click button Foobar"), *not* tool calls, decoupling planning from
grounding (`PlannerAgent.ts:175-205`). For VibeHarness this eliminates per-turn reasoning overhead
(`config.py` reserves `reason_tokens=4096` every turn) and gives the anti-loop machinery in
`agent.py` a stable target: a model that knows step 3 of 8 is "set the start date" cannot wander
back to step 1 (the exact iter-2 `page2→page1` oscillation the `_BACK_BUTTON_RE` guard fights). The
planner's rule "Always aim to minimize the number of actions" (`alum_research.md` §d) also counters
the 3B's tendency to take redundant steps.

### Source
`alum_research.md` §b `do()` loop (`alumni.py:86-133`), PlannerAgent NL-steps + zod `Plan`
(`PlannerAgent.ts:41-51, 175-205`), §d planner constraint list + few-shot
(`prompts/planner/openai/system.md`).

### Implementation in VibeHarness
```python
# Target: vibeharness/agent.py (RalphAgent.run — add an optional up-front plan)
# Current state: run() loops turns; each turn build_turn_prompt(task, memory.render(), ...)
#   then decide(). No global plan object exists.
# Proposed change: when config.plan_first is set, make one planning call before the loop and
#   thread the current step into the turn prompt. Steps are NL strings; grounding stays with
#   the existing decide()/codec path (the Actor). Re-reading the live page each turn ALREADY
#   happens (the post-turn page_snapshot capture), so we only add the plan.

def _make_plan(self, task: str, page: str) -> list[str]:
    """One stateless planning call: goal + initial compacted page -> ordered NL steps.
    Mirrors Alumnium PlannerAgent: NL strings, no element ids, minimize step count."""
    system = (
        "You plan a web task as an ordered list of short natural-language steps. "
        "Rules: do NOT include element ids/refs; ground steps in the page shown; "
        "minimize the number of steps; one action per step; if the goal is impossible "
        "from here, return an empty list. Output one step per line, no numbering.")
    user = f"Goal: {task}\n\nCurrent page:\n{page}"
    decision = self._client.decide(system, user, DecodeConstraint())  # unconstrained
    return [ln.strip("-• ").strip()
            for ln in (decision.action_json or "").splitlines() if ln.strip()]

# In run(), before the turn loop:
#   plan = self._make_plan(task, self._raw_snapshot_provider()) if self._cfg.plan_first else []
#   step_idx = 0
# In the turn body, surface the active step in the user message (recency zone):
#   if plan and step_idx < len(plan):
#       user += f"\n\n# Current plan step ({step_idx+1}/{len(plan)}): {plan[step_idx]}"
# Advance step_idx when the turn's actions all succeeded (action.ok), so a stuck step repeats.
```
```python
# Target: vibeharness/config.py
plan_first: bool = False   # opt-in: one up-front NL plan, executed step-by-step (Alumnium do())
```

---

## Takeaway 5: Change-summary compression (return "what changed", not the whole snapshot)

### What
After a turn's actions, instead of re-injecting the FULL page snapshot into history, inject a
compact 3-5 sentence "what changed" summary (first sentence = the high-level navigation/action),
keeping the full snapshot only for the *current* decision. VibeHarness today appends the entire
post-turn snapshot as a `page_snapshot` observation into memory/history every turn.

### Why
Alumnium's ChangesAnalyzer returns a 1-2 paragraph (3-5 sentence) NL diff
(`ChangesAnalyzerAgent.ts`, `prompts/changes-analyzer/openai/system.md`) that "keeps the
returned-to-orchestrator payload tiny" and "is the key to cheap MCP-subagent operation"
(`alum_research.md` §d/§e). VibeHarness's history is its biggest controllable cost: `agent.py`
`_evict_old_page_snapshot` exists solely because stale full snapshots "contradict the current page
state and waste the context window". A 3-5 sentence summary is ~100 tokens vs an ~11k-token
snapshot; keeping summaries in history while showing only the latest *full* snapshot for the active
decision matches Alumnium exactly and frees thousands of tokens for more steps within
`num_ctx=32768`. VibeHarness already computes the raw diff (`_diff_snapshot_refs`,
`_check_dom_delta`) — it just renders it verbosely; this converts that signal into a compact
summary. URL-change can be prepended deterministically with no LLM call, as Alumnium does
(`serverApp.ts:344-351`).

### Source
`alum_research.md` §d ChangesAnalyzer output rules ("Output 1-2 short paragraphs (3-5 sentences)",
ignore footer/nav chrome), §e change-summary as cross-turn memory (`alumni.py:119-133`),
deterministic URL prepend (`serverApp.ts:344-351`).

### Implementation in VibeHarness
```python
# Target: vibeharness/web.py (_check_dom_delta family) + vibeharness/agent.py (history commit)
# Current state: agent.py records the FULL post-turn snapshot as the page_snapshot observation
#   ("## Latest page state ...\n\n" + snap). _check_dom_delta already diffs before/after refs.
# Proposed change: build a compact NL summary from the diff + URL change; keep the full
#   snapshot ONLY as the current-turn page view, store the SUMMARY in history.

def summarize_change(before: str, after: str) -> str:
    """3-5 sentence 'what changed' summary from two snapshots (Alumnium ChangesAnalyzer,
    deterministic variant). First the URL/page change; then new + removed interactable
    controls, ignoring nav/footer chrome. No LLM call (a constrained tiny-LLM summary is an
    optional upgrade)."""
    new_refs = _diff_snapshot_refs(before, after)
    gone = (set(re.findall(r"\[ref=(e\d+)\]", before or "")) -
            set(re.findall(r"\[ref=(e\d+)\]", after or "")))
    url_before, url_after = parse_page_url(before), parse_page_url(after)
    parts: list[str] = []
    if url_after and url_after != url_before:
        parts.append(f"The page navigated to {url_after}.")
    alerts = _extract_validation_alerts(after)
    if alerts:
        parts.append("Validation errors are now shown: " + "; ".join(alerts[:3]) + ".")
    if new_refs:
        lines = _extract_ref_lines(new_refs, after)[:4]
        parts.append(f"{len(new_refs)} new control(s) appeared: "
                     + "; ".join(l[:60] for l in lines) + ".")
    if gone:
        parts.append(f"{len(gone)} previous control(s) are gone (the step likely advanced).")
    if not parts:
        parts.append("The page did not visibly change after your action.")
    return " ".join(parts[:5])
```
```python
# Target: vibeharness/agent.py (RalphAgent.run, the post-turn snapshot block ~line 418-429)
# Current: records the full snapshot every turn. Proposed: record the SUMMARY in narrative
#   history, while the live full snapshot is still shown via the system-prompt provider for
#   THIS turn's decision (already rebuilt fresh each turn).
if self._raw_snapshot_provider is not None and not result.finished:
    time.sleep(1)
    after = self._raw_snapshot_provider()
    summary = summarize_change(self._prev_snapshot or "", after)
    self._record(turn, Action("page_change", {}, summary, ok=True), memory)
    self._prev_snapshot = after   # full snapshot lives only as the next turn's live view
```

---

## Takeaway 6: Negative-instruction tool docstrings

### What
Bake tool *disambiguation* into each tool's `description` as an explicit "do NOT use this for X —
use Y instead" clause, rather than relying on the shared system guidance to keep tools apart.
VibeHarness has this on a couple of tools (`DrawSignatureTool`: "Do NOT use this on the 'Type your
full legal name' textbox"; `UploadTool`: "do NOT click the trigger separately first") but not
systematically across the toolset.

### Why
Alumnium's `ClickTool` docstring is one field plus a negative instruction:
`"Click an element. NEVER use ClickTool to upload files - use UploadTool instead."` — "tool
disambiguation is baked into the description, not the system prompt" (`alum_research.md` §c,
`click_tool.py:8-11`). This is the cheapest, lowest-risk recommendation and aligns with
`improvements_recommendations_analysis.md` R5 ("teach the tools" — a 3B does not have latent tool
knowledge; spend a few tokens Kira doesn't) and the live Type-F confusions documented there
(`fill` vs `select_option`, `write_file` vs `create_file`). Crucially, with native tools on
(`config.native_tools=True`), the `description` field is exactly what Ollama renders into the
model's own `# Tools` block (`hermes_codec.tools()` passes `t.description`), so a negative clause in
the description reaches the model on the native path where the harness's `system_guidance` block is
deliberately omitted (`prompt.py` `_SYSTEM_TEMPLATE_NATIVE`).

### Source
`alum_research.md` §c negative-instruction docstring (`click_tool.py:8-11`,
`tool_to_schema_converter.py:49-69`). Cross-ref `improvements_recommendations_analysis.md` R5 +
failure taxonomy rows E/F.

### Implementation in VibeHarness
```python
# Target: vibeharness/web.py (tool description strings)
# Current state: descriptions describe positive use; cross-tool disambiguation lives mostly
#   in WebToolset.system_guidance (omitted on the native_tools path) and in runtime steer
#   messages AFTER a wrong call.
# Proposed change: add a terminal negative clause to each ambiguous tool, so the model is
#   disambiguated BEFORE the wrong call, on every path (native + legacy).

class FillTool(_WebTool):
    description = ("Set a text input / textarea to an exact value, clearing it first. "
                  "NEVER use fill on a dropdown/combobox/listbox or a number stepper "
                  "(spinbutton) — use select_option for dropdowns and set_spinbutton for "
                  "steppers. " + _REF_NOTE)

class ClickTool(_WebTool):
    description = ("Click an element (a link, button, checkbox, …). NEVER use click to upload "
                  "a file — use upload. NEVER use click to set a dropdown value — use "
                  "select_option. " + _REF_NOTE)

class SelectOptionTool(_WebTool):
    description = ("Choose an option in a dropdown — a native <select> OR a custom "
                  "listbox/combobox. NEVER use select_option on a plain action button "
                  "(Continue/Submit) — use click — or on a spinbutton — use set_spinbutton. "
                  + _REF_NOTE)
```
This mirrors the runtime steers already in `agent.py`/`web.py` (the HARD STOP text, the
plain-button guard) but moves the knowledge *upstream* into the schema, so most wrong calls never
happen — eliminating the wasted turn the guards only catch *after the fact*.

---

## Takeaway 7: Small sequential id remapping at the tool-call seam

### What
Complete Takeaway 1's loop: when the compacted snapshot renumbers `e1275 → 3`, translate the
model's `target: "3"` back to `e1275` at the single `_WebTool` chokepoint before the playwright
ref guard runs, using the `id_to_ref` map. The model reasons over compact ids; the browser is
driven by stable refs — exactly Alumnium's `mapToolCallsToRawId`.

### Why
Alumnium remaps `id`/`from_id`/`to_id` back to real `raw_id`s before returning tool calls to the
client (`BaseServerAccessibilityTree.mapToolCallsToRawId:27-48`, applied at `serverApp.ts:173`),
and `#extractId` even handles model quirks (Gemini float ids, Llama string/nested ids) —
`alum_research.md` §c. VibeHarness already has the perfect single seam for this: every web tool's
`target` flows through `_WebTool._run_impl` → `normalize_ref` → `_guard_target`
(`parse_snapshot_refs`). Inserting the id→ref translation there means *no per-tool changes* (the
docstring rationale in `snapshot_prose.py` worried about a mapping table; the seam makes it one
function). It also strictly improves the existing anti-hallucination guard: a small id space means
fewer invalid targets, and any leftover invalid id is rejected against the dense map.

### Source
`alum_research.md` §c "Element-id indirection (key optimization)"
(`BaseServerAccessibilityTree.ts:17-67`, `serverApp.ts:173`), §a `simplifiedToRawId`.

### Implementation in VibeHarness
```python
# Target: vibeharness/web.py (_WebTool — translate id->ref before the existing guard)
# Current state: _run_impl validates args["target"] as a playwright ref via normalize_ref +
#   _guard_target. The CLI is bound per-run; the snapshot provider produces the compacted
#   prose. We need the current turn's id_to_ref map available to the tools.
# Proposed change: store the latest id_to_ref on the shared SessionState (already the
#   run-scoped object every tool + snapshot provider share, keyed by session name), set when
#   the snapshot is compacted, read when a tool resolves its target.

# In SessionState.__init__: self.id_to_ref: dict[str, str] = {}
# In the snapshot provider (cli.py), after compaction:
#   shared_session_state(config.web_session).id_to_ref = compacted.id_to_ref

class _WebTool(Tool):
    def _resolve_target(self, raw_target: str) -> str:
        """Map a small sequential id (from the compacted snapshot) back to a playwright ref.
        Mirrors Alumnium mapToolCallsToRawId. A value already a ref, or not in the map, is
        returned unchanged so the existing guard handles it."""
        t = (raw_target or "").strip().lstrip("[").rstrip("]")
        return self._cli.state.id_to_ref.get(t, raw_target)

    def _run_impl(self, args: dict) -> ToolResult:
        # Translate sequential-id targets to refs BEFORE missing-param / guard checks, so the
        # rest of the pipeline (normalize_ref, _guard_target, DOM-delta) is unchanged.
        for key in ("target", "end"):
            if args.get(key):
                args[key] = self._resolve_target(args[key])
        ...  # existing body unchanged
```

---

## Takeaway 8: Stateless, history-free turns (live DOM is the memory)

### What
Offer a stateless turn mode where the model carries NO accumulated chat history — only the current
compacted live DOM plus the latest change-summary (Takeaway 5) — instead of the growing
`chat_history`/narrative VibeHarness maintains today. State lives in the page, not the transcript.

### Why
"Core principle: agents are stateless and history-free" (`alum_research.md` §e). Alumnium carries
information across turns via three substitutes for history: the live DOM (re-fetched every step),
the ChangesAnalyzer summary, and few-shot examples — verified across every agent's prompt
construction (`ActorAgent.ts:50-60`, etc.). "The model always sees current truth, eliminating
stale-context drift." VibeHarness's native path keeps a full `chat_history` and FIFO-evicts it
(`agent.py:_evict_history`), and already fights stale state with `_evict_old_page_snapshot`. The
3B is especially vulnerable to stale-context confusion (the "I already did X" guards, the
`attempted`/`handled_refs` machinery). A stateless mode — current page + a short summary of the
last change — is closer to Alumnium's proven WebVoyager design and reclaims the entire history
budget for a bigger/clearer current page. This also composes with
`improvements_recommendations_analysis.md` R1 (clean the reasoning→action seam): less garbled
history at the generation point.

### Source
`alum_research.md` §e "agents are stateless and history-free" (`SessionContext.ts:22-43`,
`LlmContext.ts:13-41` — only app-id + cache-meta, no chat memory); cross-turn substitutes
(`alumni.py:105, 119-133`). Cross-ref `improvements_recommendations_analysis.md` R1.

### Implementation in VibeHarness
```python
# Target: vibeharness/agent.py (RalphAgent.run) + vibeharness/config.py
# Current state: native path appends user/assistant/tool messages to chat_history each turn
#   (_commit_turn_to_history) and FIFO-evicts (_evict_history). Legacy path embeds
#   memory.render() into every turn's user message.
# Proposed change: a stateless mode that sends ONLY [system, current-user]; the current-user
#   carries the compacted live page + the last change summary, never the full transcript.

# config.py:
stateless_turns: bool = False   # Alumnium-style: no chat history; live DOM + change summary

# In run(), when self._cfg.stateless_turns:
#   - do NOT build chat_history; call self._decide(system, user, constraint) each turn
#   - user = self._stateless_user(task, last_change_summary)   (T5 output, not memory.render())
def _stateless_user(self, task: str, last_summary: str) -> str:
    return build_turn_prompt(
        task,
        narrative=(f"Result of your last action: {last_summary}" if last_summary
                   else "This is your first action."),
        action_hint=self._codec.turn_action_hint(),
    )
```
Keep this opt-in: the existing anti-loop guards (`attempted`, `target_block_counts`) remain in
process state (not model context), so they still work without a transcript.

---

## Takeaway 9: `<SEP>`-delimited list codec for multi-value returns

### What
When a tool or extraction returns multiple values, encode them as a single separator-delimited
string (`a<SEP>b<SEP>c`) rather than a JSON array, and split on the client. Add this as a small,
isolated helper for any "get/extract a list" tool (e.g. reading all option labels, all validation
errors, a list of links).

### Why
Alumnium's RetrieverAgent returns lists as one string joined by a sentinel `<SEP>` and splits
client-side, with explicit robustness hacks for model misbehavior (trims stray separators, fixes
GPT-5 Nano brace replacement, Grok escaped-tag) — `alum_research.md` §c (`RetrieverAgent.ts:62,
147-173`). For a weak 3B this is more robust than a JSON array: there are no brackets/commas/quotes
to malform, which is precisely the failure class `improvements_recommendations_analysis.md` §2
catalogs (Type A malformed shape, Type C corrupted string node). A flat delimited string has a
trivially recoverable grammar even when the model fumbles punctuation. This belongs near the
existing codec seam (`codec.py`), which is explicitly the home for "new formats as new isolated
modules" (`get_codec`).

### Source
`alum_research.md` §c "Structured-output result decoding (Retriever)"
(`RetrieverAgent.ts:62, 147-173`; `<SEP>` separator, NOOP sentinel). Cross-ref
`improvements_recommendations_analysis.md` §2 failure taxonomy.

### Implementation in VibeHarness
```python
# Target: vibeharness/codecs/sep_list.py (NEW isolated module — next to the codec seam)
# Current state: only json/xml/tagged_json/hermes codecs exist; all parse object/array JSON.
#   No delimited-list format. List returns are tool RESULTS, not tool calls, so this is a
#   shared parser the retriever-style tools import — not a full ToolCallCodec.
# Proposed change: a tiny encode/decode pair with Alumnium's robustness trims.

_SEP = "<SEP>"

def encode_list(items: list[str]) -> str:
    """Join values with the <SEP> sentinel (Alumnium #LIST_SEPARATOR). Empty -> 'NOOP'."""
    items = [i for i in (s.strip() for s in items) if i]
    return _SEP.join(items) if items else "NOOP"

def decode_list(raw: str) -> list[str]:
    """Split a <SEP>-joined string back to a list, applying Alumnium's robustness trims:
    strip stray leading/trailing separators, tolerate the escaped &lt;SEP&gt; form some
    models emit, and treat 'NOOP' as the empty/not-found sentinel (RetrieverAgent.ts
    147-173)."""
    if not raw:
        return []
    s = raw.replace("&lt;SEP&gt;", _SEP).strip()
    if s.upper() == "NOOP":
        return []
    s = s.strip(_SEP)
    return [part.strip() for part in s.split(_SEP) if part.strip()]
```

---

## Takeaway 10: Provider-specific prompt variants (Qwen vs VibeThinker divergence)

### What
Promote prompts from inline string constants to a per-(role × provider) directory of files, so each
model family gets a tuned system prompt — directly serving the `beta`/`beta_qwen3coder`/
`beta_mythos_fast` divergence the repo already maintains. Today VibeHarness varies the *format
block* per codec (`format_instructions`) but the *system template* is a single `prompt.py` constant
(`_SYSTEM_TEMPLATE` / `_SYSTEM_TEMPLATE_NATIVE`).

### Why
Alumnium stores prompts as per-agent × per-provider markdown files
(`prompts/<agent>/<provider>/{system,user}.md`), selected via `PROVIDER_TO_PROMPTS_DEV` with an
`openai` fallback (`alum_research.md` §d). The divergence is concrete and load-bearing: the
Anthropic planner adds rule 9 "When planning to type into textbox, skip clicking it regardless of
its focused status" plus a matching example tweak — "Demonstrates per-model prompt tuning"
(`alum_research.md` §d). VibeHarness has the *exact same need*: `config.py` documents Qwen3
non-thinking sampling, `/no_think`, and the `_SYSTEM_TEMPLATE_NATIVE` that omits tool docs — all
Qwen-specific — living tangled with the generic template. A per-provider file layout makes the
mandated cross-branch divergences (CLAUDE.md §6; `QWEN3CODER_DIVERGENCE.md`, `MYTHOS_DIVERGENCE.md`)
explicit and reviewable instead of buried in conditionals, and lets the Qwen branch tune its prompt
without touching VibeThinker's.

### Source
`alum_research.md` §d "System Prompt Optimizations" (`prompts.ts` `loadAgentPrompts:55-124`,
`PROVIDER_TO_PROMPTS_DEV:35-49`, anthropic planner rule-9 divergence). Cross-ref CLAUDE.md §6 sync
rules + `QWEN3CODER_DIVERGENCE.md`.

### Implementation in VibeHarness
```python
# Target: vibeharness/prompt.py (load per-provider template) + a vibeharness/prompts/ tree
# Current state: SystemPromptBuilder.build formats one of two module-level string constants.
#   No provider dimension; Qwen-specifics (/no_think, native template) are hard-coded.
# Proposed change: a provider key on Config selects a template file; fall back to the bundled
#   default string so existing behaviour is unchanged when no override file exists. Mirrors
#   Alumnium's PROVIDER_TO_PROMPTS_DEV with an openai/default fallback. Use importlib.resources
#   (not a filesystem glob) so it survives the PyInstaller freeze — the same constraint the
#   codec discovery already handles.

# config.py:
prompt_provider: str = "qwen3"   # selects prompts/<role>/<provider>/system.md; "" = default

# prompt.py:
import importlib.resources as _res

_PROVIDER_FALLBACK = {"qwen3": "qwen3", "vibethinker": "default", "mythos_fast": "mythos"}

def _load_template(role: str, provider: str, default: str) -> str:
    """Load prompts/<role>/<dialect>/system.md, falling back to the bundled default string
    (so a missing override never breaks a build — Alumnium falls back to the openai dialect)."""
    dialect = _PROVIDER_FALLBACK.get(provider, "default")
    try:
        pkg = f"vibeharness.prompts.{role}.{dialect}"
        return _res.files(pkg).joinpath("system.md").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return default

# In SystemPromptBuilder.build, replace the literal _SYSTEM_TEMPLATE_NATIVE reference with:
#   template = _load_template("agent", self._provider, _SYSTEM_TEMPLATE_NATIVE)
#   body = template.format(tool_guidance=tool_guidance, max_actions=max_actions_label)
```
This keeps the bundled constants as the `default`, and gives each branch a clean place for its
divergence per CLAUDE.md §6.

---

## Takeaway 11: Combine the interactive/visible prune (SNAPSHOT_SCOPING) with sequential-ID renumber

### What
Unify our existing snapshot research with Alumnium's: run the **interactive/visible prune** that
`SNAPSHOT_SCOPING_ANALYSIS.md` proved lossless-on-actionable, THEN apply the **sequential-ID
renumber + name-dedup** (Takeaway 1), THEN **hoist `role=dialog` subtrees to the top** — a single
pipeline that is both smaller and more model-legible than either technique alone.

### Why
`SNAPSHOT_SCOPING_ANALYSIS.md` establishes three findings VibeHarness already trusts: (1) the prune
keeps 100% of buttons/links/inputs at 12–28% size reduction; (2) `--depth` is disqualified because
it silently clips deep controls (MDN Yes/No); (3) the deeper fix is "emit overlay/dialog nodes
FIRST" so consent controls survive any cap. Alumnium's renumber/dedup is *orthogonal* and
*composable*: prune removes noise nodes, renumber shrinks the IDs of the survivors, dedup removes
repeated label text, and dialog-first reorders for recency. Together they attack the YouTube case
from the analysis (consent buttons at char 40,300, past the old 40k cap) from two directions —
fewer chars AND the consent dialog at the top — while the dense ID space (Takeaway 7) removes the
hallucination surface that the whole `_guard_target` apparatus in `web.py` exists to patch. This is
the single highest-leverage change in this document.

### Source
`SNAPSHOT_SCOPING_ANALYSIS.md` PART A (prune lossless), "Combined recommendation" (#1 prune, #3
dialog-first), Measurement table (YouTube/BBC/MDN/w3schools). Cross-ref `alum_research.md` §a
(renumber + `#pruneRedundantName`).

### Implementation in VibeHarness
```python
# Target: vibeharness/snapshot_prose.py (single ordered pipeline) + cli.py wiring
# Current state: aria_yaml_to_prose does prune + foreground-dialog NOTE in the preamble, but
#   the dialog SUBTREE is still rendered in DOM order (last), and refs are not renumbered.
#   scripts/snapshot_prune_prototype.py (research) is not wired in.
# Proposed change: one function chaining prune (existing _walk/_is_interesting) -> dedup (T1)
#   -> dialog-hoist -> renumber (T1), returning CompactedSnapshot.

def compact_snapshot(raw: str) -> "CompactedSnapshot":
    """The unified pipeline: WebArena prune (existing) + name-dedup + dialog-first hoist +
    sequential-id renumber. Order matters — hoist BEFORE renumber so the dialog's controls
    get the LOWEST ids (top of the prose, recency-favoured), directly fixing the #24 'consent
    past the cap' case from SNAPSHOT_SCOPING_ANALYSIS.md."""
    if not raw or not raw.strip():
        return CompactedSnapshot(prose=raw, id_to_ref={})
    try:
        _url, _title, body = _split_header_and_yaml(raw)
        root = _build_tree(body)
        if not root.children:
            return CompactedSnapshot(prose=raw, id_to_ref={})
        _dedup_descendant_names(root)                       # T1: #pruneRedundantName
        dialog = _find_active_dialog(root)                  # existing helper
        if dialog is not None and dialog in root.children:  # T11: hoist dialog to front
            root.children.remove(dialog)
            root.children.insert(0, dialog)
        # renumber + render exactly as aria_yaml_to_compacted (T1): kept (pruned) nodes get
        # dense ids in walk order, dialog controls first.
        return _renumber_and_render(root)                   # shared tail of T1
    except Exception:
        return CompactedSnapshot(prose=raw, id_to_ref={})
```
```python
# cli.py: route the compacted prose AND publish its id map for Takeaway 7's tool remap.
if raw_snapshot_provider is not None and config.web_snapshot_prose:
    _inner = raw_snapshot_provider
    def _provider():
        compacted = compact_snapshot(_inner())
        shared_session_state(config.web_session).id_to_ref = compacted.id_to_ref
        return compacted.prose
    raw_snapshot_provider = _provider
```

---

## Takeaway 12: Decode-time grounding via the codec robustness seam

### What
Tie Alumnium's element-id indirection to the codec-robustness program in
`improvements_recommendations_analysis.md`: extend `ToolCallCodec.constraint(...)` with a per-turn
**context** so referential args (`target` refs, and after Takeaway 1 the small ids) become a
decode-time **enum of the ids actually on the page** — making a hallucinated/incremented id
*structurally impossible*, exactly as the analysis's R2 prescribes for filenames.

### Why
`improvements_recommendations_analysis.md` R2 is the highest-leverage codec change: bind referential
args to the real set as a JSON-schema `enum`, which "Ollama honours via `format` today". Its one
blocker is that `ToolCallCodec.constraint(registry, max_actions)` (`codec.py:97-99`) "has no access
to per-turn runtime state, so it can't build a dynamic enum today" — the seam must grow a context
argument. Alumnium independently validates the *target* of this: the model only ever picks from the
small id set the compacted tree exposes (`alum_research.md` §a/§c), and the server maps it back.
Combining them: VibeHarness's compacted snapshot (Takeaways 1/7/11) already produces the exact set
of valid ids per turn — feeding that set as the `target` enum closes the loop the `hermes` codec
currently leaves open (it returns `json_schema=None`, fully unconstrained — `hermes_codec.constraint`).
For the Qwen branch the enum would apply on a future constrained codec or a llama.cpp backend (R7);
even without enforcement, surfacing the valid-id set per turn is the data Takeaway 7 needs.

### Source
`improvements_recommendations_analysis.md` R2 (reference-binding enums; the
`constraint(registry, max_actions, context)` seam extension), R4 (discriminated decode), §0.2
("move the fragile work off the model"). Cross-ref `alum_research.md` §c element-id indirection;
current seam `codec.py:97-99`, `codecs/hermes_codec.py:261-266`.

### Implementation in VibeHarness
```python
# Target: vibeharness/codec.py (seam signature) + codecs/*_codec.py + agent.py (pass context)
# Current state: ToolCallCodec.constraint(self, registry, max_actions) -> DecodeConstraint,
#   with NO per-turn state. RalphAgent computes the constraint ONCE before the loop
#   (agent.py: constraint = self._codec.constraint(self._registry, limit)). hermes returns
#   json_schema=None (unconstrained); json builds a static oneOf schema.
# Proposed change: add an optional TurnContext carrying the current valid id/ref set, and
#   compute the constraint PER TURN. Additive default keeps every existing codec working.

from dataclasses import dataclass

@dataclass(frozen=True)
class TurnContext:
    """Per-turn runtime facts a codec may use to build a dynamic constraint (R2). Today:
    the set of element ids/refs currently on the page, from the compacted snapshot."""
    valid_target_ids: tuple[str, ...] = ()

class ToolCallCodec(ABC):
    def constraint(self, registry: "ToolRegistry", max_actions: int,
                   context: "TurnContext | None" = None) -> DecodeConstraint:
        ...  # default impls ignore context -> behaviour unchanged

# agent.py — build the constraint each turn with the live id set:
#   ctx = TurnContext(valid_target_ids=tuple(self._cli_state.id_to_ref))   # from T7/T11
#   constraint = self._codec.constraint(self._registry, limit, ctx)

# A grounded JSON codec (codecs/grounded_json_codec.py) turns target into an enum:
def _ground_target(call_schema: dict, ids: tuple[str, ...]) -> dict:
    """Replace the free-string `target` property with an enum of the real on-page ids, so a
    fabricated/incremented id is rejected at decode time (R2). New-value args (text/url/file)
    are left free — R2 covers REFERENCE, not creation."""
    props = call_schema.get("properties", {})
    if "target" in props and ids:
        props["target"] = {"type": "string", "enum": list(ids)}
    return call_schema
```
This makes the seam the home for both R2 (reference-binding) and R4 (discriminated decode), and is
the structural counterpart to the per-tool negative instructions of Takeaway 6: docstrings steer
the model *before* generation, the enum constrains it *during* generation, and the `_guard_target`
in `web.py` catches anything that still slips through *after*.

---

## Sequencing for the orchestrator

1. **Takeaways 1 + 7 + 11 together** (compaction + sequential-id remap + unified pipeline) — one
   coherent change in `snapshot_prose.py` + `web.py` + `cli.py`; highest leverage, attacks the #1
   ref-hallucination failure class head-on. Land behind the existing `web_snapshot_prose` seam.
2. **Takeaway 6** (negative docstrings) — trivial, low-risk, helps immediately on the native path.
3. **Takeaway 5 + 8** (change-summary + optional stateless turns) — reclaims the history budget;
   gate stateless mode behind a flag, keep summaries on by default.
4. **Takeaway 2** (area scoping) — composes with #1; dialog-first scope is the cheap first cut.
5. **Takeaways 3 + 4** (specialist split + plan-first) — larger; opt-in flags, A/B against the
   current single-role loop.
6. **Takeaway 12** (codec context seam) — the R2 enabler; land the signature change first, then a
   `grounded_json_codec`. **Takeaway 9** (`<SEP>` list) and **Takeaway 10** (per-provider prompts)
   are independent and parallelizable.

All changes stay within `beta_qwen3coder`'s mandate (CLAUDE.md §6): per-provider prompts and the
compaction pipeline are exactly where this branch is *meant* to diverge from `beta`/VibeThinker, so
none of this should flow back to the hub.
