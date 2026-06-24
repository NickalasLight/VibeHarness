# Token-Efficient Tool-Call Formats for Small Models (VibeHarness / VibeThinker-3B)

Research deliverable. **No production code was changed.** This is ground-truth on
alternatives to heavy JSON for LLM tool/function calls, focused on what a ~3B model
running under Ollama/llama.cpp can emit *reliably* and *cheaply*, and on what we can
keep grammar-constrained at decode time.

Every non-obvious claim is cited with a URL. Where the evidence is thin or points the
other way, that is called out explicitly — see **§7 Honest caveats**.

---

## 0. What VibeHarness does today (grounding)

So recommendations map to real code in this repo:

- The model emits a **JSON array** of actions, each `{"tool": <name>, "args": {...}}`.
  - Schema built in `vibeharness/registry.py` → `action_schema()` (`{"type":"array","minItems":1,"items":{"oneOf":[<per-tool call_schema>]}}`).
  - Per-tool call schema in `vibeharness/tools.py` → `Tool.call_schema()` = `{"tool": {"const": name}, "args": {...}}`.
- Decode is **grammar-constrained via Ollama's `format` field** = a JSON Schema, passed
  on the `/api/generate` call in `vibeharness/llm.py` (`_act()`, `"format": action_schema`).
  Ollama compiles that JSON Schema → a GBNF grammar inside llama.cpp.
- Generation is **two-phase**: phase 1 free `<think>` reasoning (stopped at `</think>`),
  phase 2 a raw ChatML continuation constrained to the schema (`llm.py` `_reason`/`_act`).
- Parsing is plain `json.loads` + light validation in `vibeharness/agent.py` `_parse()`
  (rejects non-arrays, requires `"tool"` per item).

So the three painful cases the task names map exactly to: a `read_file`/`write_file`
`args.path` holding a Windows path, a `write_file` `args.content` holding multi-line code,
and the array itself holding a multi-action batch.

---

## 1. The core problem with JSON for a 3B model

Two independent, well-supported facts:

1. **JSON is token-expensive vs. lighter formats.** XML needs ~80% more tokens than
   Markdown for the same data; Markdown uses **34–38% fewer tokens than JSON** (≈10% fewer
   than YAML) in a nested-data study across GPT-5 Nano, Llama 3.2 3B, Gemini 2.5 Flash Lite.
   ([improvingagents](https://www.improvingagents.com/blog/best-nested-data-format)) JSON's
   cost is structural: every key is quoted, every value is quoted/comma-delimited, braces
   nest. TSV/CSV-style and Markdown drop most of that punctuation
   ([D. Gilbertson, "Why JSON Costs More Than TSV"](https://david-gilbertson.medium.com/llm-output-formats-why-json-costs-more-than-tsv-ebaf590bd541)).

2. **Forcing JSON output can hurt the model's actual answer.** Practitioner write-ups
   report that constraining a model to emit JSON degrades reasoning/quality on the order of
   10–15% vs. letting it answer freely, which is why the "reason freely, then format"
   two-step is recommended — VibeHarness already does this split
   ([Hannecke, "Beyond JSON"](https://medium.com/@michael.hannecke/beyond-json-picking-the-right-format-for-llm-pipelines-b65f15f77f7d)).
   (This is a blog claim, not a peer-reviewed number — treat as directional.)

3. **The escaping problem is the real 3B killer, and it's format-intrinsic.** A JSON string
   value cannot contain a raw `\`, `"`, or newline — they must become `\\`, `\"`, `\n`.
   A Windows path `C:\Users\x\file.txt` must be emitted as `"C:\\Users\\x\\file.txt"`, and a
   multi-line file body must be a single line with every newline as `\n` and every quote
   escaped. Small models mangle exactly this — rare-token escaping and over-quoting are the
   symptoms you reported. Note: even Anthropic's *own* tool-use format sidesteps this by
   putting parameter values as **raw text between tags**, and explicitly says the output
   "is not expected to be valid XML and is parsed with regular expressions"
   ([Anthropic tool-use system prompt, leaked-system-prompts mirror](https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-api-tool-use_20250119.md)).
   The reason a format that puts content **between delimiters** instead of **inside a quoted
   string** is safer is that there is nothing to escape — the parser reads to the closing tag.

Grammar-constrained decoding (GCD) does **not** fix escaping reliability for free: a JSON
grammar still *forces* the model to produce `\\` for a backslash, and a small model fighting
the grammar can stall, loop, or emit degraded content inside the still-valid string. GCD
guarantees *syntactic* validity, not *semantic* correctness of the escaped payload. A
recent ACL industry paper found GCD improves both syntactic correctness *and* semantic
accuracy and "can serve as an effective substitute for in-context examples, especially …
for smaller models" — i.e. GCD is a real win for 3B-class models, but it's a win for any
grammar, including a cheaper non-JSON one
([Grammar-Constrained Decoding Makes LLMs Better Logical Parsers, ACL 2025 Industry](https://aclanthology.org/2025.acl-industry.34/)).

**Key implication:** we can move *off* JSON and *keep* grammar constraints, because the
constraint layer is llama.cpp GBNF, which accepts arbitrary grammars — not only JSON Schema.

---

## 2. Comparison table

Token cost is relative to the equivalent JSON tool call (↓ = cheaper). "Grammar-
constrainable" = can we express it as a GBNF grammar (or JSON-Schema) for decode-time
masking in llama.cpp.

| Format | Token cost | Escaping safety (paths/code/content) | Parse robustness | Grammar-constrainable | Small-model friendliness | Who uses it (cite) |
|---|---|---|---|---|---|---|
| **JSON array** (today) | baseline (heaviest) | **Poor** — `\`, `"`, newlines must be escaped inside quoted strings | High *if* it parses; brittle for the model to produce | **Yes** — JSON-Schema→GBNF; Ollama `format` does this natively | **Weak for content-heavy args**; fine for flat scalar args | OpenAI/Ollama/most APIs; BFCL "FC" mode ([BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)) |
| **XML / tag-based** (`<invoke name>`/`<parameter>`) | ↓ moderate (no per-key quotes; tags add some) | **Excellent** — values are raw text between tags, **nothing to escape** (paths, code, newlines all literal) | High; Anthropic parses with regex, tolerant of "invalid XML" | **Yes** — straightforward GBNF (open tag → raw run → close tag) | **Strong** for content/paths; Claude is trained on XML tags | Anthropic Claude tool use ([docs](https://platform.claude.com/docs/en/docs/use-xml-tags), [leaked prompt](https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-api-tool-use_20250119.md)) |
| **Code-as-action / CodeAct** (model writes Python that calls tools) | ↓↓ for multi-step (composability collapses steps) | **Good** — strings via Python literals, but **still needs `\\` or raw strings/triple-quotes**; can use `r"..."`/`"""..."""` to avoid escaping | Needs a real (sandboxed) interpreter; richer failure modes | **Partial** — can constrain to a Python subset via GBNF, but full Python is hard to grammar-lock | **Strong on capable models; risky on a raw 3B** (must write valid code) | CodeAct paper ([ICML 2024](https://proceedings.mlr.press/v235/wang24h.html)), smolagents `CodeAgent` ([HF](https://huggingface.co/docs/smolagents/en/index)) |
| **Terse line/shell DSL** (`tool key=value` per line; content via heredoc) | ↓↓ (least punctuation) | **Excellent** with a heredoc/sentinel for bodies; paths bare | Custom parser; ambiguity risk with spaces/quotes unless delimited | **Yes** — simple, *small* GBNF (easiest grammar of all) | **Strong** — closest to natural text a 3B emits well | SWE-agent ACI commands ([docs](https://swe-agent.com/0.7/background/aci/)), Aider edit blocks ([aider](https://aider.chat/docs/leaderboards/notes.html)) |
| **YAML** | ↓ small (≈10% vs JSON) | **Good** — block scalars (`\|`) hold multi-line/backslash content **unescaped** | Medium — indentation-sensitive; a 3B can mis-indent | **Yes-ish** — grammar is harder (indentation), but doable / or post-validate | Mixed: best on some big models, **JSON beat YAML on Llama-3B** | YAML fine-tune study ([Mohapatra](https://athekunal.medium.com/json-vs-yaml-function-calling-finetuning-comparison-25e97767cc5d)) |
| **Markdown / KV** (`key: value` lines, fenced code blocks) | ↓↓ (34–38% < JSON) | **Excellent** for bodies (fenced ```` ``` ```` blocks are literal) | Medium; needs a parser | **Yes** — GBNF for a fixed KV+fence shape | **Strong** for content; most token-efficient overall | Most token-efficient in nested-data study ([improvingagents](https://www.improvingagents.com/blog/best-nested-data-format)) |
| **TOML** | ↑ vs YAML (all strings quoted) | Good (multi-line `"""`/`'''` strings) | Medium | Yes-ish | Untested for tool calls; no clear small-model edge | general ([Hannecke](https://medium.com/@michael.hannecke/beyond-json-picking-the-right-format-for-llm-pipelines-b65f15f77f7d)) |
| **TOON** (Token-Oriented Object Notation) | ↓↓ ~40% < JSON | n/a-ish (designed for *uniform tabular input*, not arbitrary content) | Medium | Possible | **For input data, not tool-call output** — wrong tool for this job | ([toon-format/toon](https://github.com/toon-format/toon), [DigitalOcean](https://www.digitalocean.com/community/tutorials/toon-vs-json)) |

---

## 3. The same tool call, three formats, three worst cases

### Case A — a Windows path with backslashes (`read_file C:\Users\NickL\notes.txt`)

**JSON (today):** backslashes must double.
```json
[{"tool": "read_file", "args": {"path": "C:\\Users\\NickL\\notes.txt"}}]
```

**XML / tag-based (Anthropic-style):** path is raw text, nothing to escape.
```xml
<call name="read_file"><arg name="path">C:\Users\NickL\notes.txt</arg></call>
```

**Terse line DSL (heredoc reserved for bodies; scalars bare):**
```
read_file path=C:\Users\NickL\notes.txt
```

The JSON form is the only one where the model has to *invent* the `\\` escape. The 3B's
failure mode is emitting `C:\Users\...` (single backslash) inside the JSON string, which is
either invalid JSON or silently corrupts the path on parse.

### Case B — a multi-line file body (`write_file out.py` with code)

**JSON (today):** entire body collapses to one line; every newline `\n`, every quote `\"`.
```json
[{"tool":"write_file","args":{"path":"out.py","content":"import os\n\ndef main():\n    print(\"hi\\\\there\")\n"}}]
```
(Note the body needed `\n` ×4, `\"` ×2, and `\\\\` to land a literal `\` — extremely
error-prone for a small model, and token-heavy.)

**XML / tag-based:** body is literal between `<arg>` tags. No escaping at all.
```xml
<call name="write_file">
<arg name="path">out.py</arg>
<arg name="content">import os

def main():
    print("hi\there")
</arg>
</call>
```

**Terse line DSL with a heredoc sentinel:** body is verbatim until the sentinel.
```
write_file path=out.py <<END
import os

def main():
    print("hi\there")
END
```

**CodeAct (for contrast):** body via a Python triple-quoted raw string (still a literal,
but the model must produce valid Python and pick a non-colliding delimiter):
```python
write_file("out.py", r'''import os

def main():
    print("hi\there")
''')
```

Only JSON forces the body onto one line with full escaping. XML and the heredoc DSL keep
the body **byte-for-byte literal**, which is the single biggest reliability win for a 3B.

### Case C — a multi-action batch (search, then read, then write)

**JSON (today):**
```json
[
  {"tool":"search","args":{"query":"def main","path":"src"}},
  {"tool":"read_file","args":{"path":"src\\app.py"}},
  {"tool":"write_file","args":{"path":"src\\app.py","content":"# patched\n"}}
]
```

**XML / tag-based (sequence of `<call>`):**
```xml
<call name="search"><arg name="query">def main</arg><arg name="path">src</arg></call>
<call name="read_file"><arg name="path">src\app.py</arg></call>
<call name="write_file"><arg name="path">src\app.py</arg><arg name="content"># patched
</arg></call>
```

**Terse line DSL (one action per line; batch = multiple lines):**
```
search query="def main" path=src
read_file path=src\app.py
write_file path=src\app.py <<END
# patched
END
```

**CodeAct (the format's home turf — batch is just sequential statements, and it can
compose/loop):**
```python
hits = search("def main", path="src")
src = read_file("src/app.py")
write_file("src/app.py", "# patched\n")
```

CodeAct is the most expressive for batches (it can branch/loop), but the line-DSL and XML
forms get the *token* and *escaping* wins without needing an interpreter.

---

## 4. Evidence on each candidate (with numbers)

### 4.1 Code-as-action / CodeAct
- **CodeAct (Wang et al., ICML 2024):** consolidating actions into executable Python
  **"outperforms widely used alternatives like Text and JSON (up to 20% higher success
  rate)"** across **17 LLMs** on API-Bank and the curated **M3ToolEval** benchmark
  ([PMLR proceedings](https://proceedings.mlr.press/v235/wang24h.html),
  [repo](https://github.com/xingyaoww/code-act), [arXiv 2402.01030](https://arxiv.org/abs/2402.01030)).
- **smolagents (Hugging Face):** their `CodeAgent` writes actions as code; HF reports code
  agents use **~30% fewer steps / LLM calls** than JSON/text tool-calling and reach higher
  performance on hard benchmarks, citing the CodeAct line of work
  ([smolagents docs](https://huggingface.co/docs/smolagents/en/index),
  [HF blog](https://huggingface.co/blog/smolagents),
  [agents course: code vs JSON](https://huggingface.co/learn/agents-course/en/unit2/smolagents/tool_calling_agents)).
  Their own side-by-side example (a 2-query batch) is materially shorter in code than in the
  JSON tool-call list.
- **Caveat for us:** these wins are on *capable* models, with a **sandboxed Python
  interpreter** and the security surface that implies. Our action space is ~5 simple tools;
  full code-execution is a heavy hammer, and a raw VibeThinker-3B must write *valid* Python,
  which is a harder constraint to grammar-lock than a tag or a line. See §7.

### 4.2 XML / tag-based
- Anthropic's production tool-use format is literally `<function_calls>` →
  `<invoke name="...">` → `<parameter name="...">value</parameter>`, parsed by **regex**,
  with parameter values as **raw text** ("not expected to be valid XML")
  ([leaked tool-use prompt](https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-api-tool-use_20250119.md)).
- Anthropic separately documents that **XML tags help the model parse and produce structured
  content unambiguously** and that the model is steerable via tags
  ([prompting best practices / "use XML tags"](https://platform.claude.com/docs/en/docs/use-xml-tags)).
- The decisive property for *us*: **content between tags needs no escaping** — directly
  fixes Case A and Case B.

### 4.3 Terse line / shell-style DSL
- **SWE-agent** built an "Agent-Computer Interface" of **simple, LM-friendly line
  commands** (navigation, windowed file view, `edit`, search, run) rather than JSON, and
  credits this LM-centric command/feedback design for big agent-efficiency gains
  ([ACI docs](https://swe-agent.com/0.7/background/aci/),
  [paper PDF](https://arxiv.org/pdf/2405.15793v1)).
- **Aider** uses compact **search/replace edit blocks** (and `udiff`) instead of whole-file
  JSON; diff-style formats "use far fewer tokens" and let weaker models edit larger files
  ([aider edit formats](https://aider.chat/docs/leaderboards/notes.html),
  [unified diffs](https://aider.chat/docs/unified-diffs.html)).
- Line DSLs yield the **smallest GBNF grammar** of any option here, which matters for a 3B.

### 4.4 YAML / Markdown / KV
- **Nested-data study (GPT-5 Nano, Llama 3.2 3B, Gemini 2.5 Flash Lite):** YAML won
  accuracy on 2 of 3 models, **but JSON won on Llama-3.2-3B (52.7%)** with XML second; and
  **Markdown was the most token-efficient (34–38% fewer tokens than JSON)**
  ([improvingagents](https://www.improvingagents.com/blog/best-nested-data-format)).
  This is the most directly relevant small-model data point we have, and it is *mixed*: a 3B
  did not benefit from YAML/XML on *accuracy of reading* nested data.
- **YAML fine-tune comparison:** with fine-tuning, **JSON beat YAML** by 15.8% (xLAM) and
  7.44% (BFCL) on function-calling accuracy — YAML's token savings did not translate to
  accuracy ([Mohapatra](https://athekunal.medium.com/json-vs-yaml-function-calling-finetuning-comparison-25e97767cc5d)).
- **TOON** is ~40% cheaper than JSON but is designed for **uniform tabular *input***, and
  its own authors warn efficiency degrades with deep nesting — **not** a fit for emitting
  tool calls with free-text bodies
  ([toon-format/toon](https://github.com/toon-format/toon),
  [DigitalOcean](https://www.digitalocean.com/community/tutorials/toon-vs-json)).

### 4.5 Benchmarks / leaderboards for tool-calling
- **BFCL (Berkeley Function-Calling Leaderboard) V4 / Gorilla:** the standard executable
  function-calling benchmark, AST-based eval across Python/Java/JS/REST; distinguishes
  **"FC"** (native function-calling) vs **"Prompt"** (text workaround). Small-model results:
  **xLAM-2-3b-fc-r ≈ 65.7% overall**, **Qwen3-4B (Prompt) ≈ 62.0%**, Qwen3-1.7B ≈ 55.5%,
  Qwen3-0.6B ≈ 45.8% — i.e. 3B-class models are *capable but error-prone* at function
  calling, exactly our regime
  ([leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html),
  [BFCL paper, PMLR 2025](https://proceedings.mlr.press/v267/patil25a.html),
  [OpenReview](https://openreview.net/forum?id=2GmDdhBdDk)).
- **TinyLLM** (edge-device agentic SLMs) and **xLAM** both show small action-models can be
  trained/optimized for tool use, and that *format/representation choices matter* for
  reliability ([TinyLLM arXiv 2511.22138](https://arxiv.org/pdf/2511.22138),
  [xLAM arXiv 2409.03215](https://arxiv.org/pdf/2409.03215)).
- **Caveat:** BFCL evaluates the *call decision/arguments*, not "JSON vs XML token cost." I
  found **no head-to-head BFCL study isolating output *format* for a fixed small model** —
  so the format claims rest on CodeAct, the nested-data study, and the practitioner
  write-ups, not on BFCL. Stated honestly.

### 4.6 Grammar-constrained decoding (the constraint layer)
- **llama.cpp GBNF** is a BNF-derived grammar format (with regex-like extensions and token
  support) that masks invalid tokens *during sampling*, guaranteeing the output matches an
  arbitrary grammar — JSON, a programming subset, or **a custom tag/line DSL**
  ([llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md),
  [grammars guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)).
- llama.cpp converts a **subset of JSON-Schema → GBNF** for its function-calling path — this
  is exactly the path Ollama's `format` rides on
  ([DeepWiki: Grammar and Structured Output](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output)).
- GCD **helps small models specifically**: it improved syntactic *and* semantic accuracy and
  "can serve as an effective substitute for in-context examples, especially … for smaller
  models" ([ACL 2025 Industry](https://aclanthology.org/2025.acl-industry.34/)).
- **Important Ollama constraint:** Ollama's public `format` field accepts **only a JSON
  Schema** (or the string `"json"`), *not* raw GBNF. To constrain a non-JSON format we must
  either (a) drop to llama.cpp directly (its server/CLI accept `--grammar` / grammar in the
  request), or (b) constrain to JSON and transform, or (c) run unconstrained + robust parse +
  retry. See §6 for how we keep constraints if we leave JSON.

---

## 5. Recommendation

**A/B test two candidates against JSON, in this priority order. Both directly kill the
escaping problem and cut tokens, and both stay grammar-constrainable.**

### Primary candidate: **XML / tag-based calls** (top pick)
Closest thing to a "known-good" design — it's literally Anthropic's production tool format,
and the no-escaping property is exactly our pain point.

**How the model emits it** (sequence of calls = the batch; values are raw text):
```xml
<call name="write_file">
<arg name="path">C:\Users\NickL\out.py</arg>
<arg name="content">import os
print("hi\there")
</arg>
</call>
```
- No escaping of paths, quotes, or newlines anywhere.
- A batch is just multiple `<call>` blocks (replaces the JSON array).
- We keep the existing two-phase split (reason at `<think>`, then emit calls).

**How we parse it** (replaces `json.loads` in `agent.py` `_parse()`): a small regex/state
parser — scan for `<call name="X">`, then repeatedly `<arg name="K">…</arg>` reading raw
text up to the *next* `</arg>` (greedy-to-close). This is exactly what Anthropic does
("parsed with regular expressions") and is robust to the model not producing perfectly
nested XML.

**How we keep decode-time constraints:** write a **GBNF grammar** for this fixed shape
(`callseq ::= call+`, `call ::= "<call name=\"" name "\">" arg+ "</call>"`,
`arg ::= "<arg name=\"" key "\">" content "</arg>"`, with `name`/`key` restricted to the
**enum of real tool/param names** so the model can't invent tools — same guarantee the
current `oneOf`/`const` schema gives). `content` is "any run of chars not containing the
literal `</arg>`". This is a *small, easy* grammar. **Tradeoff: this grammar is not a JSON
Schema, so we lose Ollama's built-in `format` field and must feed GBNF to llama.cpp
directly** (run `llama-server`/`llama.cpp` ourselves, or use Ollama only when it later
exposes raw grammars). That is the real cost of leaving JSON — quantified in §6.

### Secondary candidate: **Terse line DSL with heredoc bodies**
Even cheaper in tokens and the *smallest* grammar, at the cost of being our own bespoke
format (no big-lab precedent for the exact syntax, though SWE-agent/Aider validate the
line-command philosophy).

**How the model emits it:**
```
search query="def main" path=src
read_file path=src\app.py
write_file path=src\app.py <<END
# any literal content, including backslashes C:\x and "quotes"
END
```
- One action per line; scalar args as `key=value` (quote only if the value has spaces);
  exactly one optional body per call via `<<SENTINEL … SENTINEL`.
- **Parse:** split into lines; for a line with `<<S`, read verbatim until a line equal to
  `S`. Trivial parser.
- **Constrain:** GBNF for `line ::= toolname (" " key "=" value)* (" <<" SENT)? "\n"` with
  `toolname`/`key` from the real enums; body = raw lines until sentinel.

### Why these two over the others
- **CodeAct** has the best *published* numbers (20% / 30%), but for *our* setup it's the
  wrong cost/benefit: it needs a sandboxed interpreter (security + complexity), full Python
  is the hardest thing to grammar-lock for a 3B, and our 5-tool action space doesn't need
  Turing-complete composition. **Keep it as a stretch/phase-2 experiment, not the first A/B.**
- **YAML/Markdown/TOON** either showed *no* small-model accuracy edge (YAML lost to JSON on
  Llama-3B and after fine-tuning) or are aimed at input data (TOON). Markdown's token win is
  real but its lack of clear delimiters for arbitrary bodies makes it weaker than a heredoc
  or a closing tag.

### Concrete A/B plan
1. Add a `format` strategy seam behind the existing `LLMClient`/`registry.action_schema()`
   boundary (the codebase is already DIP-clean: `agent.py` only needs `_parse()` and the
   schema/grammar source swapped). Strategies: `json` (control), `xml`, `linedsl`.
2. For `xml`/`linedsl`, generate the **GBNF grammar** from the same `ToolRegistry` that today
   builds the JSON Schema (one method per strategy), and route generation through a llama.cpp
   grammar request instead of Ollama `format`.
3. Metric: per-turn **valid-parse rate** and **path/content fidelity** (does the round-tripped
   path/body match byte-for-byte?), plus **tokens per action**, on a fixed task set that
   includes Windows paths and multi-line writes (the worst cases). This isolates *format*,
   which the literature does not do for us.

---

## 6. The cost of leaving JSON: Ollama `format` vs raw GBNF

| | Stay on JSON (today) | Move to XML/line-DSL + GBNF |
|---|---|---|
| Constraint mechanism | Ollama `format` = JSON Schema (Ollama compiles to GBNF for us) | Raw GBNF fed to llama.cpp directly |
| Server | Ollama `/api/generate` (as in `llm.py`) | `llama-server`/`llama-cli` with `--grammar`/grammar field, **or** Ollama *if/when* it exposes raw grammars (it does not today) |
| Validity guarantee | Syntactic JSON guaranteed | Syntactic format guaranteed (equivalent) |
| Escaping reliability | Poor (model must escape) | **Strong (no escaping)** |
| Engineering cost | none (built) | run/manage llama.cpp ourselves; write 1 small grammar per strategy + 1 parser per strategy |
| Fallback if unconstrained | n/a | unconstrained generate + tolerant parser + **retry on parse failure** (works without GBNF; GCD just makes it rarer) |

llama.cpp's grammar support is mature and the GBNF for these formats is small, so the
incremental engineering is modest; the operational change (running llama.cpp directly rather
than through Ollama's `format`) is the main thing we'd be signing up for
([llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)).
A pragmatic *intermediate*: keep Ollama, constrain to a **flat JSON shape with one
unescaped-ish field**, or run **unconstrained + tolerant parse + retry** for the XML/line
formats and measure whether the no-escaping win alone beats JSON even without GCD — then add
GBNF only if needed.

---

## 7. Honest caveats / uncertainty

- **No format-isolating small-model benchmark exists** (that I found). BFCL measures call
  correctness, not JSON-vs-XML token/escaping for a fixed 3B. Our strongest *small-model*
  format data point — the nested-data study — actually shows **JSON tied/won on Llama-3.2-3B
  for reading accuracy**, and YAML lost to JSON in a fine-tune study. So the case for moving
  rests primarily on **(a) the escaping argument** (format-intrinsic, not benchmark-
  dependent) and **(b) token cost**, with CodeAct/smolagents as the strongest *agentic*
  evidence that leaving JSON helps. We should treat the A/B as genuinely open and let our own
  valid-parse / fidelity numbers decide.
- **CodeAct's 20%/30% are on capable models with an interpreter**, not a raw constrained 3B.
  Don't assume they transfer.
- One search result returned a future-dated arXiv id for "TOON" (`2603.03306`) that I could
  **not** verify; I deliberately did **not** rely on it and cite TOON only via its repo and a
  vendor tutorial. Treat TOON numbers as vendor-reported.
- Several token-cost figures (10–15% reasoning hit; some % savings) come from practitioner
  blogs, not papers — flagged inline as directional.

---

## 8. Sources

- CodeAct — Wang et al., *Executable Code Actions Elicit Better LLM Agents*, ICML 2024:
  https://proceedings.mlr.press/v235/wang24h.html · repo https://github.com/xingyaoww/code-act · arXiv https://arxiv.org/abs/2402.01030
- smolagents: https://huggingface.co/docs/smolagents/en/index · blog https://huggingface.co/blog/smolagents · code-vs-JSON lesson https://huggingface.co/learn/agents-course/en/unit2/smolagents/tool_calling_agents
- Anthropic XML tool use / prompting with XML: https://platform.claude.com/docs/en/docs/use-xml-tags · leaked tool-use system prompt https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-api-tool-use_20250119.md
- SWE-agent ACI: https://swe-agent.com/0.7/background/aci/ · paper https://arxiv.org/pdf/2405.15793v1
- Aider edit formats / unified diffs: https://aider.chat/docs/leaderboards/notes.html · https://aider.chat/docs/unified-diffs.html
- Nested-data format study (incl. Llama-3.2-3B): https://www.improvingagents.com/blog/best-nested-data-format
- JSON-vs-YAML function-calling fine-tune: https://athekunal.medium.com/json-vs-yaml-function-calling-finetuning-comparison-25e97767cc5d
- "Beyond JSON" (formats, reasoning-hit claim): https://medium.com/@michael.hannecke/beyond-json-picking-the-right-format-for-llm-pipelines-b65f15f77f7d
- "Why JSON Costs More Than TSV": https://david-gilbertson.medium.com/llm-output-formats-why-json-costs-more-than-tsv-ebaf590bd541
- TOON: https://github.com/toon-format/toon · https://www.digitalocean.com/community/tutorials/toon-vs-json
- BFCL / Gorilla: https://gorilla.cs.berkeley.edu/leaderboard.html · paper https://proceedings.mlr.press/v267/patil25a.html · https://openreview.net/forum?id=2GmDdhBdDk
- xLAM: https://arxiv.org/pdf/2409.03215 · TinyLLM (edge SLM agents): https://arxiv.org/pdf/2511.22138
- llama.cpp GBNF grammars: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md · grammar/structured-output overview https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output
- Grammar-Constrained Decoding helps (incl. smaller models), ACL 2025 Industry: https://aclanthology.org/2025.acl-industry.34/
