# Qwen3-3B Upgrade Analysis (#139)

Branch: **`beta_qwen3coder`** (ISOLATED model line; one-way `beta → beta_qwen3coder`, never
merges back — CLAUDE.md §6). All measurements below are **live, captured on this machine**
(June 2026), not assumed.

## TL;DR

- **There is NO dense Qwen3 3B.** The Qwen3 dense line is 0.6B / 1.7B / 4B / 8B / 14B / 32B;
  the smallest MoE is 30B‑A3B. The closest ~3B peer is **`qwen3:4b`** (4.0B dense, Q4_K_M,
  2.5 GB on disk). `qwen3:3b` does not exist.
- **Recommendation: GO — conditional**, swapping to **`qwen3:4b`** with **thinking left ON**
  and **`num_ctx` dropped 32768 → 16384**. NO codec changes are required.
- VRAM at the *current* pinned `num_ctx=32768` is **7266 MiB** — it loads at 100% GPU but
  leaves only ~0.9 GB headroom on the 8 GB card, which the headed Chrome web agent can
  eat. At `num_ctx=16384` it is a comfortable **5038 MiB**.
- Biggest behavioural win: **Ollama returns STRUCTURED `message.tool_calls` for Qwen3**
  (qwen2.5‑coder returned `null` + text). The harness already prefers structured calls, so
  this path becomes primary and is *more* robust. Biggest cost: **latency** — Qwen3 thinks
  before every call (~500+ extra tokens/turn).

---

## Hardware verification

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
NVIDIA GeForce RTX 3080 Laptop GPU, 8192 MiB, 610.62
```

CUDA confirmed as the inference device. While a 3B/4B model was loaded, the GPU compute‑app
list showed Ollama's runner on the NVIDIA card:

```
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
21116, <n/a>, ...\Ollama\lib\ollama\llama-server.exe
```

`ollama ps` reports **100% GPU** for every fitting configuration below (no CPU spill). The
AMD Radeon integrated GPU is NOT used by Ollama. The true idle VRAM baseline (no model
loaded, sampled repeatedly) is **5 MiB** — essentially the whole 8 GB is free for the model.
(Earlier transient ~3.4 GB readings were a short‑lived desktop/Armoury‑Crate consumer, not
persistent — but see the Risk section: such transients DO recur and matter at `num_ctx=32768`.)

## Model availability

`qwen3:3b` does not exist on the Ollama library. Pulled and tested **`qwen3:4b`**:

```
ollama show qwen3:4b
  architecture        qwen3
  parameters          4.0B
  context length      262144           (default; we pin lower)
  quantization        Q4_K_M
  Capabilities        completion, tools, thinking
  Parameters (baked): top_k 20, top_p 0.95, temperature 0.6, repeat_penalty 1,
                      stop "<|im_start|>", stop "<|im_end|>"
  License             Apache 2.0       (vs qwen2.5-coder's "Qwen RESEARCH LICENSE")
```

Library tags: `qwen3:0.6b` (523 MB), `qwen3:1.7b` (1.4 GB), `qwen3:4b` (2.5 GB), each with
`q4_K_M` / `q8_0` / `fp16` variants. **`qwen3:4b` Q4_K_M is the chosen candidate** — it is the
nearest dense peer to a 3B (4.0B vs VibeThinker‑3B's 3B; `qwen3:1.7b` is the alternative if
VRAM/latency must shrink). Note the Apache‑2.0 license is a genuine improvement over the
current model's non‑commercial Research license.

### VRAM fit (live, settled measurements; baseline = 5 MiB)

| Model | num_ctx | nvidia‑smi used | ollama ps | Processor |
|---|---|---|---|---|
| qwen2.5-coder:3b Q4_K_M (current) | 32768 | **3380 MiB** | 3.4 GB | 100% GPU |
| **qwen3:4b** Q4_K_M | 16384 | **5038 MiB** | 5.1 GB | 100% GPU |
| **qwen3:4b** Q4_K_M | 32768 | **7266 MiB** | 7.4 GB | 100% GPU |
| **qwen3:4b** Q4_K_M | 40960 | **7930 MiB** | 8.7 GB | 100% GPU (edge) |

KV cache for Qwen3‑4B is heavy: +2.2 GB going 16k → 32k. At 32768 there is < 1 GB headroom;
at 40960 `ollama ps` reports 8.7 GB (over the card) yet still claims 100% GPU — that is the
absolute edge and is **not** recommended. **`num_ctx=16384` is the safe operating point.**

## Chat template comparison (Qwen2.5‑Coder‑3B vs Qwen3‑4B)

Fetched live from `Qwen/Qwen3-4B/tokenizer_config.json` and compared to the
`Qwen/Qwen2.5-Coder-3B-Instruct` template already ground‑truthed in `QWEN3CODER_ANALYSIS.md`.
**The tool‑calling convention is identical** — diffs only:

| Aspect | Qwen2.5‑Coder‑3B | Qwen3‑4B | Impact on harness |
|---|---|---|---|
| ChatML roles | `<|im_start|>…<|im_end|>` | same | none |
| Tool DEFINITIONS | bare `tool \| tojson` in `<tools>` | same (bare in `<tools>`) | none — `tool_definitions()` unchanged |
| Tool CALL | `<tool_call>{"name","arguments"}</tool_call>` | same | none — `parse()` unchanged |
| Tool RESULT | `<tool_response>` in a user turn | same | none — Ollama `role:"tool"` wrapping unchanged |
| **Thinking** | none (non‑reasoning instruct) | **`<think>…</think>`, ON by default** | NEW — see below |
| `arguments` serialization | always `\| tojson` | type‑checked (no double‑escape) | minor robustness gain |
| Special tokens | ChatML + FIM | adds vision/box/object tokens (unused) | none |
| License | Qwen Research | Apache 2.0 | non‑technical plus |

## Smoke test results (live `/api/chat`, num_ctx=32768, temp 0.0)

Sending the harness's exact native request (enveloped `tools:` field):

**Thinking ON (default):**
```
message.tool_calls : [{"id":"call_…","function":{"name":"fill",
                       "arguments":{"target":"e1","text":"Jason"}}}]   ← STRUCTURED, correct
message.content    : ""                                               ← clean / empty
message.thinking   : "Okay, let's see. The user wants me to use the fill tool…"  (~1840 chars)
```

**Thinking OFF (`think:false` Ollama param):** `tool_calls` still correct, BUT the reasoning
prose LEAKS into `message.content` as bare text (no `<think>` tags, no `thinking` field) —
**worse** for any content‑parse fallback. Do NOT use `think:false`.

**Multi‑turn with a tool result fed back (`role:"tool"`):** Qwen3 read the result and emitted
the correct next call (`fill e2 "Smith"`). Stateful history path works end‑to‑end.

Key contrast vs current model: **qwen2.5‑coder returns `tool_calls: null` and the call as
text in `content`**; **Qwen3‑4B returns proper structured `tool_calls` and keeps `content`
empty, with reasoning isolated in `message.thinking`.**

## Codec compatibility — NONE needed

`HermesCodec.parse()` was run against real Qwen3 output shapes:

| Input | Result |
|---|---|
| prose + `<tool_call>{…}</tool_call>` | ✅ parsed (prose ignored) |
| `<think>…</think>` + `<tool_call>{…}</tool_call>` | ✅ parsed (`_BLOCK_RE` only matches inside `<tool_call>`) |
| pure `<tool_call>` block | ✅ parsed |
| empty content | returns error — but unreachable: when thinking is ON, `content` is empty AND `tool_calls` is populated, so the agent uses the structured path (`agent.py` L270‑276) and never calls `parse("")` |
| prose, no call | graceful error → model gets steered |

- `_BLOCK_RE` (`<tool_call>(.*?)(?:</tool_call>|\Z)`) is **unaffected** by a preceding
  `<think>` block — it is non‑greedy and anchored to the tool‑call tags.
- `format_instructions()` anti‑fence clause still applies (and Ollama injects its own on the
  native path).
- `parse()` fence‑recovery still works for the non‑native path.

**The existing `hermes` codec works as‑is.** This holds because Ollama splits Qwen3's
reasoning into the separate `thinking` field, so `<think>` never pollutes `content`, and
because the agent already prefers structured `tool_calls`.

## Thinking‑mode handling

- Qwen3 has thinking ON by default. Ollama 0.30.x routes it to **`message.thinking`**, leaving
  **`message.content` empty** when a tool call is made. The harness's `_stream_chat`
  (`llm.py` L290‑298) reads only `content` + `tool_calls` — so thinking is silently dropped
  from the action text, which is fine because `tool_calls` carries the call.
- **Leave thinking ON** (do not pass `think:false`, do not append `/no_think`): `think:false`
  leaks reasoning into `content` (observed), and `/no_think` was NOT honoured under native
  tools (observed). Thinking‑ON + structured `tool_calls` is the clean, correct combination.
- **Optional polish (not required):** capture `message.thinking` into the narrative so the
  human transcript and the advisor see the model's reasoning. Today the agent reconstructs a
  preamble from `action_json` (`agent.py` L259‑264); with Qwen3 that field is empty, so the
  narrative loses the "you reasoned: …" line. Low priority — correctness is unaffected.

## Optimal config for RTX 3080 8 GB

| Setting | Current (qwen2.5‑coder) | Recommended (qwen3:4b) | Why |
|---|---|---|---|
| `model` | `qwen2.5-coder:3b-instruct` | `qwen3:4b` | nearest dense Qwen3 peer; Apache‑2.0 |
| `num_ctx` | 32768 | **16384** | 32768 = 7266 MiB (<1 GB headroom; Chrome can OOM it); 16384 = 5038 MiB, safe. Keeps single‑runner invariant (#77). |
| quantization | Q4_K_M | **Q4_K_M** | Q5/Q8 push KV+weights over budget at any useful ctx on 8 GB; Q4_K_M is the only comfortable fit. |
| `temperature` (action) | 0.0 (greedy) | **0.6** | Qwen3 card: "DO NOT use greedy decoding" (causes repetition). Use thinking‑mode defaults. |
| `top_p` | 0.95 | **0.95** | matches Qwen3 baked default |
| `top_k` | 0 | **20** | Qwen3 recommended (0 = disabled, off‑distribution for Qwen3) |
| `num_gpu` | 99 | 99 | keep full offload |
| stop tokens | (none beyond codec) | `<|im_start|>`,`<|im_end|>` baked in | no change needed |
| `reason_tokens`/`action_tokens` | 4096/4096 | keep, or raise action budget | thinking consumes output tokens; 4096 is enough for a call, but reasoning eats into the 8192 num_predict — monitor truncation |

> ⚠️ The `action_temperature: float = 0.0` divergence (greedy for "verbatim string
> fidelity", `config.py` L25) **must be revisited for Qwen3** — greedy decoding is explicitly
> discouraged by the Qwen3 card and risks repetition loops. This is the single most important
> config change beyond the model tag and num_ctx.

## GO / NO‑GO recommendation

**GO — conditional**, to **`qwen3:4b`** (NOT a mythical `qwen3:3b`), with:

1. `num_ctx` 32768 → **16384** (VRAM safety on the 8 GB card).
2. Action sampling: temperature **0.6**, top_k **20**, top_p 0.95 (drop greedy).
3. Thinking left **ON**; rely on structured `tool_calls` (already preferred by the agent).
4. **No codec changes.**

This is a documented model‑class change (4.0B dense vs the 3B‑dense parity target — flag it
in `QWEN3CODER_DIVERGENCE.md` exactly as the qwen2.5‑coder substitution was flagged). If
strict ≤3B parity with VibeThinker‑3B is mandatory, fall back to **`qwen3:1.7b`** (1.4 GB,
much lower latency) — but it is a weaker model than the current 3B coder and would likely
regress the benchmark score; `qwen3:4b` is the better quality/parity trade.

**Why not NO‑GO:** the tool dialect is byte‑compatible, it fits VRAM at 16k ctx, returns
cleaner structured calls than the incumbent, and carries a permissive license. The only real
costs are latency (thinking) and a 4B‑vs‑3B parity caveat — both acceptable and documented.

## Required code changes (if GO), by priority

1. **`vibeharness/config.py` L23** — `model = "qwen3:4b"`.
2. **`vibeharness/config.py` L104** — `num_ctx = 16384` (re‑validate the snapshot/budget math;
   the input‑budget formula is unchanged, only the window shrinks — `tests/test_snapshot_budget.py`
   constants that assume 32768 may need updating).
3. **`vibeharness/config.py` L25** — change `action_temperature` from `0.0` to `0.6`
   (Qwen3 forbids greedy). Consider a small `repeat_penalty`/`presence_penalty` if loops appear.
4. **`vibeharness/config.py` L27** — `top_k = 20` (was 0).
5. **Update divergence docs** — `QWEN3CODER_ANALYSIS.md` / `QWEN3CODER_DIVERGENCE.md`: record
   the 4.0B‑vs‑3B parity caveat, the thinking‑mode behaviour, and the new VRAM/ctx envelope.
6. *(Optional, low priority)* `vibeharness/llm.py` `_stream_chat` (L290‑298) — also accumulate
   `message.thinking` and surface it (e.g. into `Decision.reasoning`) so the narrative/advisor
   regain visibility of the model's reasoning. Not required for correctness.

No changes to `hermes_codec.py`, `prompt.py`, `registry.py`, or the agent loop are required.

## Risk assessment — MEDIUM

| Risk | Severity | Notes / mitigation |
|---|---|---|
| **VRAM at 32768** | HIGH if left at 32k | 7266 MiB + headed Chrome + transient desktop GPU use (~3.4 GB seen) → OOM / CPU spill. **Mitigated by dropping to num_ctx=16384** (5038 MiB). |
| **Latency** | MEDIUM | Thinking adds ~500+ output tokens/turn; a trivial call took ~28 s here. A 15‑turn web run with large snapshots will be noticeably slower than qwen2.5‑coder. Consider `qwen3:1.7b` if speed dominates. |
| **Greedy decoding repetition** | MEDIUM | Current `action_temperature=0.0` is off‑policy for Qwen3 and can cause loops. Mitigated by temp 0.6 / top_k 20 (change #3/#4). |
| **Smaller context window** | LOW‑MED | 16384 vs 32768 halves the input budget; the worst‑case ~11k‑token YouTube snapshot + prompt + history may get tighter. The #43 dynamic snapshot budget already trims gracefully; re‑run the heavy‑page test. |
| **Parity drift (4B vs 3B)** | LOW (documented) | Not apples‑to‑apples with VibeThinker‑3B on size; flagged, same pattern as the qwen2.5‑coder substitution. |
| **Codec breakage from `<think>`** | LOW | Verified: `_BLOCK_RE` and the structured‑calls path both handle it. No change needed. |
| **Narrative loses reasoning text** | LOW | `content` is empty under thinking‑ON; the "you reasoned:" narrative line disappears. Cosmetic; optional fix #6. |

---

*Methodology: every VRAM/behaviour figure above was captured live on the RTX 3080 Laptop GPU
via `nvidia-smi`, `ollama ps/show`, and direct `/api/chat` calls mirroring the harness's
native request; chat templates fetched live from HuggingFace. No values were assumed or
fabricated. Test scripts were run ad‑hoc and removed (not committed).*
