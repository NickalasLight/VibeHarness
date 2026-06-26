# Qwen3 Analysis — Scope Update (Urgent)

**For agent performing issue #139 analysis — read this FIRST before finalising.**

## Full hardware profile (verified 2026-06-24)

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU — **8192 MiB VRAM**, CUDA 13.3, driver 610.62 |
| CPU | AMD Ryzen 9 5900HS — 8 cores / 16 threads, 3.3 GHz |
| RAM | **40 GB total, ~22 GB FREE** |
| Virtual memory | 50.8 GB |
| AMD integrated | Radeon 512 MB — DO NOT USE, Ollama uses NVIDIA by default |
| OS | Windows 10 Home, Ollama on Windows (native) |

## Revised goal: most powerful model, not just 3B replacement

The user explicitly wants the **highest intelligence/quality Qwen3 model** that runs well on this hardware. RAM offloading is acceptable. Non-critical system resources can be commandeered. Optimize for speed AND quality.

## Candidate models to evaluate (ALL of these)

| Model | Quant | Est. VRAM | Fit | Notes |
|---|---|---|---|---|
| qwen3:8b-q5_k_m | Q5_K_M | ~6.7 GB | ✅ full GPU | Best speed+quality for 8B |
| qwen3:8b-q8_0 | Q8_0 | ~9 GB | ⚠️ partial offload | Max quality 8B, 1 GB to RAM |
| qwen3:14b-q3_k_m | Q3_K_M | ~7 GB | ✅ full GPU | Most powerful that fits, Q3 quality |
| qwen3:14b-q4_k_m | Q4_K_M | ~9 GB | ⚠️ partial offload | Most powerful at Q4, 1-2 GB to RAM |
| qwen3:30b-a3b | MoE | ~20+ GB | ❌ too large | Skip unless fits with offload |

**For each viable candidate:**
1. Pull it: `ollama pull <tag>`
2. Load it and check `nvidia-smi` — confirm VRAM on NVIDIA, note usage
3. Check `ollama ps` — confirm PROCESSOR shows GPU not CPU
4. Benchmark: `ollama run <tag> "Count to 50 in one line." 2>&1` — measure tokens/sec
5. Tool call test (below)
6. Evict: `ollama stop <tag>`

## Optimization settings to test on the winning candidate

```
OLLAMA_FLASH_ATTENTION=1       # Reduces KV cache VRAM by ~30-40%; CRITICAL for larger models + larger ctx
OLLAMA_NUM_PARALLEL=1          # Single request at a time (we're single-user)
OLLAMA_MAX_LOADED_MODELS=1     # Keep eviction tight
```

**Config settings to evaluate:**
- `num_ctx`: With flash attention enabled, test 32768 vs 65536 (22 GB free RAM handles KV overflow)
- `num_gpu_layers`: -1 (auto, Ollama decides) vs explicit count if offloading needed
- `num_thread`: 16 (all logical cores of Ryzen 9 5900HS) for CPU offload portion
- `temperature`: Check Qwen3 model card — may differ from Qwen2.5 default (0.7)
- `top_p`: Qwen3 recommendation (often 0.8–0.9 for instruct)
- `top_k`: Qwen3 recommendation (often 20 for thinking mode, higher for non-thinking)
- `repeat_penalty`: Qwen3 default (usually 1.05–1.1)
- `min_p`: Qwen3 uses min_p sampling — check if Ollama supports it

## Think-mode handling (critical for codec)

Qwen3 models in instruct mode may prepend `<think>...</think>` blocks before the `<tool_call>` output. The existing HermesCodec `_BLOCK_RE` regex will still find `<tool_call>` tags even with a `<think>` prefix, BUT:

1. Confirm empirically: does the model emit `<think>` when using tools in non-thinking mode?
2. To disable thinking: append `/no_think` to the system prompt OR set `temperature` + check model card
3. If `<think>` appears: the codec `parse()` needs to strip it before returning to caller (or regex handles it already since it scans for `<tool_call>` anywhere in the string)

Test: send the tool-call smoke test and check raw `message.content` for `<think>` presence.

## Ollama env vars — set these persistently on Windows

```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "Machine")
[System.Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "Machine")
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "Machine")
```
Restart Ollama service after setting machine-level env vars.

## Deliverable

`qwen3_upgrade_analysis.md` must include:
- Benchmark results for each candidate (VRAM used, tokens/sec, tool-call format)
- nvidia-smi confirmation screenshot/output showing NVIDIA is active
- Winning model recommendation with justification
- All config.py and env-var changes
- Exact codec delta (if any) for `<think>` stripping
- GO / NO-GO + risk level
