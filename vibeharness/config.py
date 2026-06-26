"""Runtime configuration. One immutable value object passed where needed (DIP-friendly)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Which model drives one agent role, and how to sample it (issue #163).

    A role (``base`` / ``validator`` / ``advisor`` / ``escalation`` / any future role) is
    pointed at a model by naming a registered provider plus a model id. The endpoint
    *kind* — local Ollama/llama.cpp vs an OpenAI-compatible API — is NOT stored here; it
    derives from the named provider (see :mod:`vibeharness.providers`), so flipping a role
    local↔API is a single field change (``provider``) with no other edits.

    The sampling fields are OPTIONAL per-role overrides. Resolution is layered
    spec → provider-default → :class:`Config`-default (applied by the client factory), so
    leaving them ``None`` inherits the run's global sampling exactly as before.
    """
    provider: str            # a registry name, e.g. "ollama" or "zhipuai"
    model: str               # the model id at that provider, e.g. "qwen3:4b" / "glm-4.7-flash"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # ISSUE #179 / #178 — per-MODEL tool-call codec and per-turn tool-call cap. Optional
    # per-role overrides of the wire format the model is driven with and how many tool
    # calls it may emit per turn. Resolution (see resolve_model_codec / resolve_model_limit)
    # is spec-override → per-model registry (MODEL_TOOL_POLICIES, keyed by model id) →
    # Config fallback, so leaving them None inherits the run's global codec/cap exactly as
    # before. The BASE agent's spec drives the run's codec + cap (the validator is single
    # shot = 1).
    codec: str | None = None
    max_actions_per_turn: int | None = None


@dataclass(frozen=True)
class Config:
    # model / sampling
    # ISSUE #123 / branch beta_qwen3coder (ISOLATED): the default model on this line is
    # a ~3-4B Qwen3 dense model, paired with the `hermes` codec below so the harness
    # speaks the model's native trained dialect. (On `beta` the default is "vibethinker"
    # + "json"; this branch never merges back, so these defaults stay local to it.)
    #
    # ISSUE #140 (qwen3:4b upgrade — analysis #139): upgraded from
    # qwen2.5-coder:3b-instruct to qwen3:4b. Ground truth (live on RTX 3080 8GB, CUDA
    # 13.3; see qwen3_upgrade_analysis.md):
    #   - There is NO dense `qwen3:3b` (the Qwen3 dense line is 0.6B/1.7B/4B/8B/...; the
    #     Qwen3-Coder line is MoE-only). `qwen3:4b` (4.0B dense, Q4_K_M, Apache-2.0) is
    #     the nearest dense peer to VibeThinker-3B and the cleanest 3B-class substitute.
    #     A 4.0B-vs-3.0B parity caveat is flagged in QWEN3CODER_DIVERGENCE.md.
    #   - Qwen3's tool-call dialect is byte-compatible with the hermes codec (verified —
    #     ZERO codec changes). Ollama now returns STRUCTURED message.tool_calls for
    #     Qwen3 (qwen2.5-coder returned null+text), so the native_tools path (PR #136)
    #     becomes the primary, more-robust route.
    #   - Thinking is DISABLED (think:false in payload + /no_think in system prompt).
    #     Research (arXiv:2505.09388, arXiv:2512.19585) shows thinking provides no
    #     accuracy benefit for structured tool-calling while consuming 1000-5000+ tokens
    #     per turn. Sampling tuned to Qwen3 NON-thinking mode (temp=0.7, top_p=0.8).
    model: str = "qwen3:4b"
    temperature: float = 0.3          # phase-1 reasoning temperature (some diversity helps)
    # Qwen3 NON-thinking mode sampling (research: arXiv:2505.09388, Qwen3 model card):
    # temp=0.7, top_p=0.8 are the official non-thinking recommendations.
    # (Thinking mode uses 0.6/0.95; we disabled thinking — use non-thinking values.)
    action_temperature: float = 0.7   # Qwen3 non-thinking mode (was 0.6 for thinking)
    top_p: float = 0.8                # Qwen3 non-thinking mode (was 0.95 for thinking)
    # ISSUE #140: Qwen3 recommends top_k=20 (its baked default); 0 (disabled) is
    # off-distribution for Qwen3 and was only inherited from the qwen2.5-coder setup.
    top_k: int = 20                   # Qwen3 recommendation (was 0) — #140
    num_gpu: int = 99                 # force full GPU offload (NVIDIA via CUDA)

    # loop
    max_steps: int = 15               # <= 0 means unlimited
    # arXiv:2602.07359 (W&D): 3 calls/turn is the empirical accuracy peak for web agents
    # (68% vs 60% at 5, despite 5 needing fewer turns). For sub-7B models (qwen3:4b),
    # parallel-call reliability is weaker than frontier models — 3 is the safer ceiling.
    max_actions_per_turn: int = 3     # cap tool calls per turn; 0 = unlimited

    # Per-turn wall-clock budget (seconds). 0 (default) disables the guard,
    # preserving the original behaviour exactly: decide() is called inline with no
    # threading. When > 0, each turn's blocking decide() is run in a daemon worker
    # thread and joined with this timeout; if it overruns, the turn is recorded as a
    # failure and the run ends gracefully (RunResult.finished stays False). Caps a
    # turn's wall-clock TIME, complementing reason_tokens/action_tokens which only
    # cap token COUNT. The stuck generation thread is a daemon so it cannot keep the
    # process alive after the run returns (tradeoff: that thread may still be running
    # detached until the process exits or its blocking I/O unwinds).
    turn_timeout_seconds: int = 0

    # tool-call wire format (see vibeharness.codec.get_codec). "json" is the
    # decode-constrained baseline; other codecs add alternative formats.
    # ISSUE #123 (beta_qwen3coder): default to "hermes" — the native Qwen2.5 / Hermes
    # <tool_call>{"name","arguments"} format + bare <tools> function-schema definitions
    # that Qwen2.5-Coder reads (ground-truthed from the model's tokenizer_config.json
    # chat template; see QWEN3CODER_ANALYSIS.md), so the harness aligns to the model's
    # trained dialect. On `beta` this default is "json".
    codec: str = "hermes"

    # ISSUE #125 (beta_qwen3coder): SINGLE-phase generation. The two-phase
    # reason-then-act transport (free <think> pass, then a raw continuation constrained
    # to the action) exists for VibeThinker, which emits long <think> chains. But
    # qwen2.5-coder:3b-instruct is a NON-thinking instruct model: it emits a tool call
    # immediately, so phase 1 produces a real tool call that is then DISCARDED and phase 2
    # emits another — halving throughput and dropping the model's primary intent (observed
    # in #125 iter 2: recovery clicks landed in the thrown-away phase-1 channel). With
    # two_phase=False, decide() does ONE native /api/chat generation and the codec parses
    # the call from it. On `beta`/`beta_mythos_fast` (VibeThinker) this stays True.
    two_phase: bool = False

    # ISSUE #129/#130/#131 (beta_qwen3coder): NATIVE Ollama tool calling + stateful chat.
    # When True (the default on this branch), the base agent sends the model's tools in
    # the /api/chat ``tools:`` field (so Ollama applies the model's OWN trained tool
    # template — enveloped schemas + the anti-fence clause — instead of the harness
    # hand-injecting a <tools> block) AND maintains a stateful multi-turn message history
    # (system/user/assistant/tool) across turns instead of regenerating a prose narrative.
    # Ground-truthed from live /api/chat runs: the 3B model still returns the call as text
    # (Ollama leaves tool_calls null), so the codec's tolerant parse() of the content is
    # retained as the primary path; structured tool_calls are used when present. Requires
    # the active codec to support native tools (``codec.tools()`` non-None) — only the
    # ``hermes`` codec does today; with any other codec this silently no-ops to the legacy
    # single-message path so the json/xml/etc codecs are unaffected.
    native_tools: bool = True
    # FIFO chat-history eviction cap (issue #129/#130/#131). 0 (default) = no fixed turn
    # cap; history is bounded ONLY by the token budget (num_ctx minus the output
    # reservation and safety margin), with the OLDEST non-system messages dropped first.
    # A positive value additionally caps the number of retained user/assistant/tool
    # messages, a coarse belt-and-braces limit on top of the token budget.
    chat_history_max_turns: int = 0

    # context + per-turn token budgets.
    # num_ctx is the whole window (system prompt + history + generation share it).
    #
    # ISSUE #77 (single-runner fix): num_ctx is PINNED to one value that fits the
    # 8 GB GPU. Background (diagnosed in #76 / PR #87): the old 131072 request never
    # actually fit — Ollama's auto-fit shrank each request to a VARYING size
    # (4096/16384/32768) depending on free VRAM at the moment. Ollama keys a
    # llama-server runner by (model, context-size), so every distinct auto-fit size
    # spawned a NEW runner; with MAX_LOADED_MODELS/KEEP_ALIVE unset they were never
    # evicted, stacked up, and exhausted the GPU -> the OOM/CUDA crash. Sending one
    # fixed num_ctx on EVERY request (see OllamaClient._options) means only ONE
    # runner shape is ever requested, so at most one runner exists.
    #
    # ISSUE #140 (qwen3:4b upgrade — analysis #139): num_ctx confirmed safe at 32768
    # with OLLAMA_FLASH_ATTENTION=1.
    # Live VRAM on the RTX 3080 8GB (CUDA 13.3; see qwen3_upgrade_analysis.md):
    #     qwen3:4b @ num_ctx=32768, no flash attn  -> 7266 MiB (tight on 8 GB + Chrome)
    #     qwen3:4b @ num_ctx=32768, FLASH_ATTENTION -> ~4360-5086 MiB (safe)
    #     qwen3:4b @ num_ctx=16384, no flash attn  -> 5038 MiB
    # RESOLUTION: OLLAMA_FLASH_ATTENTION=1 is now set (User env var). With flash
    # attention the 32768 context fits safely and the input_budget is restored to the
    # same 23552 tokens the qwen2.5-coder:3b-instruct runs used. Dropping to 16384
    # would shrink the input budget to 7168 tokens — too small for an 8-page form
    # requiring 15-40 turns of context (system prompt alone is ~3000 tokens). The
    # single-runner invariant (#77) is preserved: 32768 is an observed auto-fit size.
    # NOTE: OLLAMA_FLASH_ATTENTION=1 must be set BEFORE Ollama starts (it reads the
    # env at startup). Restart Ollama if the env var was just set.
    num_ctx: int = 32768
    # ISSUE #92 (token rebalance): the output reservation
    # (reason_tokens + action_tokens) is subtracted from num_ctx before anything else
    # can be fed in (see vibeharness.snapshot_budget.input_budget_tokens). With the
    # pinned 32768 window, the OLD reservation (2048 + 16384 = 18432, 56% of the
    # window) left only ~13312 input tokens — so after a heavy ~11k-token page
    # snapshot, barely ~2k tokens remained for the system prompt + history, and the
    # snapshot got trimmed in realistic multi-step turns. We rebalance:
    #
    #   action_tokens 16384 -> 4096. Phase 2 emits the CONSTRAINED JSON tool call
    #   (see llm.py::_act). It is capped at max_actions_per_turn (=4) tool calls; even
    #   four web actions with verbose string args is a few hundred tokens — well under
    #   1k. 16384 was wildly oversized. 4096 keeps ~10x headroom over the realistic
    #   worst case while reclaiming 12288 tokens of input budget.
    #
    #   reason_tokens 2048 -> 4096. Phase 1 is the model's free <think> chain (llm.py
    #   ::_reason, stops at </think>). The VibeThinker 3B emits LONG reasoning, so 2048
    #   risked truncating its thinking mid-chain. We DOUBLE it to 4096 so reasoning is
    #   not starved, while staying small enough to keep a large positive input budget.
    #
    # New reservation = 4096 + 4096 = 8192 (was 18432). With the 1024 safety margin,
    # input_budget = 32768 - 8192 - 1024 = 23552 tokens (was 13312) — a ~11k-token
    # snapshot + a realistic ~2-3k-token system prompt + several k of history all fit
    # without dropping the snapshot. See tests/test_snapshot_budget.py.
    #
    # ISSUE #123 (beta_qwen3coder) NOTE: unlike VibeThinker, Qwen2.5-Coder-3B-Instruct
    # is NOT a long-chain reasoning model — it does not emit verbose <think> chains by
    # default, so phase-1 typically returns quickly (well under 4096). We KEEP 4096 as a
    # harmless ceiling (the input-budget math is identical to beta, so all snapshot/budget
    # tests carry over unchanged); it is a cap, not a target. If a future run shows the
    # model never reasons at length, this can be lowered to reclaim input budget without
    # affecting correctness. The action reservation is unchanged — phase-2 emits the
    # compact <tool_call> blocks (a few hundred tokens), so 4096 has ~10x headroom.
    reason_tokens: int = 4096         # phase 1 (free reasoning, discarded) — see #92/#123
    action_tokens: int = 4096         # phase 2 (codec action; unconstrained for hermes) — #92
    # Qwen3 native-tools two-phase thinking cap (decide_chat path).
    # Phase 1 stops at </think> or this many tokens, whichever comes first; the capped
    # thinking is replayed as an assistant prefill so phase 2 starts after </think>.
    # 1024 is the documented sweet-spot from the Qwen3 technical report (arXiv:2505.09388):
    # sufficient for routine tool-call decisions, small enough to keep input budget healthy.
    # The previous single-phase approach used think:False which qwen3:4b ignores in some
    # Ollama versions (issue #12917); this two-phase cap is the reliable alternative.
    thinking_budget: int = 1024       # max thinking tokens (native decide_chat path)

    # ISSUE #183 (beta_qwen3coder): bounded REASON-THEN-ACT for the decide_chat path.
    # qwen3:4b is a REASONING model. Two failure modes were ground-truthed live:
    #   1. ``think:False`` does NOT stop it (Ollama #12917): it reasons anyway, the trace
    #      lands UNTAGGED in ``message.content`` and DEVOURS the action budget before any
    #      <tool_call> — every turn "could not parse a tool call" (run 20260626_210514:
    #      12/12 wasted).
    #   2. A single ``think:True`` + ``tools:`` call LEAKS: Ollama PROMOTES the <tool_call>
    #      drafts the model writes WHILE thinking into structured ``message.tool_calls``, so
    #      a call it was only CONSIDERING gets executed (run 20260626_222714 turn 1: 3
    #      drafts in the trace, EMPTY action, yet ``goto`` ran — parsed from the thinking).
    # Fix = a real two-phase split (see OllamaClient._decide_chat_reason_then_act): phase 1
    # thinks with ``think:True`` and NO ``tools:`` (so Ollama has no schema to promote
    # drafts against — verified: 4 drafts → tool_calls=[]); phase 2 replays the thinking as
    # a CLOSED <think> prefill and emits the COMMITTED call with ``tools:`` + ``think:False``
    # (verified: phase 2 → goto(...)). The thinking is captured as ``Decision.reasoning``
    # (shown to the user) and NEVER executed; only phase 2 can produce an action.
    # When True (default on this branch) ``decide_chat`` uses the two-phase split; set False
    # for a NON-thinking model (qwen2.5-coder) — ``think:True`` 400s on such a model.
    # Independent of ``two_phase`` (which must stay False so the native path stays enabled —
    # see RalphAgent._native).
    reason_then_act: bool = True

    # observation rendering
    observation_char_limit: int = 12000  # truncate big tool outputs in the narrative

    # backend
    backend: str = "ollama"           # "ollama" or "llamacpp"
    ollama_url: str = "http://127.0.0.1:11434"
    # ISSUE #77: a CONSTANT keep_alive sent on every Ollama request. Keeping the
    # value identical across requests (paired with the pinned num_ctx and
    # OLLAMA_MAX_LOADED_MODELS=1) means a fresh request reuses / re-pins the single
    # existing runner instead of leaving idle runners around to be re-loaded. "30m"
    # comfortably spans a long agent run without holding the GPU forever after it.
    ollama_keep_alive: str = "30m"
    llamacpp_url: str = "http://127.0.0.1:8080"
    request_timeout: int = 600

    # web toolset (Playwright Agent CLI)
    web_session: str = "vibe"
    web_cli_timeout: int = 90
    web_observation_char_limit: int = 14000
    # Cap on the auto-injected live page snapshot rendered into the per-turn system
    # prompt (issue #24). ARIA snapshots can be 800+ lines; truncate so the current
    # page state is shown without crowding the task out of context.
    # Raised 6000 -> 40000 (~10k tokens) per the issue #28 snapshot-size analysis
    # (SNAPSHOT_SIZE_ANALYSIS.md): 8/9 real pages exceed 6000 chars, and a 6k cap
    # truncates BEFORE the consent/Accept buttons the agent is told to click (those
    # late-DOM overlays land at the END of the ARIA tree, e.g. YouTube ~45k). 40000
    # is the knee of the coverage/budget curve. NOTE: even 40k can still clip the
    # very largest pages (e.g. w3schools ~63k) because overlay controls are emitted
    # last; the longer-term fix is to PRIORITIZE interactive/overlay controls in the
    # injected snapshot rather than rely on cap size (future enhancement, see #28).
    #
    # As of issue #43 this fixed cap is no longer the primary control: the snapshot
    # is sized DYNAMICALLY each turn so it may be as large as the remaining context
    # window allows (a 200k snapshot beside a 50k message is fine; a 500k snapshot
    # beside the same message is truncated only because together they would overflow
    # num_ctx). ``web_snapshot_char_limit`` is kept as an ABSOLUTE CEILING / safety
    # fallback: the dynamic budget is min(dynamic_budget, web_snapshot_char_limit).
    # Set it very high (or rely on the dynamic budget alone) to let the window be the
    # only limit. The dynamic computation lives in vibeharness.snapshot_budget.
    web_snapshot_char_limit: int = 2_000_000

    # --- dynamic snapshot budget (issue #43) ---
    # The live page snapshot injected into the per-turn system prompt is truncated
    # ONLY when including it whole would push the full model message (system prompt +
    # per-turn user/history) past the usable input window. The usable input window is
    #   input_budget = num_ctx - (reason_tokens + action_tokens) - snapshot_safety_margin_tokens
    # all in TOKENS. We estimate tokens from characters with a fixed, conservative
    # ratio (chars per token). 4.0 is the long-standing English rule of thumb; we keep
    # it configurable and deliberately treat it as a *floor* (real ARIA snapshots —
    # refs, URLs, punctuation — often pack FEWER chars per token, i.e. more tokens per
    # char, so a too-high ratio would UNDER-count tokens and risk overflow). Combined
    # with the explicit token safety margin this keeps us safely under num_ctx.
    snapshot_chars_per_token: float = 4.0
    # Extra tokens held back on top of the output reservation, absorbing chat-template
    # wrapping, role tokens, and tokenizer estimate error so we never reach num_ctx.
    snapshot_safety_margin_tokens: int = 1024
    web_headless: bool = False        # headed by default so a human can watch
    web_browser: str = "chrome"

    # --- same-turn duplicate tool-call suppression (issue #162) ---
    # Names of web tools for which an EXACT duplicate call within the SAME assistant turn
    # (same tool name + identical args, at ANY position — not only back-to-back) is
    # suppressed: the duplicate is NOT executed, is stripped from the assistant tool-call
    # block, and produces NO tool_response — so the model's replayed history shows no sign
    # it ever requested a duplicate (a history that implied it did would bias a small model
    # toward repeating itself). Per-tool OPT-IN: only the tools named here participate, so
    # the default (empty) is a complete no-op — behaviour is unchanged unless a run opts a
    # tool in. Applies ONLY to the web-agent flow (a live snapshot provider is present) and
    # never to the snapshot/validate pseudo-tools. CROSS-turn duplicates are deliberately
    # out of scope (those are handled by the anti-loop steer). The existing consecutive
    # filter still runs first; this generalises it to non-adjacent repeats for opted-in
    # tools only. See RalphAgent._suppress_same_turn_duplicates.
    #
    # Stored as a frozenset of tool names (the persisted-settings path supplies a list,
    # which the agent coerces with set(); membership is all that is required).
    web_dedup_same_turn_tools: frozenset = frozenset()

    # VibeThinker advisor — periodic free-text hint injector (beta_qwen3coder only).
    # When advisor_enabled=True, every advisor_interval Qwen turns an advisor model is called
    # (free-text, no schema) and its advice is injected into Qwen's next turn user message
    # as <user_advice>...</user_advice>.
    # Empty string means "same model as the base agent" (Qwen self-advises, one model in VRAM).
    # Set to "vibethinker:latest" to use VibeThinker as the advisor (requires model-swap mode).
    advisor_model: str = "vibethinker:latest"

    advisor_temperature: float = 1.0   # high diversity for advice
    advisor_interval: int = 5          # call advisor after N accumulated tool calls (across turns)
    advisor_enabled: bool = False      # opt-in; set True via CLI or settings

    # --- snapshot prose rendering (issue #64) ---
    # When True, the auto-injected live page snapshot is run through the deterministic
    # WebArena-style ARIA->prose transform (vibeharness.snapshot_prose) before injection,
    # instead of the raw Playwright ARIA-YAML. The prose prunes generic/img noise and
    # emits one ref-keyed line per interesting control, which small models reason over
    # far more reliably. Refs are preserved inline so the discrete web subtools (click/
    # fill/…) keep working unchanged. This is an A/B SEAM, not a replacement: set False
    # (default) to inject the raw ARIA snapshot exactly as before. Budgeting/diagnostics
    # are unchanged — only the text fed into the page section differs.
    #
    # ISSUE #125 (beta_qwen3coder): default TRUE. Iter 3 showed qwen2.5-coder:3b-instruct
    # could not reliably pick input refs from the RAW ARIA tree — it looped on a
    # hallucinated ref, clicked headings, and tried to `fill` a label <div> (e42) instead
    # of the input. The pruned, ref-keyed prose (one line per interactable, with fillable
    # affordances per #70) is exactly what a small model needs to map field -> correct ref.
    web_snapshot_prose: bool = True

    # --- Escalation / API provider ---
    # When the local model gets stuck (same tool call repeated escalation_stuck_threshold
    # times in a row), the run escalates mid-session to an external API model — same
    # browser, same session, just a stronger LLM answering the next turn.
    escalation_enabled: bool = True
    # ISSUE #197: default escalator → DeepSeek. The previous default (zhipuai/glm-5.2)
    # is unreachable in the standard setup — the shared ZHIPUAI_API_KEY is account
    # rate-limited (HTTP 429 {"code":"1302","message":"Rate limit reached for requests"}),
    # so the escalation machinery (#191/PR #195) silently no-ops out of the box. DeepSeek
    # answers (proven live: a full qwen3:4b → deepseek-v4-flash take-over ran to FINISHED),
    # which is what makes escalation_enabled=True meaningful by default (#191 acceptance).
    escalation_provider: str = "deepseek"          # key into providers.PROVIDERS
    escalation_model: str = "deepseek-v4-flash"    # empty = use provider default
    escalation_stuck_threshold: int = 3            # consecutive identical calls → stuck
    escalation_on_premature_validate: bool = True  # escalate on first premature validate

    # --- Validation provider ---
    # LLMValidator uses this provider's API model for a stronger, independent verdict.
    # Falls back to the main Ollama client when the provider key env var is absent.
    validation_provider: str = "zhipuai"
    validation_model: str = "glm-5.2"

    # --- Per-role model endpoints (issue #163) ---
    # Nested, role-keyed override of which model drives each agent role. When a role has
    # an entry here it WINS; otherwise the role falls back to the legacy flat keys above
    # via ``resolve_role_spec`` (so every pre-existing run/test behaves identically). Keyed
    # by role name ("base", "validator", "advisor", "escalation"). Empty by default — the
    # whole feature is opt-in and backward compatible.
    models: dict[str, ModelSpec] = field(default_factory=dict)


# Maps each role to the LEGACY flat Config keys it historically read, so a role with no
# explicit ``models`` entry resolves to exactly the same model/provider it always used.
# (provider-field, model-field) — provider-field is read as the registry name; for the
# local base/advisor roles it is the backend ("ollama"/"llamacpp").
_LEGACY_ROLE_KEYS: dict[str, tuple[str, str]] = {
    "base": ("backend", "model"),
    "validator": ("validation_provider", "validation_model"),
    "advisor": ("backend", "advisor_model"),
    "escalation": ("escalation_provider", "escalation_model"),
}


def resolve_role_spec(config: Config, role: str) -> ModelSpec:
    """Resolve the :class:`ModelSpec` that drives ``role`` (issue #163).

    Precedence: an explicit ``config.models[role]`` entry WINS; otherwise the role falls
    back to its LEGACY flat keys (``model``/``backend``, ``validation_*``, ``advisor_model``,
    ``escalation_*``) so every existing run and test keeps working unchanged. This
    backward-compat fallback is the contract that lets the new per-role seam ship without a
    behaviour change for any config that uses the old flat keys.

    The ``advisor`` role's model falls back to the base ``model`` when ``advisor_model`` is
    empty (the "Qwen self-advises" default), mirroring the historical advisor resolution.
    """
    explicit = config.models.get(role)
    if explicit is not None:
        return explicit
    try:
        provider_field, model_field = _LEGACY_ROLE_KEYS[role]
    except KeyError:
        raise KeyError(f"unknown agent role {role!r}; "
                       f"known roles: {', '.join(sorted(_LEGACY_ROLE_KEYS))}") from None
    provider = getattr(config, provider_field, "")
    model = getattr(config, model_field, "")
    if role == "advisor" and not model:
        model = config.model       # empty advisor_model => self-advise on the base model
    return ModelSpec(provider=provider, model=model)


# --------------------------------------------------------------------------- #
# Per-MODEL tool-call policy (issues #179 + #178).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelToolPolicy:
    """The per-MODEL tool-call codec + per-turn tool-call cap, keyed by model id.

    A single global codec/cap cannot serve a tiny local Ollama model AND a frontier API
    model: the local qwen3:4b speaks the native Hermes ``<tool_call>`` dialect and is a
    sub-7B model whose parallel-call reliability peaks at 3 calls/turn, whereas the z.ai/GLM
    API models expose a SEPARATE thinking trace (``reasoning_content``) and are best driven
    by a JSON-schema-constrained tool call (the ``json`` codec) and can safely batch more
    calls per turn. This registry makes both choices EXPLICIT per model.
    """
    codec: str
    max_actions_per_turn: int
    # ISSUE #193 — per-MODEL context window (TOKENS) for the input/snapshot budget.
    # This is the model's DOCUMENTED context length (ground-truthed from the provider API
    # docs; URLs cited in MODEL_TOOL_POLICIES below) and is used ONLY to size the input
    # budget (how much page snapshot / history to keep), via
    # ``vibeharness.snapshot_budget.input_budget_tokens``. It is NEVER sent to a provider:
    #   * the LOCAL Ollama path keeps sending ``Config.num_ctx`` as the runner ``num_ctx``
    #     option (the 8 GB-GPU ceiling, #77/#140) — so for local models the resolver returns
    #     ``config.num_ctx``, NOT this field;
    #   * the API path (DeepSeek/GLM) sends no context size at all — this window is purely a
    #     budgeting input, so the API never receives a giant ``num_ctx`` Ollama option.
    # 0 means "unknown / inherit config.num_ctx" (no behaviour change).
    context_window: int = 0
    rationale: str = ""


# Keyed by model id (matched case-insensitively, exact first; see ``model_tool_policy``).
#
# qwen3:4b — hermes + 3 (UNCHANGED behaviour, made explicit):
#   The native Qwen2.5/Hermes <tool_call> dialect is what qwen3:4b was trained on (codec
#   ground-truthed from its chat template; see QWEN3CODER_ANALYSIS.md). 3 calls/turn is the
#   empirical accuracy peak for sub-7B web agents (arXiv:2602.07359 "W&D": 68% at 3 vs 60%
#   at 5); a sub-7B model's parallel-call reliability is weaker than a frontier model's.
#
# z.ai/GLM API models — `json` (schema-constrained) + ground-truthed caps:
#   The OpenAI-compatible API path is single-shot and the GLM reasoning models stream their
#   thinking SEPARATELY as `reasoning_content` (consumed as reasoning by ApiLLMClient), so
#   the tool call must be constrained with a JSON-schema action array — the `json` codec —
#   not the native-only `hermes` codec (whose DecodeConstraint.json_schema is None → the
#   model emits free prose, the #179 root-cause failure). Caps are HARNESS-side policy
#   justified by documented capability, NOT a vendor-mandated per-turn limit (no such hard
#   limit is documented — z.ai exposes a `parallel_tool_calls` switch and allows up to 128
#   tool DEFINITIONS per request, AI/ML API GLM-4.7 ref). GLM-4.5 scored 90.6% on the
#   BrowseComp web benchmark and GLM-4.6 "expects fewer unnecessary errors when handling
#   multiple tool calls" (z.ai GLM-4.6 notes), so these frontier models tolerate more
#   batching than a sub-7B local model. Best practice (z.ai): only parallelise truly
#   independent calls — so we stay conservative, well under the 128-definition ceiling:
#     glm-4.7-flash → 5  (lighter/faster "flash" tier; W&D's 5-call setting is safe given
#                         native parallel_tool_calls support, kept modest for the small tier)
#     glm-4.7       → 8  (flagship agentic-coding model: 200K ctx, native parallel_tool_calls)
#     glm-5.2       → 8  (newest flagship, long-horizon agentic line — z.ai GLM-5.x)
# Sources (cited in the PR): https://docs.z.ai/guides/capabilities/function-calling ,
#   https://docs.aimlapi.com/api-references/text-models-llm/zhipu/glm-4.7 ,
#   https://z.ai/blog/glm-4.6 , arXiv:2602.07359.
#
# DeepSeek API models — `json` (schema-constrained) + ground-truthed caps (issue #182):
#   DeepSeek is OpenAI-compatible (base_url https://api.deepseek.com). As of DeepSeek-V3.1
#   (release 2025-08-21) BOTH exposed model ids map to V3.1: `deepseek-chat` is the
#   non-thinking mode, `deepseek-reasoner` is the thinking mode, 128K context for both.
#     deepseek-chat (V3.1 non-thinking) — supports function/tool calling (the official
#       function-calling guide's example uses exactly model="deepseek-chat"). Single-shot
#       OpenAI-compatible path → the `json` schema-constrained codec (the native-only hermes
#       codec emits free prose on this path — the #179 root cause), NOT hermes.
#     deepseek-reasoner (V3.1 thinking) — TOOL-CALLING GROUND TRUTH (issue #182): the LEGACY
#       R1 reasoning model did NOT support function calling (old reasoning_model guide listed
#       Function Calling as UNSUPPORTED), but V3.1 CHANGED this — the current Thinking-Mode
#       guide states: "The DeepSeek model's thinking mode supports tool calls. Before
#       outputting the final answer, the model can perform multiple turns of reasoning and
#       tool calls." It streams its chain-of-thought SEPARATELY as `reasoning_content`
#       (consumed as REASONING by ApiLLMClient, never fed to the tool-call parser — exactly
#       the GLM path), so it gets the same `json` codec. CAVEAT (documented, sidestepped): in
#       NATIVE multi-turn tool calling the assistant's `reasoning_content` MUST be replayed in
#       every subsequent request or the API returns 400; the harness uses the SINGLE-SHOT
#       constrained-JSON protocol (it never sends `tools` and never replays an assistant
#       reasoning_content — it re-derives state from the live page snapshot each turn), so the
#       400 condition cannot arise. Reasoner is given a slightly more conservative cap because
#       it reasons BETWEEN calls (parallel batching helps a thinking model less).
#   Caps are HARNESS-side policy: DeepSeek documents NO hard per-request tool-count limit, so
#   we stay conservative (best practice: parallelise only truly independent calls), modestly
#   above the local-model default and below GLM's flagship cap given the single-shot path.
# Sources (cited in the PR): https://api-docs.deepseek.com/guides/function_calling ,
#   https://api-docs.deepseek.com/guides/thinking_mode (reasoner tool-call support) ,
#   https://api-docs.deepseek.com/guides/reasoning_model (legacy R1: FC unsupported) ,
#   https://api-docs.deepseek.com/news/news250821 (V3.1 release: 128K ctx, chat/reasoner modes).
# ISSUE #193 — per-MODEL context window (TOKENS), ground-truthed from the LIVE provider
# docs (verified 2026-06, URLs below). Used ONLY to size the input/snapshot budget
# (resolve_model_context_window → input_budget_tokens); NEVER sent to a provider. The
# qwen3:4b window is the GPU-pinned 32768 (a hardware/VRAM limit, #77/#140 — NOT the
# model's capability) and the LOCAL resolver returns config.num_ctx for it (so a
# --num-ctx override is honoured); the API models get their documented windows so a large
# (~45k-char YouTube) ARIA snapshot is no longer truncated to the 32768-derived budget.
#
# DeepSeek (api-docs.deepseek.com):
#   deepseek-chat / deepseek-reasoner — DeepSeek-V3.1 documents 128K context for BOTH the
#   non-thinking (chat) and thinking (reasoner) modes
#   (https://api-docs.deepseek.com/news/news250821 — "128K context for both"). As of 2026
#   these two model IDs additionally ROUTE to deepseek-v4-flash (1M context) and are slated
#   to retire 2026-07-24 (https://api-docs.deepseek.com/news/news260424). We pin the budget
#   window to the STABLE, model-id-native 128K (131072): it is the documented baseline, it
#   already exceeds the largest real snapshot by ~25x (so the qwen 32768 cap is fully
#   lifted), and it stays safely valid through the alias retirement. (Sources cited in PR:
#   https://api-docs.deepseek.com/news/news250821 , .../news/news260424 .)
# z.ai/GLM (docs.z.ai):
#   glm-4.7       → 200K (204800): GLM-4.6 "expands the context window to 200K"
#                   (https://docs.z.ai/release-notes/new-released ,
#                    https://docs.z.ai/guides/llm/glm-4.6); glm-4.7 is the flagship
#                   successor and inherits the 200K flagship window.
#   glm-4.7-flash → 128K (131072): the lighter "flash" tier (GLM-4.5-Flash documents 128K
#                   context — https://github.com/zai-org/GLM-4.5); conservative for the
#                   small tier, still ~25x the largest snapshot.
#   glm-5.2       → 1M (1048576): z.ai release notes — "GLM-5.2 supports 1M lossless context"
#                   (https://docs.z.ai/release-notes/new-released).
MODEL_TOOL_POLICIES: dict[str, ModelToolPolicy] = {
    # ISSUE #197 ("simple fixes for now"): max_actions_per_turn=99 on EVERY policy below
    # (and both family fallbacks) — a deliberate, temporary lift of the per-turn batching
    # cap while we evaluate. This intentionally SUPERSEDES the #178-ground-truthed per-model
    # caps (3/5/8/6/4 etc.) for now; the documented per-model rationales are retained for
    # when we re-tighten. codec + context_window are unchanged from #178/#182/#193.
    "qwen3:4b": ModelToolPolicy(
        codec="hermes", max_actions_per_turn=99, context_window=32768,
        rationale="native Hermes dialect; sub-7B 3-call accuracy peak (arXiv:2602.07359); "
                  "ctx 32768 = GPU-pinned num_ctx (#77/#140), NOT a model limit; "
                  "cap lifted to 99 for now (#197)"),
    "glm-4.7-flash": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=131072,
        rationale="API reasoning model; schema-constrained JSON; lighter flash tier; "
                  "128K ctx (GLM-4.5-Flash documented 128K); cap lifted to 99 for now (#197)"),
    "glm-4.7": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=204800,
        rationale="flagship agentic-coding; 200K ctx (GLM-4.6 documented 200K) + native "
                  "parallel_tool_calls; cap lifted to 99 for now (#197)"),
    "glm-5.2": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=1048576,
        rationale="newest flagship long-horizon agentic line; 1M lossless ctx "
                  "(z.ai release notes); cap lifted to 99 for now (#197)"),
    "deepseek-chat": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=131072,
        rationale="DeepSeek-V3.1 non-thinking; OpenAI-compat function calling; "
                  "128K ctx (V3.1 documented baseline); cap lifted to 99 for now (#197)"),
    "deepseek-reasoner": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=131072,
        rationale="DeepSeek-V3.1 thinking; tool calls supported since V3.1 (was unsupported "
                  "on legacy R1); reasoning_content consumed as reasoning; 128K ctx (V3.1 "
                  "documented baseline); cap lifted to 99 for now (#197)"),
    # ISSUE #197: explicit DeepSeek-V4 entries (the V4 successors the deepseek-chat /
    # deepseek-reasoner aliases now route to, and the default escalator model). Both natively
    # support a 1M-token context window and tool/function calling — pinning explicit entries
    # (instead of the 128K _DEEPSEEK_FAMILY_POLICY fallback) lets the #193 budget-follows-
    # escalation track the true 1M window rather than under-sizing it.
    # Sources: https://api-docs.deepseek.com/news/news260424 (V4-Pro 1.6T/49B + V4-Flash
    #   284B/13B; "1M context is now the default across all official DeepSeek services";
    #   Tool Calls guide + enhanced agentic capabilities) ,
    #   https://openrouter.ai/deepseek/deepseek-v4-flash .
    "deepseek-v4-flash": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=1_000_000,
        rationale="DeepSeek-V4-Flash (284B/13B-active); OpenAI-compat tool calling; native 1M "
                  "context (news260424); json single-shot codec; cap lifted to 99 for now (#197)"),
    "deepseek-v4-pro": ModelToolPolicy(
        codec="json", max_actions_per_turn=99, context_window=1_000_000,
        rationale="DeepSeek-V4-Pro (1.6T/49B-active); OpenAI-compat tool calling; native 1M "
                  "context (news260424); json single-shot codec; cap lifted to 99 for now (#197)"),
}

# Family fallback for any other z.ai/GLM API model id (e.g. a future glm-4.7-air): the API
# path still needs a JSON-schema-constrained codec (never the native-only hermes), so an
# unrecognised GLM model resolves to the conservative `json` + 5 policy rather than silently
# inheriting the local `hermes` default. The qwen-local default is the global Config codec.
_GLM_FAMILY_POLICY = ModelToolPolicy(
    codec="json", max_actions_per_turn=99, context_window=131072,
    rationale="unrecognised GLM/z.ai model: API JSON codec + 128K ctx; "
              "cap lifted to 99 for now (#197)")

# Family fallback for any other DeepSeek API model id (issue #182) — e.g. a future
# `deepseek-v4-flash` (the V4 successor the deepseek-chat/deepseek-reasoner aliases now point
# at). The OpenAI-compatible single-shot path always needs a JSON-schema-constrained codec
# (never the native-only hermes), so an unrecognised DeepSeek model resolves to the
# conservative `json` + 5 policy rather than silently inheriting the local `hermes` default.
_DEEPSEEK_FAMILY_POLICY = ModelToolPolicy(
    codec="json", max_actions_per_turn=99, context_window=131072,
    rationale="unrecognised DeepSeek model: API JSON codec + 128K ctx; "
              "cap lifted to 99 for now (#197)")


def model_tool_policy(model: str | None) -> ModelToolPolicy | None:
    """Return the :class:`ModelToolPolicy` for ``model`` (issues #179/#178/#182), or ``None``.

    Exact (case-insensitive) match in :data:`MODEL_TOOL_POLICIES` wins; otherwise an API
    family id falls back to a conservative schema-constrained policy so a new variant is still
    driven with the `json` codec (never the native-only hermes): a ``glm`` prefix → the GLM
    family policy, a ``deepseek`` prefix → the DeepSeek family policy. Everything else returns
    ``None`` → the caller uses the :class:`Config` fallback (no behaviour change)."""
    if not model:
        return None
    key = model.strip().lower()
    for name, policy in MODEL_TOOL_POLICIES.items():
        if key == name.lower():
            return policy
    if key.startswith("glm"):
        return _GLM_FAMILY_POLICY
    if key.startswith("deepseek"):
        return _DEEPSEEK_FAMILY_POLICY
    return None


def resolve_model_codec(config: Config, spec: ModelSpec) -> str:
    """Resolve the tool-call codec NAME for ``spec`` (issue #179).

    Precedence: the spec's explicit ``codec`` field WINS; else the per-model registry
    (:func:`model_tool_policy`) keyed by the model id; else ``config.codec`` (the global
    fallback). The BASE agent's spec is what drives the run's codec."""
    if spec.codec:
        return spec.codec
    policy = model_tool_policy(spec.model)
    if policy is not None:
        return policy.codec
    return config.codec


def resolve_model_limit(config: Config, spec: ModelSpec) -> int:
    """Resolve the per-turn tool-call cap for ``spec`` (issue #178).

    Precedence: the spec's explicit ``max_actions_per_turn`` field WINS; else the per-model
    registry (:func:`model_tool_policy`); else ``config.max_actions_per_turn`` (the global
    fallback). The BASE agent's spec is what drives the run's cap (the validator stays
    single-shot = 1)."""
    if spec.max_actions_per_turn is not None:
        return spec.max_actions_per_turn
    policy = model_tool_policy(spec.model)
    if policy is not None:
        return policy.max_actions_per_turn
    return config.max_actions_per_turn


def _provider_is_local(name: str) -> bool:
    """True if ``name`` is a LOCAL (ollama/llamacpp) endpoint (issue #193).

    Resolved through the provider registry so it tracks one source of truth; an unknown
    provider name degrades to a literal kind-token check (``"ollama"``/``"llamacpp"``). The
    import is local to avoid any import-ordering coupling between config and providers."""
    from .providers import LOCAL_KINDS, get_endpoint, is_local
    try:
        return is_local(get_endpoint(name))
    except KeyError:
        return name in LOCAL_KINDS


def resolve_model_context_window(config: Config, spec: ModelSpec) -> int:
    """Resolve the context window (TOKENS) that bounds the INPUT/snapshot budget for
    ``spec`` (issue #193). This is a BUDGETING input only — it is never sent to a provider.

    Precedence:
      * a LOCAL endpoint (ollama/llamacpp) → ``config.num_ctx`` — the runner is literally
        created at that size (the 8 GB-GPU ceiling, #77/#140), so it IS the real window and
        a ``--num-ctx`` override is honoured. qwen3:4b therefore stays 32768, NOT raised.
      * else the per-model registry's documented ``context_window``
        (:func:`model_tool_policy`, ground-truthed from the provider docs) when set (> 0);
      * else ``config.num_ctx`` (unknown API model → conservative, no behaviour change).
    """
    if _provider_is_local(spec.provider):
        return config.num_ctx
    policy = model_tool_policy(spec.model)
    if policy is not None and policy.context_window > 0:
        return policy.context_window
    return config.num_ctx


def effective_context_window(config: Config, spec: ModelSpec | None = None) -> int:
    """The context window (TOKENS) for the ACTIVE base model (issue #193).

    When ``spec`` is None the BASE role's spec is resolved (:func:`resolve_role_spec`), so
    the input/snapshot budget reflects whichever model is currently driving the run (e.g. a
    DeepSeek/GLM API model gets its large documented window instead of qwen3:4b's 32768).
    Callers that have already escalated to a different model (the agent) pass that model's
    ``spec`` explicitly so the budget tracks the live client."""
    if spec is None:
        spec = resolve_role_spec(config, "base")
    return resolve_model_context_window(config, spec)
