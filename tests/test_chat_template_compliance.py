"""Integration test: validate the harness's tool schema and message construction against
the ground-truth HuggingFace chat templates for Qwen2.5-Coder and VibeThinker.

This is the living ground-truth validator. It fetches the real ``chat_template`` from the
HuggingFace API, renders it with Jinja2 using fixtures, and asserts what the template
ACTUALLY produces — so prompt/schema divergence is caught before it reaches a real run.

Why this matters on ``beta_qwen3coder``
---------------------------------------
The default model here is ``qwen2.5-coder:3b-instruct`` and the active codec is the
``hermes`` codec (``vibeharness/codecs/hermes_codec.py``). The codec speaks the model's
NATIVE tool-calling dialect, which it claims was ground-truthed from the model's chat
template:

  * tool DEFINITIONS rendered as **bare** ``{"name","description","parameters"}`` JSON
    (NO ``{"type":"function","function":{...}}`` envelope), one per line in a
    ``<tools>...</tools>`` block (``ToolRegistry.tools_block(style="hermes")`` /
    ``HermesCodec.tool_definitions``),
  * each tool CALL is ``{"name": <tool>, "arguments": {...}}`` wrapped in a
    ``<tool_call>...</tool_call>`` tag (``HermesCodec.format_instructions``),
  * tool RESULTS fed back inside ``<tool_response>...</tool_response>``.

This test re-verifies those three claims against the LIVE template (not a copy), and
checks the harness's own codec output renders cleanly through the real template — the one
place a wrong assumption silently degrades tool-use quality.

Requires: network access and jinja2. An HF_TOKEN env var is OPTIONAL — these public
templates are fetchable without auth; if a request is rejected (rate limit / gated), the
test skips rather than fails. (No token is hard-coded; supply one via the environment if
you hit anonymous rate limits.)
Run with: pytest tests/test_chat_template_compliance.py -v -m integration
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
from jinja2 import BaseLoader, Environment  # noqa: E402

pytestmark = pytest.mark.integration

# Optional: only used if present. Never hard-code a token here (push protection / secret
# scanning will reject it, and the public templates fetch fine anonymously).
HF_TOKEN = os.getenv("HF_TOKEN", "")
CACHE_DIR = Path(__file__).parent.parent / ".vibe" / "hf_templates"

# --- model config (verified against the live HF API while authoring this test) ---
# Qwen2.5-Coder-3B-Instruct keeps its chat_template inside tokenizer_config.json.
QWEN_HF_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
# VibeThinker (WeiboAI/VibeThinker-1.5B) is a Qwen2ForCausalLM ("qwen2") derivative whose
# chat template lives in a SEPARATE chat_template.jinja file (tokenizer_config.json has no
# chat_template key) — fetch_template falls back to it automatically.
VIBETHINKER_HF_ID = "WeiboAI/VibeThinker-1.5B"

# --- fixtures ---
# IMPORTANT (ground-truthed): the native template renders each tool with `tool | tojson`,
# i.e. it dumps WHATEVER object is in the `tools` list verbatim. The hermes codec passes
# the BARE shape below, so that is what this test exercises. (A bare schema and an
# enveloped one BOTH render verbatim — the template imposes no envelope; the harness's
# choice of bare-vs-enveloped is the thing under test.)
SAMPLE_TOOLS_BARE = [
    {
        "name": "fill",
        "description": "Fill a form field with text.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Element ref"},
                "text": {"type": "string", "description": "Text to fill"},
            },
            "required": ["target", "text"],
        },
    }
]

SAMPLE_MESSAGES_NO_TOOLS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# NOTE: the template renders tool-call arguments via `tool_call.arguments | tojson`, so
# `arguments` must be a JSON-serialisable OBJECT, not a pre-stringified JSON blob (a string
# would be double-encoded and rendered with surrounding quotes).
SAMPLE_MESSAGES_WITH_TOOL_CALL = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Fill the name field."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "fill", "arguments": {"target": "e12", "text": "Jason"}}}
        ],
    },
    {"role": "tool", "content": "OK - filled 'e12' with 'Jason'", "tool_call_id": "call_1"},
]


# --------------------------------------------------------------------------- helpers
def fetch_template(hf_model_id: str) -> str:
    """Fetch a model's chat template from HuggingFace, caching it to disk.

    Tries tokenizer_config.json first (most models embed the template there), then falls
    back to a standalone chat_template.jinja file (used by e.g. VibeThinker). Skips the
    test when the template cannot be fetched (offline / auth / not present)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{hf_model_id.replace('/', '_')}_chat_template.txt"
    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8")
        if cached:
            return cached

    base = f"https://huggingface.co/{hf_model_id}/resolve/main"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    template, last_err = "", None
    try:  # 1) tokenizer_config.json -> chat_template
        req = urllib.request.Request(f"{base}/tokenizer_config.json", headers=headers)
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        template = data.get("chat_template", "") or ""
    except Exception as e:  # noqa: BLE001
        last_err = e
    if not template:  # 2) fallback: standalone chat_template.jinja
        try:
            req = urllib.request.Request(f"{base}/chat_template.jinja", headers=headers)
            template = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            last_err = e
    if not template:
        pytest.skip(
            f"Could not fetch a chat template for {hf_model_id} "
            f"(tokenizer_config.json + chat_template.jinja unavailable): {last_err}"
        )

    cache_file.write_text(template, encoding="utf-8")
    return template


def render_template(template_str: str, messages: list, tools: list | None = None,
                    add_generation_prompt: bool = True) -> str:
    """Render a HuggingFace Jinja2 chat template, matching HF's rendering environment."""
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True,
                      keep_trailing_newline=True)
    # HF's `tojson` mirrors json.dumps; templates may pass kwargs (indent), so accept them.
    env.filters["tojson"] = lambda v, **kwargs: json.dumps(v, **kwargs)
    tmpl = env.from_string(template_str)
    return tmpl.render(messages=messages, tools=tools,
                       add_generation_prompt=add_generation_prompt,
                       bos_token="", eos_token="<|im_end|>")


def _tools_block_body(rendered: str) -> str:
    """Return the contents of the POPULATED <tools> ... </tools> block. The literal text
    "<tools></tools>" also appears in the instructional sentence, so anchor on the real
    block which the template opens with a newline."""
    return rendered.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]


# --------------------------------------------------------------------------- tests
# 1. Template fetch + cache + sanity
def test_fetch_qwen_template():
    """Fetch the Qwen2.5-Coder template from HuggingFace and sanity-check its shape."""
    template = fetch_template(QWEN_HF_ID)
    assert template
    assert "tool_call" in template          # native tool-calling protocol present
    assert "<|im_start|>" in template       # ChatML boundaries present


def test_qwen_template_is_cached():
    """A second fetch is served from the on-disk cache (no second network round-trip)."""
    fetch_template(QWEN_HF_ID)
    cache_file = CACHE_DIR / f"{QWEN_HF_ID.replace('/', '_')}_chat_template.txt"
    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8")


# 2. Native tool-calling protocol (the ground truth the model was trained on)
def test_template_declares_native_tool_call_protocol():
    """The native template wraps tools in <tools>, calls in <tool_call>, results in
    <tool_response> — exactly the dialect the hermes codec targets."""
    template = fetch_template(QWEN_HF_ID)
    assert "<tools>" in template and "</tools>" in template
    assert "<tool_call>" in template and "</tool_call>" in template
    assert "<tool_response>" in template and "</tool_response>" in template


def test_template_renders_tools_verbatim_via_tojson():
    """The template renders each tool with `tool | tojson` (verbatim dump). This is WHY
    the hermes codec is free to choose the bare schema shape — the template imposes no
    envelope. Asserting this pins the assumption the codec's design depends on."""
    template = fetch_template(QWEN_HF_ID)
    assert "tool | tojson" in template
    # Rendering the bare fixture comes out unwrapped (no function envelope injected).
    rendered = render_template(template, SAMPLE_MESSAGES_NO_TOOLS, tools=SAMPLE_TOOLS_BARE)
    body = _tools_block_body(rendered)
    assert '"name": "fill"' in body
    assert '"type": "function"' not in body      # template did NOT add an envelope
    assert '"function":' not in body


# 3. Harness <tools> definitions render cleanly through the native template
def test_harness_tools_block_renders_through_native_template():
    """The hermes codec's <tools> block (bare per-line function schemas) must parse back
    into JSON objects that the native template renders verbatim — i.e. the harness and the
    template agree on the tool-DEFINITION wire format."""
    from vibeharness.codec import get_codec
    from vibeharness.config import Config
    from vibeharness.toolset import default_catalog

    catalog = default_catalog()
    registry = catalog.build_registry(catalog.select(["fs"]), Config())
    block = get_codec("hermes").tool_definitions(registry)
    assert block is not None and block.startswith("<tools>") and block.rstrip().endswith("</tools>")

    # Each line between the tags must be a BARE {"name","description","parameters"} object.
    lines = [ln for ln in block.splitlines() if ln not in ("<tools>", "</tools>")]
    assert lines
    parsed = [json.loads(ln) for ln in lines]
    for obj in parsed:
        assert set(("name", "description", "parameters")) <= set(obj)
        assert "function" not in obj and obj.get("type") != "function"  # NOT enveloped

    # Feed those exact bare objects to the native template; they render verbatim.
    template = fetch_template(QWEN_HF_ID)
    rendered = render_template(template, SAMPLE_MESSAGES_NO_TOOLS, tools=parsed)
    body = _tools_block_body(rendered)
    for obj in parsed:
        assert f'"name": "{obj["name"]}"' in body
    assert '"type": "function"' not in body


# 4. Harness call/result wire format matches the template's expectation
def test_harness_format_instructions_match_template_call_format():
    """The hermes codec's format_instructions must describe the SAME <tool_call> JSON
    object the native template renders for an assistant tool call."""
    from vibeharness.codec import get_codec

    instr = get_codec("hermes").format_instructions(max_actions=5)
    assert "<tool_call>" in instr and "</tool_call>" in instr
    assert '"name"' in instr and '"arguments"' in instr

    # The template emits exactly this shape for an assistant tool call.
    template = fetch_template(QWEN_HF_ID)
    rendered = render_template(template, SAMPLE_MESSAGES_WITH_TOOL_CALL, tools=SAMPLE_TOOLS_BARE)
    assert "<tool_call>" in rendered
    assert '"name": "fill"' in rendered
    # Object-form arguments (arguments | tojson), not a double-encoded string.
    assert '"arguments": {"target": "e12", "text": "Jason"}' in rendered


def test_format_instructions_contains_antifence_clause():
    """After fix(#129/#130) the hermes format_instructions carry the anti-fence clause so
    the model emits raw <tool_call> blocks, never a ```json fence. (The native template
    itself has NO such clause — see test_template_has_no_antifence_clause — so the harness
    must supply it.)"""
    from vibeharness.codec import get_codec

    instr = get_codec("hermes").format_instructions(max_actions=5)
    assert "backtick" in instr.lower() or "```json" in instr
    assert "no other text" in instr.lower()


def test_template_has_no_antifence_clause():
    """Ground-truth check: the NATIVE Qwen/VibeThinker template contains no anti-fence
    ("backticks") instruction; the harness owns that guidance."""
    template = fetch_template(QWEN_HF_ID)
    assert "backtick" not in template.lower()
    assert "```" not in template


# 5. ChatML boundaries
def test_chatml_boundaries_present():
    """Rendered prompt uses the correct ChatML control tokens."""
    template = fetch_template(QWEN_HF_ID)
    rendered = render_template(template, SAMPLE_MESSAGES_NO_TOOLS)
    assert "<|im_start|>system" in rendered
    assert "<|im_start|>user" in rendered
    assert "<|im_end|>" in rendered
    assert rendered.rstrip().endswith("<|im_start|>assistant")  # generation prompt


# 6. Tool-result rendering (role:"tool" -> <tool_response> inside a user turn)
def test_tool_result_rendered_as_tool_response_in_user_turn():
    """Tool results (role:"tool") are wrapped in <tool_response> tags inside a
    <|im_start|>user turn — NOT a dedicated ChatML `tool` role."""
    template = fetch_template(QWEN_HF_ID)
    rendered = render_template(template, SAMPLE_MESSAGES_WITH_TOOL_CALL, tools=SAMPLE_TOOLS_BARE)
    assert "OK - filled" in rendered
    assert "<tool_response>" in rendered and "</tool_response>" in rendered
    assert "<|im_start|>tool" not in rendered     # no dedicated tool role
    before_resp = rendered.split("<tool_response>", 1)[0]
    assert before_resp.rstrip().endswith("<|im_start|>user")


# 7. VibeThinker template compatibility with Qwen
def test_vibethinker_template_shares_chatml_structure():
    """VibeThinker (Qwen2 derivative) uses a compatible ChatML + tool-call template."""
    vt_template = fetch_template(VIBETHINKER_HF_ID)
    assert "<|im_start|>" in vt_template
    assert "<tool_call>" in vt_template and "<tools>" in vt_template
    rendered = render_template(vt_template, SAMPLE_MESSAGES_NO_TOOLS)
    assert "<|im_start|>system" in rendered and "<|im_end|>" in rendered


def test_vibethinker_and_qwen_tool_protocol_match():
    """VibeThinker and Qwen must agree on the tool-call protocol tags so the single hermes
    codec can serve both models. (Default system prompts differ; the protocol does not.)"""
    qwen = fetch_template(QWEN_HF_ID)
    vt = fetch_template(VIBETHINKER_HF_ID)
    for tag in ("<tools>", "<tool_call>", "<tool_response>", "<|im_start|>", "<|im_end|>"):
        assert tag in qwen, f"missing {tag} in Qwen template"
        assert tag in vt, f"missing {tag} in VibeThinker template"
