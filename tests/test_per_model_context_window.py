"""Per-MODEL context window for the input/snapshot budget (issue #193).

Covers, WITHOUT a live model or a browser:
  * the per-model context-window registry (qwen3:4b → GPU-pinned 32768; DeepSeek/GLM →
    their documented windows; family fallbacks);
  * resolve_model_context_window precedence (LOCAL provider → config.num_ctx;
    API model → documented window; unknown API model → config.num_ctx);
  * effective_context_window resolves the BASE role's window (and honours config.models);
  * input_budget_tokens reflects the per-model window (DeepSeek/GLM >> 32768-derived;
    qwen3:4b unchanged at the 32768-derived 23552);
  * the end-to-end fit_chat_history effect: a large (~45k-char YouTube) ARIA snapshot is
    TRUNCATED on qwen3:4b's 32768 window but kept WHOLE on a DeepSeek/GLM window;
  * the API path does NOT emit a num_ctx Ollama option (the window is budgeting-only).
"""
from __future__ import annotations

import unittest
from dataclasses import replace

from vibeharness.config import (Config, ModelSpec, effective_context_window,
                                model_tool_policy, resolve_model_context_window)
from vibeharness.snapshot_budget import fit_chat_history, input_budget_tokens


def _api(provider: str, model: str) -> Config:
    """A Config whose BASE role is the given API model (issue #163 per-role seam)."""
    return replace(Config(), models={"base": ModelSpec(provider=provider, model=model)})


class RegistryWindowTest(unittest.TestCase):
    def test_qwen_window_is_gpu_pinned_32768(self):
        # The qwen3:4b policy documents 32768 = the GPU-pinned num_ctx (NOT a model limit).
        self.assertEqual(model_tool_policy("qwen3:4b").context_window, 32768)

    def test_deepseek_windows_are_128k(self):
        self.assertEqual(model_tool_policy("deepseek-chat").context_window, 131072)
        self.assertEqual(model_tool_policy("deepseek-reasoner").context_window, 131072)

    def test_glm_documented_windows(self):
        self.assertEqual(model_tool_policy("glm-4.7-flash").context_window, 131072)  # 128K
        self.assertEqual(model_tool_policy("glm-4.7").context_window, 204800)        # 200K
        self.assertEqual(model_tool_policy("glm-5.2").context_window, 1048576)       # 1M

    def test_family_fallbacks_carry_a_window(self):
        # An unrecognised GLM / DeepSeek id still resolves to a documented-floor 128K window.
        self.assertEqual(model_tool_policy("glm-9.9-air").context_window, 131072)
        self.assertEqual(model_tool_policy("deepseek-v9-flash").context_window, 131072)


class ResolveContextWindowTest(unittest.TestCase):
    def test_local_provider_uses_num_ctx(self):
        cfg = Config()
        spec = ModelSpec(provider="ollama", model="qwen3:4b")
        self.assertEqual(resolve_model_context_window(cfg, spec), cfg.num_ctx)

    def test_local_provider_honours_num_ctx_override(self):
        # A --num-ctx override flows through for the local model (it IS the runner size).
        cfg = replace(Config(), num_ctx=16384)
        spec = ModelSpec(provider="ollama", model="qwen3:4b")
        self.assertEqual(resolve_model_context_window(cfg, spec), 16384)

    def test_api_model_uses_documented_window(self):
        cfg = Config()
        self.assertEqual(
            resolve_model_context_window(cfg, ModelSpec("deepseek", "deepseek-chat")), 131072)
        self.assertEqual(
            resolve_model_context_window(cfg, ModelSpec("zhipuai", "glm-5.2")), 1048576)

    def test_unknown_api_model_falls_back_to_num_ctx(self):
        # An API provider with a model id we have no policy for → conservative num_ctx.
        cfg = Config()
        spec = ModelSpec(provider="zhipuai", model="totally-unknown-model")
        self.assertEqual(resolve_model_context_window(cfg, spec), cfg.num_ctx)

    def test_effective_window_resolves_base_role(self):
        base = Config()                                  # default base = local qwen3:4b
        self.assertEqual(effective_context_window(base), 32768)
        self.assertEqual(effective_context_window(_api("deepseek", "deepseek-chat")), 131072)
        self.assertEqual(effective_context_window(_api("zhipuai", "glm-5.2")), 1048576)


class InputBudgetTest(unittest.TestCase):
    def test_qwen_input_budget_unchanged(self):
        # 32768 - (reason 4096 + action 4096 + margin 1024) = 23552 (pre-#193 value).
        self.assertEqual(input_budget_tokens(Config()), 23552)

    def test_api_models_get_far_larger_budget(self):
        ds = input_budget_tokens(_api("deepseek", "deepseek-chat"))
        glm = input_budget_tokens(_api("zhipuai", "glm-5.2"))
        self.assertEqual(ds, 131072 - 9216)
        self.assertEqual(glm, 1048576 - 9216)
        self.assertGreater(ds, 23552)
        self.assertGreater(glm, ds)

    def test_explicit_window_override(self):
        self.assertEqual(input_budget_tokens(Config(), window=200000), 200000 - 9216)


class SnapshotKeptWholeTest(unittest.TestCase):
    """End-to-end: the ~45k-char YouTube ARIA snapshot is truncated on qwen3:4b's 32768
    window but kept WHOLE on a DeepSeek/GLM window (the #193 acceptance demo)."""

    def _scenario(self):
        # A large system prompt + the latest 45k-char page snapshot. On the 32768 window the
        # snapshot cannot fit whole (it must be truncated); on a 128K+ window it fits.
        system = "S" * 60000
        user = "U" * 8000
        snap = "<tool_response>\n## Latest page state — NOW\n" + "P" * 45000 + "\n</tool_response>"
        return [{"role": "user", "content": snap}], system, user

    def test_qwen_truncates_but_api_keeps_whole(self):
        hist, system, user = self._scenario()
        raw_chars = len(hist[0]["content"])

        q = fit_chat_history(hist, system, user, Config(),
                             effective_context_window(Config()))
        self.assertTrue(q.snapshot_truncated)
        self.assertLess(q.snapshot_kept_chars, raw_chars)
        self.assertEqual(q.num_ctx, 32768)

        for provider, model in [("deepseek", "deepseek-chat"), ("zhipuai", "glm-5.2")]:
            cfg = _api(provider, model)
            hist, system, user = self._scenario()
            r = fit_chat_history(hist, system, user, cfg, effective_context_window(cfg))
            self.assertFalse(r.snapshot_truncated, f"{model} should keep the snapshot whole")
            self.assertEqual(r.snapshot_kept_chars, raw_chars)
            self.assertGreater(r.num_ctx, 32768)


class ApiPathDoesNotSendNumCtxTest(unittest.TestCase):
    """The per-model window is a BUDGETING input only: it must never be sent to the API as
    an Ollama num_ctx option. The OpenAI-compatible client builds NO num_ctx; the local
    Ollama client's options still carry exactly config.num_ctx (the GPU-pinned runner size,
    unchanged by this feature)."""

    def test_api_client_emits_no_num_ctx(self):
        from vibeharness.api_llm import ApiLLMClient
        from vibeharness.providers import get_endpoint
        client = ApiLLMClient(provider=get_endpoint("deepseek"), api_key="x",
                              model="deepseek-chat")
        payload = client._chat_kwargs(system="s", user="u", constraint=None) \
            if hasattr(client, "_chat_kwargs") else {}
        # Whatever the API request shape, it must never contain an Ollama 'num_ctx' option.
        self.assertNotIn("num_ctx", payload)
        # And the class source carries no num_ctx wiring at all.
        import inspect
        self.assertNotIn("num_ctx", inspect.getsource(ApiLLMClient))

    def test_ollama_options_still_send_config_num_ctx(self):
        from vibeharness.llm import OllamaClient
        cfg = Config()
        opts = OllamaClient(cfg)._options()
        self.assertEqual(opts["num_ctx"], cfg.num_ctx)   # 32768, unchanged by #193


if __name__ == "__main__":
    unittest.main()
