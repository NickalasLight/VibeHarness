import os
import unittest
from unittest import mock

from vibeharness import providers
from vibeharness.providers import (ApiProviderConfig, api_key_present,
                                   get_provider, make_api_client)


class ProvidersTest(unittest.TestCase):
    def test_get_known_provider(self):
        p = get_provider("zhipuai")
        self.assertEqual(p.name, "zhipuai")
        self.assertEqual(p.model, "glm-5.2")
        self.assertEqual(p.api_key_env, "ZHIPUAI_API_KEY")
        self.assertTrue(p.base_url.startswith("https://"))

    def test_get_known_deepseek_provider(self):
        # Issue #182: DeepSeek registered as an OpenAI-compatible provider, key from env only.
        p = get_provider("deepseek")
        self.assertEqual(p.name, "deepseek")
        self.assertEqual(p.model, "deepseek-chat")
        self.assertEqual(p.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(p.base_url, "https://api.deepseek.com")

    def test_deepseek_missing_key_raises_runtime_error(self):
        # The key is read from DEEPSEEK_API_KEY at construction; absent → RuntimeError.
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                make_api_client("deepseek")
            self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_deepseek_constructs_client_with_env_key(self):
        fake = object()
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"}, clear=False):
            with mock.patch("vibeharness.api_llm.ApiLLMClient",
                            return_value=fake) as ctor:
                client = make_api_client("deepseek")
        self.assertIs(client, fake)
        _, kwargs = ctor.call_args
        self.assertEqual(kwargs["api_key"], "secret")
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["provider"].name, "deepseek")

    def test_get_unknown_provider_raises_with_known_names(self):
        with self.assertRaises(KeyError) as ctx:
            get_provider("nope")
        self.assertIn("zhipuai", str(ctx.exception))

    def test_no_api_key_in_provider_config(self):
        # The dataclass must hold only coordinates, never a secret value.
        p = get_provider("zhipuai")
        for value in (p.name, p.base_url, p.api_key_env, p.model):
            self.assertNotIn("sk-", value.lower())

    def test_api_key_present(self):
        p = ApiProviderConfig("x", "https://x/", "X_TEST_KEY_ENV", "m")
        with mock.patch.dict(os.environ, {"X_TEST_KEY_ENV": ""}, clear=False):
            self.assertFalse(api_key_present(p))
        with mock.patch.dict(os.environ, {"X_TEST_KEY_ENV": "secret"}, clear=False):
            self.assertTrue(api_key_present(p))

    def test_make_api_client_missing_key_raises_runtime_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                make_api_client("zhipuai")
            self.assertIn("ZHIPUAI_API_KEY", str(ctx.exception))

    def test_make_api_client_unknown_provider_raises_key_error(self):
        with self.assertRaises(KeyError):
            make_api_client("does-not-exist")

    def test_make_api_client_constructs_client_when_key_present(self):
        fake = object()
        with mock.patch.dict(os.environ, {"ZHIPUAI_API_KEY": "secret"}, clear=False):
            with mock.patch.object(providers, "ApiProviderConfig", ApiProviderConfig):
                with mock.patch("vibeharness.api_llm.ApiLLMClient",
                                return_value=fake) as ctor:
                    client = make_api_client("zhipuai", model_override="glm-x")
        self.assertIs(client, fake)
        _, kwargs = ctor.call_args
        self.assertEqual(kwargs["api_key"], "secret")
        self.assertEqual(kwargs["model"], "glm-x")
        self.assertEqual(kwargs["provider"].name, "zhipuai")


if __name__ == "__main__":
    unittest.main()
