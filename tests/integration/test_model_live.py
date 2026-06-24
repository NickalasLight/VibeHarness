"""Live model tests: confirm Ollama is available and the model actually generates.

This is the core-functionality smoke test. The fast unit suite mocks the LLM, so a
crashed/stopped Ollama or a broken generation path would otherwise pass unnoticed.
This test really talks to Ollama, generates text, and stops inference early.

The generation tests auto-skip when Ollama isn't reachable (so CI stays green); the
error-path test always runs (it points at a dead port and needs no server).
"""
import unittest
from dataclasses import replace

import pytest

from vibeharness.config import Config
from vibeharness.llm import OllamaClient, OllamaUnavailable


@pytest.mark.needs_ollama
class ModelGenerationTest(unittest.TestCase):
    def test_model_generates_text(self):
        text = OllamaClient(Config()).generate(
            "Write one short sentence about the ocean.", max_chars=100)
        self.assertTrue(text.strip(), "the model produced no text — is Ollama healthy?")

    def test_generation_stops_after_about_100_chars(self):
        seen = []
        text = OllamaClient(Config()).generate(
            "Count upward in words, slowly, one number per line.",
            max_chars=100, on_token=lambda t: seen.append(t))
        self.assertGreater(len(text), 0)              # it generated
        self.assertTrue(seen)                          # it streamed token by token
        self.assertLess(len(text), 200, "early-stop at 100 chars did not bound the output")


class OllamaUnavailableTest(unittest.TestCase):
    """Always runs (uses a dead port): a down Ollama must fail fast and clearly."""
    def test_generate_raises_when_server_down(self):
        client = OllamaClient(replace(Config(), ollama_url="http://127.0.0.1:1"))
        with self.assertRaises(OllamaUnavailable):
            client.generate("hello", max_chars=20)


if __name__ == "__main__":
    unittest.main()
