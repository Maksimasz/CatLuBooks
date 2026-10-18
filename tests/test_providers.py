"""Проверки реестра провайдеров и смены модели."""
import os
import unittest
from unittest.mock import patch

from config import LLMConfig, resolve_llm
from core.providers import base_url_for, key_for, needs_key, provider_catalog


class ProviderTests(unittest.TestCase):
    def test_ollama_does_not_need_key(self):
        self.assertFalse(needs_key("ollama"))
        self.assertTrue(key_for("ollama"))
        self.assertIn("localhost", base_url_for("ollama"))

    def test_cloud_providers_have_urls(self):
        for name in ("groq", "gemini", "openrouter", "deepseek"):
            with self.subTest(provider=name):
                self.assertTrue(base_url_for(name).startswith("https://"))

    def test_full_url_can_be_used_as_provider(self):
        url = "http://192.168.1.5:1234/v1"
        self.assertEqual(base_url_for(url), url)

    def test_unknown_provider_without_url_fails(self):
        with self.assertRaises(ValueError):
            base_url_for("нет-такого-провайдера")

    def test_key_is_taken_from_environment(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            self.assertEqual(key_for("groq"), "test-key")

    def test_resolve_llm_switches_provider_keeping_parameters(self):
        base = LLMConfig(provider="ollama", model="local",
                         temperature=0.5, top_p=0.8, max_tokens=100)
        llm = resolve_llm(base, provider="groq", model="llama-3.3-70b-versatile")
        self.assertEqual(llm.provider, "groq")
        self.assertEqual(llm.model, "llama-3.3-70b-versatile")
        self.assertIn("groq.com", llm.base_url)
        self.assertEqual(llm.temperature, 0.5)
        self.assertEqual(llm.max_tokens, 100)

    def test_catalog_marks_local_provider_configured(self):
        catalog = {p["name"]: p for p in provider_catalog()}
        self.assertIn("ollama", catalog)
        self.assertTrue(catalog["ollama"]["configured"])
        self.assertTrue(catalog["ollama"]["models"])


if __name__ == "__main__":
    unittest.main()
