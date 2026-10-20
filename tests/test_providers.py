"""Проверки реестра провайдеров и смены модели."""
import os
import unittest
from unittest.mock import patch

from config import LLMConfig, resolve_llm
from core.llm import _is_chat_model, _is_free_pricing
from core.providers import base_url_for, key_for, model_is_free, needs_key, provider_catalog


class ProviderTests(unittest.TestCase):
    def test_ollama_does_not_need_key(self):
        self.assertFalse(needs_key("ollama"))
        self.assertTrue(key_for("ollama"))
        self.assertIn("localhost", base_url_for("ollama"))

    def test_llamacpp_is_local_and_needs_no_key(self):
        # llama-server из llama.cpp: тот же OpenAI-совместимый API, порт 8080.
        self.assertFalse(needs_key("llamacpp"))
        self.assertEqual(key_for("llamacpp"), "local")
        self.assertEqual(base_url_for("llamacpp"), "http://localhost:8080/v1")
        self.assertTrue(model_is_free("llamacpp", "qwen2.5-3b"))

    def test_catalog_lists_llamacpp_as_configured(self):
        catalog = {p["name"]: p for p in provider_catalog()}

        self.assertIn("llamacpp", catalog)
        self.assertTrue(catalog["llamacpp"]["configured"])
        self.assertFalse(catalog["llamacpp"]["needs_key"])

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
                         temperature=0.5, top_p=0.8, max_tokens=100,
                         max_continuations=1, prompt_timeout=240)
        llm = resolve_llm(base, provider="groq", model="llama-3.3-70b-versatile")
        self.assertEqual(llm.provider, "groq")
        self.assertEqual(llm.model, "llama-3.3-70b-versatile")
        self.assertIn("groq.com", llm.base_url)
        self.assertEqual(llm.temperature, 0.5)
        self.assertEqual(llm.max_tokens, 100)
        # Таймауты и продолжение не сбрасываются при смене модели на лету.
        self.assertEqual(llm.max_continuations, 1)
        self.assertEqual(llm.prompt_timeout, 240)

    def test_catalog_marks_local_provider_configured(self):
        catalog = {p["name"]: p for p in provider_catalog()}
        self.assertIn("ollama", catalog)
        self.assertTrue(catalog["ollama"]["configured"])
        self.assertTrue(catalog["ollama"]["models"])


class FreeModelTests(unittest.TestCase):
    def test_free_by_zero_pricing(self):
        self.assertTrue(_is_free_pricing({"prompt": "0", "completion": "0"}))
        self.assertFalse(_is_free_pricing({"prompt": "0.000001", "completion": "0"}))
        self.assertIsNone(_is_free_pricing(None))
        self.assertIsNone(_is_free_pricing({"prompt": "0"}))

    def test_free_all_beats_pricing(self):
        # У Groq весь API бесплатный, поэтому цены из ответа не показательны.
        self.assertTrue(model_is_free("groq", "openai/gpt-oss-120b", False))

    def test_local_models_count_as_free(self):
        # Локальные модели (Ollama, llama.cpp) денег не стоят.
        self.assertTrue(model_is_free("ollama", "qwen2.5:3b"))
        self.assertTrue(model_is_free("llamacpp", "qwen2.5-3b"))

    def test_free_suffix_and_flash_hint(self):
        self.assertTrue(model_is_free("openrouter", "meta-llama/llama-3.3-70b:free"))
        self.assertFalse(model_is_free("openrouter", "openai/gpt-4o", False))
        self.assertTrue(model_is_free("gemini", "gemini-2.5-flash"))
        self.assertFalse(model_is_free("gemini", "gemini-2.5-pro"))

    def test_chat_model_filters_speech_and_embeddings(self):
        self.assertFalse(_is_chat_model({"id": "whisper-large-v3",
                                         "output_modalities": ["transcription"]}))
        self.assertFalse(_is_chat_model({"id": "google/lyria",
                                         "output_modalities": ["text", "audio"]}))
        self.assertFalse(_is_chat_model({"id": "nomic-embed-text:latest"}))
        self.assertFalse(_is_chat_model({"id": "guard", "context_length": 512}))
        self.assertTrue(_is_chat_model({"id": "openai/gpt-oss-120b", "context_length": 131072,
                                        "output_modalities": ["text"]}))
        self.assertTrue(_is_chat_model({"id": "yandex/llm:latest"}))


if __name__ == "__main__":
    unittest.main()
