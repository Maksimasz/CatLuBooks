"""Проверки сборки списка моделей для интерфейса (сеть подменяется)."""
import unittest
from unittest.mock import AsyncMock, patch

from config import Config, LLMConfig
from core import models
from core.llm import LLMError


class ProviderModelsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Кэш живёт между вызовами — начинаем каждый тест с чистого.
        models.forget_models()

    async def test_live_list_drops_models_that_cannot_answer(self):
        live = [
            {"id": "whisper-large-v3", "chat": False},           # речь, не для чата
            {"id": "z-модель", "chat": True, "free": None},
            {"id": "openai/gpt-oss-120b", "chat": True, "free": None},
        ]
        with patch.object(models, "fetch_models", AsyncMock(return_value=live)):
            result = await models.provider_models("groq", Config())

        self.assertEqual(result["source"], "provider")
        self.assertEqual([m["id"] for m in result["models"]],
                         ["openai/gpt-oss-120b", "z-модель"])
        # У Groq весь API бесплатный, поэтому бесплатны все оставшиеся.
        self.assertTrue(all(m["free"] for m in result["models"]))
        self.assertIsNone(result["error"])

    async def test_paid_models_are_after_free_ones(self):
        live = [
            {"id": "openai/gpt-4o", "chat": True, "free": False},
            {"id": "meta/llama-3:free", "chat": True, "free": True},
        ]
        with patch.object(models, "fetch_models", AsyncMock(return_value=live)):
            result = await models.provider_models("openrouter", Config())

        self.assertEqual([m["id"] for m in result["models"]],
                         ["meta/llama-3:free", "openai/gpt-4o"])

    async def test_failed_request_falls_back_to_registry(self):
        with patch.object(models, "fetch_models",
                          AsyncMock(side_effect=LLMError("нет ключа"))):
            result = await models.provider_models("gemini", Config())

        self.assertEqual(result["source"], "fallback")
        self.assertIn("gemini-2.5-flash", [m["id"] for m in result["models"]])
        self.assertEqual(result["error"], "нет ключа")

    async def test_list_is_cached_until_forgotten(self):
        fetch = AsyncMock(return_value=[{"id": "модель", "chat": True}])
        with patch.object(models, "fetch_models", fetch):
            await models.provider_models("deepseek", Config())
            await models.provider_models("deepseek", Config())
            self.assertEqual(fetch.await_count, 1)

            await models.provider_models("deepseek", Config(), fresh=True)
            self.assertEqual(fetch.await_count, 2)

            await models.provider_models("deepseek", Config())
            self.assertEqual(fetch.await_count, 2)
            models.forget_models("deepseek")
            await models.provider_models("deepseek", Config())
            self.assertEqual(fetch.await_count, 3)

    async def test_unknown_provider_is_not_requested(self):
        with patch.object(models, "fetch_models", AsyncMock()) as fetch:
            result = await models.provider_models("http://192.168.1.5:1234/v1", Config())

        fetch.assert_not_called()
        self.assertEqual(result["models"], [])
        self.assertEqual(result["source"], "fallback")

    async def test_parameters_come_from_config(self):
        """Адрес и ключ берутся из конфига запроса, а не из реестра."""
        cfg = Config(llm=LLMConfig(provider="ollama", model="local"))
        with patch.object(models, "fetch_models", AsyncMock(return_value=[])) as fetch:
            await models.provider_models("groq", cfg)

        sent = fetch.await_args.args[0]
        self.assertEqual(sent.provider, "groq")
        self.assertIn("groq.com", sent.base_url)


if __name__ == "__main__":
    unittest.main()
