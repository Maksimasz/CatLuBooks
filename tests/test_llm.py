"""Проверки клиента модели: таймауты и понятные ошибки."""
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from config import LLMConfig
from core.llm import LLMError, OpenAICompatClient


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = LLMConfig(provider="ollama", model="тест",
                             base_url="http://localhost:11434/v1")
        self.client = OpenAICompatClient(self.cfg)

    async def tearDown(self):
        await self.client.aclose()

    def test_timeout_is_taken_from_config(self):
        self.assertEqual(self.client._client.timeout.read, 300.0)

    async def test_timeout_can_be_overridden(self):
        """Для сведения нужно больше времени, чем для обычного ответа."""
        patient = OpenAICompatClient(self.cfg, timeout=900)
        try:
            self.assertEqual(patient._client.timeout.read, 900.0)
        finally:
            await patient.aclose()

    async def test_timeout_error_explains_what_to_do(self):
        """Просто «не удалось обратиться» бесполезно — нужно сказать, что делать."""
        with patch.object(self.client._client, "post",
                          new=AsyncMock(side_effect=httpx.ReadTimeout(""))):
            with self.assertRaises(LLMError) as caught:
                await self.client.chat([{"role": "user", "content": "привет"}])
        message = str(caught.exception)
        self.assertIn("не ответила", message)
        self.assertIn("llm.timeout", message)

    async def test_network_error_keeps_details(self):
        with patch.object(self.client._client, "post",
                          new=AsyncMock(side_effect=httpx.ConnectError("порт закрыт"))):
            with self.assertRaises(LLMError) as caught:
                await self.client.chat([{"role": "user", "content": "привет"}])
        self.assertIn("порт закрыт", str(caught.exception))

    async def test_draft_can_get_its_own_token_limit(self):
        """Черновику слияния хватает меньше токенов, чем полному ответу."""
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "черновик"}}]}
        post = AsyncMock(return_value=response)
        with patch.object(self.client._client, "post", new=post):
            text = await self.client.chat([{"role": "user", "content": "черновик"}],
                                          max_tokens=900)

        self.assertEqual(text, "черновик")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 900)

    def test_stream_chunk_reports_text_and_finish_reason(self):
        """Причина остановки нужна, чтобы заметить обрыв по лимиту токенов."""
        piece, reason = OpenAICompatClient._parse_chunk(
            'data: {"choices":[{"delta":{"content":"рецепт"},"finish_reason":null}]}')
        self.assertEqual((piece, reason), ("рецепт", None))

        piece, reason = OpenAICompatClient._parse_chunk(
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}')
        self.assertEqual((piece, reason), ("", "length"))

        self.assertEqual(OpenAICompatClient._parse_chunk("data: [DONE]"), ("", None))

    async def test_chat_remembers_why_the_model_stopped(self):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "готово"}, "finish_reason": "stop"}]}
        post = AsyncMock(return_value=response)
        with patch.object(self.client._client, "post", new=post):
            await self.client.chat([{"role": "user", "content": "x"}])

        self.assertEqual(self.client.finish_reason, "stop")

    def test_rate_limit_is_explained_in_words(self):
        """429 бесплатного тарифа: показываем текст провайдера и когда повторять."""
        body = json.dumps({"error": {
            "message": "Rate limit reached on tokens per minute (TPM): Limit 8000"}})

        text = OpenAICompatClient._error_text(429, body, "25.4")

        self.assertIn("tokens per minute", text)
        self.assertIn("25.4", text)
        self.assertNotIn("{", text)

    def test_error_without_json_keeps_the_body(self):
        text = OpenAICompatClient._error_text(502, "Bad gateway")

        self.assertIn("502", text)
        self.assertIn("Bad gateway", text)

    async def test_chat_error_reaches_the_user(self):
        """Ошибка провайдера — это LLMError с понятным текстом, а не сбой."""
        response = Mock()
        response.status_code = 429
        response.headers = {"retry-after": "12"}
        response.text = json.dumps({"error": {"message": "Rate limit reached"}})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)):
            with self.assertRaises(LLMError) as caught:
                await self.client.chat([{"role": "user", "content": "x"}])

        self.assertIn("Rate limit reached", str(caught.exception))
        self.assertIn("12", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
