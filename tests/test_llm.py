"""Проверки клиента модели: таймауты и понятные ошибки."""
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
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "черновик"}}]}
        post = AsyncMock(return_value=response)
        with patch.object(self.client._client, "post", new=post):
            text = await self.client.chat([{"role": "user", "content": "черновик"}],
                                          max_tokens=900)

        self.assertEqual(text, "черновик")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 900)


if __name__ == "__main__":
    unittest.main()
