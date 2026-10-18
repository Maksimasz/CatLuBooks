#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Клиент генеративной модели.

Работает с любым сервером, поддерживающим OpenAI-совместимый
эндпоинт /chat/completions: облачные API (OpenAI, OpenRouter, DeepSeek…),
а также локальные Ollama, LM Studio, llama.cpp, vLLM.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Dict, List

import httpx

from config import LLMConfig


class LLMError(RuntimeError):
    """Понятная пользователю ошибка обращения к модели."""


class OpenAICompatClient:
    """Обёртка над OpenAI-совместимым /chat/completions."""

    def __init__(self, cfg: LLMConfig):
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_tokens = cfg.max_tokens

        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        # base_url без завершающего слэша — httpx склеит корректно.
        base_url = cfg.base_url.rstrip("/") + "/"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    def _payload(self, messages: List[Dict[str, str]], stream: bool) -> Dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Один запрос без стриминга. Возвращает готовый текст ответа."""
        try:
            response = await self._client.post(
                "chat/completions", json=self._payload(messages, False)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Не удалось обратиться к модели «{self.model}»: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("Модель вернула ответ в неизвестном формате") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Неожиданная структура ответа модели: {data}") from exc

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Стриминг ответа: выдаёт кусочки текста по мере генерации."""
        try:
            async with self._client.stream(
                "POST", "chat/completions", json=self._payload(messages, True)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"Модель ответила ошибкой {response.status_code}: {body[:300]}"
                    )
                async for line in response.aiter_lines():
                    piece = self._parse_line(line)
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"Ошибка соединения с моделью: {exc}") from exc

    @staticmethod
    def _parse_line(line: str) -> str:
        """Разбирает одну строку SSE и возвращает кусок текста (или пусто)."""
        if not line or not line.startswith("data:"):
            return ""
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            return ""
        try:
            obj = json.loads(chunk)
        except ValueError:
            return ""
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""

    async def aclose(self) -> None:
        await self._client.aclose()
