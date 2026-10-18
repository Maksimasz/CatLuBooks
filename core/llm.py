#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Клиент генеративной модели.

Работает с любым сервером, поддерживающим OpenAI-совместимый
эндпоинт /chat/completions: облачные API (OpenAI, OpenRouter, DeepSeek…),
а также локальные Ollama, LM Studio, llama.cpp, vLLM.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

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


async def fetch_models(cfg: LLMConfig) -> List[Dict[str, Any]]:
    """Спрашивает у провайдера список доступных моделей (/models).

    Список моделей у облачных провайдеров меняется, поэтому его нельзя
    жёстко прописывать — берём актуальный у самого сервиса. Для моделей,
    у которых провайдер отдаёт цены (например OpenRouter), отмечаем бесплатные.
    """
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    base_url = cfg.base_url.rstrip("/") + "/"
    try:
        async with httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=httpx.Timeout(15.0, connect=8.0)
        ) as client:
            response = await client.get("models")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise LLMError(f"Не удалось получить список моделей: {exc}") from exc
    except ValueError as exc:
        raise LLMError("Провайдер вернул список моделей в неизвестном формате") from exc

    items = data.get("data") if isinstance(data, dict) else data
    models: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            if item:
                models.append({"id": str(item), "free": None, "chat": True})
            continue
        ident = item.get("id")
        if not ident:
            continue
        models.append({
            "id": str(ident),
            "free": _is_free_pricing(item.get("pricing")),
            "chat": _is_chat_model(item),
        })
    return models


def _is_chat_model(item: Dict[str, Any]) -> bool:
    """Годится ли модель для чата (не картинки/речь/расшифровка звука)."""
    arch = item.get("architecture")
    arch = arch if isinstance(arch, dict) else {}
    ident = str(item.get("id", "")).lower()
    if "embed" in ident:
        return False  # модель эмбеддингов, для ответов не подходит
    outputs = item.get("output_modalities") or arch.get("output_modalities")
    if isinstance(outputs, list) and outputs:
        if "text" not in outputs:
            return False
        if any(kind in ("audio", "image", "video") for kind in outputs):
            return False
    for key in ("context_length", "context_window", "context"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return value >= 1024  # совсем короткий контекст для RAG бесполезен
    return True


def _is_free_pricing(pricing) -> bool | None:
    """Бесплатна ли модель по данным о ценах (None — неизвестно)."""
    if not isinstance(pricing, dict):
        return None
    prompt = pricing.get("prompt")
    completion = pricing.get("completion")
    if prompt is None or completion is None:
        return None
    try:
        return float(prompt) == 0 and float(completion) == 0
    except (TypeError, ValueError):
        return None
