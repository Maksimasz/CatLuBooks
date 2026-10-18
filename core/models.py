#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Список моделей для интерфейса: что показать в поле выбора модели.

Живой список запрашивается у самого провайдера (у него всегда актуальный
набор), а если сервис его не отдаёт без ключа или недоступен — берём запасной
из реестра (`core.providers`). Список показывается в шапке интерфейса, поэтому
бесплатные модели идут первыми, а служебные (речь, картинки, эмбеддинги)
отсеиваются: отвечать ими нельзя.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from config import Config, resolve_llm
from core.llm import fetch_models
from core.providers import PROVIDERS, key_for, model_is_free

MODELS_TTL = 300.0  # секунд; список моделей меняется нечасто

# провайдер -> (время получения, готовый ответ для интерфейса)
_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}


def forget_models(provider: str | None = None) -> None:
    """Сбрасывает кэш: изменился ключ или нужен свежий список."""
    if provider is None:
        _cache.clear()
    else:
        _cache.pop(provider, None)


async def provider_models(name: str, cfg: Config,
                          fresh: bool = False) -> Dict[str, Any]:
    """Модели провайдера: живой список от сервиса, иначе статический.

    Бесплатные модели помечаются и идут первыми. Без ключа большинство
    облачных сервисов список не отдаёт — тогда берём запасной из реестра.
    """
    if not fresh:
        cached = _cache.get(name)
        if cached and time.monotonic() - cached[0] < MODELS_TTL:
            return cached[1]

    info = PROVIDERS.get(name, {})
    live: List = []
    error = None
    if name in PROVIDERS:
        try:
            live = await fetch_models(resolve_llm(cfg.llm, name, None))
        except Exception as exc:  # нет ключа, нет сети и т.п.
            error = str(exc)

    models = []
    for item in (live or info.get("models", [])):
        if isinstance(item, dict):
            if item.get("chat") is False:
                continue  # речь, расшифровка звука, картинки — не для чата
            ident, pricing_free = item.get("id"), item.get("free")
        else:
            ident, pricing_free = item, None
        if not ident:
            continue
        ident = str(ident)
        models.append({"id": ident, "free": model_is_free(name, ident, pricing_free)})
    models.sort(key=lambda m: (not m["free"], m["id"].lower()))

    result = {
        "provider": name,
        "label": info.get("label", name),
        "models": models,
        "preferred": info.get("preferred", ""),
        "source": "provider" if live else "fallback",
        "configured": not info.get("needs_key", True) or bool(key_for(name)),
        "error": error,
    }
    _cache[name] = (time.monotonic(), result)
    return result
