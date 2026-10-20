#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Реестр провайдеров генеративной модели и хранение API-ключей.

Все провайдеры ниже отдают OpenAI-совместимый /chat/completions,
поэтому работают через один и тот же клиент (core.llm).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT / "secrets.json"

# name -> {label, base_url, env, models, needs_key}
# base_url пустой у ollama — подставится адрес локального сервера.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "label": "Ollama (локально, без ключа)",
        "base_url": "http://localhost:11434/v1",
        "env": None,
        "models": [
            "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest",
            "qwen2.5:3b",
            "qwen2.5:latest",
            "gemma2:2b",
            "llama3.2:3b",
        ],
        "needs_key": False,
        # Локальные модели денег не стоят: в списке они идут как бесплатные.
        "free_all": True,
    },
    "llamacpp": {
        "label": "llama.cpp (локально, без ключа)",
        "base_url": "http://127.0.0.1:8080/v1",
        "env": None,
        # Имя модели задаёт сам llama-server (--alias или путь к GGUF), поэтому
        # запасного списка нет: живой список с сервера важнее.
        "models": [],
        "needs_key": False,
        "free_all": True,
    },
    "groq": {
        "label": "Groq (бесплатно, очень быстро)",
        "base_url": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY",
        "models": [
        ],
        "needs_key": True,
        "free_all": True,
        # Модель, которую предлагаем по умолчанию.
        "preferred": "openai/gpt-oss-120b",
    },
    "gemini": {
        "label": "Google Gemini (бесплатный тариф)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": "GEMINI_API_KEY",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ],
        "needs_key": True,
        # Модели flash входят в бесплатный тариф.
        "free_models": ["flash"],
        "preferred": "gemini-2.5-flash",
    },
    "openrouter": {
        "label": "OpenRouter (есть бесплатные модели)",
        "base_url": "https://openrouter.ai/api/v1",
        "env": "OPENROUTER_API_KEY",
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-r1:free",
            "google/gemini-2.0-flash-exp:free",
        ],
        "needs_key": True,
        # Список моделей и цены OpenRouter отдаёт без ключа, поэтому список
        # подтягивается вживую, а бесплатные определяются по цене.
    },
    "deepseek": {
        "label": "DeepSeek (дешёвый API)",
        "base_url": "https://api.deepseek.com/v1",
        "env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat"],
        "needs_key": True,
    },
    "mistral": {
        "label": "Mistral (есть бесплатный тариф)",
        "base_url": "https://api.mistral.ai/v1",
        "env": "MISTRAL_API_KEY",
        "models": ["mistral-small-latest"],
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini"],
        "needs_key": True,
    },
}


def load_secrets() -> Dict[str, str]:
    """Читает сохранённые ключи (secrets.json). Файл в .gitignore."""
    if not SECRETS_PATH.exists():
        return {}
    try:
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_secret(provider: str, key: str) -> None:
    """Сохраняет (или удаляет при пустом ключе) ключ провайдера."""
    secrets = load_secrets()
    if key:
        secrets[provider] = key
    else:
        secrets.pop(provider, None)
    SECRETS_PATH.write_text(
        json.dumps(secrets, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def key_for(provider: str) -> str:
    """Ключ провайдера: сначала переменная окружения, затем secrets.json."""
    info = PROVIDERS.get(provider, {})
    env = info.get("env")
    if env and os.environ.get(env):
        return os.environ[env]
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    saved = load_secrets().get(provider)
    if saved:
        return saved
    if not info.get("needs_key", True):
        # Локальному серверу ключ не нужен, но заголовок требует значения.
        return "local"
    return ""


def base_url_for(provider: str) -> str:
    """Адрес OpenAI-совместимого эндпоинта для провайдера."""
    info = PROVIDERS.get(provider)
    if info and info.get("base_url"):
        return info["base_url"]
    if provider.startswith(("http://", "https://")):
        # Провайдер задан как полный URL: подходит любой совместимый сервер.
        return provider
    raise ValueError(
        f"Неизвестный провайдер «{provider}». Укажите известное имя из реестра "
        f"или полный URL OpenAI-совместимого сервера в поле base_url."
    )


def set_provider_base_url(provider: str, url: str) -> None:
    """Меняет адрес провайдера на лету (порт управляемого llama-server)."""
    if provider in PROVIDERS and url:
        PROVIDERS[provider]["base_url"] = url


def needs_key(provider: str) -> bool:
    return bool(PROVIDERS.get(provider, {}).get("needs_key", True))


def model_is_free(provider: str, ident: str, pricing_free: bool | None = None) -> bool:
    """Бесплатна ли модель.

    pricing_free — сведения о цене от провайдера (OpenRouter), если есть.
    Если цен нет, смотрим суффикс ":free", флаг free_all и подсказки free_models.
    """
    info = PROVIDERS.get(provider, {})
    if info.get("free_all"):
        # Например, у Groq весь API бесплатный (с ограничением по частоте),
        # поэтому цены из ответа не показательны.
        return True
    if pricing_free is not None:
        return bool(pricing_free)
    if ident.endswith(":free"):
        return True
    needles = info.get("free_models") or ()
    low = ident.lower()
    return any(str(needle).lower() in low for needle in needles)


def provider_catalog() -> List[Dict[str, Any]]:
    """Список провайдеров для интерфейса (с признаком настроенного ключа)."""
    catalog = []
    for name, info in PROVIDERS.items():
        catalog.append({
            "name": name,
            "label": info["label"],
            "models": info.get("models", []),
            "needs_key": info.get("needs_key", True),
            "configured": not info.get("needs_key", True) or bool(key_for(name)),
        })
    return catalog
