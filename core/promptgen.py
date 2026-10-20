#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Промпт для агента: свой у каждого раздела знаний.

Агент по даташитам разъёмов и агент по кулинарным книгам не могут отвечать по
одному и тому же тексту: одному нужны «распиновка, допуски, температура пайки»,
другому — «ингредиенты и порядок приготовления». Поэтому системный промпт
хранится в `agents/<slug>/agent.json` и составляется при создании агента:

1. пользователь указывает название и тему раздела;
2. модель пишет промпт по заданию `prompt.compose_template` из `config.yaml`;
3. готовый промпт виден в интерфейсе (⚙ → Агенты) и его можно поправить.

Если модель недоступна (не выбран провайдер, нет ключа, сервер offline),
агент всё равно создаётся — с промптом по шаблону
`prompt.agent_default_system`, и его можно отредактировать вручную.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Больше этого промпт агента не сохраняем: модель иногда уходит в рассуждения.
MAX_PROMPT_CHARS = 1500
# Хватает на 4–7 строк промпта, но не даёт модели разогнаться.
COMPOSE_MAX_TOKENS = 700

# Обёртки, которые модель любит добавлять вокруг готового промпта:
# ```markdown … ``` и подпись «Системный промпт:».
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$")
_PREFIXES = ("системный промпт:", "system prompt:", "промпт:", "prompt:")


def compose_messages(prompt_cfg, title: str, description: str) -> List[Dict[str, str]]:
    """Задание модели: составить системный промпт для такого агента."""
    topic = (description or "").strip() or "не указана — определи по названию"
    question = prompt_cfg.compose_template.format(
        title=(title or "").strip(), topic=topic)
    return [{"role": "user", "content": question}]


def fallback_prompt(prompt_cfg, title: str, description: str) -> str:
    """Промпт по шаблону: когда модель недоступна, агент всё равно работает."""
    topic = (description or "").strip() or (title or "").strip()
    return prompt_cfg.agent_default_system.format(topic=topic).strip()


def clean(text: str) -> str:
    """Убирает markdown-обёртки и подписи, оставляя только промпт."""
    out = (text or "").strip()
    for _ in range(3):          # ``` с двух сторон — снимаем по шагу
        stripped = _FENCE_RE.sub("", out).strip()
        if stripped == out:
            break
        out = stripped
    low = out.lower()
    for prefix in _PREFIXES:
        if low.startswith(prefix):
            out = out[len(prefix):].strip()
            break
    return out[:MAX_PROMPT_CHARS].strip()


async def compose(prompt_cfg, client, title: str, description: str) -> str:
    """Просит модель написать промпт; пустой ответ заменяется шаблоном."""
    answer = await client.chat(compose_messages(prompt_cfg, title, description),
                               max_tokens=COMPOSE_MAX_TOKENS)
    return clean(answer) or fallback_prompt(prompt_cfg, title, description)
