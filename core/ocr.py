#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Состояние OCR: установлен ли tesseract и какие языки к нему есть.

Интерфейс спрашивает это перед включением распознавания сканов: без
tesseract настройка бесполезна, и об этом нужно сказать до, а не после.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.parsers import Tesseract

# Официальная страница установщика для Windows (там же языковые пакеты).
INSTALL_URL = "https://github.com/UB-Mannheim/tesseract/wiki"
RUSSIAN = "rus"


def status() -> Dict[str, Any]:
    """Что известно про tesseract: есть ли, какая версия, какие языки."""
    engine = Tesseract()
    available = engine.available
    languages: List[str] = engine.languages if available else []
    return {
        "available": available,
        "path": engine.path,
        "version": engine.version if available else "",
        "languages": languages,
        "has_russian": RUSSIAN in languages,
        "install_url": INSTALL_URL,
    }
