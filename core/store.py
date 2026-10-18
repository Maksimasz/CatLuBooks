#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Обёртка над векторным хранилищем.

Изолирует приложение от конкретной реализации (сейчас — ChromaDB
через RAGDatabase). Чтобы сменить базу или движок поиска, достаточно
переписать этот класс, не трогая веб-слой.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import Config
from core.database import RAGDatabase


class VectorStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._db = RAGDatabase(
            db_path=cfg.store.db_path,
            collection_name=cfg.store.collection,
            embedding_model=cfg.embeddings.model,
            chunk_size=cfg.store.chunk_size,
            chunk_overlap=cfg.store.chunk_overlap,
            formats=cfg.store.formats,
            ocr=cfg.store.ocr,
        )
        self._db.lexical = cfg.search.lexical

    def search(self, query: str, n_results: int | None = None) -> List[Dict[str, Any]]:
        """Поиск с расширением контекста соседними фрагментами."""
        n = n_results or self.cfg.search.n_results
        return self._db.expand_context(self._db.search_similar(query, n_results=n))

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Превращает найденные фрагменты в текст для промпта.

        Фрагменты добавляются целиком: разрезанный пополам рецепт модели
        бесполезен. Если в лимит не влезает — фрагмент пропускается; частично
        показывается только самый важный (первый), чтобы ответ не остался
        совсем без данных.
        """
        budget = self.cfg.search.max_context_chars
        parts: List[str] = []
        used = 0
        for i, result in enumerate(results, 1):
            meta = result.get("metadata") or {}
            source = meta.get("source", "Неизвестный источник")
            page = meta.get("page")
            if page is not None:
                source += f", {meta.get('unit') or 'страница'} {page}"
            piece = f"Источник {i} ({source}):\n{result.get('document', '')}"
            gap = 2 if parts else 0
            room = budget - used - gap
            if room <= 0:
                break
            if len(piece) > room:
                if parts:
                    continue  # не режем рецепт посередине — попробуем следующий
                piece = piece[:room]
            parts.append(piece)
            used += len(piece) + gap
        return "\n\n".join(parts)

    def sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Краткое описание источников (без дублей) для интерфейса."""
        out: List[Dict[str, Any]] = []
        seen = set()
        for i, result in enumerate(results, 1):
            meta = result.get("metadata") or {}
            source = meta.get("source", "Неизвестно")
            page = meta.get("page")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "index": i,
                "source": source,
                "page": page,
                "unit": meta.get("unit") or "стр.",
                "snippet": (result.get("document") or "")[:200],
            })
        return out

    def stats(self) -> Dict[str, Any]:
        return self._db.get_database_stats()

    def index(self, folder: str | None = None) -> bool:
        """Пересобирает базу из PDF-файлов папки."""
        return self._db.process_research_folder(folder or self.cfg.store.research_folder)
