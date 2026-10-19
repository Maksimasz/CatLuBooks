#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Хранилище знаний агента.

VectorStore — поиск по базе одного агента: веб-слой не знает, что внутри
ChromaDB. StoreRegistry открывает базы агентов по требованию; модель
эмбеддингов грузится один раз на процесс и переиспользуется всеми агентами.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config import Config
from core import parsers
from core.agents import Agents
from core.database import BOOKS_COLLECTION, RAGDatabase, embedding_model_for


class VectorStore:
    """База одного агента: поиск, контекст для модели, статистика."""

    def __init__(self, cfg: Config, db: RAGDatabase):
        self.cfg = cfg
        self._db = db

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

    def source_counts(self) -> Dict[str, int]:
        """Сколько фрагментов в базе по каждой книге."""
        return self._db.source_counts()

    def index(self, books_folder: str,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None,
              full: bool = False) -> bool:
        """Индексирует книги агента (запускается фоновой задачей).

        full=False — читает только новые и изменившиеся книги: уже готовые
        пропускаются, поэтому добавить одну книгу можно за минуты.
        """
        return self._db.process_research_folder(books_folder, progress=progress,
                                                full=full)

    def delete_source(self, name: str) -> int:
        """Убирает из базы фрагменты книги; файл на диске не трогает."""
        return self._db.delete_source(name)

    def clear(self) -> int:
        """Стирает базу агента: книги на диске остаются."""
        return self._db.clear_base()

    def close(self) -> None:
        self._db.close()


class StoreRegistry:
    """Базы агентов по требованию: одному агенту — одна открытая база."""

    def __init__(self, cfg: Config, agents: Agents):
        self.cfg = cfg
        self.agents = agents
        self._stores: Dict[str, VectorStore] = {}

    def get(self, slug: str) -> VectorStore:
        """Открывает базу агента (и создаёт пустую, если её ещё нет)."""
        store = self._stores.get(slug)
        if store is None:
            agent = self.agents.get(slug)  # AgentError, если агента нет
            db = RAGDatabase(
                db_path=str(agent.db_dir),
                collection_name=BOOKS_COLLECTION,
                embedding_model=self.cfg.embeddings.model,
                chunk_size=self.cfg.store.chunk_size,
                chunk_overlap=self.cfg.store.chunk_overlap,
                formats=self.cfg.store.formats,
                ocr=self.cfg.store.ocr,
            )
            db.lexical = self.cfg.search.lexical
            store = VectorStore(self.cfg, db)
            self._stores[slug] = store
        return store

    def ensure_model(self):
        """Заранее грузит модель эмбеддингов: первый вопрос не будет ждать."""
        return embedding_model_for(self.cfg.embeddings.model)

    def set_ocr(self, ocr) -> None:
        """Переключает распознавание сканов у всех уже открытых баз."""
        ocr = parsers.ocr_mode(ocr)
        self.cfg.store.ocr = ocr
        for store in self._stores.values():
            store._db.set_ocr(ocr)

    def forget(self, slug: str) -> None:
        """Закрывает базу агента — нужно перед удалением его папки."""
        store = self._stores.pop(slug, None)
        if store is not None:
            store.close()

    def close(self) -> None:
        for slug in list(self._stores):
            self.forget(slug)
