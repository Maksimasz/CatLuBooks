#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Удаление данных агентов из консоли (то же самое есть в интерфейсе, ⚙).

Из папки проекта:

    python tools/drop.py --list                    # агенты, книги, фрагменты
    python tools/drop.py cooking                   # что внутри одного агента
    python tools/drop.py cooking "Плов.pdf"        # убрать книгу целиком
    python tools/drop.py cooking --clear           # очистить базу агента
    python tools/drop.py cooking --delete --yes    # удалить агента целиком

Обычно это не нужно: агенты, книги и базу можно удалить в интерфейсе
(⚙ → Агенты, Книги, База). Инструмент выручает, если сервер не запускается.
Перед запуском остановите сервер: Windows держит файлы базы открытыми.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Файл лежит в tools/, а config и core — в корне проекта.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, load_config
from core.agents import Agent, AgentError, Agents
from core.database import BOOKS_COLLECTION, RAGDatabase


def counts_by_source(collection: Any) -> Dict[str, int]:
    """Сколько фрагментов в базе по каждому файлу."""
    counts: Dict[str, int] = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        name = (meta or {}).get("source", "неизвестно")
        counts[name] = counts.get(name, 0) + 1
    return counts


def drop_sources(collection: Any, names: Iterable[str]) -> Dict[str, int]:
    """Убирает все фрагменты указанных файлов; возвращает, сколько убрано."""
    removed: Dict[str, int] = {}
    for name in names:
        found = collection.get(where={"source": name}, include=["metadatas"])
        removed[name] = len(found.get("metadatas") or [])
        if removed[name]:
            collection.delete(where={"source": name})
    return removed


def drop_all(client: Any, collection_name: str) -> int:
    """Стирает базу целиком и создаёт пустую; возвращает, сколько было."""
    from core.database import COLLECTION_META  # как у остальных коллекций

    collection = client.get_or_create_collection(collection_name)
    before = collection.count()
    client.delete_collection(collection_name)
    client.create_collection(name=collection_name, metadata=COLLECTION_META)
    return before


def open_agent_db(cfg: Config, agent: Agent) -> RAGDatabase:
    """Открывает базу агента: для удаления модель эмбеддингов не нужна."""
    db = RAGDatabase(
        db_path=str(agent.db_dir),
        collection_name=BOOKS_COLLECTION,
        embedding_model=cfg.embeddings.model,
        chunk_size=cfg.store.chunk_size,
        chunk_overlap=cfg.store.chunk_overlap,
        formats=cfg.store.formats,
        ocr="off",  # распознавание при удалении не нужно
    )
    db.lexical = cfg.search.lexical
    return db


def show_agent(cfg: Config, agents: Agents, agent: Agent) -> None:
    """Печатает книги агента и число фрагментов каждой."""
    books = agents.books(agent.slug)
    counts: Dict[str, int] = {}
    db: Optional[RAGDatabase] = None
    if books and agent.has_base():
        db = open_agent_db(cfg, agent)
        try:
            counts = counts_by_source(db.collection)
        finally:
            db.close()
    print(f"\nАгент «{agent.title}» (папка {agent.slug})")
    if not books:
        print("  книг нет — загрузите их в ⚙ → Книги")
        return
    for book in books:
        number = counts.get(book["name"], 0)
        mark = "проиндексирована" if number else "не в базе"
        print(f"  {number:>6}  {book['name']}  ({mark})")


def main(argv: List[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    cfg = load_config()
    agents = Agents(cfg.store.agents_root)

    if args[0] == "--list":
        known = agents.list()
        if not known:
            print(f"В {cfg.store.agents_root} нет ни одного агента — "
                  f"создайте его в интерфейсе (⚙ → Агенты)")
            return 0
        for agent in known:
            show_agent(cfg, agents, agent)
        print()
        return 0

    slug = args[0]
    rest = args[1:]
    names = [name for name in rest if not name.startswith("--")]
    try:
        agent = agents.get(slug)
    except AgentError as exc:
        print(f"❌ {exc}. Список агентов: python tools/drop.py --list")
        return 1

    if "--delete" in rest:
        if "--yes" not in rest:
            answer = input(f"Удалить агента «{agent.title}» вместе с книгами и базой? "
                           f"Отменить будет нельзя [да/нет]: ")
            if answer.strip().lower() not in ("да", "д", "yes", "y"):
                print("Отменено")
                return 1
        agents.delete(agent.slug)
        print(f"Агент «{agent.title}» удалён вместе с папкой {agent.slug}")
        return 0

    if not names:
        if "--clear" not in rest:
            show_agent(cfg, agents, agent)
            return 0

    if not agent.has_base():
        if "--clear" in rest:
            print(f"База агента «{agent.title}» и так пуста")
            return 0
        print(f"❌ В базе агента «{agent.title}» нет данных, "
              f"индексировать пока нечего")
        return 1

    db = open_agent_db(cfg, agent)
    try:
        if "--clear" in rest:
            before = drop_all(db.client, db.collection_name)
            print(f"База агента «{agent.title}» очищена: было {before} фрагментов. "
                  f"Книги на диске остались.")
            return 0

        for name, number in drop_sources(db.collection, names).items():
            if number:
                path = agent.books_dir / name
                if path.is_file():
                    path.unlink()
                print(f"Убрано фрагментов: {number:>6}  {name} (файл удалён)")
            else:
                print(f"В базе нет файла «{name}» — точное имя покажет "
                      f"python tools/drop.py {agent.slug}")
        print(f"Осталось в базе: {db.collection.count()}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
