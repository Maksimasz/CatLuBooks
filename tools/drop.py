#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Удаляет данные из векторной базы: одну книгу или всю базу сразу.

Из папки проекта:

    python tools/drop.py --list                  # что сейчас в базе
    python tools/drop.py "Плов.pdf"              # убрать книгу целиком
    python tools/drop.py "Плов.pdf" "Гриль.pdf"  # несколько книг сразу
    python tools/drop.py --all                   # стереть всю базу

Перед запуском остановите сервер (он держит базу открытой), а сам файл уберите
из папки `research/`: иначе следующая пересборка (`python -m core.database`)
вернёт книгу в базу.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Файл лежит в tools/, а config и core — в корне проекта.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from config import load_config


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


def main(argv: List[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    cfg = load_config()
    client = chromadb.PersistentClient(path=cfg.store.db_path)

    if args[0] == "--list":
        counts = counts_by_source(client.get_or_create_collection(cfg.store.collection))
        if not counts:
            print("База пуста")
            return 0
        for name, number in sorted(counts.items(), key=lambda item: -item[1]):
            print(f"{number:>6}  {name}")
        print(f"{sum(counts.values()):>6}  всего фрагментов в {len(counts)} файлах")
        return 0

    if args[0] == "--all":
        if "--yes" not in args:
            answer = input("Стереть всю базу? Отменить будет нельзя [да/нет]: ")
            if answer.strip().lower() not in ("да", "д", "yes", "y"):
                print("Отменено")
                return 1
        before = drop_all(client, cfg.store.collection)
        print(f"База очищена: было {before} фрагментов. "
              f"Собрать заново: python -m core.database")
        return 0

    names = [name for name in args if not name.startswith("--")]
    collection = client.get_or_create_collection(cfg.store.collection)
    for name, number in drop_sources(collection, names).items():
        if number:
            print(f"Убрано фрагментов: {number:>6}  {name}")
        else:
            print(f"В базе нет файла «{name}» — точное имя покажет "
                  f"python tools/drop.py --list")
    print(f"Осталось в базе: {collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
