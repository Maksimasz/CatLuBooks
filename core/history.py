"""История поиска агента: сохранённые ответы, чтобы не искать снова.

Хранится в папке агента (`history.json`), поэтому переносится вместе с
базой и книгами и исчезает вместе с агентом. Файл небольшой: ответы
обрезаются до разумной длины, а старые записи вытесняются новыми.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Сколько записей держим: больше — файл растёт, меньше — теряются старые.
MAX_ENTRIES = 200
# Сколько символов ответа сохраняем: длинные ответы режем, чтобы файл не
# раздувался; полный ответ всё равно виден в чате в момент получения.
MAX_ANSWER_CHARS = 12000


def _path(folder: str) -> Path:
    return Path(folder) / "history.json"


def load(folder: str) -> List[Dict[str, Any]]:
    """Читает историю агента; при битом файле возвращает пустой список."""
    path = _path(folder)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _save(folder: str, entries: List[Dict[str, Any]]) -> None:
    """Атомарная запись: сначала во временный файл, потом переименование."""
    path = _path(folder)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def add(folder: str, *, question: str, answer: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        provider: str = "", model: str = "") -> Dict[str, Any]:
    """Добавляет запись в начало истории и возвращает её."""
    entry: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "question": (question or "").strip(),
        "answer": (answer or "")[:MAX_ANSWER_CHARS],
        "sources": list(sources or []),
        "provider": provider or "",
        "model": model or "",
        "created": time.time(),
    }
    entries = load(folder)
    entries.insert(0, entry)
    if len(entries) > MAX_ENTRIES:
        entries = entries[:MAX_ENTRIES]
    _save(folder, entries)
    return entry


def delete(folder: str, entry_id: str) -> bool:
    """Удаляет одну запись; возвращает True, если она была."""
    entries = load(folder)
    new = [entry for entry in entries if entry.get("id") != entry_id]
    if len(new) == len(entries):
        return False
    _save(folder, new)
    return True


def clear(folder: str) -> int:
    """Очищает историю целиком; возвращает, сколько записей было."""
    count = len(load(folder))
    _save(folder, [])
    return count