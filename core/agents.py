#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Агенты: у каждого своя база знаний в своей папке.

Агент — это папка с латинским именем:

    agents/dokumenty/
        agent.json     — название, описание и системный промпт агента
        books/         — книги агента (PDF, EPUB, TXT, FB2)
        db/            — векторная база (ChromaDB) этого агента

Папка самодостаточна: скопировал папку — перенёс агента вместе с базой,
удалил папку — удалил агента. Ничего вне своей папки агент не занимает,
поэтому агенты не мешают друг другу и не смешивают знания.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core import parsers

# Русские буквы в латиницу: имя папки должно быть латиницей.
TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Имя папки агента: латиница, цифры, дефис и подчёркивание.
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
AGENT_FILE = "agent.json"
BOOKS_DIR = "books"
DB_DIR = "db"


class AgentError(ValueError):
    """Понятная пользователю ошибка работы с агентами."""


class AgentNotFound(AgentError):
    """Агента с таким именем нет."""


@dataclass
class Agent:
    slug: str
    title: str
    path: Path
    description: str = ""
    created: str = ""
    # Свой системный промпт агента: у раздела знаний своя тема, общий
    # промпт подошёл бы только одному агенту из всех.
    system_prompt: str = ""

    @property
    def books_dir(self) -> Path:
        return self.path / BOOKS_DIR

    @property
    def db_dir(self) -> Path:
        return self.path / DB_DIR

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "prompt": self.system_prompt,
            "created": self.created,
            "folder": self.path.name,
        }

    def has_base(self) -> bool:
        """Появлялись ли в папке базы файлы (агент уже индексировался)."""
        try:
            return any(self.db_dir.iterdir())
        except OSError:
            return False

    def size(self) -> int:
        """Сколько занимает папка агента: книги и база вместе."""
        total = 0
        for path in self.path.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total


class Agents:
    """Папки агентов на диске: создать, переименовать, удалить, показать."""

    def __init__(self, root: str | Path = "./agents"):
        self.root = Path(root).expanduser()

    # --- чтение ---------------------------------------------------------
    def list(self) -> List[Agent]:
        """Все агенты, по алфавиту названий."""
        if not self.root.is_dir():
            return []
        found = [self._load(path) for path in sorted(self.root.iterdir())
                 if path.is_dir() and SLUG_RE.match(path.name)]
        return sorted(found, key=lambda agent: agent.title.lower())

    def get(self, slug: str) -> Agent:
        path = self._path(slug)
        if not path.is_dir():
            raise AgentNotFound(f"Агента «{slug}» нет")
        return self._load(path)

    def exists(self, slug: str) -> bool:
        try:
            return self._path(slug).is_dir()
        except AgentError:
            return False

    # --- изменение ------------------------------------------------------
    def create(self, title: str, description: str = "") -> Agent:
        """Создаёт папку агента: имя папки — латиницей, из названия."""
        title = (title or "").strip()
        if not re.search(r"[A-Za-zА-Яа-яЁё0-9]", title):
            raise AgentError("Укажите название агента — хотя бы одну букву или цифру")
        slug = self._free_slug(title)
        path = self.root / slug
        (path / BOOKS_DIR).mkdir(parents=True, exist_ok=True)
        (path / DB_DIR).mkdir(exist_ok=True)
        agent = Agent(
            slug=slug,
            title=title,
            path=path,
            description=(description or "").strip(),
            created=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        self._save(agent)
        return agent

    def rename(self, slug: str, title: str) -> Agent:
        """Меняет название для интерфейса; имя папки остаётся прежним."""
        agent = self.get(slug)
        title = (title or "").strip()
        if not title:
            raise AgentError("Название не может быть пустым")
        agent.title = title
        self._save(agent)
        return agent

    def save_prompt(self, slug: str, system_prompt: str,
                    description: str | None = None) -> Agent:
        """Записывает системный промпт агента (пустой — вернуться к общему)."""
        agent = self.get(slug)
        agent.system_prompt = (system_prompt or "").strip()
        if description is not None:
            agent.description = (description or "").strip()
        self._save(agent)
        return agent

    def delete(self, slug: str) -> None:
        """Удаляет папку агента вместе с книгами и базой."""
        agent = self.get(slug)
        last: OSError | None = None
        for _ in range(3):
            try:
                shutil.rmtree(agent.path)
                return
            except OSError as exc:  # Windows может держать файлы базы
                last = exc
                time.sleep(0.3)
        raise AgentError(f"Не удалось удалить папку «{agent.path}»: {last}")

    # --- книги ----------------------------------------------------------
    def books(self, slug: str) -> List[Dict[str, Any]]:
        """Файлы в папке книг агента (и в подпапках): имя, размер, формат.

        Имя — путь внутри папки книг («инструкции/руководство.pdf»): так книга из
        подпапки отличается от одноимённой в корне и её видно в интерфейсе —
        иначе она попадала бы в базу, не показываясь в списке.
        """
        agent = self.get(slug)
        if not agent.books_dir.is_dir():
            return []
        books = []
        for path in sorted(agent.books_dir.rglob("*")):
            if not path.is_file() or path.name.endswith(".part"):
                continue
            books.append({
                "name": parsers.relative_name(path, agent.books_dir),
                "size": path.stat().st_size,
                "supported": path.suffix.lower() in parsers.SUPPORTED,
            })
        return books

    def book_path(self, slug: str, name: str) -> Path:
        """Путь к книге внутри папки агента: чужие пути не принимаются.

        Путь внутри папки разрешён («инструкции/руководство.pdf»), выход за её пределы
        и скрытые папки — нет.
        """
        agent = self.get(slug)
        raw = str(name or "").strip().replace("\\", "/")
        if raw.startswith("/") or Path(raw).is_absolute() or ":" in raw:
            raise AgentError(f"Недопустимое имя файла: {name!r}")
        parts = [part for part in raw.split("/") if part]
        if not parts or any(part in (".", "..") or part.startswith(".") for part in parts):
            raise AgentError(f"Недопустимое имя файла: {name!r}")
        return agent.books_dir.joinpath(*parts)

    def write_book(self, slug: str, name: str,
                   chunks: Iterable[bytes]) -> Dict[str, Any]:
        """Сохраняет загруженную книгу: сначала во временный файл, потом замена."""
        path = self.book_path(slug, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".part")
        replaced = path.exists()
        size = 0
        try:
            with temp.open("wb") as out:
                for chunk in chunks:
                    out.write(chunk)
                    size += len(chunk)
        except OSError:
            temp.unlink(missing_ok=True)
            raise
        if size == 0:
            temp.unlink(missing_ok=True)
            raise AgentError(f"Файл «{path.name}» пустой")
        temp.replace(path)
        return {
            "name": path.name,
            "size": size,
            "replaced": replaced,
            "supported": path.suffix.lower() in parsers.SUPPORTED,
        }

    def remove_book(self, slug: str, name: str) -> bool:
        """Удаляет файл книги; фрагменты из базы убирает веб-слой."""
        path = self.book_path(slug, name)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # --- внутреннее -----------------------------------------------------
    @staticmethod
    def slugify(title: str) -> str:
        """«Документы» -> «dokumenty»; всё неподходящее становится дефисом."""
        out = []
        for char in title.strip().lower():
            if char in TRANSLIT:
                out.append(TRANSLIT[char])
            elif char.isascii() and char.isalnum():
                out.append(char)
            else:
                out.append("-")
        return re.sub(r"-{2,}", "-", "".join(out)).strip("-")[:32].strip("-")

    def _free_slug(self, title: str) -> str:
        base = self.slugify(title) or "agent"
        if not SLUG_RE.match(base):
            base = "agent"
        slug = base
        number = 2
        while (self.root / slug).exists():
            slug = f"{base}-{number}"[:32]
            number += 1
        return slug

    def _path(self, slug: str) -> Path:
        if not SLUG_RE.match(str(slug or "")):
            raise AgentError(f"Недопустимое имя агента: {slug!r}")
        return self.root / slug

    def _load(self, path: Path) -> Agent:
        data: Dict[str, Any] = {}
        try:
            loaded = json.loads((path / AGENT_FILE).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            pass  # файла нет или он испорчен — названием станет имя папки
        return Agent(
            slug=path.name,
            title=str(data.get("title") or path.name),
            path=path,
            description=str(data.get("description") or ""),
            created=str(data.get("created") or ""),
            system_prompt=str(data.get("prompt") or ""),
        )

    @staticmethod
    def _save(agent: Agent) -> None:
        agent.path.mkdir(parents=True, exist_ok=True)
        (agent.path / AGENT_FILE).write_text(
            json.dumps({"title": agent.title, "description": agent.description,
                        "prompt": agent.system_prompt, "created": agent.created},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        agent.books_dir.mkdir(exist_ok=True)
        agent.db_dir.mkdir(exist_ok=True)
