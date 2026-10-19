#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Фоновые задачи для интерфейса: сейчас — индексация книг агента.

Индексация идёт минуты, поэтому кнопка «Индексировать» не держит запрос:
она запускает задачу, а интерфейс спрашивает её состояние и показывает
ход работы. Состояние живёт в памяти процесса — его достаточно, чтобы
пережить перезагрузку страницы, но не перезапуск сервера.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOG_LINES = 12  # сколько последних событий показывать в интерфейсе
KEEP_JOBS = 30  # сколько завершённых задач держать в памяти


@dataclass
class Job:
    """Одна задача индексации агента."""

    slug: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "running"  # running | done | error
    total: int = 0          # сколько книг читать
    index: int = 0          # какая по счёту сейчас читается
    file: str = ""          # имя текущей книги
    chunks: int = 0         # сколько фрагментов уже записано
    failed: int = 0
    skipped: int = 0        # книги, убранные из папки во время работы
    ready: int = 0          # книги, которые уже были в базе и не изменились
    added: int = 0          # сколько книг прочитано за этот запуск
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    log: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "state": self.state,
            "total": self.total,
            "index": self.index,
            "file": self.file,
            "chunks": self.chunks,
            "failed": self.failed,
            "skipped": self.skipped,
            "ready": self.ready,
            "added": self.added,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "log": self.log[-LOG_LINES:],
        }

    def _log(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-LOG_LINES]


class Jobs:
    """Задачи индексации: начать, показать, закончить."""

    def __init__(self, keep: int = KEEP_JOBS):
        self.keep = keep
        self._jobs: Dict[str, Job] = {}

    def start(self, slug: str) -> Job:
        """Начинает задачу для агента (предыдущие завершённые забываются)."""
        job = Job(slug=slug)
        self._jobs[job.id] = job
        self._trim()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def latest(self, slug: str) -> Optional[Job]:
        """Последняя задача агента — чтобы страница показала ход работ."""
        found = [job for job in self._jobs.values() if job.slug == slug]
        return max(found, key=lambda job: job.started) if found else None

    def apply(self, job: Job, event: Dict[str, Any]) -> None:
        """Обновляет задачу событием от RAGDatabase (см. _report)."""
        kind = event.get("event")
        if kind == "files":
            job.total = int(event.get("total") or 0)
            ready = int(event.get("ready") or 0)
            line = f"Файлов к чтению: {job.total}"
            if ready:
                line += f" (уже в базе: {ready})"
            job._log(line)
        elif kind == "ready":
            job.ready = int(event.get("total") or 0)
            job._log(f"Уже в базе, читать заново не нужно: {job.ready}")
        elif kind == "file":
            job.index = int(event.get("index") or 0)
            job.file = str(event.get("source") or "")
            # Счётчик фрагментов — про текущую книгу: у новой книги он с нуля,
            # иначе интерфейс показывал бы ей число прошлой книги.
            job.chunks = 0
            job._log(f"[{job.index}/{job.total}] {job.file}")
        elif kind == "chunks":
            job.chunks = int(event.get("saved") or 0)
        elif kind == "file_done":
            job._log(f"{event.get('source')}: {event.get('chunks')} фрагментов")
        elif kind == "failed":
            job.failed += 1
            job._log(f"ошибка: {event.get('source')} — {event.get('message')}")
        elif kind == "skipped":
            # Файл убрали из папки, пока шла индексация: это не ошибка.
            job.skipped += 1
            job._log(f"пропущено (файла больше нет): {event.get('source')}")
        elif kind == "done":
            job.chunks = int(event.get("chunks") or job.chunks)
            job.failed = int(event.get("failed") or job.failed)
            job.skipped = int(event.get("skipped") or job.skipped)
            job.ready = int(event.get("ready") or job.ready)
            job.added = int(event.get("added") or job.added)
            job.state = "done"
            job.finished = time.time()
            line = f"Готово: в базе {job.chunks} фрагментов, прочитано {job.added}"
            if job.ready:
                line += f", уже было {job.ready}"
            job._log(line)
        elif kind == "error":
            job.state = "error"
            job.error = str(event.get("message") or "не получилось")
            job.finished = time.time()
            job._log(f"Ошибка: {job.error}")

    def fail(self, job: Job, message: str) -> None:
        """Отмечает задачу упавшей (исключение вне обработки файлов)."""
        job.state = "error"
        job.error = message
        job.finished = time.time()
        job._log(f"Ошибка: {message}")

    def _trim(self) -> None:
        """Оставляет только свежие задачи: память не растёт бесконечно."""
        if len(self._jobs) <= self.keep:
            return
        order = sorted(self._jobs.values(), key=lambda job: job.started)
        for job in order[:len(self._jobs) - self.keep]:
            if job.state != "running":
                self._jobs.pop(job.id, None)
