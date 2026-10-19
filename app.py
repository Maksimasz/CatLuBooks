#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Веб-оболочка RAG-ассистента: FastAPI, чат (PWA) и админка агентов.

Запуск:
    python app.py
Затем откройте http://localhost:8000 (с телефона — http://<IP-ПК>:8000).
Агенты, книги и модель — кнопка ⚙ в правом верхнем углу.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config, load_config, resolve_llm, save_setting
from core import ocr, parsers
from core.agents import Agent, AgentError, AgentNotFound, Agents
from core.jobs import Jobs
from core.llm import LLMError, OpenAICompatClient
from core.models import forget_models, provider_models
from core.providers import (
    PROVIDERS,
    key_for,
    needs_key,
    provider_catalog,
    save_secret,
)
from core.store import StoreRegistry, VectorStore

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

WEB_DIR = Path(__file__).resolve().parent / "web"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LAST_REQUEST = LOG_DIR / "last_request.json"
UPLOAD_CHUNK = 1 << 20  # загружаем книги по мегабайту, а не целиком в память

CONFIG: Config = load_config()
AGENTS = Agents(CONFIG.store.agents_root)
REGISTRY = StoreRegistry(CONFIG, AGENTS)
JOBS = Jobs()


def save_last_request(record: dict) -> None:
    """Записывает разбор последнего запроса: вопрос, источники и текст контекста.

    Нужен, чтобы разбираться с ответами: видно, что именно ушло в модель.
    """
    try:
        LOG_DIR.mkdir(exist_ok=True)
        LAST_REQUEST.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"не удалось записать {LAST_REQUEST}: {exc}")


def sse(event: str, data) -> str:
    """Формирует одно SSE-событие."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def human_size(size: int) -> str:
    """«1536» -> «1,5 КБ»: размеры в интерфейсе читаются глазом."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def http_error(exc: AgentError) -> HTTPException:
    """Ошибки агентов: «нет такого» — 404, остальное — просьба исправить ввод."""
    if isinstance(exc, AgentNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def agent_store(slug: Optional[str]) -> Tuple[Agent, VectorStore]:
    """Агент и его база: slug из запроса, иначе первый по алфавиту.

    Нет агентов или нет такого агента — это ошибка запроса (404), а не сбой
    сервера: интерфейс показывает её текстом.
    """
    if not slug:
        found = AGENTS.list()
        if not found:
            raise HTTPException(
                status_code=404,
                detail="Агентов пока нет. Нажмите ⚙ в правом верхнем углу и создайте первого.")
        slug = found[0].slug
    try:
        agent = AGENTS.get(slug)
        return agent, REGISTRY.get(agent.slug)
    except AgentError as exc:
        raise http_error(exc) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clients = {}  # (provider, model) -> OpenAICompatClient
    known = AGENTS.list()
    if known:
        print(f"Агентов: {len(known)} ({', '.join(agent.title for agent in known)})")
        await asyncio.to_thread(REGISTRY.ensure_model)
        agent, store = agent_store(known[0].slug)
        stats = store.stats()
        print(f"«{agent.title}»: {stats['total_chunks']} фрагментов, "
              f"{stats['sources_count']} книг")
    else:
        print("Агентов пока нет — создайте первого в настройках (⚙)")
    engine = await asyncio.to_thread(ocr.status)
    if engine["available"]:
        languages = ", ".join(engine["languages"]) or "языки не найдены"
        state = "включён" if CONFIG.store.ocr != parsers.OCR_OFF else "выключен"
        print(f"🔎 OCR: tesseract {engine['version']} ({languages}) — {state}")
    else:
        print("🔎 OCR: tesseract не найден — сканы распознаваться не будут")
    print(f"Модель по умолчанию: {CONFIG.llm.model} ({CONFIG.llm.provider})")
    print(f"Открой http://localhost:{CONFIG.server.port}")
    try:
        yield
    finally:
        for client in app.state.clients.values():
            await client.aclose()
        REGISTRY.close()


app = FastAPI(title="RAG Ассистент", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    agent: str | None = None


class SearchRequest(BaseModel):
    query: str
    n_results: int | None = None
    agent: str | None = None


class KeyRequest(BaseModel):
    provider: str
    key: str = ""


class AgentRequest(BaseModel):
    title: str
    description: str = ""


class OcrRequest(BaseModel):
    enabled: bool = True


def get_client(app: FastAPI, provider: str, model: str) -> OpenAICompatClient:
    """Клиент для пары (провайдер, модель), создаётся один раз и кэшируется."""
    cache_key = (provider, model)
    client = app.state.clients.get(cache_key)
    if client is None:
        client = OpenAICompatClient(resolve_llm(CONFIG.llm, provider, model))
        app.state.clients[cache_key] = client
    return client


def build_messages(req: ChatRequest, context: str) -> list[dict]:
    """Собирает сообщения для модели: system + история + текущий вопрос."""
    prompt = CONFIG.prompt
    template = prompt.answer_template if context else prompt.no_context_template
    user_content = template.format(context=context, question=req.message)

    messages = [{"role": "system", "content": prompt.system}]
    history, size = [], 0
    for msg in reversed(req.history[-CONFIG.search.history_messages:]):
        if msg.role not in ("user", "assistant"):
            continue
        size += len(msg.content)
        if size > CONFIG.search.history_chars:
            break
        history.append({"role": msg.role, "content": msg.content})
    messages.extend(reversed(history))
    messages.append({"role": "user", "content": user_content})
    return messages


async def stream_chat(req: ChatRequest, app: FastAPI):
    """Генератор SSE-событий: проверка модели → поиск → источники → ответ."""
    provider = req.provider or CONFIG.llm.provider
    model = req.model or CONFIG.llm.model

    if needs_key(provider) and not key_for(provider):
        yield sse("error", {"message": f"Для «{provider}» нужен API-ключ. "
                                      f"Введите его в настройках (⚙ → Модели)."})
        yield sse("done", {})
        return

    try:
        agent, store = agent_store(req.agent)
    except HTTPException as exc:
        yield sse("error", {"message": exc.detail})
        yield sse("done", {})
        return

    try:
        llm = get_client(app, provider, model)
    except Exception as exc:
        yield sse("error", {"message": f"Не удалось настроить «{provider}»: {exc}"})
        yield sse("done", {})
        return

    try:
        results = await asyncio.to_thread(store.search, req.message, None)
        context = store.format_context(results)
    except Exception as exc:
        yield sse("error", {"message": f"Ошибка поиска: {exc}"})
        yield sse("done", {})
        return

    yield sse("model", {"provider": provider, "model": model,
                        "agent": agent.title, "slug": agent.slug})

    found = store.sources(results) if results else []
    print(f"❓ [{agent.slug}] {req.message}  ({provider}/{model}, "
          f"фрагментов {len(results)}, контекст {len(context)} символов)", flush=True)
    record = {
        "время": time.strftime("%Y-%m-%d %H:%M:%S"),
        "агент": agent.title,
        "провайдер": provider,
        "модель": model,
        "вопрос": req.message,
        "история": [{"role": msg.role, "content": msg.content} for msg in req.history],
        "контекст_символов": len(context),
        "источники": found,
        "контекст": context,
    }
    save_last_request(record)

    if results:
        yield sse("sources", found)
    elif CONFIG.search.require_context:
        yield sse("token", {"text": "В базе этого агента не найдено фрагментов "
                                    "для ответа на этот вопрос."})
        yield sse("done", {})
        return

    answer: list[str] = []
    try:
        async for piece in llm.stream(build_messages(req, context)):
            answer.append(piece)
            yield sse("token", {"text": piece})
    except LLMError as exc:
        yield sse("error", {"message": str(exc)})
    record["ответ"] = "".join(answer)
    save_last_request(record)
    yield sse("done", {})


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    return StreamingResponse(
        stream_chat(req, request.app),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/search")
async def search(req: SearchRequest):
    _, store = agent_store(req.agent)
    results = await asyncio.to_thread(store.search, req.query, req.n_results)
    return {"results": store.sources(results)}


@app.get("/api/stats")
async def stats(agent: str | None = None):
    found, store = agent_store(agent)
    data = await asyncio.to_thread(store.stats)
    data["agent"] = found.as_dict()
    data["books"] = len(AGENTS.books(found.slug))
    return data


@app.get("/health")
async def health():
    """Ответ «сервер жив».

    Адрес спрашивают сторонние инструменты (например, обозреватель, из
    которого запускают сервер) — лучше ответить, чем оставлять 404 в журнале.
    """
    return {"ok": True}


@app.get("/api/config")
async def config():
    return {
        "provider": CONFIG.llm.provider,
        "model": CONFIG.llm.model,
        "agents_root": CONFIG.store.agents_root,
        "embedding_model": CONFIG.embeddings.model,
    }


@app.get("/api/providers")
async def providers():
    return {
        "providers": provider_catalog(),
        "default": {"provider": CONFIG.llm.provider, "model": CONFIG.llm.model},
    }


@app.get("/api/ocr")
async def ocr_status():
    """Проверка tesseract: установлен ли, версия, языки, включён ли OCR."""
    data = await asyncio.to_thread(ocr.status)
    data["enabled"] = CONFIG.store.ocr != parsers.OCR_OFF
    return data


@app.post("/api/ocr")
async def set_ocr(req: OcrRequest):
    """Включает или выключает распознавание сканов — сразу и навсегда."""
    mode = parsers.OCR_ON if req.enabled else parsers.OCR_OFF
    data = await asyncio.to_thread(ocr.status)
    if req.enabled and not data["available"]:
        raise HTTPException(
            status_code=409,
            detail="Tesseract не найден. Установите его и нажмите «Проверить снова».")
    save_setting("store", "ocr", mode)   # переживёт перезапуск сервера
    CONFIG.store.ocr = mode
    REGISTRY.set_ocr(mode)               # уже открытые базы тоже
    data["enabled"] = mode != parsers.OCR_OFF
    return data


@app.post("/api/keys")
async def set_key(req: KeyRequest):
    if req.provider not in {item["name"] for item in provider_catalog()}:
        raise HTTPException(status_code=400, detail="Неизвестный провайдер")
    save_secret(req.provider, req.key.strip())
    forget_models(req.provider)  # ключ изменился — список моделей обновим
    return {"ok": True, "configured": bool(key_for(req.provider))}


@app.get("/api/models")
async def models(provider: str | None = None, all: int = 0):
    """Список моделей: ?provider=groq для одного, ?all=1 — сразу для всех."""
    if all:
        results = await asyncio.gather(
            *(provider_models(name, CONFIG) for name in PROVIDERS))
        return {"providers": list(results)}
    name = provider or CONFIG.llm.provider
    if name not in PROVIDERS:
        return {"provider": name, "label": name, "models": [], "source": "fallback",
                "configured": True, "error": "нет реестра моделей"}
    return await provider_models(name, CONFIG)


# --- Агенты: своя папка, свои книги, своя база ------------------------------

@app.get("/api/agents")
async def list_agents():
    """Агенты для интерфейса: книги, фрагменты, размер папки."""
    out: list[Dict[str, Any]] = []
    for agent in AGENTS.list():
        info = agent.as_dict()
        info["books"] = len(AGENTS.books(agent.slug))
        info["size"] = await asyncio.to_thread(agent.size)
        info["size_text"] = human_size(info["size"])
        try:
            data = await asyncio.to_thread(REGISTRY.get(agent.slug).stats)
            info["chunks"] = data["total_chunks"]
            info["indexed"] = data["sources_count"]
        except Exception as exc:  # база не открылась — покажем причину
            info["chunks"] = 0
            info["indexed"] = 0
            info["error"] = str(exc)
        out.append(info)
    return {"agents": out}


@app.post("/api/agents")
async def create_agent(req: AgentRequest):
    try:
        agent = await asyncio.to_thread(AGENTS.create, req.title, req.description)
    except AgentError as exc:
        raise http_error(exc) from exc
    return agent.as_dict()


@app.patch("/api/agents/{slug}")
async def rename_agent(slug: str, req: AgentRequest):
    try:
        agent = await asyncio.to_thread(AGENTS.rename, slug, req.title)
    except AgentError as exc:
        raise http_error(exc) from exc
    return agent.as_dict()


@app.delete("/api/agents/{slug}")
async def delete_agent(slug: str):
    try:
        AGENTS.get(slug)
        await asyncio.to_thread(REGISTRY.forget, slug)  # закрыть базу агента
        await asyncio.to_thread(AGENTS.delete, slug)
    except AgentError as exc:
        raise http_error(exc) from exc
    return {"ok": True}


@app.get("/api/agents/{slug}/books")
async def list_books(slug: str):
    """Книги агента и сколько фрагментов каждой уже в базе."""
    try:
        agent = AGENTS.get(slug)
        counts = await asyncio.to_thread(REGISTRY.get(slug).source_counts)
    except AgentError as exc:
        raise http_error(exc) from exc
    books = await asyncio.to_thread(AGENTS.books, slug)
    for book in books:
        book["chunks"] = counts.get(book["name"], 0)
        book["size_text"] = human_size(book["size"])

    def inspect(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Проверяет, есть ли в книге текст: скан без текста — повод для OCR."""
        for book in items:
            report = parsers.probe(agent.books_dir / book["name"])
            book["needs_ocr"] = report["needs_ocr"]
        return items

    await asyncio.to_thread(inspect, books)
    return {"slug": slug, "title": agent.title, "books": books,
            "chunks": sum(counts.values()),
            "indexed": len(counts),
            "needs_ocr": sum(1 for book in books if book["needs_ocr"])}


@app.post("/api/agents/{slug}/books")
async def upload_books(slug: str, files: list[UploadFile] = File(...)):
    """Загружает книги в папку агента (файл пишется по частям, не в память)."""
    uploaded: list[Dict[str, Any]] = []
    failed: list[Dict[str, Any]] = []
    for upload in files:
        name = Path(upload.filename or "").name
        try:
            reader = iter(lambda: upload.file.read(UPLOAD_CHUNK), b"")
            info = await asyncio.to_thread(AGENTS.write_book, slug, name, reader)
            uploaded.append(info)
        except (AgentError, OSError) as exc:
            failed.append({"name": name, "error": str(exc)})
        finally:
            await upload.close()
    return {"uploaded": uploaded, "failed": failed}


@app.delete("/api/agents/{slug}/books/{name}")
async def delete_book(slug: str, name: str):
    """Удаляет книгу: сначала фрагменты из базы, потом сам файл."""
    try:
        removed = await asyncio.to_thread(REGISTRY.get(slug).delete_source, name)
        gone = await asyncio.to_thread(AGENTS.remove_book, slug, name)
    except AgentError as exc:
        raise http_error(exc) from exc
    if not gone:
        raise HTTPException(status_code=404, detail=f"Файла «{name}» нет")
    return {"ok": True, "removed": removed}


@app.post("/api/agents/{slug}/index")
async def index_agent(slug: str):
    """Запускает индексацию в фоне: интерфейс следит за ходом через /job."""
    try:
        agent = AGENTS.get(slug)
    except AgentError as exc:
        raise http_error(exc) from exc

    running = JOBS.latest(slug)
    if running is not None and running.state == "running":
        return {"job": running.as_dict()}

    job = JOBS.start(slug)

    def run() -> None:
        try:
            store = REGISTRY.get(slug)
            done = store.index(str(agent.books_dir),
                               progress=lambda event: JOBS.apply(job, event))
            if not done and job.state == "running":
                JOBS.fail(job, "Ни одну книгу не удалось прочитать")
        except Exception as exc:  # база не открылась, файл не прочитался и т.п.
            JOBS.fail(job, str(exc))

    asyncio.create_task(asyncio.to_thread(run))
    return {"job": job.as_dict()}


@app.get("/api/agents/{slug}/job")
async def job_status(slug: str):
    """Ход последней индексации агента (интерфейс спрашивает раз в секунду)."""
    try:
        AGENTS.get(slug)
    except AgentError as exc:
        raise http_error(exc) from exc
    job = JOBS.latest(slug)
    return {"job": job.as_dict() if job else None}


@app.delete("/api/agents/{slug}/base")
async def clear_base(slug: str):
    """Очищает базу агента; книги на диске остаются."""
    try:
        removed = await asyncio.to_thread(REGISTRY.get(slug).clear)
    except AgentError as exc:
        raise http_error(exc) from exc
    return {"ok": True, "removed": removed}


# Статика (index.html, app.js, PWA) — монтируется последней, чтобы
# не перехватывать запросы к /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import uvicorn

    uvicorn.run(app, host=CONFIG.server.host, port=CONFIG.server.port, log_level="info")


if __name__ == "__main__":
    main()
