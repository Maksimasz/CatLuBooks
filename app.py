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
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config, load_config, resolve_llm, save_setting
from core import ocr, parsers, promptgen
from core.agents import Agent, AgentError, AgentNotFound, Agents
from core.database import UNKNOWN_SOURCE
from core.history import add as history_add, clear as history_clear
from core.history import delete as history_delete, load as history_load
from core.jobs import Jobs
from core.llm import LLMError, OpenAICompatClient
from core.models import forget_models, provider_models
from core.server import ServerControl, SUCCESSOR_ENV, wait_until_free
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

# Просьба дописать ответ, если модель упёрлась в лимит токенов.
CONTINUE_PROMPT = (
    "Продолжи ответ ровно с того места, где он оборвался. "
    "Не повторяй уже написанное и не начинай заново."
)

CONFIG: Config = load_config()
AGENTS = Agents(CONFIG.store.agents_root)
REGISTRY = StoreRegistry(CONFIG, AGENTS)
JOBS = Jobs()
SERVER = ServerControl(CONFIG.server.host, CONFIG.server.port)


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


def mark_scans(agent: Agent, books: list[Dict[str, Any]]) -> None:
    """Помечает сканы без текстового слоя: им поможет OCR."""
    for book in books:
        book["needs_ocr"] = (False if book.get("missing")
                             else parsers.probe(agent.books_dir / book["name"])["needs_ocr"])


def agent_store(slug: Optional[str]) -> Tuple[Agent, VectorStore]:
    """Агент и его база: slug из запроса, иначе первый по алфавиту.

    Нет агентов или нет такого агента — это ошибка запроса (404), а не сбой
    сервера: интерфейс показывает её текстом.
    """
    if not slug:
        found = AGENTS.list()
        if not found:
            raise AgentNotFound(
                "Агентов пока нет. Нажмите ⚙ в правом верхнем углу и создайте первого.")
        slug = found[0].slug
    agent = AGENTS.get(slug)
    return agent, REGISTRY.get(agent.slug)


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


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Ошибки агентов — понятным текстом: «нет такого» это 404, остальное — 400."""
    status = 404 if isinstance(exc, AgentNotFound) else 400
    return JSONResponse(status_code=status, content={"detail": str(exc)})


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
    provider: str | None = None
    model: str | None = None
    generate_prompt: bool = True


class PromptRequest(BaseModel):
    prompt: str = ""
    description: str | None = None
    provider: str | None = None
    model: str | None = None


class OcrRequest(BaseModel):
    enabled: bool = True


class MergeRequest(BaseModel):
    mode: str = "off"


def get_client(app: FastAPI, provider: str, model: str,
               timeout: int | None = None) -> OpenAICompatClient:
    """Клиент для пары (провайдер, модель), создаётся один раз и кэшируется."""
    cache_key = (provider, model, timeout)
    client = app.state.clients.get(cache_key)
    if client is None:
        client = OpenAICompatClient(resolve_llm(CONFIG.llm, provider, model),
                                    timeout=timeout)
        app.state.clients[cache_key] = client
    return client


def search_query(req: ChatRequest) -> str:
    """Текст для поиска: короткое уточнение ищется вместе с прошлым вопросом.

    «Почему рецепт не до конца» само по себе не содержит темы — без прошлого
    вопроса поиск уходит в случайные книги и отвечает «такого рецепта нет».
    """
    question = req.message.strip()
    low = question.lower()
    cues = ("продолж", "подробн", "уточн", "не до конца", "не доконца",
            "ещё", "еще", "почему", "зачем")
    if not (low.startswith(("почему", "зачем")) or any(cue in low for cue in cues)):
        return question
    previous = next((msg.content.strip() for msg in reversed(req.history)
                     if msg.role == "user" and msg.content.strip()), "")
    return f"{previous} {question}".strip() if previous else question


def agent_system(agent) -> str:
    """Системный промпт агента; если своего нет — общий из config.yaml."""
    own = getattr(agent, "system_prompt", "") or ""
    if isinstance(own, str) and own.strip():
        return own.strip()
    return CONFIG.prompt.system


def build_messages(req: ChatRequest, context: str,
                   template: str | None = None,
                   system: str | None = None) -> list[dict]:
    """Собирает сообщения для модели: system + история + текущий вопрос."""
    prompt = CONFIG.prompt
    if template is None:
        template = prompt.answer_template if context else prompt.no_context_template
    user_content = template.format(context=context, question=req.message)

    messages = [{"role": "system", "content": system or prompt.system}]
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
    except AgentError as exc:
        yield sse("error", {"message": str(exc)})
        yield sse("done", {})
        return

    merge_mode = CONFIG.search.merge_mode
    merging = merge_mode in ("fast", "full")
    system = agent_system(agent)   # у каждого агента свой промпт

    try:
        # При сведении контекст большой, и первый токен приходит позже:
        # модели нужно время, чтобы переварить найденное.
        llm = get_client(app, provider, model,
                        CONFIG.llm.merge_timeout if merging else None)
    except Exception as exc:
        yield sse("error", {"message": f"Не удалось настроить «{provider}»: {exc}"})
        yield sse("done", {})
        return


    try:
        query = search_query(req)
        if merging:
            # Разные книги: один рецепт в десяти источниках — десять
            # вариантов, из которых можно собрать один хороший.
            results = await asyncio.to_thread(
                store.search_diverse, query, CONFIG.search.merge_sources)
            context = store.format_context(results, CONFIG.search.merge_context_chars)
        else:
            results = await asyncio.to_thread(store.search, query, None)
            context = store.format_context(results)
    except Exception as exc:
        yield sse("error", {"message": f"Ошибка поиска: {exc}"})
        yield sse("done", {})
        return

    yield sse("model", {"provider": provider, "model": model,
                        "agent": agent.title, "slug": agent.slug})

    found = store.sources(results) if results else []
    print(f"❓ [{agent.slug}] {req.message}  ({provider}/{model}, "
          f"фрагментов {len(results)}, контекст {len(context)} символов"
          f"{f', сведение: {merge_mode}' if merging else ''})", flush=True)
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

    # Полное сведение: сначала черновик из всех найденных вариантов,
    # потом ответ по нему. Модель видит целиком то, что нашлось, и не
    # выбрасывает половину рецепта, чтобы уложиться в один проход.
    draft = ""
    if merge_mode == "full" and context:
        try:
            yield sse("stage", {"text": "Свожу найденное в один ответ…"})
            # Черновик идёт без стрима и с большим контекстом — времени и
            # таймаута ему нужно больше, чем обычному ответу. Свой лимит
            # длины: черновику хватает компактного рецепта, а на CPU длинная
            # генерация идёт слишком долго.
            draft = await llm.chat(
                build_messages(req, context, CONFIG.prompt.synthesis_template,
                               system),
                max_tokens=CONFIG.llm.merge_max_tokens)
        except LLMError as exc:
            print(f"⚠️ Модель: {exc}")
            yield sse("error", {"message": str(exc)})
            yield sse("done", {})
            return
        if draft.strip():
            record["черновик"] = draft
            context = draft

    template = None
    if merging:
        template = (CONFIG.prompt.answer_from_draft_template if draft
                    else CONFIG.prompt.merge_template)

    answer: list[str] = []
    messages = build_messages(req, context, template, system)
    try:
        async for piece in llm.stream(messages):
            answer.append(piece)
            yield sse("token", {"text": piece})
        # Модель упёрлась в лимит токенов — просим продолжение: полный ответ
        # по-русски длиннее одного ответа, а обрыв на полуслове бесполезен.
        for _ in range(CONFIG.llm.max_continuations):
            if llm.finish_reason != "length":
                break
            yield sse("stage", {"text": "Ответ длинный — дописываю продолжение…"})
            messages = build_messages(req, context, template, system) + [
                {"role": "assistant", "content": "".join(answer)},
                {"role": "user", "content": CONTINUE_PROMPT},
            ]
            async for piece in llm.stream(messages):
                answer.append(piece)
                yield sse("token", {"text": piece})
    except LLMError as exc:
        print(f"⚠️ Модель: {exc}")
        record["ошибка"] = str(exc)
        yield sse("error", {"message": str(exc)})
    record["ответ"] = "".join(answer)
    record["обрыв"] = llm.finish_reason == "length"
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


@app.get("/api/merge")
async def merge_status():
    """Режим сведения источников: выключен, быстрый или полный."""
    return {"mode": CONFIG.search.merge_mode,
            "sources": CONFIG.search.merge_sources,
            "options": ["off", "fast", "full"]}


@app.post("/api/merge")
async def set_merge(req: MergeRequest):
    """Включает или выключает сведение нескольких источников в один ответ."""
    mode = (req.mode or "").strip().lower()
    if mode not in ("off", "fast", "full"):
        raise HTTPException(status_code=400,
                            detail="Режим сведения: off, fast или full")
    save_setting("search", "merge_mode", mode)   # переживёт перезапуск сервера
    CONFIG.search.merge_mode = mode
    return {"mode": mode, "sources": CONFIG.search.merge_sources}


@app.post("/api/keys")
async def set_key(req: KeyRequest):
    if req.provider not in {item["name"] for item in provider_catalog()}:
        raise HTTPException(status_code=400, detail="Неизвестный провайдер")
    save_secret(req.provider, req.key.strip())
    forget_models(req.provider)  # ключ изменился — список моделей обновим
    return {"ok": True, "configured": bool(key_for(req.provider))}


@app.get("/api/models")
async def models(provider: str | None = None, all: int = 0, fresh: int = 0):
    """Список моделей: ?provider=groq для одного, ?all=1 — сразу для всех.

    ?fresh=1 — не брать кэш: список нужен прямо сейчас (например, только что
    запустили локальный сервер модели).
    """
    if all:
        results = await asyncio.gather(
            *(provider_models(name, CONFIG, bool(fresh)) for name in PROVIDERS))
        return {"providers": list(results)}
    name = provider or CONFIG.llm.provider
    if name not in PROVIDERS:
        return {"provider": name, "label": name, "models": [], "source": "fallback",
                "configured": True, "error": "нет реестра моделей"}
    return await provider_models(name, CONFIG, bool(fresh))


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
async def create_agent(req: AgentRequest, request: Request):
    """Создаёт агента и сразу составляет ему промпт под указанную тему."""
    agent = await asyncio.to_thread(AGENTS.create, req.title, req.description)
    info = agent.as_dict()
    if not req.generate_prompt:
        return info
    prompt, error = await compose_prompt_for(request.app, agent.title,
                                             agent.description,
                                             req.provider, req.model)
    agent = await asyncio.to_thread(AGENTS.save_prompt, agent.slug, prompt)
    info = agent.as_dict()
    if error:
        info["prompt_error"] = error
    return info


async def compose_prompt_for(app: FastAPI, title: str, description: str,
                             provider: str | None,
                             model: str | None) -> tuple[str, str]:
    """Промпт агента от модели. Возвращает промпт и причину замены шаблоном.

    Недоступная модель — не повод не создать агента: тогда промпт берётся из
    шаблона, а в интерфейсе видно, почему, и его можно поправить руками.
    """
    provider = provider or CONFIG.llm.provider
    model = model or CONFIG.llm.model
    fallback = promptgen.fallback_prompt(CONFIG.prompt, title, description)
    if needs_key(provider) and not key_for(provider):
        return fallback, (f"Промпт составлен по шаблону: для «{provider}» нужен "
                          f"API-ключ (⚙ → Модели).")
    try:
        llm = get_client(app, provider, model, CONFIG.llm.prompt_timeout)
        written = await promptgen.compose(CONFIG.prompt, llm, title, description)
    except Exception as exc:   # сеть, ключ, формат ответа — агент важнее
        return fallback, f"Промпт составлен по шаблону: модель не ответила ({exc})"
    return written or fallback, ""


@app.put("/api/agents/{slug}/prompt")
async def save_agent_prompt(slug: str, req: PromptRequest):
    """Сохраняет промпт, отредактированный в интерфейсе."""
    agent = await asyncio.to_thread(AGENTS.save_prompt, slug, req.prompt,
                                    req.description)
    return agent.as_dict()


@app.post("/api/agents/{slug}/prompt")
async def compose_agent_prompt(slug: str, req: PromptRequest, request: Request):
    """Составляет промпт агента заново: по новой теме или текущей."""
    agent = await asyncio.to_thread(AGENTS.get, slug)
    description = agent.description if req.description is None else req.description
    prompt, error = await compose_prompt_for(request.app, agent.title, description,
                                             req.provider, req.model)
    agent = await asyncio.to_thread(AGENTS.save_prompt, slug, prompt, description)
    info = agent.as_dict()
    if error:
        info["prompt_error"] = error
    return info


@app.patch("/api/agents/{slug}")
async def rename_agent(slug: str, req: AgentRequest):
    agent = await asyncio.to_thread(AGENTS.rename, slug, req.title)
    return agent.as_dict()


@app.delete("/api/agents/{slug}")
async def delete_agent(slug: str):
    AGENTS.get(slug)
    await asyncio.to_thread(REGISTRY.forget, slug)  # закрыть базу агента
    await asyncio.to_thread(AGENTS.delete, slug)
    return {"ok": True}


@app.get("/api/agents/{slug}/books")
async def list_books(slug: str):
    """Книги агента и сколько фрагментов каждой уже в базе."""
    agent = AGENTS.get(slug)
    counts = await asyncio.to_thread(REGISTRY.get(slug).source_counts)
    books = await asyncio.to_thread(AGENTS.books, slug)
    for book in books:
        book["chunks"] = counts.get(book["name"], 0)
        book["size_text"] = human_size(book["size"])

    # Фрагменты книг, которых в папке больше нет (файл убрали мимо интерфейса):
    # показываем их отдельными строками — иначе они молча занимают место и
    # отвечают на вопросы, хотя книги уже нет.
    present = {book["name"] for book in books}
    for name in sorted(counts):
        if name in present or name == UNKNOWN_SOURCE:
            continue
        books.append({"name": name, "size": 0, "size_text": "",
                      "supported": True, "chunks": counts[name], "missing": True})

    await asyncio.to_thread(mark_scans, agent, books)
    return {"slug": slug, "title": agent.title, "books": books,
            "chunks": sum(counts.values()),
            "indexed": len(counts),
            "missing": sum(1 for book in books if book.get("missing")),
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


@app.delete("/api/agents/{slug}/book")
async def delete_book(slug: str, name: str = Query(min_length=1), mode: str = Query("full")):
    """Удаляет книгу: сначала фрагменты из базы, потом сам файл.

    Имя приходит строкой запроса, а не частью адреса: так удаляется и книга
    из подпапки, и фрагменты книги, файл которой уже убрали руками.
    """
    if mode not in ("file", "full"):
        raise HTTPException(status_code=400,
                            detail="Режим удаления: file (только файл) или full (файл и база)")
    if mode == "file":
        gone = await asyncio.to_thread(AGENTS.remove_book, slug, name)
        if not gone:
            raise HTTPException(status_code=404, detail=f"Файла «{name}» нет")
        return {"ok": True, "removed": 0, "file": True, "mode": "file"}
    removed = await asyncio.to_thread(REGISTRY.get(slug).delete_source, name)
    gone = await asyncio.to_thread(AGENTS.remove_book, slug, name)
    if not gone and not removed:
        raise HTTPException(status_code=404, detail=f"Книги «{name}» нет ни в базе, ни в папке")
    return {"ok": True, "removed": removed, "file": gone, "mode": "full"}


@app.post("/api/agents/{slug}/index")
async def index_agent(slug: str, full: bool = False):
    """Запускает индексацию в фоне: интерфейс следит за ходом через /job.

    Уже прочитанные книги пропускаются; full=true перечитывает всё заново.
    """
    agent = AGENTS.get(slug)

    running = JOBS.latest(slug)
    if running is not None and running.state == "running":
        return {"job": running.as_dict()}

    job = JOBS.start(slug)

    def run() -> None:
        try:
            store = REGISTRY.get(slug)
            done = store.index(str(agent.books_dir),
                               progress=lambda event: JOBS.apply(job, event),
                               full=full)
            if not done and job.state == "running":
                JOBS.fail(job, "Ни одну книгу не удалось прочитать")
        except Exception as exc:  # база не открылась, файл не прочитался и т.п.
            JOBS.fail(job, str(exc))

    asyncio.create_task(asyncio.to_thread(run))
    return {"job": job.as_dict()}


@app.get("/api/agents/{slug}/job")
async def job_status(slug: str):
    """Ход последней индексации агента (интерфейс спрашивает раз в секунду)."""
    AGENTS.get(slug)
    job = JOBS.latest(slug)
    return {"job": job.as_dict() if job else None}


@app.delete("/api/agents/{slug}/base")
async def clear_base(slug: str):
    """Очищает базу агента; книги на диске остаются."""
    removed = await asyncio.to_thread(REGISTRY.get(slug).clear)
    return {"ok": True, "removed": removed}


# --- История поиска -----------------------------------------------------
class HistoryEntry(BaseModel):
    question: str
    answer: str = ""
    sources: list[dict] = []
    provider: str = ""
    model: str = ""


@app.get("/api/agents/{slug}/history")
async def list_history(slug: str, limit: int = 50, offset: int = 0):
    """Сохранённые ответы агента: чтобы не искать снова."""
    agent = AGENTS.get(slug)
    entries = await asyncio.to_thread(history_load, str(agent.path))
    return {"slug": slug, "total": len(entries),
            "entries": entries[offset:offset + limit]}


@app.post("/api/agents/{slug}/history")
async def save_history(slug: str, req: HistoryEntry):
    """Сохраняет ответ (вопрос + ответ + источники) в историю агента."""
    agent = AGENTS.get(slug)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос пуст")
    entry = await asyncio.to_thread(
        history_add, str(agent.path),
        question=req.question, answer=req.answer, sources=req.sources,
        provider=req.provider, model=req.model)
    return {"ok": True, "entry": entry}


@app.delete("/api/agents/{slug}/history/{entry_id}")
async def delete_history_entry(slug: str, entry_id: str):
    """Удаляет одну запись из истории."""
    agent = AGENTS.get(slug)
    removed = await asyncio.to_thread(history_delete, str(agent.path), entry_id)
    if not removed:
        raise HTTPException(status_code=404,
                            detail=f"Записи «{entry_id}» в истории нет")
    return {"ok": True}


@app.delete("/api/agents/{slug}/history")
async def clear_history(slug: str):
    """Очищает всю историю агента."""
    agent = AGENTS.get(slug)
    removed = await asyncio.to_thread(history_clear, str(agent.path))
    return {"ok": True, "removed": removed}


@app.get("/api/server")
async def server_status():
    """Что за сервер запущен: адрес для телефона, порт, сколько работает.

    Интерфейс показывает это на вкладке «Сервер», чтобы не пришлось искать
    IP компьютера в командной строке.
    """
    return SERVER.status()


@app.post("/api/server/restart")
async def server_restart():
    """Перезапускает сервер: нужен после правки config.yaml или git pull.

    Сначала запускается копия (свежий код подхватится), затем этот процесс
    завершается. Ответ уходит до остановки, чтобы браузер успел его показать.
    """
    try:
        pid = SERVER.restart()
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Не удалось перезапустить сервер: {exc}")
    return {"ok": True, "restarting": True, "pid": pid,
            "urls": SERVER.status()["urls"]}


@app.post("/api/server/stop")
async def server_stop():
    """Останавливает сервер: запустить обратно можно только на компьютере.

    Из браузера поднять остановленный сервер нельзя — на это есть команда.
    """
    SERVER.stop()
    return {"ok": True, "stopping": True, "start_command": SERVER.status()["start_command"]}


# Статика (index.html, app.js, PWA) — монтируется последней, чтобы
# не перехватывать запросы к /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import uvicorn

    if os.environ.get(SUCCESSOR_ENV):
        # Это перезапущенный сервер: ждём, пока предыдущий отпустит порт.
        print("Перезапуск: жду, пока предыдущий сервер освободит порт…")
        if not wait_until_free(CONFIG.server.host, CONFIG.server.port, timeout=30.0):
            print("Порт всё ещё занят — пробую запуститься всё равно")

    config = uvicorn.Config(app, host=CONFIG.server.host, port=CONFIG.server.port,
                            log_level="info")
    server = uvicorn.Server(config)
    SERVER.uvicorn = server
    server.run()


if __name__ == "__main__":
    main()
