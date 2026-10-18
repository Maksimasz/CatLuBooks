#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Веб-оболочка RAG-ассистента: FastAPI + статический чат (PWA).

Запуск:
    python app.py
Затем откройте http://localhost:8000 (с телефона — http://<IP-ПК>:8000).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config, load_config, resolve_llm
from core.llm import LLMError, OpenAICompatClient, fetch_models
from core.providers import (
    PROVIDERS,
    key_for,
    model_is_free,
    needs_key,
    provider_catalog,
    save_secret,
)
from core.store import VectorStore

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

WEB_DIR = Path(__file__).resolve().parent / "web"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LAST_REQUEST = LOG_DIR / "last_request.json"
CONFIG: Config = load_config()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Загружаю RAG-ядро (коллекция «{CONFIG.store.collection}»)...")
    app.state.store = await asyncio.to_thread(VectorStore, CONFIG)
    app.state.clients = {}  # (provider, model) -> OpenAICompatClient
    stats = app.state.store.stats()
    print(f"Готово: {stats['total_chunks']} фрагментов, источников: {stats['sources_count']}")
    print(f"Модель по умолчанию: {CONFIG.llm.model} ({CONFIG.llm.provider})")
    print(f"Открой http://localhost:{CONFIG.server.port}")
    try:
        yield
    finally:
        for client in app.state.clients.values():
            await client.aclose()


app = FastAPI(title="RAG Ассистент", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class SearchRequest(BaseModel):
    query: str
    n_results: int | None = None


class KeyRequest(BaseModel):
    provider: str
    key: str = ""


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
    store: VectorStore = app.state.store
    provider = req.provider or CONFIG.llm.provider
    model = req.model or CONFIG.llm.model

    if needs_key(provider) and not key_for(provider):
        yield sse("error", {"message": f"Для «{provider}» нужен API-ключ. Введите его кнопкой 🔑 в шапке."})
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

    yield sse("model", {"provider": provider, "model": model})

    found = store.sources(results) if results else []
    print(f"❓ {req.message}  ({provider}/{model}, фрагментов {len(results)}, "
          f"контекст {len(context)} символов)", flush=True)
    record = {
        "время": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        yield sse("token", {"text": "В базе не найдено фрагментов для ответа на этот вопрос."})
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
async def search(req: SearchRequest, request: Request):
    store = request.app.state.store
    results = await asyncio.to_thread(store.search, req.query, req.n_results)
    return {"results": store.sources(results)}


@app.get("/api/stats")
async def stats(request: Request):
    return request.app.state.store.stats()


@app.get("/api/config")
async def config():
    return {
        "provider": CONFIG.llm.provider,
        "model": CONFIG.llm.model,
        "collection": CONFIG.store.collection,
        "embedding_model": CONFIG.embeddings.model,
    }


@app.get("/api/providers")
async def providers():
    return {
        "providers": provider_catalog(),
        "default": {"provider": CONFIG.llm.provider, "model": CONFIG.llm.model},
    }


@app.post("/api/keys")
async def set_key(req: KeyRequest):
    if req.provider not in {item["name"] for item in provider_catalog()}:
        raise HTTPException(status_code=400, detail="Неизвестный провайдер")
    save_secret(req.provider, req.key.strip())
    _models_cache.pop(req.provider, None)  # ключ изменился — список моделей обновим
    return {"ok": True, "configured": bool(key_for(req.provider))}


MODELS_TTL = 300.0  # секунд; список моделей меняется нечасто
_models_cache: dict[str, tuple[float, dict]] = {}


async def provider_models(name: str, fresh: bool = False) -> dict:
    """Модели провайдера: живой список от сервиса, иначе статический.

    Бесплатные модели помечаются и идут первыми. Без ключа большинство
    облачных сервисов список не отдаёт — тогда берём запасной из реестра.
    """
    if not fresh:
        cached = _models_cache.get(name)
        if cached and time.monotonic() - cached[0] < MODELS_TTL:
            return cached[1]

    info = PROVIDERS.get(name, {})
    live: list = []
    error = None
    if name in PROVIDERS:
        try:
            live = await fetch_models(resolve_llm(CONFIG.llm, name, None))
        except Exception as exc:  # нет ключа, нет сети и т.п.
            error = str(exc)

    source = "provider" if live else "fallback"
    models = []
    for item in (live or info.get("models", [])):
        if isinstance(item, dict):
            if item.get("chat") is False:
                continue  # речь, расшифровка звука, картинки — не для чата
            ident, pricing_free = item.get("id"), item.get("free")
        else:
            ident, pricing_free = item, None
        if not ident:
            continue
        ident = str(ident)
        models.append({"id": ident, "free": model_is_free(name, ident, pricing_free)})
    models.sort(key=lambda m: (not m["free"], m["id"].lower()))

    result = {
        "provider": name,
        "label": info.get("label", name),
        "models": models,
        "preferred": info.get("preferred", ""),
        "source": source,
        "configured": not info.get("needs_key", True) or bool(key_for(name)),
        "error": error,
    }
    _models_cache[name] = (time.monotonic(), result)
    return result


@app.get("/api/models")
async def models(provider: str | None = None, all: int = 0):
    """Список моделей: ?provider=groq для одного, ?all=1 — сразу для всех."""
    if all:
        results = await asyncio.gather(*(provider_models(name) for name in PROVIDERS))
        return {"providers": list(results)}
    name = provider or CONFIG.llm.provider
    if name not in PROVIDERS:
        return {"provider": name, "label": name, "models": [], "source": "fallback",
                "configured": True, "error": "нет реестра моделей"}
    return await provider_models(name)


# Статика (index.html, app.js, PWA) — монтируется последней, чтобы
# не перехватывать запросы к /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import uvicorn

    uvicorn.run(app, host=CONFIG.server.host, port=CONFIG.server.port, log_level="info")


if __name__ == "__main__":
    main()
