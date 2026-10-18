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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config, load_config
from core.llm import LLMError, OpenAICompatClient
from core.store import VectorStore

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

WEB_DIR = Path(__file__).resolve().parent / "web"
CONFIG: Config = load_config()


def sse(event: str, data) -> str:
    """Формирует одно SSE-событие."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Загружаю RAG-ядро (коллекция «{CONFIG.store.collection}»)...")
    app.state.store = await asyncio.to_thread(VectorStore, CONFIG)
    app.state.llm = OpenAICompatClient(CONFIG.llm)
    stats = app.state.store.stats()
    print(f"Готово: {stats['total_chunks']} фрагментов, источников: {stats['sources_count']}")
    print(f"Модель: {CONFIG.llm.model} ({CONFIG.llm.provider})")
    print(f"Открой http://localhost:{CONFIG.server.port}")
    yield
    await app.state.llm.aclose()


app = FastAPI(title="RAG Ассистент", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    n_results: int | None = None


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
    """Генератор SSE-событий: поиск → источники → стриминг ответа."""
    store: VectorStore = app.state.store
    llm: OpenAICompatClient = app.state.llm

    try:
        results = await asyncio.to_thread(store.search, req.message, None)
        context = store.format_context(results)
    except Exception as exc:
        yield sse("error", {"message": f"Ошибка поиска: {exc}"})
        yield sse("done", {})
        return

    if results:
        yield sse("sources", store.sources(results))
    elif CONFIG.search.require_context:
        yield sse("token", {"text": "В базе не найдено фрагментов для ответа на этот вопрос."})
        yield sse("done", {})
        return

    try:
        async for piece in llm.stream(build_messages(req, context)):
            yield sse("token", {"text": piece})
    except LLMError as exc:
        yield sse("error", {"message": str(exc)})
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


# Статика (index.html, app.js, PWA) — монтируется последней, чтобы
# не перехватывать запросы к /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import uvicorn

    uvicorn.run(app, host=CONFIG.server.host, port=CONFIG.server.port, log_level="info")


if __name__ == "__main__":
    main()
