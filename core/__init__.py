"""Ядро RAG-ассистента.

Подмодули импортируются напрямую (core.database, core.parsers, core.llm,
core.providers, core.models, core.store), чтобы импорт пакета не тянул
тяжёлые зависимости — модели эмбеддингов и ChromaDB.
"""
