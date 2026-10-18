#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для создания и управления векторной базой данных для RAG системы
Обрабатывает PDF файлы исследований и создает векторные представления
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from core import parsers

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# На консоли Windows (cp866/cp1251) печать эмодзи падает с UnicodeEncodeError
# и прерывает работу. Заменяем непечатаемые символы вместо ошибки.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

def _set_alarm(seconds: int):
    """Ограничение времени на обработку файла (на Windows не поддерживается)."""
    if os.name == "nt":
        return
    import signal

    if seconds:
        def timeout_handler(signum, frame):
            raise TimeoutError("Превышен лимит времени обработки файла")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    else:
        signal.alarm(0)


class RAGDatabase:
    # Значения по умолчанию на уровне класса: их видят и экземпляры,
    # созданные через __new__ (например, в тестах).
    chunk_size = 1000
    chunk_overlap = 200
    # Точные совпадения по словам (через индекс Chroma). Быстро на большой
    # базе; можно отключить, чтобы искать только векторно.
    lexical = True
    # Служебные слова запроса: по ним рецепт не ищут, они только добавляют шум.
    STOPWORDS = frozenset({
        "как", "что", "это", "такое", "для", "или", "мне", "про", "расскажи",
        "приготовить", "приготовление", "готовить", "рецепт", "пожалуйста",
        "какие", "есть", "найди", "можно", "нужно", "надо", "хочу", "сделать",
    })

    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "research_papers",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 formats: Optional[List[str]] = None,
                 ocr: str = "auto"):
        """
        Инициализация RAG базы данных
        
        Args:
            db_path: Путь к базе данных ChromaDB
            collection_name: Название коллекции
            embedding_model: Модель для создания эмбеддингов
            chunk_size: Размер фрагмента в символах
            chunk_overlap: Перекрытие между фрагментами в символах
            formats: Расширения файлов, которые индексируются
            ocr: "auto" — распознавать страницы без текста (нужен tesseract),
                 "off" — не распознавать
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.formats = [suffix.lower() for suffix in (formats or parsers.SUPPORTED)]
        self.tesseract = None if ocr == "off" else parsers.Tesseract()
        if self.tesseract is not None and not self.tesseract.available:
            self.tesseract = None
        
        # Инициализация ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Инициализация модели эмбеддингов
        print(f"🔄 Загружаю модель эмбеддингов: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✅ Модель эмбеддингов загружена")
        
        # Создание или получение коллекции
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Найдена существующая коллекция: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Векторная база исследований для RAG"}
            )
            print(f"✅ Создана новая коллекция: {collection_name}")
    
    def iter_sections(self, file_path: str):
        """Текст документа по частям: страница PDF, глава EPUB, часть TXT.

        Ошибка чтения прерывает импорт: лучше остановиться, чем молча
        записать в базу половину книги.
        """
        yield from parsers.read_sections(file_path, self.tesseract)

    def extract_text(self, file_path: str) -> str:
        """Весь текст файла одной строкой (для внешних вызовов)."""
        return "\n".join(text for _, text in self.iter_sections(file_path))

    def clean_text(self, text: str) -> str:
        """Очищает и нормализует текст (колонтитулы, переносы, мусор)."""
        return parsers.clean_text(text)
    
    def iter_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        """Выдаёт фрагменты без накопления списка и всегда движется вперёд."""
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError("Требуется chunk_size > 0 и 0 <= overlap < chunk_size")
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                sentence_end = text.rfind('.', max(start + overlap, end - 100), end + 100)
                if sentence_end != -1:
                    end = sentence_end + 1
            chunk = text[start:end].strip()
            if chunk:
                yield chunk
            if end >= len(text):
                break
            start = end - overlap

    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Совместимый интерфейс для получения списка фрагментов."""
        return list(self.iter_text_chunks(text, chunk_size, overlap))

    def add_file_to_database(self, file_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Читает файл (PDF/EPUB/TXT/FB2) и сразу сохраняет порции по 32 фрагмента.

        При ошибке импорт прерывается; уже записанные порции остаются в базе.
        """
        path = Path(file_path)
        print(f"📄 Обрабатываю файл: {path.name}")
        source = path.name
        unit = parsers.unit_for(path)
        file_size = path.stat().st_size
        chunks = []
        metadatas = []
        saved = 0

        def save_batch():
            nonlocal saved
            embeddings = self.embedding_model.encode(
                chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            self.collection.upsert(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{source}_chunk_{i}" for i in range(saved, saved + len(chunks))]
            )
            saved += len(chunks)
            print(f"  💾 Сохранено фрагментов: {saved}", flush=True)
            chunks.clear()
            metadatas.clear()

        for number, section in self.iter_sections(str(path)):
            text = self.clean_text(section)
            if not parsers.keep_section(text):
                continue  # пустая страница, обложка, оглавление или указатель
            for chunk in self.iter_text_chunks(text, self.chunk_size, self.chunk_overlap):
                # Обрывки оглавления и мусор распознавания в базу не пишем.
                # Явные признаки проверяем и здесь: страница может быть
                # смешанной, тогда на фрагменты она не отсеется целиком.
                if len(chunk) <= 50 or parsers.has_leader_dots(chunk) \
                        or parsers.has_broken_glyphs(chunk):
                    continue
                chunks.append(chunk)
                metadatas.append({
                    **(metadata or {}),
                    "source": source,
                    "chunk_id": saved + len(chunks) - 1,
                    "chunk_length": len(chunk),
                    "page": number,
                    "unit": unit,
                    "file_size": file_size,  # Размер файла в байтах.
                })
                if len(chunks) == 32:
                    save_batch()
        if chunks:
            save_batch()
        if not saved:
            raise ValueError(f"В файле {source} не найден текст для индексации")
        # При повторном импорте более короткого документа удаляем старый хвост.
        self.collection.delete(where={"$and": [
            {"source": source}, {"chunk_id": {"$gte": saved}}
        ]})
        print(f"✅ {source} добавлен: {saved} фрагментов")

    @staticmethod
    def _resolve_folder(folder: str) -> Path:
        """Находит папку, если отличается только регистр имени (Windows/Linux)."""
        path = Path(folder)
        if path.exists() or not path.parent.exists():
            return path
        name = path.name.casefold()
        for child in path.parent.iterdir():
            if child.is_dir() and child.name.casefold() == name:
                return child
        return path

    def process_research_folder(self, research_folder: str = "./research"):
        """Индексирует все поддерживаемые файлы в папке (и в подпапках).

        Args:
            research_folder: Путь к папке с книгами (PDF, EPUB, TXT, FB2)
        """
        research_path = self._resolve_folder(research_folder)
        if not research_path.exists():
            print(f"❌ Папка {research_folder} не найдена")
            return False

        files = [path for path in sorted(research_path.rglob("*"))
                 if path.is_file() and path.suffix.lower() in self.formats]
        skipped = [path for path in sorted(research_path.rglob("*"))
                   if path.is_file() and path.suffix.lower() not in self.formats]
        if not files:
            print(f"❌ Книги не найдены в {research_folder} "
                  f"(поддерживаются: {', '.join(self.formats)})")
            return False

        print(f"🔍 Найдено файлов: {len(files)} "
              f"({', '.join(sorted({p.suffix.lower() for p in files}))})")
        if skipped:
            print(f"⏭️ Пропускаю файлы, формат которых не читаю: "
                  f"{len(skipped)} ({', '.join(sorted({p.suffix.lower() for p in skipped}))})")
        if self.tesseract is not None:
            print("🔎 Страницы без текста будут распознаны через tesseract "
                  "(это самая долгая часть)")

        successful_files = 0
        failed_files = []

        for i, file_path in enumerate(files, 1):
            print(f"\n📋 Обрабатываю файл {i}/{len(files)}: {file_path.name}")

            try:
                _set_alarm(300)  # ограничение времени на файл, только для UNIX
                self.add_file_to_database(str(file_path))
                _set_alarm(0)
                successful_files += 1

            except TimeoutError:
                print(f"⏰ Превышен лимит времени для {file_path.name}")
                failed_files.append((file_path.name, "Таймаут"))
                _set_alarm(0)

            except Exception as e:
                print(f"❌ Ошибка обработки {file_path.name}: {e}")
                failed_files.append((file_path.name, str(e)))
                _set_alarm(0)

        # Итоговая статистика
        print(f"\n🎉 Обработка завершена!")
        print(f"✅ Успешно обработано файлов: {successful_files}/{len(files)}")
        
        if failed_files:
            print(f"❌ Не удалось обработать {len(failed_files)} файлов:")
            for file_name, error in failed_files:
                print(f"  - {file_name}: {error}")
        
        total_docs = self.collection.count()
        print(f"📊 Всего документов в базе: {total_docs}")
        
        return successful_files > 0
    
    @staticmethod
    def _search_text(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))

    @staticmethod
    def _stem(word: str) -> str:
        """Грубая основа слова: убирает типичное окончание и лишний хвост.

        Нужна, чтобы «бёдрышки» находили «бёдрышек», а «в мёде» — «мёдом».
        Слово уже должно быть приведено к нижнему регистру (см. _search_text).
        """
        if len(word) > 3 and word[-1] in "аеёиоуыэюяй":
            word = word[:-1]
        return word if len(word) <= 5 else word[:5]

    def _matches_stem(self, token: str, stem: str) -> bool:
        """Слово подходит основе, если начинается с неё и хвост не слишком длинный."""
        return token.startswith(stem) and len(token) - len(stem) <= max(4, len(stem))

    @staticmethod
    def _contains_variants(stem: str) -> List[str]:
        """Варианты основы для $contains: регистр и «е»/«ё».

        В книгах пишут и «бёдрышки», и «бедрышки», поэтому проверяем обе буквы.
        """
        forms = {stem}
        positions = [index for index, char in enumerate(stem) if char == "е"]
        for index in positions:
            forms.add(stem[:index] + "ё" + stem[index + 1:])
        if 1 < len(positions) <= 4:
            forms.add(stem.replace("е", "ё"))
        variants = set()
        for form in forms:
            variants.update({form, form.capitalize(), form.upper()})
        return sorted(variants)

    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Гибридный поиск: слова запроса + векторная близость.

        Слова ищутся по основам (учитываются падежи и формы), поэтому редкое
        слово из вопроса поднимает нужный фрагмент, даже если векторная модель
        ошиблась. Списки ранжируются и объединяются по Reciprocal Rank Fusion.
        """
        normalized = self._search_text(query)
        if not normalized or n_results <= 0:
            return []
        count = self.collection.count()
        if not count:
            return []

        words = {
            word for word in normalized.split()
            if len(word) > 2 and word not in self.STOPWORDS
        }

        embedding = self.embedding_model.encode([query], show_progress_bar=False)
        vector = embedding.tolist()
        candidates = self._lexical_candidates(words) if self.lexical else {}

        found = self.collection.query(
            query_embeddings=vector,
            n_results=min(count, max(12, n_results)),
        )
        semantic = [
            {'id': ident, 'document': doc, 'metadata': meta or {}, 'distance': distance}
            for ident, doc, meta, distance in zip(found['ids'][0], found['documents'][0],
                                                  found['metadatas'][0], found['distances'][0])
        ]
        # Оглавления, указатели и OCR-мусор — в конец списка, но не выбрасываем:
        # если ничего другого нет, лучше такой фрагмент, чем ничего.
        semantic.sort(key=lambda item: self._is_noise(item['document']))

        # Фрагмент, поднявшийся высоко в обоих списках, обгоняет того, кто
        # оказался высоко только в одном. У словесного списка приоритет выше:
        # векторная модель слабее на русских формулировках. Берём только верх
        # словесного списка, иначе слабые частые слова перевешивают точные.
        lexical = self._rank_lexical(candidates, normalized)[:max(10, n_results * 5)]
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}
        for list_weight, ranked in ((1.0, lexical), (0.6, semantic)):
            for rank, item in enumerate(ranked, 1):
                ident = item['id']
                items.setdefault(ident, item)
                scores[ident] = scores.get(ident, 0.0) + list_weight / (60 + rank)

        order = sorted(scores, key=lambda ident: scores[ident], reverse=True)
        return [items[ident] for ident in order[:n_results]]

    def _lexical_candidates(self, words, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Кандидаты, содержащие слова запроса (индекс Chroma), с оценкой редкости.

        Слова ищутся по основам и без ограничения на порядок выдачи, поэтому
        редкая книга не теряется среди тысяч частых совпадений. Выборка не
        обрезается: обрезанная, она искажает вес слова (чем реже слово, тем
        оно важнее), а сама проверка всё равно просматривает всю базу.
        """
        candidates: Dict[str, Dict[str, Any]] = {}
        texts: Dict[str, str] = {}
        limit = limit or max(1000, self.collection.count())
        for word in words:
            stem = self._stem(word)
            if len(stem) < 3:
                continue
            variants = self._contains_variants(stem)
            where = (
                {"$or": [{"$contains": variant} for variant in variants]}
                if len(variants) > 1 else {"$contains": variants[0]}
            )
            try:
                batch = self.collection.get(
                    where_document=where, limit=limit,
                    include=["documents", "metadatas"],
                )
            except Exception:
                # Старые версии Chroma могут не поддерживать $contains.
                continue
            ids = batch['ids']
            if not ids:
                continue
            # Чем в меньшем числе фрагментов встречается слово, тем оно важнее.
            weight = 1.0 / len(ids)
            for ident, doc, meta in zip(ids, batch['documents'], batch['metadatas']):
                text = texts.get(ident)
                if text is None:
                    text = self._search_text(doc)
                    texts[ident] = text
                occurrences = sum(
                    1 for token in text.split() if self._matches_stem(token, stem)
                )
                if not occurrences:
                    continue
                entry = candidates.get(ident)
                if entry is None:
                    entry = {'id': ident, 'document': doc, 'metadata': meta or {},
                             'distance': None, 'matched': 0, 'weight': 0.0, 'count': 0}
                    candidates[ident] = entry
                entry['matched'] += 1
                entry['weight'] += weight
                entry['count'] += occurrences
        return candidates

    def _rank_lexical(self, candidates, normalized: str) -> List[Dict[str, Any]]:
        """Сортирует словесных кандидатов: осмысленный текст, затем точность.

        Оглавления и служебные маркеры отодвигаются в конец: в оглавлении
        название блюда встречается рядом с номерами страниц, и по числу
        совпадений оно обгоняло сам рецепт. Плотность совпадений важнее
        их абсолютного числа, иначе длинный фрагмент побеждает короткий.
        """
        items = list(candidates.values())
        for item in items:
            text = self._search_text(item['document'])
            item['phrase'] = normalized in text
            item['noise'] = self._is_noise(item['document'])
            item['density'] = item['count'] * 1000 / max(1, len(item['document']))
        items.sort(
            key=lambda item: (
                item['noise'],
                not item['phrase'],
                -item['weight'],
                -item['matched'],
                -item['count'],
                -item['density'],  # добивка: при равном числе совпадений короче — точнее
            )
        )
        return items

    @staticmethod
    def _is_noise(text: str) -> bool:
        """Оглавление, указатель, мусор распознавания — не ответ на вопрос.

        Такие фрагменты не содержат рецепта, но могут много раз повторять
        название блюда и обгонять сам рецепт. Поэтому они понижаются при
        поиске, а при индексации такие страницы и вовсе не пишутся в базу
        (см. parsers.keep_section, parsers.is_listing).
        """
        return parsers.is_noise(text)

    def expand_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Добавляет по одному соседнему фрагменту с каждой стороны."""
        expanded = []
        seen = set()
        for result in results:
            meta = result['metadata']
            source, number = meta.get('source'), meta.get('chunk_id')
            nearby = [result]
            if source is not None and isinstance(number, int):
                ids = [f"{source}_chunk_{i}" for i in range(max(0, number - 1), number + 2)]
                batch = self.collection.get(ids=ids, include=["documents", "metadatas"])
                nearby = [
                    {'id': ident, 'document': doc, 'metadata': metadata or {}, 'distance': None}
                    for ident, doc, metadata in zip(batch['ids'], batch['documents'], batch['metadatas'])
                ]
                nearby = [item for item in nearby if item['metadata'].get('source') == source]
                if not any(item['id'] == result['id'] for item in nearby):
                    nearby.append(result)
                nearby.sort(key=lambda item: item['metadata'].get('chunk_id', number))
            for item in nearby:
                if item['id'] not in seen:
                    seen.add(item['id'])
                    expanded.append(item)
        return expanded

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Получает статистику базы данных
        
        Returns:
            Словарь со статистикой
        """
        count = self.collection.count()
        
        # Получаем информацию о источниках
        sources = set()
        for offset in range(0, count, 1000):
            batch = self.collection.get(include=["metadatas"], limit=1000, offset=offset)
            sources.update(meta.get('source', 'unknown') for meta in batch['metadatas'] if meta)
        
        return {
            'total_chunks': count,
            'sources': list(sources),
            'sources_count': len(sources),
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name
        }


def main():
    """Пересобирает базу из папки с книгами (см. config.yaml)."""
    print("🚀 Создание векторной базы данных для RAG системы")
    print("=" * 60)

    from config import load_config

    cfg = load_config()
    rag_db = RAGDatabase(
        db_path=cfg.store.db_path,
        collection_name=cfg.store.collection,
        embedding_model=cfg.embeddings.model,
        chunk_size=cfg.store.chunk_size,
        chunk_overlap=cfg.store.chunk_overlap,
        formats=cfg.store.formats,
        ocr=cfg.store.ocr,
    )
    rag_db.process_research_folder(cfg.store.research_folder)

    # Показываем статистику
    stats = rag_db.get_database_stats()
    print("\n📊 Статистика базы данных:")
    print(f"Всего чанков: {stats['total_chunks']}")
    print(f"Источников: {stats['sources_count']}")
    print(f"Файлы: {', '.join(stats['sources'])}")

    # Тестовый поиск
    print("\n🔍 Тестовый поиск...")
    test_query = "machine learning"
    results = rag_db.search_similar(test_query, n_results=3)
    
    print(f"Результаты для запроса '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Источник: {result['metadata']['source']}")
        print(f"   Текст: {result['document'][:200]}...")
        print(f"   Расстояние: {result['distance']:.4f}" if result['distance'] else "")
        print()


if __name__ == "__main__":
    main()