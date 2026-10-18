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

import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# На консоли Windows (cp866/cp1251) печать эмодзи падает с UnicodeEncodeError
# и прерывает работу. Заменяем непечатаемые символы вместо ошибки.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

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
                 chunk_overlap: int = 200):
        """
        Инициализация RAG базы данных
        
        Args:
            db_path: Путь к базе данных ChromaDB
            collection_name: Название коллекции
            embedding_model: Модель для создания эмбеддингов
            chunk_size: Размер фрагмента в символах
            chunk_overlap: Перекрытие между фрагментами в символах
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
    
    def iter_pdf_pages(self, pdf_path: str):
        """Извлекает по одной странице; ошибки прерывают импорт."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # shortcut: PyPDF2 кеширует объекты PDF; особо тяжёлые страницы
            # требуют отдельного процесса с ограничением памяти.
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    yield page_num, f"--- Страница {page_num} ---\n{text}"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Возвращает весь текст для внешних вызовов; импорт использует итератор."""
        return "\n".join(text for _, text in self.iter_pdf_pages(pdf_path))

    def clean_text(self, text: str) -> str:
        """
        Очищает и нормализует текст
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем специальные символы, но сохраняем пунктуацию
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\%\=\+\*\&\#\@]', ' ', text)
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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

    def add_pdf_to_database(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Читает страницы и сразу сохраняет порции по 32 фрагмента.

        При ошибке импорт прерывается; уже записанные порции остаются в базе.
        """
        print(f"📄 Обрабатываю PDF: {pdf_path}")
        pdf_name = Path(pdf_path).name
        file_size = Path(pdf_path).stat().st_size
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
                ids=[f"{pdf_name}_chunk_{i}" for i in range(saved, saved + len(chunks))]
            )
            saved += len(chunks)
            print(f"  💾 Сохранено фрагментов: {saved}", flush=True)
            chunks.clear()
            metadatas.clear()

        for page_num, page_text in self.iter_pdf_pages(pdf_path):
            for chunk in self.iter_text_chunks(
                self.clean_text(page_text), self.chunk_size, self.chunk_overlap
            ):
                if len(chunk) <= 50:
                    continue
                chunks.append(chunk)
                metadatas.append({
                    **(metadata or {}),
                    "source": pdf_name,
                    "chunk_id": saved + len(chunks) - 1,
                    "chunk_length": len(chunk),
                    "page": page_num,
                    "file_size": file_size,  # Размер PDF в байтах.
                })
                if len(chunks) == 32:
                    save_batch()
        if chunks:
            save_batch()
        if not saved:
            raise ValueError(f"В PDF {pdf_name} не найден текст для индексации")
        # При повторном импорте более короткого документа удаляем старый хвост.
        self.collection.delete(where={"$and": [
            {"source": pdf_name}, {"chunk_id": {"$gte": saved}}
        ]})
        print(f"✅ PDF {pdf_name} добавлен: {saved} фрагментов")

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

    def process_research_folder(self, research_folder: str = "./Research"):
        """
        Обрабатывает все PDF файлы в папке исследований
        
        Args:
            research_folder: Путь к папке с PDF файлами
        """
        research_path = self._resolve_folder(research_folder)
        if not research_path.exists():
            print(f"❌ Папка {research_folder} не найдена")
            return
        
        pdf_files = list(research_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ PDF файлы не найдены в {research_folder}")
            return
        
        print(f"🔍 Найдено PDF файлов: {len(pdf_files)}")
        
        successful_files = 0
        failed_files = []
        
        # Сигналы таймаута доступны только на UNIX-системах (Linux / macOS)
        has_signal_alarm = hasattr(os, 'name') and os.name != 'nt'
        if has_signal_alarm:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Превышен лимит времени обработки файла")

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📋 Обрабатываю файл {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                if has_signal_alarm:
                    import signal
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 минут
                
                self.add_pdf_to_database(str(pdf_file))
                
                if has_signal_alarm:
                    import signal
                    signal.alarm(0)
                successful_files += 1
                
            except TimeoutError:
                print(f"⏰ Превышен лимит времени для {pdf_file.name}")
                failed_files.append((pdf_file.name, "Таймаут"))
                if has_signal_alarm:
                    import signal
                    signal.alarm(0)
                
            except Exception as e:
                print(f"❌ Ошибка обработки {pdf_file.name}: {e}")
                failed_files.append((pdf_file.name, str(e)))
                if has_signal_alarm:
                    import signal
                    signal.alarm(0)
        
        # Итоговая статистика
        print(f"\n🎉 Обработка завершена!")
        print(f"✅ Успешно обработано файлов: {successful_files}/{len(pdf_files)}")
        
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

    def _lexical_candidates(self, words, limit: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Кандидаты, содержащие слова запроса (индекс Chroma), с оценкой редкости.

        Слова ищутся по основам и без ограничения на порядок выдачи, поэтому
        редкая книга не теряется среди тысяч частых совпадений.
        """
        candidates: Dict[str, Dict[str, Any]] = {}
        texts: Dict[str, str] = {}
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
        """Сортирует словесных кандидатов: фраза, редкость слов, число попаданий."""
        items = list(candidates.values())
        for item in items:
            item['phrase'] = normalized in self._search_text(item['document'])
        items.sort(
            key=lambda item: (item['phrase'], item['weight'], item['matched'], item['count']),
            reverse=True,
        )
        return items

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
    """Главная функция для создания базы данных"""
    print("🚀 Создание векторной базы данных для RAG системы")
    print("=" * 60)
    
    # Создаем экземпляр базы данных
    rag_db = RAGDatabase()
    
    # Обрабатываем папку исследований
    rag_db.process_research_folder("./Research")
    
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