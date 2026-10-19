#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Векторная база книг: индексация (PDF, EPUB, TXT, FB2) и гибридный поиск по ней.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from core import parsers

# На консоли Windows (cp866/cp1251) печать эмодзи падает с UnicodeEncodeError
# и прерывает работу. Заменяем непечатаемые символы вместо ошибки.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# Описание коллекции: видно в базе, помогает понять, что за данные внутри.
COLLECTION_META = {"description": "Векторная база знаний агента"}

# Имя коллекции внутри базы агента: у каждого агента своя папка db/.
BOOKS_COLLECTION = "books"

# Модель эмбеддингов одна на процесс: агентов может быть много, а модель
# тяжёлая — грузим её при первом обращении и переиспользуем.
MODELS: Dict[str, Any] = {}


def embedding_model_for(name: str):
    """Локальная модель эмбеддингов (общая для всех агентов)."""
    if name not in MODELS:
        print(f"🔄 Загружаю модель эмбеддингов: {name}")
        MODELS[name] = SentenceTransformer(name)
        print("✅ Модель эмбеддингов загружена")
    return MODELS[name]

# Надбавка фрагменту, в котором совпали все слова вопроса. Больше любой суммы
# мест RRF (места дают в лучшем случае 1/60 + 0,6/60), поэтому полное
# совпадение слов всегда идёт впереди — см. search_similar.
EXACT_MATCH_BONUS = 10.0


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
                 ocr: str = "auto",
                 embedding_model_instance: Any | None = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Открывает базу агента.

        Args:
            db_path: папка базы агента (агент/база, обычно agents/<имя>/db)
            collection_name: имя коллекции внутри базы
            embedding_model: имя локальной модели эмбеддингов
            chunk_size: размер фрагмента в символах
            chunk_overlap: перекрытие соседних фрагментов
            formats: расширения файлов, которые читаем из папки с книгами
            ocr: "auto" — распознавать страницы-картинки (нужен tesseract),
                 "off" — не распознавать
            embedding_model_instance: уже загруженная модель (общая на процесс)
            progress: куда сообщать о ходе индексации (для интерфейса)
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._embedding_model = embedding_model_instance
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.formats = [suffix.lower() for suffix in (formats or parsers.SUPPORTED)]
        self.set_ocr(ocr)
        self.progress = progress

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)

        # Коллекция открывается или создаётся — отдельный случай «уже есть»
        # не нужен: get_or_create сам вернёт существующую.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=COLLECTION_META,
        )
        print(f"✅ База «{Path(db_path).name}» готова: "
              f"{self.collection.count()} фрагментов")

    @property
    def embedding_model(self):
        """Модель эмбеддингов: грузится при первом обращении, одна на процесс."""
        if getattr(self, "_embedding_model", None) is None:
            self._embedding_model = embedding_model_for(self.embedding_model_name)
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value) -> None:
        self._embedding_model = value

    def set_ocr(self, ocr) -> None:
        """Включает или выключает распознавание сканов на ходу.

        Настройку меняют в интерфейсе, а база уже открыта: пересоздавать её
        незачем — достаточно убрать или поднять движок распознавания.
        """
        self.ocr = parsers.ocr_mode(ocr)
        self.tesseract = None if self.ocr == parsers.OCR_OFF else parsers.Tesseract()
        if self.tesseract is not None and not self.tesseract.available:
            self.tesseract = None

    def _report(self, **event: Any) -> None:
        """Сообщает о ходе индексации: веб-слой показывает это в интерфейсе."""
        reporter = getattr(self, "progress", None)
        if reporter is not None:
            reporter(event)

    def close(self) -> None:
        """Закрывает базу: освобождает файлы (нужно перед удалением папки)."""
        try:
            self.client.close()
        except Exception as exc:  # уже закрыта или закрыть нечем
            print(f"не удалось закрыть базу {self.db_path}: {exc}")

    def iter_sections(self, file_path: str):
        """Текст документа по частям: страница PDF, глава EPUB, часть TXT.

        Ошибка чтения прерывает импорт: лучше остановиться, чем молча
        записать в базу половину книги.
        """
        yield from parsers.read_sections(file_path, self.tesseract)

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
            self._report(event="chunks", source=source, saved=saved)
            chunks.clear()
            metadatas.clear()

        for number, section in self.iter_sections(str(path)):
            text = parsers.clean_text(section)
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
        self._report(event="file_done", source=source, chunks=saved)

    def delete_source(self, source: str) -> int:
        """Убирает из базы все фрагменты файла: сколько их было, столько вернёт."""
        found = self.collection.get(where={"source": source}, include=["metadatas"])
        count = len(found.get("metadatas") or [])
        if count:
            self.collection.delete(where={"source": source})
        return count

    def source_counts(self) -> Dict[str, int]:
        """Сколько фрагментов в базе по каждому файлу (один проход)."""
        counts: Dict[str, int] = {}
        for meta in self.collection.get(include=["metadatas"]).get("metadatas") or []:
            name = (meta or {}).get("source", "неизвестно")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def clear_base(self) -> int:
        """Стирает базу целиком: книги на диске остаются; вернёт число фрагментов."""
        count = self.collection.count()
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata=COLLECTION_META)
        return count

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

    def process_research_folder(self, books_folder: str = "./books",
                                progress: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Индексирует все поддерживаемые файлы в папке (и в подпапках).

        Args:
            books_folder: Путь к папке с книгами агента (PDF, EPUB, TXT, FB2)
            progress: куда сообщать о ходе работы (интерфейс показывает это)
        """
        if progress is not None:
            self.progress = progress
        research_path = self._resolve_folder(books_folder)
        if not research_path.exists():
            print(f"❌ Папка {books_folder} не найдена")
            self._report(event="error", message=f"Папка {books_folder} не найдена")
            return False

        files, skipped = [], []
        for path in sorted(research_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in self.formats:
                files.append(path)
            else:
                skipped.append(path)
        if not files:
            print(f"❌ Книги не найдены в {books_folder} "
                  f"(поддерживаются: {', '.join(self.formats)})")
            self._report(event="error",
                         message="В папке агента нет книг, которые можно прочитать")
            return False

        print(f"🔍 Найдено файлов: {len(files)} "
              f"({', '.join(sorted({p.suffix.lower() for p in files}))})")
        if skipped:
            print(f"⏭️ Пропускаю файлы, формат которых не читаю: "
                  f"{len(skipped)} ({', '.join(sorted({p.suffix.lower() for p in skipped}))})")
        if self.tesseract is not None:
            print("🔎 Страницы без текста будут распознаны через tesseract "
                  "(это самая долгая часть)")
        self._report(event="files", total=len(files), skipped=len(skipped))

        successful_files = 0
        failed_files = []
        missing_files = []

        for i, file_path in enumerate(files, 1):
            print(f"\n📋 Обрабатываю файл {i}/{len(files)}: {file_path.name}")
            self._report(event="file", source=file_path.name, index=i, total=len(files))

            if not file_path.exists():
                # Книгу убрали из папки, пока шла индексация: это не сбой,
                # и в базу про неё ничего не попало.
                print(f"⏭️ Файла больше нет: {file_path.name}")
                missing_files.append(file_path.name)
                self._report(event="skipped", source=file_path.name)
                continue

            try:
                _set_alarm(300)  # ограничение времени на файл, только для UNIX
                try:
                    self.add_file_to_database(str(file_path))
                finally:
                    _set_alarm(0)
                successful_files += 1

            except TimeoutError:
                print(f"⏰ Превышен лимит времени для {file_path.name}")
                failed_files.append((file_path.name, "Таймаут"))
                self._report(event="failed", source=file_path.name, message="Таймаут")

            except Exception as e:
                print(f"❌ Ошибка обработки {file_path.name}: {e}")
                failed_files.append((file_path.name, str(e)))
                self._report(event="failed", source=file_path.name, message=str(e))

        # Итоговая статистика
        print("\n🎉 Обработка завершена!")
        print(f"✅ Успешно обработано файлов: {successful_files}/{len(files)}")

        if failed_files:
            print(f"❌ Не удалось обработать {len(failed_files)} файлов:")
            for file_name, error in failed_files:
                print(f"  - {file_name}: {error}")

        if missing_files:
            print(f"⏭️ Пропущены файлы, убранные из папки во время работы: "
                  f"{len(missing_files)}")
            for file_name in missing_files:
                print(f"  - {file_name}")

        total_docs = self.collection.count()
        print(f"📊 Всего документов в базе: {total_docs}")
        self._report(event="done", chunks=total_docs, files=len(files),
                     failed=len(failed_files), added=successful_files,
                     skipped=len(missing_files))

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
        ошиблась. Списки объединяются по Reciprocal Rank Fusion, но фрагмент,
        где совпали все слова вопроса, идёт впереди остальных.
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
        # По словам ищутся только основы длиной от трёх букв (см.
        # _lexical_candidates): столько слов может совпасть во фрагменте.
        searched = [word for word in words if len(self._stem(word)) >= 3]

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

        # Совпадение по всем словам вопроса важнее любого одного списка: в
        # техкарте есть и «борщ», и «московский», а в биографии шефа — только
        # «московский». Бонус больше любой суммы мест в RRF, поэтому такие
        # фрагменты всегда впереди; если совпадений по всем словам нет,
        # порядок прежний.
        if len(searched) > 1:
            for ident, item in items.items():
                if item.get('matched', 0) == len(searched):
                    scores[ident] += EXACT_MATCH_BONUS

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
        """Статистика базы: сколько фрагментов и из каких файлов.

        Источники собираются порциями: у больших баз один запрос за всеми
        метаданными займёт слишком много памяти.
        """
        count = self.collection.count()

        sources = set()
        for offset in range(0, count, 1000):
            batch = self.collection.get(include=["metadatas"], limit=1000, offset=offset)
            sources.update(meta.get('source', 'unknown') for meta in batch['metadatas'] if meta)

        return {
            'total_chunks': count,
            'sources': list(sources),
            'sources_count': len(sources),
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
        }


def build_agent_database(cfg, agent, model):
    """Открывает базу агента для индексации (модель эмбеддингов — общая)."""
    db = RAGDatabase(
        db_path=str(agent.db_dir),
        collection_name=BOOKS_COLLECTION,
        embedding_model=cfg.embeddings.model,
        chunk_size=cfg.store.chunk_size,
        chunk_overlap=cfg.store.chunk_overlap,
        formats=cfg.store.formats,
        ocr=cfg.store.ocr,
        embedding_model_instance=model,
    )
    db.lexical = cfg.search.lexical
    return db


def main(argv: Optional[List[str]] = None):
    """Индексирует агентов из консоли: python -m core.database [агент ...].

    Без аргументов индексирует всех агентов по очереди. Повторный запуск
    обновляет уже проиндексированные книги, а не добавляет их заново.
    Обычно это не нужно: то же самое делает кнопка «Индексировать» в ⚙.
    """
    from config import load_config
    from core.agents import Agents

    cfg = load_config()
    agents = Agents(cfg.store.agents_root)
    slugs = list(sys.argv[1:] if argv is None else argv)
    if not slugs:
        slugs = [agent.slug for agent in agents.list()]
    if not slugs:
        print(f"❌ В {cfg.store.agents_root} нет ни одного агента — создайте его в ⚙")
        return 1

    model = embedding_model_for(cfg.embeddings.model)
    ok = True
    for slug in slugs:
        agent = agents.get(slug)
        print(f"\n🚀 Агент «{agent.title}» ({slug}): {agent.books_dir}")
        rag_db = build_agent_database(cfg, agent, model)
        ok = rag_db.process_research_folder(str(agent.books_dir)) and ok
        stats = rag_db.get_database_stats()
        print(f"📊 В базе {stats['total_chunks']} фрагментов "
              f"из {stats['sources_count']} файлов")
        rag_db.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())