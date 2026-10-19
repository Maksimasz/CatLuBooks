"""Автономные проверки импорта: без Ollama, моделей и рабочей базы."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

# Подменяем только тяжёлые зависимости; исполняем настоящий модуль целиком.
DATABASE_PATH = Path(__file__).resolve().parents[1] / 'core' / 'database.py'
spec = importlib.util.spec_from_file_location('rag_database_under_test', DATABASE_PATH)
module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {
    'chromadb': types.SimpleNamespace(),
    'sentence_transformers': types.SimpleNamespace(SentenceTransformer=Mock()),
}):
    spec.loader.exec_module(module)


def book_page(marker, words=60):
    """Страница, похожая на настоящую: обычные слова, а не «xxxx»."""
    return f"Рецепт {marker}. " + " ".join(f"ингредиент{number}" for number in range(words))


class StreamingImportTests(unittest.TestCase):
    def setUp(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)
        self.db.collection = Mock()
        self.db.embedding_model = Mock()
        self.db.embedding_model.encode.side_effect = lambda chunks, **kw: types.SimpleNamespace(
            tolist=lambda: [[len(chunk)] for chunk in chunks]
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pdf = Path(self.temp.name) / 'book.pdf'
        self.pdf.write_bytes(b'placeholder')

    def import_book(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.db.add_file_to_database(str(self.pdf))

    def record_batches(self):
        """Запоминает, что реально ушло в базу: списки в моках очищаются."""
        documents, metadatas = [], []
        def save(**data):
            documents.extend(data['documents'])
            metadatas.extend(data['metadatas'])
        self.db.collection.upsert.side_effect = save
        return documents, metadatas

    def test_short_sentence_does_not_loop(self):
        text = 'Short. ' + 'x' * 5000
        chunks = self.db.split_text_into_chunks(text)
        self.assertLess(len(chunks), 10)
        self.assertTrue(chunks[0].startswith('Short.'))
        self.assertEqual(chunks[-1][-200:], text[-200:])

    def test_overlap_preserves_text(self):
        text = ''.join(str(i % 10) for i in range(5000))
        chunks = self.db.split_text_into_chunks(text)
        self.assertEqual(chunks[0] + ''.join(c[200:] for c in chunks[1:]), text)
        self.assertEqual(self.db.split_text_into_chunks(''), [])

    def test_invalid_chunk_parameters(self):
        for size, overlap in [(0, 0), (100, -1), (100, 100), (100, 200)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                self.db.split_text_into_chunks('text', size, overlap)

    def test_batches_are_saved_before_next_pages_are_read(self):
        stored = []
        def save(**data):
            self.assertLessEqual(len(data['documents']), 32)
            for doc, vector, meta, chunk_id in zip(data['documents'], data['embeddings'], data['metadatas'], data['ids']):
                self.assertEqual(vector, [len(doc)])
                self.assertEqual(meta['chunk_id'], len(stored))
                self.assertEqual(chunk_id, f'book.pdf_chunk_{len(stored)}')
                stored.append(doc)
        self.db.collection.upsert.side_effect = save
        def pages():
            for i in range(65):
                if i == 32:
                    self.assertEqual(len(stored), 32)
                yield i + 1, book_page(i)
        self.db.iter_sections = lambda path: pages()
        self.import_book()
        self.assertEqual(len(stored), 65)
        self.assertEqual(self.db.collection.upsert.call_count, 3)
        self.db.collection.delete.assert_called_once_with(where={'$and': [
            {'source': 'book.pdf'}, {'chunk_id': {'$gte': 65}}
        ]})

    def test_failure_stops_import_without_deleting_old_tail(self):
        self.db.iter_sections = lambda path: ((i, book_page(i)) for i in range(100))
        calls = 0
        def encode(chunks, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('encoding failed')
            return types.SimpleNamespace(tolist=lambda: [[1]] * len(chunks))
        self.db.embedding_model.encode.side_effect = encode
        with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
            self.import_book()
        self.assertEqual(self.db.collection.upsert.call_count, 1)
        self.db.collection.delete.assert_not_called()

    def test_storage_error_is_propagated(self):
        self.db.iter_sections = lambda path: iter([(1, book_page('первый'))])
        self.db.collection.upsert.side_effect = RuntimeError('disk full')
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            self.import_book()
        self.db.collection.delete.assert_not_called()

    def test_empty_file_is_not_successful(self):
        self.db.iter_sections = lambda path: iter([])
        with self.assertRaises(ValueError):
            self.import_book()
        self.db.collection.upsert.assert_not_called()
        self.db.collection.delete.assert_not_called()

    def test_junk_pages_are_not_indexed(self):
        """Оглавление и указатель в базу не пишутся, а рецепт — пишется."""
        pages = [
            (4, 'СУПЫ Борщ .......... 27 Гаспачо .......... 30 Суп ......... 31'),
            (5, '177. Борщ с капустой 178. Борщ с черносливом 179. Борщ московский '
                '180. Борщ флотский 181. Борщ с фасолью 182. Борщ с грибами '
                '183. Борщ украинский 184. Борщ сибирский 185. Борщ полтавский '
                '186. Борщ с квашеной капустой 187. Борщ с картофелем '
                '188. Борщ с фрикадельками 189. Борщ летний'),
            (38, book_page('куриный суп-лапша')),
        ]
        self.db.iter_sections = lambda path: iter(pages)
        documents, _ = self.record_batches()
        self.import_book()
        self.assertEqual(len(documents), 1)
        self.assertIn('куриный суп-лапша', documents[0])

    def test_page_number_is_stored_as_metadata(self):
        self.db.iter_sections = lambda path: iter([(38, book_page('борщ'))])
        _, metadatas = self.record_batches()
        self.import_book()
        self.assertEqual(metadatas[0]['page'], 38)
        self.assertEqual(metadatas[0]['source'], 'book.pdf')
        self.assertEqual(metadatas[0]['unit'], 'страница')

    def test_reimport_replaces_chunks_and_removes_old_tail(self):
        records = {}
        def upsert(**data):
            records.update(zip(data['ids'], data['documents']))
        def delete(where):
            cutoff = where['$and'][1]['chunk_id']['$gte']
            for key in list(records):
                if int(key.rsplit('_', 1)[1]) >= cutoff:
                    del records[key]
        self.db.collection.upsert.side_effect = upsert
        self.db.collection.delete.side_effect = delete
        self.db.iter_sections = lambda path: ((i, book_page(f'старый{i}')) for i in range(65))
        self.import_book()
        self.assertEqual(len(records), 65)
        self.db.iter_sections = lambda path: iter([(1, book_page('новый'))])
        self.import_book()
        self.assertEqual(list(records), ['book.pdf_chunk_0'])
        self.assertIn('новый', records['book.pdf_chunk_0'])

    def test_stats_reads_only_metadata_in_batches(self):
        self.db.collection_name = 'test'
        self.db.embedding_model_name = 'test-model'
        self.db.collection.count.return_value = 2001
        self.db.collection.get.side_effect = lambda **kwargs: {'metadatas': [{'source': 'book.pdf'}]}
        stats = self.db.get_database_stats()
        self.assertEqual(stats['total_chunks'], 2001)
        self.assertEqual(stats['sources'], ['book.pdf'])
        self.assertEqual([c.kwargs for c in self.db.collection.get.call_args_list], [
            {'include': ['metadatas'], 'limit': 1000, 'offset': offset}
            for offset in [0, 1000, 2000]
        ])


class InitTests(unittest.TestCase):
    """Открытие базы: существующая коллекция переиспользуется, не пересоздаётся."""

    def test_collection_is_opened_or_created_with_description(self):
        client = Mock()
        collection = client.get_or_create_collection.return_value
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.chromadb, "PersistentClient",
                              return_value=client, create=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    db = module.RAGDatabase(db_path=folder, collection_name="книги",
                                            embedding_model="test", ocr="off")

        self.assertIs(db.collection, collection)
        self.assertIsNone(db.tesseract)  # ocr: off — распознавание не поднимаем
        client.get_or_create_collection.assert_called_once_with(
            name="книги", metadata=module.COLLECTION_META)

    def test_shared_embedding_model_is_reused_without_loading(self):
        """Модель, загруженную для другого агента, повторно не грузим."""
        model = Mock()
        client = Mock()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.chromadb, "PersistentClient",
                              return_value=client, create=True):
                with patch.object(module, "SentenceTransformer") as loader:
                    with contextlib.redirect_stdout(io.StringIO()):
                        db = module.RAGDatabase(
                            db_path=folder, collection_name="книги",
                            embedding_model="test", ocr="off",
                            embedding_model_instance=model)
                    self.assertIs(db.embedding_model, model)
                    loader.assert_not_called()

    def test_embedding_model_loads_once_for_everyone(self):
        """Модель эмбеддингов одна на процесс: второй агент её не перезагружает."""
        client = Mock()
        module.MODELS.clear()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.chromadb, "PersistentClient",
                              return_value=client, create=True):
                with patch.object(module, "SentenceTransformer") as loader:
                    with contextlib.redirect_stdout(io.StringIO()):
                        first = module.RAGDatabase(
                            db_path=folder, collection_name="книги",
                            embedding_model="test", ocr="off")
                        second = module.RAGDatabase(
                            db_path=folder, collection_name="книги",
                            embedding_model="test", ocr="off")
                        self.assertIs(first.embedding_model, second.embedding_model)
                    loader.assert_called_once_with("test")
        module.MODELS.clear()

    def test_base_can_be_cleared_and_source_deleted(self):
        client = Mock()
        collection = client.get_or_create_collection.return_value
        collection.count.return_value = 12
        collection.get.return_value = {"metadatas": [{"source": "книга.pdf"}] * 7}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.chromadb, "PersistentClient",
                              return_value=client, create=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    db = module.RAGDatabase(db_path=folder, collection_name="книги",
                                            embedding_model="test", ocr="off")

        self.assertEqual(db.delete_source("книга.pdf"), 7)
        collection.delete.assert_called_once_with(where={"source": "книга.pdf"})

        self.assertEqual(db.clear_base(), 12)
        client.delete_collection.assert_called_once_with("книги")
        client.create_collection.assert_called_once_with(
            name="книги", metadata=module.COLLECTION_META)

    def test_source_counts_groups_chunks_by_file(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)
        self.db.collection = Mock()
        self.db.collection.get.return_value = {"metadatas": [
            {"source": "Плов.pdf"}, {"source": "Плов.pdf"}, {"source": "Борщ.pdf"},
            {},
        ]}
        self.assertEqual(self.db.source_counts(),
                         {"Плов.pdf": 2, "Борщ.pdf": 1, "неизвестно": 1})


class FolderTests(unittest.TestCase):
    def setUp(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)
        self.db.collection = Mock()
        self.db.formats = ['.pdf', '.epub', '.txt']
        self.db.tesseract = None
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_folder_reports_failed_file(self):
        Path(self.temp.name, 'broken.pdf').write_bytes(b'x')
        self.db.add_file_to_database = Mock(side_effect=RuntimeError('broken PDF'))
        self.db.collection.count.return_value = 0
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.db.process_research_folder(self.temp.name)
        self.assertFalse(result)
        self.assertIn('0/1', output.getvalue())

    def test_folder_reads_supported_formats_and_skips_others(self):
        for name in ('книга.epub', 'заметки.txt', 'рецепты.pdf', 'скан.djvu'):
            Path(self.temp.name, name).write_bytes(b'x')
        calls = []
        self.db.add_file_to_database = lambda path: calls.append(Path(path).name)
        self.db.collection.count.return_value = 0
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.db.process_research_folder(self.temp.name)
        self.assertEqual(sorted(calls), ['заметки.txt', 'книга.epub', 'рецепты.pdf'])
        self.assertIn('.djvu', output.getvalue())

    def test_file_removed_from_folder_is_skipped_not_failed(self):
        """Книгу убрали из папки, пока шла индексация: это пропуск, а не сбой."""
        for name in ('а.txt', 'б.txt'):
            Path(self.temp.name, name).write_bytes(b'x')
        removed = Path(self.temp.name, 'б.txt')

        def read(path):
            removed.unlink()      # пока читалась первая книга, вторую убрали
            return None

        self.db.add_file_to_database = read
        self.db.collection.count.return_value = 10
        events = []
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.db.process_research_folder(
                self.temp.name, progress=events.append)

        self.assertTrue(result)
        self.assertIn('Пропущены файлы', output.getvalue())
        self.assertEqual([event['event'] for event in events],
                         ['files', 'file', 'file', 'skipped', 'done'])
        self.assertEqual(events[-1]['skipped'], 1)
        self.assertEqual(events[-1]['failed'], 0)

    def test_empty_folder_is_reported(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.db.process_research_folder(self.temp.name)
        self.assertFalse(result)
        self.assertIn('не найдены', output.getvalue())

    def test_progress_reports_every_file_and_finish(self):
        """Ход индексации уходит в интерфейс: файлы, текущий файл, итог."""
        for name in ('а.txt', 'б.txt'):
            Path(self.temp.name, name).write_bytes(b'x')
        self.db.add_file_to_database = Mock()
        self.db.collection.count.return_value = 42
        events = []
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.db.process_research_folder(
                self.temp.name, progress=events.append)

        self.assertTrue(result)
        self.assertEqual([event['event'] for event in events],
                         ['files', 'file', 'file', 'done'])
        self.assertEqual(events[0]['total'], 2)
        self.assertEqual(events[1]['source'], 'а.txt')
        self.assertEqual(events[-1]['chunks'], 42)


class OcrSwitchTests(unittest.TestCase):
    """Распознавание сканов переключается у уже открытой базы."""

    def setUp(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)

    def test_tesseract_appears_and_disappears_on_demand(self):
        engine = Mock(available=True)
        with patch.object(module.parsers, 'Tesseract',
                          return_value=engine) as factory:
            self.db.set_ocr('auto')
            self.assertIs(self.db.tesseract, engine)
            self.assertEqual(self.db.ocr, 'auto')

            self.db.set_ocr('off')

        self.assertIsNone(self.db.tesseract)
        self.assertEqual(self.db.ocr, 'off')
        factory.assert_called_once()          # второй раз движок не создаём

    def test_enabling_without_tesseract_does_not_break(self):
        with patch.object(module.parsers, 'Tesseract',
                          return_value=Mock(available=False)):
            self.db.set_ocr('auto')

        self.assertIsNone(self.db.tesseract)

    def test_yaml_off_switches_ocr_off(self):
        """`ocr: off` из config.yaml приходит как False: OCR должен выключиться."""
        with patch.object(module.parsers, 'Tesseract') as factory:
            self.db.set_ocr(False)

        self.assertEqual(self.db.ocr, 'off')
        self.assertIsNone(self.db.tesseract)
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
