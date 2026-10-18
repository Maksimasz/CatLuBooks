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
                '183. Борщ украинский 184. Борщ сибирский'),
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

    def test_empty_folder_is_reported(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.db.process_research_folder(self.temp.name)
        self.assertFalse(result)
        self.assertIn('не найдены', output.getvalue())


if __name__ == '__main__':
    unittest.main()
