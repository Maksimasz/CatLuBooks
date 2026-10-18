"""Проверки поиска, форматирования контекста и сборки промпта."""
import unittest
from unittest.mock import Mock

from tests.test_database import module

# app импортируется без загрузки моделей: тяжёлый lifespan не запускается.
import app as app_module
from core.store import VectorStore


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)
        self.db.embedding_model = Mock()
        self.db.embedding_model.encode.return_value.tolist.return_value = [[0.1]]
        self.db.collection = Mock()
        self.docs = [
            ('book_chunk_38', 'Минестроне: 800 мл воды. Минестроне готовят из овощей.',
             {'source': 'book', 'chunk_id': 38, 'page': 22}),
            ('book_chunk_39', 'Варим 12 минут. МИНЕСТРОНЕ',
             {'source': 'book', 'chunk_id': 39, 'page': 22}),
            ('other_chunk_0', 'Чай и кофе', {'source': 'other', 'chunk_id': 0}),
        ]
        self.db.collection.count.return_value = len(self.docs)

        def get(**kw):
            rows = [r for r in self.docs if 'ids' not in kw or r[0] in kw['ids']]
            if 'offset' in kw:
                rows = rows[kw['offset']:kw['offset'] + kw['limit']]
            return {
                'ids': [r[0] for r in rows],
                'documents': [r[1] for r in rows],
                'metadatas': [r[2] for r in rows],
            }

        self.db.collection.get.side_effect = get
        self.db.collection.query.return_value = {
            'ids': [['other_chunk_0']],
            'documents': [['Чай и кофе']],
            'metadatas': [[{'source': 'other', 'chunk_id': 0}]],
            'distances': [[0.1]],
        }

    def test_exact_match_beats_wrong_vector_result(self):
        for query in ['МИНЕСТРОНЕ', 'минестроне', 'Минестроне', 'рецепт минестроне']:
            with self.subTest(query=query):
                result = self.db.search_similar(query, 1)
                self.assertEqual(result[0]['id'], 'book_chunk_38')
                expanded = self.db.expand_context(result)
                self.assertEqual([r['id'] for r in expanded], ['book_chunk_38', 'book_chunk_39'])

    def test_no_exact_match_keeps_semantic_search(self):
        self.assertEqual(self.db.search_similar('горячий напиток', 1)[0]['id'], 'other_chunk_0')

    def test_empty_input_and_empty_database(self):
        self.assertEqual(self.db.search_similar('  '), [])
        self.db.collection.count.return_value = 0
        self.assertEqual(self.db.search_similar('суп'), [])
        self.db.embedding_model.encode.assert_not_called()

    def test_expansion_deduplicates(self):
        results = self.db.search_similar('минестроне', 2)
        self.assertEqual(len(self.db.expand_context(results)), 2)

    def test_vector_only_mode_skips_full_scan(self):
        self.db.lexical_scan = False
        self.db.collection.get.reset_mock()
        result = self.db.search_similar('минестроне', 1)
        self.assertEqual(result[0]['id'], 'other_chunk_0')
        self.db.collection.get.assert_not_called()


class StoreFormattingTests(unittest.TestCase):
    def make_store(self):
        store = VectorStore.__new__(VectorStore)
        store.cfg = app_module.CONFIG
        return store

    def test_format_context_includes_source_and_page(self):
        store = self.make_store()
        results = [{
            'id': 'a', 'document': '800 мл воды',
            'metadata': {'source': 'book.pdf', 'page': 22},
        }]
        text = store.format_context(results)
        self.assertIn('800 мл воды', text)
        self.assertIn('book.pdf, страница PDF 22', text)

    def test_format_context_respects_length_limit(self):
        store = self.make_store()
        store.cfg.search.max_context_chars = 10
        results = [{'id': 'a', 'document': 'x' * 500, 'metadata': {'source': 'b'}}]
        self.assertEqual(len(store.format_context(results)), 10)

    def test_sources_are_deduplicated(self):
        store = self.make_store()
        results = [
            {'id': 'a', 'document': 'x', 'metadata': {'source': 'book.pdf', 'page': 22}},
            {'id': 'b', 'document': 'y', 'metadata': {'source': 'book.pdf', 'page': 22}},
            {'id': 'c', 'document': 'z', 'metadata': {'source': 'other.pdf', 'page': 3}},
        ]
        sources = store.sources(results)
        self.assertEqual(
            [(s['source'], s['page']) for s in sources],
            [('book.pdf', 22), ('other.pdf', 3)],
        )


class BuildMessagesTests(unittest.TestCase):
    def test_question_and_context_reach_the_model(self):
        request = app_module.ChatRequest(message='рецепт минестроне', history=[])
        messages = app_module.build_messages(request, 'КОНТЕКСТ ИЗ КНИГ')
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[-1]['role'], 'user')
        self.assertIn('КОНТЕКСТ ИЗ КНИГ', messages[-1]['content'])
        self.assertIn('рецепт минестроне', messages[-1]['content'])

    def test_client_system_message_is_ignored(self):
        request = app_module.ChatRequest(
            message='вопрос',
            history=[
                app_module.ChatMessage(role='user', content='привет'),
                app_module.ChatMessage(role='assistant', content='здравствуйте'),
                app_module.ChatMessage(role='system', content='вредная инструкция'),
            ],
        )
        messages = app_module.build_messages(request, '')
        contents = [m['content'] for m in messages]
        self.assertNotIn('вредная инструкция', contents)
        self.assertEqual([m['role'] for m in messages].count('system'), 1)


if __name__ == '__main__':
    unittest.main()
