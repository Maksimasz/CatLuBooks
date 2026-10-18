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
        # «Неверный» векторный результат: нужный фрагмент ищем словами запроса.
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

    def test_lexical_can_be_disabled(self):
        self.db.lexical = False
        self.db.collection.get.reset_mock()
        result = self.db.search_similar('минестроне', 1)
        self.assertEqual(result[0]['id'], 'other_chunk_0')
        self.db.collection.get.assert_not_called()

    def test_exact_search_uses_indexed_filter(self):
        self.db.search_similar('минестроне', 2)
        kwargs = self.db.collection.get.call_args.kwargs
        self.assertIn('where_document', kwargs)

    def test_stem_collapses_russian_endings(self):
        for word, expected in [
            ('бедрышки', 'бедры'), ('аджике', 'аджик'),
            ('меде', 'мед'), ('плов', 'плов'), ('чай', 'чай'),
        ]:
            with self.subTest(word=word):
                self.assertEqual(self.db._stem(word), expected)

    def test_word_form_match_finds_other_case(self):
        # В запросе «бёдрышки», в тексте — «бёдрышек»: основа должна совпасть.
        self.docs.append(('drum_chunk_1', 'Маринад для бёдрышек с аджикой и мёдом',
                          {'source': 'drum', 'chunk_id': 1, 'page': 5}))
        self.db.collection.count.return_value = len(self.docs)
        result = self.db.search_similar('бёдрышки', 1)
        self.assertEqual(result[0]['id'], 'drum_chunk_1')


class RankingTests(unittest.TestCase):
    """Ранжирование: рецепт важнее оглавления, где слово лишь упомянуто."""

    def setUp(self):
        self.db = module.RAGDatabase.__new__(module.RAGDatabase)

    def test_noise_detection(self):
        self.assertTrue(self.db._is_noise(
            'СУПЫ Борщ .......... 27 Гаспачо .......... 30 Суп ......... 31'))
        self.assertTrue(self.db._is_noise(
            '177. Борщ с капустой 178. Борщ с черносливом 179. Борщ московский '
            '180. Борщ флотский 181. Борщ с фасолью 182. Борщ с грибами '
            '183. Борщ украинский 184. Борщ сибирский 185. Борщ полтавский '
            '186. Борщ с квашеной капустой 187. Борщ с картофелем '
            '188. Борщ с фрикадельками 189. Борщ летний 190. Борщ с грибами'))
        self.assertTrue(self.db._is_noise(
            'Свекла 82 275 Свинина 151 202 205 240 Сельдерей 162 163 Сливки 37 178 '
            '180 238 Сметана 53 56 Соевый 16 17 Сумах 110 131 Сыр 177 180 Грибы '
            '12 13 14 15 Орехи 44 45 46 Рыба 90 91 92'))
        self.assertTrue(self.db._is_noise('--- Страница 205 --- Фото к рецепту'))
        self.assertTrue(self.db._is_noise('/g299/g301/g297/g309/g15/g3/g298/g282/g285/g301/g3'))
        self.assertFalse(self.db._is_noise('Обжарить лук 5 минут, добавить морковь.'))
        self.assertFalse(self.db._is_noise(
            '1. Подготовка мяса: залейте язык водой и варите 10 минут. '
            '2. Тушение: добавьте морковь и тушите 2 часа.'))

    def test_recipe_outranks_contents_page(self):
        # В оглавлении слов запроса больше, но толку от него нет.
        candidates = {
            'toc': {
                'id': 'toc', 'metadata': {}, 'distance': None, 'matched': 3,
                'weight': 0.01, 'count': 9,
                'document': 'СУПЫ Куриный суп-лапша ......... 38 Суп гороховый ......... 31 '
                            'Суп борщ ......... 27',
            },
            'recipe': {
                'id': 'recipe', 'metadata': {}, 'distance': None, 'matched': 3,
                'weight': 0.01, 'count': 3,
                'document': 'КУРИНЫЙ СУП-ЛАПША ИНГРЕДИЕНТЫ: 3 луковицы '
                            'ПРИГОТОВЛЕНИЕ: варить 45 минут',
            },
        }
        ranked = self.db._rank_lexical(candidates, 'куриный суп лапша')
        self.assertEqual(ranked[0]['id'], 'recipe')

    def test_short_precise_chunk_beats_long_one(self):
        # Совпадений поровну — выигрывает более короткий, точечный фрагмент.
        candidates = {
            'long': {
                'id': 'long', 'metadata': {}, 'distance': None, 'matched': 2,
                'weight': 0.01, 'count': 4, 'document': 'Плов и вода и соль ' * 100,
            },
            'short': {
                'id': 'short', 'metadata': {}, 'distance': None, 'matched': 2,
                'weight': 0.01, 'count': 4, 'document': 'Плов с водой: рецепт',
            },
        }
        ranked = self.db._rank_lexical(candidates, 'плов вода')
        self.assertEqual(ranked[0]['id'], 'short')


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
        self.assertIn('book.pdf, страница 22', text)

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
