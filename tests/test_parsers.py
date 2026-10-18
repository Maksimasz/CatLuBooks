"""Проверки чтения файлов: PDF, EPUB, TXT, FB2 — без моделей и базы."""
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from core import parsers


class CleanTextTests(unittest.TestCase):
    def test_soft_hyphen_is_joined(self):
        self.assertEqual(parsers.clean_text('нарезать ку\u00ad рицу'), 'нарезать курицу')

    def test_word_broken_by_line_break_is_joined(self):
        self.assertIn('приготовления', parsers.clean_text('до пригото-\nвления'))

    def test_running_header_and_page_numbers_are_dropped(self):
        text = parsers.clean_text('38 | РЕЦЕПТЫ\nОбжарить лук.\n120\n')
        self.assertEqual(text, 'РЕЦЕПТЫ Обжарить лук.')

    def test_lines_are_joined_into_a_paragraph(self):
        self.assertEqual(parsers.clean_text('первая\nвторая'), 'первая вторая')

    def test_table_values_survive(self):
        """В технологических картах граммовки напечатаны отдельными строками."""
        text = parsers.clean_text('126\nНаименование сырья\nКартофель\n80\n60\n'
                                  'Морковь\n15\n12\n127')
        self.assertEqual(text, 'Наименование сырья Картофель 80 60 Морковь 15 12')


class NoiseTests(unittest.TestCase):
    RECIPE = ('Обжарить лук и морковь пять минут, добавить бульон и варить двадцать '
              'минут. Посолить по вкусу, положить лавровый лист и перец горошком. '
              'Подавать со сметаной и мелко нарубленной зеленью.')
    INDEX = ('177. Борщ с капустой 178. Борщ с черносливом 179. Борщ московский '
             '180. Борщ флотский 181. Борщ с фасолью 182. Борщ с грибами '
             '183. Борщ украинский 184. Борщ сибирский 185. Борщ полтавский '
             '186. Борщ с квашеной капустой 187. Борщ с картофелем '
             '188. Борщ с фрикадельками 189. Борщ летний 190. Борщ с грибами')
    TECH_CARD = (
        'РЕЦЕПТУРА 182, БОРЩ С ФАСОЛЬЮ И КАРТОФЕЛЕМ Расход сырья на 1 порцию, г '
        'Наименование сырья 1 и 1 брутто нетто брутто [ нето брутто нетто Свекла '
        '60 48 100 89 100 80 Картофель 40 30 67 50 67 50 Фасоль 12 12 20 20 20 20 '
        'Морковь 15 12 25 20 25 20 Петрушка (корень) 4 3 7 5 со В Лук репчатый '
        '14 12 24 20 24 20 Томат-пюре 9 9 15 15 15 15 Кулинарный жир 6 6 10 10 '
        '10 10 или маргарин 7,3 7,3 12,2 12,2 12,2 12,2 Чеснок 1,2 1 2 1,5 2 1,5 '
        'Сахар 2 2 В 3 К 3 Уксус 9% 1,65 1,65 2,64 2,64 2,64 2,64 Бульон или '
        'воде 240 240 400 400 400 400 Перец горошком 0,03 0,03 0,05 0,05 0,05 '
        '0,05 Соль 2 2 4 4 4 4 Сметана 10 10 10 10 10 10 Выход 300/10 500/10 500')

    def test_contents_page_is_listing(self):
        self.assertTrue(parsers.is_listing(
            'СУПЫ Борщ .......... 27 Гаспачо .......... 30 Суп ......... 31'))

    def test_numbered_index_is_listing(self):
        self.assertTrue(parsers.is_listing(self.INDEX))

    def test_index_without_leader_dots_is_listing(self):
        """Указатель без точек: «Клафути 118 Авокадо ...»."""
        self.assertTrue(parsers.is_listing(
            'Абрикосы Клафути 118 Авокадо Филе индейки 45 Аджика Гриль 62 '
            'Баранина с алычой 100 Кеци 103 Айва Баранья нога 191 Бекон 63 '
            'Брокколи 71 Говядина 88 Грибы 92 Капуста 19 Картофель 44 '
            'Курица 46 Лук 31 Морковь 55 Рыба 117 Свекла 82 Томаты 108 Ю'))

    def test_short_index_is_not_listing(self):
        """Пара номеров подряд — это ещё не указатель (может быть список шагов)."""
        self.assertFalse(parsers.is_listing('10. Обжарьте лук 11. Добавьте морковь'))

    def test_broken_glyphs_are_listing(self):
        self.assertTrue(parsers.is_listing(
            '/g299/g301/g297/g309/g15/g3/g298/g282/g285/g301/g3/g290/g3'))

    def test_recipe_is_not_listing(self):
        self.assertFalse(parsers.is_listing(self.RECIPE))

    def test_recipe_table_is_not_listing(self):
        """Техкарта с граммовками — не указатель, хотя чисел на ней много."""
        self.assertFalse(parsers.is_listing(self.TECH_CARD))

    def test_table_marker_wins_over_numbered_index(self):
        """Те же номера страниц, но со столбцами техкарты — это не указатель."""
        index_like = ' '.join(f'{100 + i}. Борщ {i}' for i in range(12))
        self.assertTrue(parsers.is_listing(index_like))
        self.assertFalse(parsers.is_listing(index_like + ' брутто нетто'))

    def test_recipe_table_is_kept(self):
        self.assertTrue(parsers.keep_section(self.TECH_CARD))

    def test_recipe_table_is_not_noise(self):
        """Техкарта — не служебный текст: её числа это граммовки, а не страницы.

        Пока таблица считалась «числовой», при поиске она уезжала в конец
        выдачи, и модель отвечала «в таблицах граммовок нет».
        """
        self.assertTrue(parsers.digits_heavy(self.TECH_CARD))
        self.assertFalse(parsers.is_noise(self.TECH_CARD))

    def test_title_list_without_numbers_is_listing(self):
        """Оглавление без номеров страниц: «Хоровац Гриль салат Террин …»."""
        self.assertTrue(parsers.is_listing(
            'Квас из копченой сливы Винегрет из печеных овощей Печеная картошка '
            'Закуска к вину из печеной свеклы Печеный лук Салат из печеных перцев '
            'Салат из свежих овощей с брынзой Свекла с орехами Морковь по-корейски '
            'Капуста с яблоками Грибы жареные Огурцы соленые Помидоры в собственном '
            'соку Баклажаны с чесноком Кабачки гриль Перец запеченный Лук маринованный'))

    def test_ingredient_list_is_not_title_list(self):
        """Строчки ингредиентов пишутся со строчной буквы — их не выбрасываем."""
        self.assertFalse(parsers.is_listing(
            'Ингредиенты мука 200 г масло сливочное 100 г сахар 150 г яйца 2 шт '
            'молоко 100 мл соль по вкусу сода пищевая щепотка ванилин по вкусу '
            'сметана 200 г творог 300 г изюм 50 г орехи грецкие 70 г мед 2 ст ложки '
            'перец черный молотый по вкусу корица молотая 1 ч ложка кефир 250 мл '
            'яблоки кислые 3 шт груши 2 шт слива 300 г абрикосы 200 г вишня 250 г'))

    def test_step_numbers_are_not_listing(self):
        steps = ' '.join(f'{number}. Обжарьте лук и тушите 20 минут.' for number in range(10, 16))
        self.assertFalse(parsers.is_listing(steps))

    def test_alphabetical_index_is_digits_heavy(self):
        self.assertTrue(parsers.digits_heavy(
            'Свекла 82 275 Свинина 151 202 205 240 Сельдерей 162 163 Сливки 37 178 '
            '180 238 Сметана 53 56 Соевый 16 17 Сумах 110 131 Сыр 177 180 Грибы '
            '12 13 14 15 Орехи 44 45 46 Рыба 90 91 92 Тесто 71 72 73 Мясо 44 45'))

    def test_recipe_is_not_digits_heavy(self):
        self.assertFalse(parsers.digits_heavy(
            'На 1 порцию: кабачки 249/200, масло 20/20, сметана 30/30, соль 4/4, '
            'обжарить и довести до готовности, подать с соусом.'))

    def test_latin_soup_is_garbled(self):
        self.assertTrue(parsers.looks_garbled(
            'БЫСТРЫЙ КУРИНЫЙ БУЛЬОН ' + ' '.join(['WJQ RQKI IZQW NVRT PLMO'] * 5)))

    def test_short_russian_page_is_not_garbled(self):
        self.assertFalse(parsers.looks_garbled('Супы Борщ Варить бульон двадцать минут'))

    def test_russian_text_is_not_garbled(self):
        self.assertFalse(parsers.looks_garbled(
            'Обжарить лук и морковь, добавить бульон и варить двадцать минут.'))

    def test_english_text_is_not_garbled(self):
        self.assertFalse(parsers.looks_garbled(
            'Heat the olive oil in a large pot, add the onion and cook for ten minutes.'))

    def test_recipe_is_kept(self):
        self.assertTrue(parsers.keep_section(self.RECIPE))

    def test_empty_page_is_dropped(self):
        self.assertFalse(parsers.keep_section('38 | РЕЦЕПТЫ'))
        self.assertFalse(parsers.keep_section(''))

    def test_contents_page_is_dropped(self):
        self.assertFalse(parsers.keep_section(
            'СУПЫ Борщ .......... 27 Гаспачо .......... 30 Суп ......... 31'))

    def test_search_noise_catches_short_marker(self):
        self.assertTrue(parsers.is_noise('Страница 205 фотография к рецепту'))
        self.assertFalse(parsers.is_noise(self.RECIPE))


class TextFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, name, data):
        path = Path(self.temp.name) / name
        path.write_bytes(data)
        return path

    def test_utf8_text_is_split_into_parts(self):
        path = self.write('книга.txt', ('абзац с текстом\n\n' * 200).encode('utf-8'))
        parts = list(parsers.read_txt(path, limit=500))
        self.assertGreater(len(parts), 1)
        self.assertEqual([number for number, _ in parts], list(range(1, len(parts) + 1)))
        self.assertTrue(all(len(text) <= 700 for _, text in parts))

    def test_cp1251_text_is_read(self):
        path = self.write('старое.txt', 'Борщ со сметаной'.encode('cp1251'))
        self.assertEqual(list(parsers.read_txt(path)), [(1, 'Борщ со сметаной')])

    def test_fb2_text_is_read(self):
        body = ('<?xml version="1.0" encoding="utf-8"?><FictionBook>'
                '<body><section><title><p>Супы</p></title>'
                '<p>Варить бульон.</p><p>Добавить лапшу.</p></section></body>'
                '<binary>AAAA</binary></FictionBook>')
        path = self.write('книга.fb2', body.encode('utf-8'))
        parts = list(parsers.read_fb2(path))
        self.assertEqual(parts, [(1, 'Супы\n\nВарить бульон.\n\nДобавить лапшу.')])

    def test_html_to_text_keeps_paragraphs(self):
        markup = ('<html><head><title>Нет</title><style>p{}</style></head>'
                  '<body><h1>Супы</h1><p>Варить <b>бульон</b>.</p><script>x()</script></body></html>')
        text = parsers.html_to_text(markup)
        self.assertIn('Супы', text)
        self.assertIn('бульон', text)
        self.assertNotIn('x()', text)
        self.assertNotIn('p{}', text)

    def test_unknown_format_is_reported(self):
        path = self.write('скан.djvu', b'x')
        with self.assertRaises(ValueError):
            list(parsers.read_sections(path))

    def test_unit_names(self):
        self.assertEqual(parsers.unit_for('a.pdf'), 'страница')
        self.assertEqual(parsers.unit_for('a.EPUB'), 'раздел')
        self.assertEqual(parsers.unit_for('a.txt'), 'часть')
        self.assertEqual(parsers.unit_for('a.fb2'), 'часть')


class EpubTests(unittest.TestCase):
    CONTAINER = ('<?xml version="1.0"?><container><rootfiles>'
                 '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
    OPF = ('<?xml version="1.0"?><package><manifest>'
           '<item id="c1" href="ch1.xhtml"/><item id="c2" href="text/ch2.xhtml"/>'
           '</manifest><spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'книга.epub'
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('META-INF/container.xml', self.CONTAINER)
            archive.writestr('OEBPS/content.opf', self.OPF)
            archive.writestr('OEBPS/ch1.xhtml', '<html><body><p>Первая глава</p></body></html>')
            archive.writestr('OEBPS/text/ch2.xhtml', '<html><body><p>Вторая глава</p></body></html>')

    def test_chapters_follow_spine_order(self):
        parts = list(parsers.read_epub(self.path))
        self.assertEqual([number for number, _ in parts], [1, 2])
        self.assertIn('Вторая глава', parts[0][1])
        self.assertIn('Первая глава', parts[1][1])


class FakePage:
    def __init__(self, text):
        self.text = text
        self.rendered = False

    def get_text(self):
        return self.text

    def get_pixmap(self, dpi=200):
        self.rendered = True
        return types.SimpleNamespace(tobytes=lambda fmt: b'png')


class FakeDocument:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.pages)


class PdfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'книга.pdf'
        self.path.write_bytes(b'placeholder')

    def read(self, pages, tess=None, opened=None):
        def open_document(path):
            document = FakeDocument(pages)
            if opened is not None:
                opened.append(document)
            return document
        with patch.object(parsers, 'pymupdf', types.SimpleNamespace(open=open_document)):
            return list(parsers.read_pdf(self.path, tess))

    def test_pages_are_read_one_by_one(self):
        pages = [FakePage('первая страница'), FakePage('вторая страница')]
        opened = []
        self.read(pages, opened=opened)
        self.assertTrue(opened[0].closed)  # файл закрывается
        self.assertEqual(pages[0].text, 'первая страница')

    def test_pages_without_text_are_recognized(self):
        pages = [FakePage(''), FakePage('Обжарить лук и морковь пять минут, добавить '
                                        'бульон, положить лавровый лист, чёрный перец '
                                        'горошком и варить двадцать минут, посолить '
                                        'по вкусу и подать со сметаной.')]
        class FakeOcr:
            available = True
            dpi = 200
            def __init__(self):
                self.images = 0
            def texts(self, images):
                self.images += len(images)
                return ['РАСПОЗНАНО НА КАРТИНКЕ' for _ in images]
        tess = FakeOcr()
        result = self.read(pages, tess)
        self.assertEqual(tess.images, 1)          # распознавали только пустую страницу
        self.assertEqual(result[0], (1, 'РАСПОЗНАНО НА КАРТИНКЕ'))
        self.assertFalse(pages[1].rendered)

    def test_without_ocr_pages_are_read_as_is(self):
        pages = [FakePage(''), FakePage('текст')]
        self.assertEqual(self.read(pages), [(1, ''), (2, 'текст')])


if __name__ == '__main__':
    unittest.main()
