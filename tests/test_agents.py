"""Проверки агентов: папки, имена, книги (без базы и моделей)."""
import tempfile
import unittest
from pathlib import Path

from core.agents import AgentError, AgentNotFound, Agents


class CreateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)

    def test_agent_gets_latin_folder_with_books_and_database(self):
        agent = self.agents.create("Кулинария")
        self.assertEqual(agent.slug, "kulinariya")
        self.assertEqual(agent.title, "Кулинария")
        self.assertTrue(agent.books_dir.is_dir())
        self.assertTrue(agent.db_dir.is_dir())
        self.assertTrue((agent.path / "agent.json").is_file())

    def test_second_agent_with_same_title_gets_number(self):
        first = self.agents.create("Кулинария")
        second = self.agents.create("Кулинария")
        self.assertEqual(first.slug, "kulinariya")
        self.assertEqual(second.slug, "kulinariya-2")

    def test_title_keeps_cyrillic_and_folder_mixes_words(self):
        agent = self.agents.create("Программирование на Python")
        self.assertEqual(agent.slug, "programmirovanie-na-python")

    def test_title_without_letters_or_digits_is_rejected(self):
        for title in ("", "   ", "!!!", "???"):
            with self.subTest(title=title):
                with self.assertRaises(AgentError):
                    self.agents.create(title)

    def test_only_latin_characters_reach_the_folder_name(self):
        agent = self.agents.create("Книги — 2026 (том №1)")
        self.assertRegex(agent.slug, r"^[a-z0-9][a-z0-9_-]*$")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)

    def test_list_is_sorted_by_title_and_ignores_strangers(self):
        self.agents.create("Ягоды")
        self.agents.create("Борщ")
        Path(self.temp.name, "заметки.txt").write_text("не агент", encoding="utf-8")
        Path(self.temp.name, "чужое").mkdir()
        self.assertEqual([agent.title for agent in self.agents.list()], ["Борщ", "Ягоды"])

    def test_rename_changes_title_but_keeps_folder(self):
        agent = self.agents.create("Кулинария")
        renamed = self.agents.rename(agent.slug, "Кухня народов мира")
        self.assertEqual(renamed.title, "Кухня народов мира")
        self.assertEqual(renamed.slug, "kulinariya")
        self.assertEqual(self.agents.get("kulinariya").title, "Кухня народов мира")

    def test_empty_rename_is_rejected(self):
        agent = self.agents.create("Кулинария")
        with self.assertRaises(AgentError):
            self.agents.rename(agent.slug, "  ")
        self.assertEqual(self.agents.get(agent.slug).title, "Кулинария")

    def test_missing_agent_is_not_found(self):
        with self.assertRaises(AgentNotFound):
            self.agents.get("missing")
        self.assertFalse(self.agents.exists("missing"))

    def test_bad_folder_name_is_rejected(self):
        for slug in ("../секрет", "Кулинария", "a/b", "", "агент 1"):
            with self.subTest(slug=slug):
                with self.assertRaises(AgentError):
                    self.agents.get(slug)

    def test_delete_removes_folder_together_with_books_and_base(self):
        agent = self.agents.create("Кулинария")
        (agent.books_dir / "книга.txt").write_text("текст", encoding="utf-8")
        (agent.db_dir / "chroma.sqlite3").write_bytes(b"db")

        self.agents.delete(agent.slug)

        self.assertFalse(agent.path.exists())
        self.assertEqual(self.agents.list(), [])

    def test_broken_agent_json_falls_back_to_folder_name(self):
        agent = self.agents.create("Кулинария")
        (agent.path / "agent.json").write_text("не json", encoding="utf-8")
        self.assertEqual(self.agents.get(agent.slug).title, agent.slug)

    def test_size_counts_books_and_database(self):
        agent = self.agents.create("Кулинария")
        (agent.books_dir / "книга.txt").write_bytes(b"12345")
        (agent.db_dir / "chroma.sqlite3").write_bytes(b"123")
        self.assertEqual(agent.size(),
                         8 + (agent.path / "agent.json").stat().st_size)


class PromptTests(unittest.TestCase):
    """Промпт агента живёт в его папке и переживает перезапуск."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)
        self.agent = self.agents.create("Схемы", "даташиты разъёмов")

    def test_new_agent_has_no_prompt_of_its_own(self):
        self.assertEqual(self.agent.system_prompt, "")
        self.assertEqual(self.agent.as_dict()["prompt"], "")

    def test_saved_prompt_is_written_to_agent_json_and_read_back(self):
        self.agents.save_prompt(self.agent.slug, "  Ты — ассистент по схемам.  ")

        again = Agents(self.temp.name).get(self.agent.slug)
        self.assertEqual(again.system_prompt, "Ты — ассистент по схемам.")
        self.assertEqual(again.as_dict()["prompt"], "Ты — ассистент по схемам.")
        self.assertEqual(again.description, "даташиты разъёмов")

    def test_prompt_can_be_cleared_to_go_back_to_the_common_one(self):
        self.agents.save_prompt(self.agent.slug, "Свой промпт")
        self.agents.save_prompt(self.agent.slug, "   ")

        self.assertEqual(self.agents.get(self.agent.slug).system_prompt, "")

    def test_topic_is_updated_together_with_the_prompt(self):
        updated = self.agents.save_prompt(self.agent.slug, "Промпт", "пайка и разъёмы")

        self.assertEqual(updated.description, "пайка и разъёмы")
        self.assertEqual(self.agents.get(self.agent.slug).description, "пайка и разъёмы")

    def test_agent_json_without_prompt_falls_back_to_the_common_one(self):
        (self.agent.path / "agent.json").write_text(
            '{"title": "Схемы", "description": "даташиты"}', encoding="utf-8")

        self.assertEqual(self.agents.get(self.agent.slug).system_prompt, "")


class BookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)
        self.agent = self.agents.create("Кулинария")

    def test_uploaded_book_is_written_read_and_removed(self):
        info = self.agents.write_book(self.agent.slug, "Плов.pdf", [b"abc", b"de"])
        self.assertEqual(info["size"], 5)
        self.assertFalse(info["replaced"])
        self.assertTrue(info["supported"])

        books = self.agents.books(self.agent.slug)
        self.assertEqual([book["name"] for book in books], ["Плов.pdf"])
        self.assertTrue(self.agents.remove_book(self.agent.slug, "Плов.pdf"))
        self.assertFalse(self.agents.remove_book(self.agent.slug, "Плов.pdf"))
        self.assertEqual(self.agents.books(self.agent.slug), [])

    def test_replacing_book_keeps_single_file(self):
        self.agents.write_book(self.agent.slug, "книга.txt", [b"old"])
        info = self.agents.write_book(self.agent.slug, "книга.txt", [b"new content"])
        self.assertTrue(info["replaced"])
        self.assertEqual(len(self.agents.books(self.agent.slug)), 1)

    def test_empty_upload_is_rejected_and_leaves_no_file(self):
        with self.assertRaises(AgentError):
            self.agents.write_book(self.agent.slug, "пусто.txt", [b""])
        self.assertEqual(self.agents.books(self.agent.slug), [])

    def test_unsupported_format_is_listed_but_marked(self):
        self.agents.write_book(self.agent.slug, "книга.djvu", [b"x"])
        book = self.agents.books(self.agent.slug)[0]
        self.assertFalse(book["supported"])

    def test_names_outside_the_agent_folder_are_rejected(self):
        for name in ("../секрет.txt", r"..\секрет.txt", "books/../x.txt",
                     "подпапка/../../x.txt", "/абсолютный/путь.txt",
                     r"C:\Windows\x.txt", "", "   ", ".скрытый",
                     ".скрытая/книга.txt"):
            with self.subTest(name=name):
                with self.assertRaises(AgentError):
                    self.agents.book_path(self.agent.slug, name)

    def test_book_in_subfolder_is_listed_and_removed(self):
        """Книга в подпапке видна в интерфейсе путём и удаляется по нему.

        Индексация читает и подпапки, поэтому такая книга должна быть видна
        в списке — иначе она попадает в базу, не показываясь пользователю.
        """
        folder = self.agent.books_dir / "выпечка"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "плов.pdf").write_bytes(b"x")

        books = self.agents.books(self.agent.slug)
        self.assertEqual([book["name"] for book in books], ["выпечка/плов.pdf"])
        self.assertTrue(self.agents.remove_book(self.agent.slug, "выпечка/плов.pdf"))
        self.assertFalse((folder / "плов.pdf").exists())

    def test_surrounding_spaces_are_trimmed_from_names(self):
        self.agents.write_book(self.agent.slug, " книга.txt ", [b"x"])
        books = self.agents.books(self.agent.slug)
        self.assertEqual([book["name"] for book in books], ["книга.txt"])


if __name__ == '__main__':
    unittest.main()
