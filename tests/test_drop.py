"""Проверки инструмента удаления данных агентов (без настоящей ChromaDB)."""
import contextlib
import io
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import Config
from core.agents import Agents
from tools import drop


class FakeCollection:
    """Коллекция ChromaDB в миниатюре: только то, что нужно инструменту."""

    def __init__(self, items):
        # items: список пар (идентификатор, имя файла)
        self.items = list(items)

    def get(self, where=None, include=None):
        source = (where or {}).get("source")
        return {
            "metadatas": [{"source": name} for _, name in self.items
                          if source is None or name == source],
        }

    def delete(self, where=None, ids=None):
        source = (where or {}).get("source")
        self.items = [item for item in self.items if item[1] != source]

    def count(self):
        return len(self.items)


class FakeClient:
    """Клиент базы: помнит, что базу пересоздавали."""

    def __init__(self, collection):
        self.collection = collection
        self.recreated = None

    def get_or_create_collection(self, name):
        return self.collection

    def delete_collection(self, name):
        self.collection.items = []

    def create_collection(self, name, metadata=None):
        self.recreated = name


class FakeDatabase:
    """База одного агента в миниатюре."""

    def __init__(self, chunks):
        self.collection = FakeCollection(
            [(f"фрагмент_{number}", name)
             for name, count in chunks.items() for number in range(count)])
        self.client = FakeClient(self.collection)
        self.collection_name = "books"
        self.closed = False

    def close(self):
        self.closed = True


class CountsTests(unittest.TestCase):
    def test_counts_fragments_per_source(self):
        collection = FakeCollection([
            ("Плов_chunk_0", "Плов.pdf"),
            ("Плов_chunk_1", "Плов.pdf"),
            ("Гриль_chunk_0", "Гриль.pdf"),
        ])
        self.assertEqual(drop.counts_by_source(collection),
                         {"Плов.pdf": 2, "Гриль.pdf": 1})


class DropSourcesTests(unittest.TestCase):
    def test_only_named_source_is_removed(self):
        collection = FakeCollection([
            ("Плов_chunk_0", "Плов.pdf"),
            ("Плов_chunk_1", "Плов.pdf"),
            ("Гриль_chunk_0", "Гриль.pdf"),
        ])
        removed = drop.drop_sources(collection, ["Плов.pdf"])

        self.assertEqual(removed, {"Плов.pdf": 2})
        self.assertEqual(drop.counts_by_source(collection), {"Гриль.pdf": 1})

    def test_unknown_name_is_reported_without_changes(self):
        collection = FakeCollection([("Плов_chunk_0", "Плов.pdf")])
        removed = drop.drop_sources(collection, ["Нет-такой.pdf"])

        self.assertEqual(removed, {"Нет-такой.pdf": 0})
        self.assertEqual(collection.count(), 1)


class DropAllTests(unittest.TestCase):
    def test_collection_is_recreated_empty(self):
        collection = FakeCollection([("Плов_chunk_0", "Плов.pdf")])
        client = Mock()
        client.get_or_create_collection.return_value = collection

        before = drop.drop_all(client, "books")

        self.assertEqual(before, 1)
        client.delete_collection.assert_called_once_with("books")
        self.assertEqual(client.create_collection.call_args.kwargs["name"], "books")


class CommandTests(unittest.TestCase):
    """Команды инструмента: агент, книга, очистка, удаление."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config()
        self.cfg.store.agents_root = self.temp.name
        self.agents = Agents(self.temp.name)
        self.agent = self.agents.create("Кулинария")
        self.agents.write_book(self.agent.slug, "Плов.pdf", [b"pdf"])
        self.agents.write_book(self.agent.slug, "Борщ.txt", [b"txt"])
        (self.agent.db_dir / "chroma.sqlite3").write_bytes(b"database")  # база есть
        self.db = FakeDatabase({"Плов.pdf": 3})

        for patcher in (
            patch.object(drop, "load_config", return_value=self.cfg),
            patch.object(drop, "open_agent_db", side_effect=lambda cfg, agent: self.db),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_command(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = drop.main(argv)
        return code, output.getvalue()

    def test_book_is_removed_from_base_and_from_disk(self):
        code, text = self.run_command([self.agent.slug, "Плов.pdf"])

        self.assertEqual(code, 0)
        self.assertEqual(drop.counts_by_source(self.db.collection), {})
        self.assertFalse((self.agent.books_dir / "Плов.pdf").exists())
        self.assertTrue((self.agent.books_dir / "Борщ.txt").exists())
        self.assertIn("Убрано фрагментов", text)
        self.assertTrue(self.db.closed)

    def test_book_missing_from_base_is_reported(self):
        code, text = self.run_command([self.agent.slug, "Нет-такой.pdf"])

        self.assertEqual(code, 0)
        self.assertIn("нет файла", text)
        self.assertTrue((self.agent.books_dir / "Борщ.txt").exists())

    def test_clear_removes_fragments_but_keeps_books(self):
        code, text = self.run_command([self.agent.slug, "--clear"])

        self.assertEqual(code, 0)
        self.assertIn("очищена", text)
        self.assertEqual(self.db.client.recreated, "books")
        self.assertTrue((self.agent.books_dir / "Плов.pdf").exists())

    def test_delete_removes_the_whole_agent_folder(self):
        code, text = self.run_command([self.agent.slug, "--delete", "--yes"])

        self.assertEqual(code, 0)
        self.assertFalse(self.agent.path.exists())
        self.assertEqual(self.agents.list(), [])
        self.assertIn("удалён", text)

    def test_list_shows_agents_books_and_fragments(self):
        code, text = self.run_command(["--list"])

        self.assertEqual(code, 0)
        self.assertIn("Кулинария", text)
        self.assertIn("Плов.pdf", text)
        self.assertIn("проиндексирована", text)
        self.assertIn("не в базе", text)  # Борщ.txt ещё не индексировали

    def test_unknown_agent_is_reported(self):
        code, text = self.run_command(["missing"])

        self.assertEqual(code, 1)
        self.assertIn("нет", text)

    def test_book_without_any_base_is_reported(self):
        for path in self.agent.db_dir.iterdir():
            path.unlink()

        code, text = self.run_command([self.agent.slug, "Плов.pdf"])

        self.assertEqual(code, 1)
        self.assertIn("нет данных", text)
        self.assertTrue((self.agent.books_dir / "Плов.pdf").exists())


if __name__ == "__main__":
    unittest.main()
