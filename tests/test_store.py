"""Проверки реестра баз агентов: ChromaDB и модель эмбеддингов подменены."""
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import Config
from core.agents import Agents
from core.store import StoreRegistry, VectorStore


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config()
        self.cfg.store.agents_root = self.temp.name
        self.agents = Agents(self.temp.name)
        self.agents.create("Кулинария")

    def test_database_is_opened_once_for_the_same_agent(self):
        with patch("core.store.RAGDatabase") as database:
            registry = StoreRegistry(self.cfg, self.agents)
            first = registry.get("kulinariya")
            second = registry.get("kulinariya")

        self.assertIsInstance(first, VectorStore)
        self.assertIs(first, second)
        database.assert_called_once()
        settings = database.call_args.kwargs
        self.assertEqual(settings["db_path"],
                         str(self.agents.get("kulinariya").db_dir))
        self.assertEqual(settings["collection_name"], "books")
        self.assertEqual(settings["embedding_model"], self.cfg.embeddings.model)
        self.assertIs(first._db.lexical, self.cfg.search.lexical)

    def test_every_agent_gets_its_own_database(self):
        self.agents.create("Программирование")
        with patch("core.store.RAGDatabase", side_effect=lambda **kw: Mock()) as database:
            registry = StoreRegistry(self.cfg, self.agents)
            registry.get("kulinariya")
            registry.get("programmirovanie")

        paths = [call.kwargs["db_path"] for call in database.call_args_list]
        self.assertEqual(database.call_count, 2)
        self.assertEqual(len(set(paths)), 2)

    def test_unknown_agent_is_not_opened(self):
        from core.agents import AgentNotFound

        with patch("core.store.RAGDatabase") as database:
            registry = StoreRegistry(self.cfg, self.agents)
            with self.assertRaises(AgentNotFound):
                registry.get("missing")
        database.assert_not_called()

    def test_forget_closes_the_base_and_lets_it_open_again(self):
        with patch("core.store.RAGDatabase"):
            registry = StoreRegistry(self.cfg, self.agents)
            store = registry.get("kulinariya")
            registry.forget("kulinariya")

            store._db.close.assert_called_once()
            self.assertIsNot(registry.get("kulinariya"), store)

    def test_close_closes_everything(self):
        with patch("core.store.RAGDatabase", side_effect=lambda **kw: Mock()):
            registry = StoreRegistry(self.cfg, self.agents)
            cooking = registry.get("kulinariya")
            self.agents.create("Программирование")
            coding = registry.get("programmirovanie")

            registry.close()

        cooking._db.close.assert_called_once()
        coding._db.close.assert_called_once()
        self.assertEqual(registry._stores, {})

    def test_index_and_delete_are_delegated_to_the_database(self):
        with patch("core.store.RAGDatabase") as database:
            registry = StoreRegistry(self.cfg, self.agents)
            store = registry.get("kulinariya")
            store._db.process_research_folder.return_value = True
            store._db.delete_source.return_value = 7
            store._db.clear_base.return_value = 12

            reporter = lambda event: None
            self.assertTrue(store.index("путь", progress=reporter))
            self.assertEqual(store.delete_source("Плов.pdf"), 7)
            self.assertEqual(store.clear(), 12)

        store._db.process_research_folder.assert_called_once_with(
            "путь", progress=reporter)
        store._db.delete_source.assert_called_once_with("Плов.pdf")


if __name__ == '__main__':
    unittest.main()
