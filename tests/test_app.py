"""Проверки веб-слоя: ошибки агентов отдаются текстом, а не сбоем сервера.

Базы агентов подменены заглушкой: тесты не поднимают ни ChromaDB, ни модель.
"""
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app as app_module
from core.agents import Agents
from core.store import StoreRegistry


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)
        self.registry = Mock(spec=StoreRegistry)
        self.client = TestClient(app_module.app)

        for patcher in (
            patch.object(app_module, "AGENTS", self.agents),
            patch.object(app_module, "REGISTRY", self.registry),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_stats_without_agents_asks_to_create_one(self):
        response = self.client.get("/api/stats")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Агентов пока нет", response.json()["detail"])

    def test_stats_of_unknown_agent_is_not_a_server_error(self):
        self.agents.create("Кулинария")

        response = self.client.get("/api/stats", params={"agent": "missing"})

        self.assertEqual(response.status_code, 404)
        self.assertIn("missing", response.json()["detail"])

    def test_search_of_unknown_agent_is_not_a_server_error(self):
        response = self.client.post("/api/search",
                                    json={"query": "борщ", "agent": "missing"})

        self.assertEqual(response.status_code, 404)

    def test_chat_without_agents_explains_itself_in_the_stream(self):
        response = self.client.post("/api/chat", json={"message": "привет"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn("Агентов пока нет", response.text)

    def test_agent_list_shows_books_chunks_and_size(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "Плов.pdf", [b"pdf"])
        self.registry.get.return_value.stats.return_value = {
            "total_chunks": 42, "sources_count": 3,
        }

        data = self.client.get("/api/agents").json()

        self.assertEqual(data["agents"][0]["title"], "Кулинария")
        self.assertEqual(data["agents"][0]["folder"], "kulinariya")
        self.assertEqual(data["agents"][0]["books"], 1)
        self.assertEqual(data["agents"][0]["chunks"], 42)
        self.assertIn("Б", data["agents"][0]["size_text"])

    def test_agent_is_created_renamed_and_deleted(self):
        created = self.client.post("/api/agents", json={"title": "Кулинария"}).json()
        self.assertEqual(created["slug"], "kulinariya")

        renamed = self.client.patch(f"/api/agents/{created['slug']}",
                                    json={"title": "Кухня"}).json()
        self.assertEqual(renamed["title"], "Кухня")

        answer = self.client.delete(f"/api/agents/{created['slug']}")
        self.assertEqual(answer.status_code, 200)
        self.assertFalse(self.agents.exists("kulinariya"))

    def test_title_without_letters_is_rejected(self):
        response = self.client.post("/api/agents", json={"title": "!!!"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("название", response.json()["detail"].lower())

    def test_books_are_listed_with_their_fragment_counts(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "Плов.pdf", [b"pdf"])
        self.registry.get.return_value.source_counts.return_value = {"Плов.pdf": 7}

        data = self.client.get(f"/api/agents/{agent.slug}/books").json()

        self.assertEqual(data["books"][0]["name"], "Плов.pdf")
        self.assertEqual(data["books"][0]["chunks"], 7)
        self.assertEqual(data["chunks"], 7)

    def test_uploaded_book_is_saved_to_the_agent_folder(self):
        agent = self.agents.create("Кулинария")

        response = self.client.post(
            f"/api/agents/{agent.slug}/books",
            files=[("files", ("Плов.txt", "рецепт плова".encode("utf-8"), "text/plain"))])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["uploaded"][0]["name"], "Плов.txt")
        self.assertTrue((agent.books_dir / "Плов.txt").is_file())

    def test_book_is_removed_from_base_and_from_disk(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "Плов.pdf", [b"pdf"])
        self.registry.get.return_value.delete_source.return_value = 5

        response = self.client.delete(f"/api/agents/{agent.slug}/books/Плов.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed"], 5)
        self.assertFalse((agent.books_dir / "Плов.pdf").exists())

    def test_clearing_base_keeps_books(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "Плов.pdf", [b"pdf"])
        self.registry.get.return_value.clear.return_value = 12

        response = self.client.delete(f"/api/agents/{agent.slug}/base")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed"], 12)
        self.assertTrue((agent.books_dir / "Плов.pdf").exists())

    def test_config_and_providers_describe_where_agents_live(self):
        info = self.client.get("/api/config").json()
        self.assertIn("agents_root", info)

        catalog = self.client.get("/api/providers").json()
        self.assertTrue(catalog["providers"])
        self.assertTrue(catalog["default"]["provider"])


if __name__ == '__main__':
    unittest.main()
