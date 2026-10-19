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

# Как отвечает проверка tesseract: установлен со всеми языками или нет.
OCR_READY = {"available": True, "path": "tesseract", "version": "5.5.3",
             "languages": ["eng", "rus"], "has_russian": True,
             "install_url": "https://example.org/tesseract"}
OCR_MISSING = {"available": False, "path": "", "version": "", "languages": [],
               "has_russian": False, "install_url": "https://example.org/tesseract"}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agents = Agents(self.temp.name)
        self.registry = Mock(spec=StoreRegistry)
        self.client = TestClient(app_module.app)
        self.ocr_before = app_module.CONFIG.store.ocr
        self.addCleanup(setattr, app_module.CONFIG.store, "ocr", self.ocr_before)

        self.server = Mock()
        self.server.status.return_value = {
            "pid": 1234, "host": "0.0.0.0", "port": 8000, "uptime": 42,
            "urls": ["http://127.0.0.1:8000", "http://192.168.1.5:8000"],
            "lan_url": "http://192.168.1.5:8000", "python": "3.12.0",
            "folder": "/tmp/project",
            "start_command": "cd /tmp/project\npython app.py",
            "can_restart": True, "can_start": False,
        }
        self.server.restart.return_value = 9999

        for patcher in (
            patch.object(app_module, "AGENTS", self.agents),
            patch.object(app_module, "REGISTRY", self.registry),
            patch.object(app_module, "SERVER", self.server),
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

        response = self.client.delete(f"/api/agents/{agent.slug}/book",
                                      params={"name": "Плов.pdf"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed"], 5)
        self.assertTrue(response.json()["file"])
        self.assertFalse((agent.books_dir / "Плов.pdf").exists())

    def test_fragments_of_a_removed_book_are_listed_and_can_be_dropped(self):
        """Файл убрали мимо интерфейса: фрагменты видно и можно убрать."""
        agent = self.agents.create("Кулинария")
        self.registry.get.return_value.source_counts.return_value = {"Старая.pdf": 9}
        self.registry.get.return_value.delete_source.return_value = 9

        listed = self.client.get(f"/api/agents/{agent.slug}/books").json()

        self.assertEqual(listed["missing"], 1)
        self.assertTrue(listed["books"][0]["missing"])
        self.assertEqual(listed["books"][0]["chunks"], 9)

        response = self.client.delete(f"/api/agents/{agent.slug}/book",
                                      params={"name": "Старая.pdf"})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["file"])
        self.assertEqual(response.json()["removed"], 9)

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

    def test_health_answers_for_outside_checkers(self):
        """Адрес /health спрашивают сторонние инструменты — 404 им не нужен."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    # --- OCR ----------------------------------------------------------

    def test_ocr_status_is_reported(self):
        with patch.object(app_module.ocr, "status", return_value=dict(OCR_READY)):
            data = self.client.get("/api/ocr").json()

        self.assertTrue(data["available"])
        self.assertTrue(data["has_russian"])
        self.assertIn("enabled", data)

    def test_ocr_cannot_be_enabled_without_tesseract(self):
        with patch.object(app_module.ocr, "status", return_value=dict(OCR_MISSING)):
            response = self.client.post("/api/ocr", json={"enabled": True})

        self.assertEqual(response.status_code, 409)
        self.assertIn("Tesseract", response.json()["detail"])

    def test_ocr_can_be_turned_off_even_without_tesseract(self):
        with patch.object(app_module.ocr, "status", return_value=dict(OCR_MISSING)), \
             patch.object(app_module, "save_setting") as save:
            response = self.client.post("/api/ocr", json={"enabled": False})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["enabled"])
        save.assert_called_once_with("store", "ocr", "off")

    def test_enabled_ocr_is_saved_and_applied_to_open_bases(self):
        with patch.object(app_module.ocr, "status", return_value=dict(OCR_READY)), \
             patch.object(app_module, "save_setting") as save:
            response = self.client.post("/api/ocr", json={"enabled": True})

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["enabled"])
            self.assertEqual(app_module.CONFIG.store.ocr, "auto")

        save.assert_called_once_with("store", "ocr", "auto")
        self.registry.set_ocr.assert_called_once_with("auto")

    def test_books_report_scans_without_text_layer(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "скан.pdf", [b"pdf"])
        self.registry.get.return_value.source_counts.return_value = {}

        with patch.object(app_module.parsers, "probe",
                          return_value={"readable": False, "needs_ocr": True,
                                        "reason": "scan"}):
            data = self.client.get(f"/api/agents/{agent.slug}/books").json()

        self.assertTrue(data["books"][0]["needs_ocr"])
        self.assertEqual(data["needs_ocr"], 1)

    def test_readable_book_does_not_ask_for_ocr(self):
        agent = self.agents.create("Кулинария")
        self.agents.write_book(agent.slug, "книга.pdf", [b"pdf"])
        self.registry.get.return_value.source_counts.return_value = {}

        with patch.object(app_module.parsers, "probe",
                          return_value={"readable": True, "needs_ocr": False,
                                        "reason": "text"}):
            data = self.client.get(f"/api/agents/{agent.slug}/books").json()

        self.assertFalse(data["books"][0]["needs_ocr"])
        self.assertEqual(data["needs_ocr"], 0)


    # --- сервер -----------------------------------------------------------
    def test_server_status_is_reported(self):
        data = self.client.get("/api/server").json()

        self.assertEqual(data["port"], 8000)
        self.assertEqual(data["lan_url"], "http://192.168.1.5:8000")
        self.assertTrue(data["can_restart"])
        self.assertFalse(data["can_start"])

    def test_server_restart_schedules_shutdown(self):
        response = self.client.post("/api/server/restart")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["restarting"])
        self.assertEqual(response.json()["pid"], 9999)
        self.server.restart.assert_called_once()

    def test_server_stop_returns_start_command(self):
        response = self.client.post("/api/server/stop")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["stopping"])
        self.assertIn("python app.py", response.json()["start_command"])
        self.server.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
