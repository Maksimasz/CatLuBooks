"""Проверки веб-слоя: ошибки агентов отдаются текстом, а не сбоем сервера.

Базы агентов подменены заглушкой: тесты не поднимают ни ChromaDB, ни модель.
"""
import json
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


class _FakeChatClient:
    """Модель с готовым ответом на запрос вне потока (промпт агента)."""

    def __init__(self, answer):
        self.answer = answer
        self.messages = []

    async def chat(self, messages, max_tokens=None):
        self.messages.append(messages)
        return self.answer


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
        created = self.client.post("/api/agents",
                                   json={"title": "Кулинария", "generate_prompt": False}).json()
        self.assertEqual(created["slug"], "kulinariya")

        renamed = self.client.patch(f"/api/agents/{created['slug']}",
                                    json={"title": "Кухня"}).json()
        self.assertEqual(renamed["title"], "Кухня")

        answer = self.client.delete(f"/api/agents/{created['slug']}")
        self.assertEqual(answer.status_code, 200)
        self.assertFalse(self.agents.exists("kulinariya"))

    def test_agent_prompt_is_written_by_the_model(self):
        llm = _FakeChatClient("Ты — ассистент по схемам и даташитам.")

        with patch.object(app_module, "get_client", return_value=llm):
            created = self.client.post("/api/agents", json={
                "title": "Схемы", "description": "даташиты разъёмов"}).json()

        self.assertEqual(created["prompt"], "Ты — ассистент по схемам и даташитам.")
        self.assertNotIn("prompt_error", created)
        self.assertIn("даташиты разъёмов", llm.messages[0][0]["content"])
        self.assertEqual(self.agents.get(created["slug"]).system_prompt,
                         "Ты — ассистент по схемам и даташитам.")
        self.assertEqual(self.client.get("/api/agents").json()["agents"][0]["prompt"],
                         "Ты — ассистент по схемам и даташитам.")

    def test_agent_without_model_gets_a_template_prompt(self):
        with patch.object(app_module, "needs_key", return_value=True), \
             patch.object(app_module, "key_for", return_value=""):
            created = self.client.post("/api/agents", json={
                "title": "Схемы", "description": "даташиты"}).json()

        self.assertIn("даташиты", created["prompt"])   # агент всё равно создан
        self.assertIn("ключ", created["prompt_error"])
        self.assertEqual(self.agents.get(created["slug"]).system_prompt, created["prompt"])

    def test_agent_prompt_is_composed_again_and_edited(self):
        agent = self.agents.create("Схемы", "даташиты разъёмов")
        llm = _FakeChatClient("Промпт от модели.")

        with patch.object(app_module, "get_client", return_value=llm):
            composed = self.client.post(f"/api/agents/{agent.slug}/prompt",
                                        json={"description": "пайка и разъёмы"}).json()

        self.assertEqual(composed["prompt"], "Промпт от модели.")
        self.assertEqual(composed["description"], "пайка и разъёмы")   # тема обновилась
        self.assertIn("пайка и разъёмы", llm.messages[0][0]["content"])

        saved = self.client.put(f"/api/agents/{agent.slug}/prompt",
                                json={"prompt": "  Мой промпт.  "}).json()
        self.assertEqual(saved["prompt"], "Мой промпт.")
        self.assertEqual(self.agents.get(agent.slug).system_prompt, "Мой промпт.")

        cleared = self.client.put(f"/api/agents/{agent.slug}/prompt",
                                  json={"prompt": ""}).json()
        self.assertEqual(cleared["prompt"], "")

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

    # --- сведение источников -------------------------------------------

    def test_merge_status_is_reported(self):
        data = self.client.get("/api/merge").json()

        self.assertIn(data["mode"], ("off", "fast", "full"))
        self.assertIn("fast", data["options"])

    def test_merge_mode_can_be_switched(self):
        before = app_module.CONFIG.search.merge_mode
        with patch.object(app_module, "save_setting") as save:
            response = self.client.post("/api/merge", json={"mode": "fast"})
        self.addCleanup(setattr, app_module.CONFIG.search, "merge_mode", before)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "fast")
        self.assertEqual(app_module.CONFIG.search.merge_mode, "fast")
        save.assert_called_once_with("search", "merge_mode", "fast")

    def test_unknown_merge_mode_is_rejected(self):
        response = self.client.post("/api/merge", json={"mode": "nonsense"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("off", response.json()["detail"])

    def test_merge_prompts_are_configured(self):
        """Режимы сведения используют свои промпты: слияние и ответ по черновику."""
        prompt = app_module.CONFIG.prompt

        self.assertIn("{context}", prompt.merge_template)
        self.assertIn("{question}", prompt.merge_template)
        self.assertIn("{context}", prompt.synthesis_template)
        self.assertIn("ЧЕРНОВИК", prompt.answer_from_draft_template)

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


    # --- история поиска ---------------------------------------------------
    def test_history_is_loaded_and_saved(self):
        agent = self.agents.create("Кулинария")
        saved = self.client.post(
            f"/api/agents/{agent.slug}/history",
            json={"question": "как варить бульон", "answer": "варите 2 часа",
                  "sources": [{"file": "book.pdf", "page": 5}],
                  "provider": "ollama", "model": "llama3"}).json()
        self.assertTrue(saved["ok"])
        self.assertTrue(saved["entry"]["id"])

        data = self.client.get(f"/api/agents/{agent.slug}/history").json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["entries"][0]["question"], "как варить бульон")
        self.assertEqual(data["entries"][0]["answer"], "варите 2 часа")

    def test_empty_question_is_rejected(self):
        agent = self.agents.create("Кулинария")
        response = self.client.post(f"/api/agents/{agent.slug}/history",
                                    json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_history_entry_can_be_deleted(self):
        agent = self.agents.create("Кулинария")
        entry = self.client.post(f"/api/agents/{agent.slug}/history",
                                 json={"question": "q", "answer": "a"}).json()["entry"]
        response = self.client.delete(
            f"/api/agents/{agent.slug}/history/{entry['id']}")
        self.assertEqual(response.status_code, 200)
        data = self.client.get(f"/api/agents/{agent.slug}/history").json()
        self.assertEqual(data["total"], 0)

    def test_missing_history_entry_returns_404(self):
        agent = self.agents.create("Кулинария")
        response = self.client.delete(f"/api/agents/{agent.slug}/history/nope")
        self.assertEqual(response.status_code, 404)

    def test_history_can_be_cleared(self):
        agent = self.agents.create("Кулинария")
        self.client.post(f"/api/agents/{agent.slug}/history", json={"question": "q1", "answer": "a"})
        self.client.post(f"/api/agents/{agent.slug}/history", json={"question": "q2", "answer": "b"})
        response = self.client.delete(f"/api/agents/{agent.slug}/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed"], 2)
        data = self.client.get(f"/api/agents/{agent.slug}/history").json()
        self.assertEqual(data["total"], 0)


class _FakeLLM:
    """Модель, которая отвечает заранее заданными кусками."""

    def __init__(self, answers):
        self.answers = list(answers)   # [(куски текста, finish_reason), ...]
        self.calls = 0
        self.finish_reason = None
        self.messages = None

    async def stream(self, messages):
        pieces, reason = self.answers[self.calls]
        self.calls += 1
        self.finish_reason = reason
        self.messages = messages
        for piece in pieces:
            yield piece


class ChatContinuationTests(unittest.IsolatedAsyncioTestCase):
    """Ответ, упёршийся в лимит токенов, дописывается, а не обрывается."""

    @staticmethod
    def make_store():
        store = Mock()
        store.search.return_value = [
            {"id": "1", "document": "рецепт", "metadata": {"source": "книга.pdf"}}]
        store.format_context.return_value = "КОНТЕКСТ"
        store.sources.return_value = [{"source": "книга.pdf"}]
        return store

    async def ask(self, answers, system_prompt=None):
        request = app_module.ChatRequest(message="котлета по киевски", history=[])
        llm = _FakeLLM(answers)
        agent = Mock(title="Кулинария", slug="kulinariya", system_prompt=system_prompt)
        merge_before = app_module.CONFIG.search.merge_mode
        app_module.CONFIG.search.merge_mode = "off"
        self.addCleanup(setattr, app_module.CONFIG.search, "merge_mode", merge_before)
        with patch.object(app_module, "agent_store",
                          return_value=(agent, self.make_store())), \
             patch.object(app_module, "get_client", return_value=llm), \
             patch.object(app_module, "save_last_request"), \
             patch("builtins.print"):
            events = [event async for event in
                      app_module.stream_chat(request, app_module.app)]
        return llm, events

    @staticmethod
    def answer(events):
        pieces = []
        for event in events:
            if event.startswith("event: token"):
                pieces.append(json.loads(event.split("data: ", 1)[1])["text"])
        return "".join(pieces)

    async def test_truncated_answer_is_continued(self):
        llm, events = await self.ask([(["Первая часть."], "length"),
                                      ([" Вторая часть."], "stop")])

        self.assertEqual(self.answer(events), "Первая часть. Вторая часть.")
        self.assertEqual(llm.calls, 2)
        self.assertIn("event: stage", "".join(events))

    async def test_finished_answer_is_not_continued(self):
        llm, events = await self.ask([(["Целый ответ."], "stop")])

        self.assertEqual(self.answer(events), "Целый ответ.")
        self.assertEqual(llm.calls, 1)
        self.assertNotIn("event: stage", "".join(events))

    async def test_agent_prompt_replaces_the_common_one(self):
        llm, _ = await self.ask([(["Ответ."], "stop")],
                                system_prompt="Ты — ассистент по схемам и даташитам.")

        self.assertEqual(llm.messages[0],
                         {"role": "system",
                          "content": "Ты — ассистент по схемам и даташитам."})

    async def test_agent_without_prompt_uses_the_common_one(self):
        llm, _ = await self.ask([(["Ответ."], "stop")], system_prompt="   ")

        self.assertEqual(llm.messages[0]["content"], app_module.CONFIG.prompt.system)


if __name__ == '__main__':
    unittest.main()
