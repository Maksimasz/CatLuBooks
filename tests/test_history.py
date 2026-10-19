"""Проверки истории поиска: сохранение, чтение, удаление, очистка."""
import json
import tempfile
import unittest
from pathlib import Path

from core import history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = self.temp.name

    def test_empty_folder_has_empty_history(self):
        self.assertEqual(history.load(self.folder), [])

    def test_add_creates_entry_with_id_and_timestamp(self):
        entry = history.add(self.folder, question="как варить бульон",
                            answer="варите 2 часа", sources=[{"file": "a.pdf", "page": 5}])
        self.assertTrue(entry["id"])
        self.assertEqual(entry["question"], "как варить бульон")
        self.assertEqual(entry["answer"], "варите 2 часа")
        self.assertEqual(entry["sources"], [{"file": "a.pdf", "page": 5}])
        self.assertGreater(entry["created"], 0)

        loaded = history.load(self.folder)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], entry["id"])

    def test_newest_entry_comes_first(self):
        history.add(self.folder, question="первый", answer="a")
        history.add(self.folder, question="второй", answer="b")
        entries = history.load(self.folder)
        self.assertEqual([e["question"] for e in entries], ["второй", "первый"])

    def test_answer_is_trimmed_to_limit(self):
        long_answer = "x" * (history.MAX_ANSWER_CHARS + 500)
        entry = history.add(self.folder, question="q", answer=long_answer)
        self.assertEqual(len(entry["answer"]), history.MAX_ANSWER_CHARS)

    def test_delete_removes_entry(self):
        entry = history.add(self.folder, question="q", answer="a")
        self.assertTrue(history.delete(self.folder, entry["id"]))
        self.assertEqual(history.load(self.folder), [])
        # Повторное удаление — False.
        self.assertFalse(history.delete(self.folder, entry["id"]))

    def test_clear_returns_count_and_empties(self):
        history.add(self.folder, question="q1", answer="a")
        history.add(self.folder, question="q2", answer="b")
        self.assertEqual(history.clear(self.folder), 2)
        self.assertEqual(history.load(self.folder), [])

    def test_corrupt_file_returns_empty(self):
        Path(self.folder, "history.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(history.load(self.folder), [])

    def test_trim_to_max_entries(self):
        for i in range(history.MAX_ENTRIES + 5):
            history.add(self.folder, question=f"q{i}", answer="a")
        self.assertEqual(len(history.load(self.folder)), history.MAX_ENTRIES)


if __name__ == "__main__":
    unittest.main()
