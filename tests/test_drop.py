"""Проверки инструмента удаления данных из базы (без настоящей ChromaDB)."""
import unittest
from unittest.mock import Mock

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

        before = drop.drop_all(client, "research_papers")

        self.assertEqual(before, 1)
        client.delete_collection.assert_called_once_with("research_papers")
        self.assertEqual(client.create_collection.call_args.kwargs["name"],
                         "research_papers")


if __name__ == "__main__":
    unittest.main()
