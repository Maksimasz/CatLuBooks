"""Проверки локальных настроек: выбор из интерфейса важнее config.yaml."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.settings = self.folder / "settings.json"
        patcher = patch.object(config, "SETTINGS_PATH", self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, text: str) -> str:
        path = self.folder / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_missing_file_means_no_overrides(self):
        self.assertEqual(config.load_settings(), {})

    def test_saved_setting_survives_and_is_read_back(self):
        config.save_setting("store", "ocr", "auto")

        self.assertEqual(config.load_settings(), {"store": {"ocr": "auto"}})
        self.assertTrue(self.settings.is_file())

    def test_interface_setting_wins_over_yaml(self):
        path = self.write_config("store:\n  ocr: 'off'\n  chunk_size: 1000\n")
        config.save_setting("store", "ocr", "auto")

        cfg = config.load_config(path)

        self.assertEqual(cfg.store.ocr, "auto")
        self.assertEqual(cfg.store.chunk_size, 1000)

    def test_unknown_keys_are_ignored(self):
        path = self.write_config("store:\n  ocr: 'off'\n")
        config.save_setting("store", "нет_такого_поля", 5)
        config.save_setting("чужой_блок", "поле", 1)

        cfg = config.load_config(path)

        self.assertFalse(hasattr(cfg.store, "нет_такого_поля"))
        self.assertFalse(hasattr(cfg, "чужой_блок"))

    def test_number_stored_as_text_becomes_number(self):
        path = self.write_config("store:\n  chunk_size: 1000\n")
        config.save_setting("store", "chunk_size", "1500")

        cfg = config.load_config(path)

        self.assertEqual(cfg.store.chunk_size, 1500)

    def test_off_without_quotes_is_still_off(self):
        """YAML читает `off` как False: настройка не должна молча означать «включено»."""
        path = self.write_config("store:\n  ocr: off\n")

        self.assertEqual(config.load_config(path).store.ocr, "off")

    def test_interface_choice_beats_yaml_boolean(self):
        path = self.write_config("store:\n  ocr: off\n")

        config.save_setting("store", "ocr", "auto")
        self.assertEqual(config.load_config(path).store.ocr, "auto")

        config.save_setting("store", "ocr", "off")
        self.assertEqual(config.load_config(path).store.ocr, "off")

    def test_yaml_on_means_ocr_is_on(self):
        path = self.write_config("store:\n  ocr: on\n")

        self.assertEqual(config.load_config(path).store.ocr, "auto")

    def test_broken_file_is_ignored(self):
        self.settings.write_text("{это не json", encoding="utf-8")
        path = self.write_config("store:\n  ocr: 'off'\n")

        self.assertEqual(config.load_settings(), {})
        self.assertEqual(config.load_config(path).store.ocr, "off")


if __name__ == "__main__":
    unittest.main()
