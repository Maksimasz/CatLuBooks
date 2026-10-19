"""Проверки OCR: читается ли книга без распознавания и что есть в системе."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

from core import ocr, parsers


def make_pdf(folder: Path, name: str, pages: int = 1, text: str = "") -> Path:
    """Создаёт PDF: страницы с текстом или пустые (как у скана)."""
    path = folder / name
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


TEXT = " ".join(["Recipe of borscht with beetroot cabbage and beef"] * 6)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_text_pdf_is_readable_by_itself(self):
        path = make_pdf(self.folder, "книга.pdf", pages=3, text=TEXT)
        report = parsers.probe(path)

        self.assertTrue(report["readable"])
        self.assertFalse(report["needs_ocr"])
        self.assertGreaterEqual(report["words"], parsers.PROBE_MIN_WORDS)

    def test_scan_without_text_layer_suggests_ocr(self):
        path = make_pdf(self.folder, "скан.pdf", pages=4)
        report = parsers.probe(path)

        self.assertFalse(report["readable"])
        self.assertTrue(report["needs_ocr"])
        self.assertEqual(report["reason"], "scan")
        self.assertEqual(report["words"], 0)

    def test_text_formats_never_need_ocr(self):
        path = self.folder / "заметки.txt"
        path.write_text("Рецепт борща", encoding="utf-8")

        report = parsers.probe(path)

        self.assertTrue(report["readable"])
        self.assertFalse(report["needs_ocr"])

    def test_unsupported_format_is_not_readable(self):
        path = self.folder / "книга.djvu"
        path.write_bytes(b"x")

        report = parsers.probe(path)

        self.assertFalse(report["readable"])
        self.assertFalse(report["needs_ocr"])
        self.assertEqual(report["reason"], "format")

    def test_broken_file_does_not_raise(self):
        path = self.folder / "битая.pdf"
        path.write_bytes(b"not a pdf")

        report = parsers.probe(path)

        self.assertFalse(report["readable"])
        self.assertFalse(report["needs_ocr"])

    def test_missing_file_does_not_raise(self):
        report = parsers.probe(self.folder / "нет.pdf")

        self.assertFalse(report["readable"])
        self.assertEqual(report["reason"], "missing")


class ModeTests(unittest.TestCase):
    def test_words_are_understood(self):
        self.assertEqual(parsers.ocr_mode("auto"), "auto")
        self.assertEqual(parsers.ocr_mode(" ON "), "auto")
        self.assertEqual(parsers.ocr_mode("off"), "off")
        self.assertEqual(parsers.ocr_mode("no"), "off")

    def test_yaml_booleans_are_understood(self):
        """`ocr: off` в YAML — это логическая ложь, и это тот же выключатель."""
        self.assertEqual(parsers.ocr_mode(False), "off")
        self.assertEqual(parsers.ocr_mode(True), "auto")

    def test_unclear_value_means_off(self):
        """Опечатка не должна включать распознавание на всю ночь."""
        self.assertEqual(parsers.ocr_mode(None), "off")
        self.assertEqual(parsers.ocr_mode(""), "off")
        self.assertEqual(parsers.ocr_mode("аutto"), "off")


class EngineTests(unittest.TestCase):
    def test_missing_tesseract_is_reported_plainly(self):
        with patch.object(parsers.shutil, "which", return_value=None), \
             patch.object(parsers, "WINDOWS_TESSERACT", "C:\\nope\\tesseract.exe"):
            engine = parsers.Tesseract()

        self.assertFalse(engine.available)
        self.assertEqual(engine.version, "")
        self.assertEqual(engine.languages, [])

    def test_installed_languages_are_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder, "tesseract.exe")
            binary.write_bytes(b"not a program")
            data = Path(folder, "tessdata")
            data.mkdir()
            (data / "rus.traineddata").write_bytes(b"x")
            (data / "eng.traineddata").write_bytes(b"x")

            with patch.object(parsers.shutil, "which", return_value=None), \
                 patch.object(parsers, "WINDOWS_TESSERACT", str(binary)):
                engine = parsers.Tesseract()

            self.assertTrue(engine.available)
            self.assertEqual(engine.languages, ["eng", "rus"])
            self.assertEqual(engine.version, "")     # запустить не получилось


class StatusTests(unittest.TestCase):
    def fake(self, **fields) -> Mock:
        engine = Mock(available=True, path="tesseract", version="5.5.3",
                      languages=["eng", "rus"])
        for name, value in fields.items():
            setattr(engine, name, value)
        return engine

    def test_ready_tesseract_with_russian(self):
        with patch.object(ocr, "Tesseract", return_value=self.fake()):
            data = ocr.status()

        self.assertTrue(data["available"])
        self.assertTrue(data["has_russian"])
        self.assertEqual(data["version"], "5.5.3")
        self.assertIn("tesseract", data["install_url"])

    def test_english_only_installation_warns_about_russian(self):
        engine = self.fake(languages=["eng"])
        with patch.object(ocr, "Tesseract", return_value=engine):
            data = ocr.status()

        self.assertTrue(data["available"])
        self.assertFalse(data["has_russian"])

    def test_absent_tesseract_reports_nothing(self):
        engine = self.fake(available=False, path="", version="", languages=[])
        with patch.object(ocr, "Tesseract", return_value=engine):
            data = ocr.status()

        self.assertFalse(data["available"])
        self.assertFalse(data["has_russian"])
        self.assertEqual(data["version"], "")


if __name__ == "__main__":
    unittest.main()
