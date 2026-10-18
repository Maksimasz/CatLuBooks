#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Чтение документов для индексации: PDF, EPUB, TXT, FB2.

Модуль отвечает только за то, чтобы достать из файла текст по частям
(страница PDF, глава EPUB, часть TXT). Как резать текст на фрагменты и что
выбрасывать, решает RAGDatabase.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pymupdf

# Расширения, которые умеет читать этот модуль.
SUPPORTED = (".pdf", ".epub", ".txt", ".md", ".fb2")

# Подпись части документа: страница у PDF, глава у EPUB, часть у текста.
UNITS = {".pdf": "страница", ".epub": "раздел", ".txt": "часть",
         ".md": "часть", ".fb2": "часть"}

# --- Признаки служебного текста -------------------------------------------
DOTS_LEADER = re.compile(r"\.{4,}")                       # «Борщ ......... 27»
NUMBERED_INDEX = re.compile(r"\b\d{2,4}\.\s*[А-ЯЁ]")      # «177. Борщ с картофелем»
GLYPH_GARBAGE = re.compile(r"/g\d+")                      # сломанный шрифт PDF
PAGE_MARKER = re.compile(r"^[-—–\s]*(Страница|Фото|Содержание|Оглавление)\b",
                         re.IGNORECASE)
PAGE_HEADER = re.compile(r"^\d{1,4}\s*[|•·]\s*")          # колонтитул «38 | РЕЦЕПТЫ»
NUMBER_ONLY_LINE = re.compile(r"^[\d\s.,;:|/\\\-–—()\[\]]+$")
LINE_BREAK_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")      # «ку-рицу» -> «курицу»
ANY_LETTER_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")
LATIN_WORD = re.compile(r"^[A-Za-z]{2,}$")
CYRILLIC_ANY = re.compile(r"[А-Яа-яЁё]")
WORD = re.compile(r"\w+")

MIN_WORDS = 20        # меньше слов — страница пустая (обложка, пустой скан)
DIGIT_SHARE = 0.35    # доля чисел, при которой текст похож на алфавитный указатель
LISTING_DIGIT_SHARE = 0.15   # порог доли чисел для «177. Борщ … 178. Борщ»
NUMBERED_MIN = 8      # столько записей «177. Борщ» подряд — это указатель, не рецепт
OCR_LANGUAGE = "rus+eng"
OCR_DPI = 200
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def unit_for(path) -> str:
    """Подпись части документа: «страница», «раздел» или «часть»."""
    return UNITS.get(Path(path).suffix.lower(), "часть")


def clean_text(text: str) -> str:
    """Убирает колонтитулы и переносы, оставляя читаемый текст."""
    text = re.sub(r"\u00ad\s*", "", text)      # мягкий перенос внутри слова
    lines = []
    for line in text.splitlines():
        line = PAGE_HEADER.sub("", line.strip())
        if not line or NUMBER_ONLY_LINE.match(line):
            continue                            # пустая строка или колонцифра
        lines.append(line)
    text = "\n".join(lines)
    text = LINE_BREAK_HYPHEN.sub(r"\1\2", text)  # перенос по слогам
    text = re.sub(r"\n+", " ", text)
    # Оставляем буквы, цифры и обычную пунктуацию, остальное — в пробел.
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\%\=\+\*\&\#\@]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _digit_share(text: str) -> Tuple[float, int]:
    tokens = WORD.findall(text)
    share = sum(1 for token in tokens if token.isdigit()) / len(tokens) if tokens else 0.0
    return share, len(tokens)


def is_listing(text: str) -> bool:
    """Оглавление, указатель или мусор распознавания, но не текст рецепта."""
    if len(DOTS_LEADER.findall(text)) >= 3:
        return True
    if len(GLYPH_GARBAGE.findall(text)) >= 10:
        return True
    share, _ = _digit_share(text)
    return len(NUMBERED_INDEX.findall(text)) >= NUMBERED_MIN and share >= LISTING_DIGIT_SHARE


def digits_heavy(text: str) -> bool:
    """Похоже на алфавитный указатель: «Свекла 82 275 Свинина 151».

    Настоящие рецептурные таблицы тоже состоят из чисел, поэтому при
    индексации такие страницы не выбрасываются — только понижается их вес
    при поиске.
    """
    share, count = _digit_share(text)
    return count >= 30 and share >= DIGIT_SHARE


def looks_garbled(text: str) -> bool:
    """Шрифт без кириллицы: в русском тексте слова подменены латиницей.

    Нужен минимум кириллицы (заголовки-то читаются) и заметный перевес
    «латинских» слов — иначе это просто страница на английском языке.
    """
    words = ANY_LETTER_WORD.findall(text)
    if len(words) < MIN_WORDS:
        return False
    cyrillic = sum(1 for word in words if CYRILLIC_ANY.search(word))
    latin = sum(1 for word in words if LATIN_WORD.match(word))
    return cyrillic >= 3 and latin / (latin + cyrillic) > 0.5


def is_noise(text: str) -> bool:
    """Служебный текст, который не может быть ответом: понижается при поиске."""
    if is_listing(text) or digits_heavy(text) or looks_garbled(text):
        return True
    stripped = text.strip()
    return bool(len(stripped) < 200 and PAGE_MARKER.match(stripped))


def keep_section(text: str) -> bool:
    """Стоит ли хранить этот текст в базе."""
    if len(ANY_LETTER_WORD.findall(text)) < MIN_WORDS:
        return False                            # пустой скан, обложка, реклама
    return not (looks_garbled(text) or is_listing(text))


class Tesseract:
    """Распознавание текста со страниц-картинок через tesseract."""

    def __init__(self, language: str = OCR_LANGUAGE, dpi: int = OCR_DPI,
                 workers: int = 4):
        self.language = language
        self.dpi = dpi
        self.workers = workers
        found = shutil.which("tesseract")
        if not found and Path(WINDOWS_TESSERACT).exists():
            found = WINDOWS_TESSERACT
        self.path = found or ""

    @property
    def available(self) -> bool:
        return bool(self.path) and Path(self.path).exists()

    def text(self, image: bytes) -> str:
        """Распознаёт одну картинку (PNG) как текст."""
        try:
            done = subprocess.run(
                [self.path, "stdin", "stdout", "-l", self.language, "--psm", "3"],
                input=image, capture_output=True,
            )
        except OSError:
            return ""
        return done.stdout.decode("utf-8", "replace")

    def texts(self, images: List[bytes]) -> List[str]:
        """Распознаёт несколько страниц: tesseract считает в своих процессах."""
        if not images:
            return []
        workers = max(1, min(self.workers, len(images)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.text, images))


def _decode(raw: bytes) -> str:
    """Определяет кодировку текстового файла по содержимому."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    for encoding in ("utf-8-sig", "cp1251", "koi8-r"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", "replace")


class _HtmlText(HTMLParser):
    """Собирает текст из HTML, разбивая абзацы и пропуская script/style."""

    SKIP = {"script", "style", "head", "title"}
    BREAK = {"p", "div", "br", "li", "tr", "td", "h1", "h2", "h3", "h4",
             "h5", "h6", "section", "article", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    """Превращает HTML (главу EPUB) в обычный текст."""
    parser = _HtmlText()
    parser.feed(markup)
    return "".join(parser.parts)


def _split_parts(text: str, limit: int) -> Iterator[Tuple[int, str]]:
    """Режет текст на части по абзацам, не разрывая абзац посередине."""
    part: List[str] = []
    size = 0
    number = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if part and size + len(paragraph) > limit:
            number += 1
            yield number, "\n\n".join(part)
            part, size = [], 0
        part.append(paragraph)
        size += len(paragraph) + 2
    if part:
        number += 1
        yield number, "\n\n".join(part)


def _recognize(buffer: List[Tuple[int, "pymupdf.Page"]],
               tess: Tesseract) -> Iterator[Tuple[int, str]]:
    """Распознаёт накопленные страницы-картинки и отдаёт их по порядку."""
    if not buffer:
        return
    images = [page.get_pixmap(dpi=tess.dpi).tobytes("png") for _, page in buffer]
    for (number, _), text in zip(buffer, tess.texts(images)):
        yield number, text
    buffer.clear()


def read_pdf(path: Path, tess: Optional[Tesseract] = None,
             batch: int = 32) -> Iterator[Tuple[int, str]]:
    """Страницы PDF по порядку; страницы без текста распознаются, если есть OCR."""
    with pymupdf.open(path) as doc:
        buffer: List[Tuple[int, "pymupdf.Page"]] = []
        for number, page in enumerate(doc, 1):
            text = page.get_text()
            if tess is not None and tess.available and not keep_section(clean_text(text)):
                buffer.append((number, page))
                if len(buffer) >= batch:
                    yield from _recognize(buffer, tess)
                continue
            # Сначала отдаём накопленные страницы: они идут раньше текущей.
            yield from _recognize(buffer, tess)
            yield number, text
        yield from _recognize(buffer, tess)


def _opf_path(archive: zipfile.ZipFile) -> str:
    """Находит в EPUB файл описания книги (OPF)."""
    container = ET.fromstring(archive.read("META-INF/container.xml"))
    for node in container.iter():
        if node.tag.endswith("rootfile") and node.attrib.get("full-path"):
            return node.attrib["full-path"]
    raise ValueError("В EPUB не найден файл описания (OPF)")


def _spine_files(archive: zipfile.ZipFile, opf: str) -> List[str]:
    """Список глав EPUB в порядке чтения (пути внутри архива — через «/»)."""
    package = ET.fromstring(archive.read(opf))
    manifest = {}
    spine = []
    for node in package.iter():
        tag = node.tag.split("}")[-1]
        if tag == "item" and node.attrib.get("id"):
            manifest[node.attrib["id"]] = node.attrib.get("href", "")
        elif tag == "itemref" and node.attrib.get("idref"):
            spine.append(node.attrib["idref"])
    folder = PurePosixPath(opf).parent
    names = []
    for item in spine:
        href = manifest.get(item)
        if not href:
            continue
        names.append(href if str(folder) == "." else str(folder / href))
    return names


def read_epub(path: Path) -> Iterator[Tuple[int, str]]:
    """Главы EPUB по порядку чтения."""
    with zipfile.ZipFile(path) as archive:
        opf = _opf_path(archive)
        for number, name in enumerate(_spine_files(archive, opf), 1):
            try:
                markup = archive.read(name).decode("utf-8", "replace")
            except KeyError:
                continue
            yield number, html_to_text(markup)


def read_txt(path: Path, limit: int = 6000) -> Iterator[Tuple[int, str]]:
    """Текстовый файл частями по абзацам."""
    yield from _split_parts(_decode(path.read_bytes()), limit)


def read_fb2(path: Path, limit: int = 6000) -> Iterator[Tuple[int, str]]:
    """FB2 (XML) частями по абзацам."""
    yield from _split_parts(fb2_to_text(path.read_bytes()), limit)


def fb2_to_text(raw: bytes) -> str:
    """Достаёт текст книги из FB2, не трогая картинки в base64."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        root = ET.fromstring(_decode(raw))
    paragraphs = [node.text for node in root.iter()
                  if node.tag.split("}")[-1] == "p" and node.text]
    return "\n\n".join(paragraphs)


def read_sections(path, tess: Optional[Tesseract] = None) -> Iterator[Tuple[int, str]]:
    """Текст документа по частям: (номер части, текст)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from read_pdf(path, tess)
    elif suffix == ".epub":
        yield from read_epub(path)
    elif suffix in {".txt", ".md"}:
        yield from read_txt(path)
    elif suffix == ".fb2":
        yield from read_fb2(path)
    else:
        raise ValueError(f"Формат {suffix or path.name} не поддерживается")
