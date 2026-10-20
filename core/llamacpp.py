"""Управляемый llama-server: запуск и остановка локальной модели из интерфейса.

llama.cpp отдаёт тот же OpenAI-совместимый API, что и Ollama, поэтому в
провайдерах он есть отдельным пунктом (`llamacpp`, порт 8080). Здесь — вторая
половина: приложение само поднимает `llama-server` с выбранной GGUF-моделью,
чтобы не открывать консоль.

    models/                        — сюда кладут .gguf (папка не в git)
    llama.cpp/llama-server.exe     — сборка llama.cpp (или путь в config.yaml)
    logs/llamacpp.log              — журнал сервера
    logs/llamacpp.json             — номер процесса: состояние переживает
                                     перезапуск приложения

Процесс не привязан к приложению: перезапуск сервера его не останавливает,
а остановить можно кнопкой в ⚙ → Модели.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from core.format import human_size
from core.server import port_is_free

# Куда смотрит встроенный провайдер llamacpp (core/providers.py).
INSTALL_URL = "https://github.com/ggml-org/llama.cpp/releases"
# Где искать llama-server.exe, если путь не задан в config.yaml.
SEARCH_PATHS = (
    "llama.cpp/llama-server.exe",
    "llama.cpp/llama-server",
    "llama.cpp/build/bin/llama-server.exe",
    "llama.cpp/build/bin/Release/llama-server.exe",
    "llama.cpp/build/bin/llama-server",
    "bin/llama-server.exe",
    "bin/llama-server",
    "llama-server.exe",
    "llama-server",
)
LOG_LINES = 40
LOG_BYTES = 16_000


class LlamaCppError(Exception):
    """Понятная человеку ошибка запуска: показывается в интерфейсе."""


def _alive(pid: int) -> bool:
    """Живой ли процесс с таким номером (на Windows — без внешних программ)."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _kill(pid: int) -> None:
    """Останавливает процесс (на Windows — вместе с потомками)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    try:
        os.kill(pid, 15)      # SIGTERM
    except OSError:
        pass


@dataclass
class LlamaServer:
    """llama-server, которым управляет приложение."""

    root: Path
    models_dir: Path
    binary: str = ""
    port: int = 8080
    args: List[str] = field(default_factory=list)
    log_path: Path = field(init=False)
    state_path: Path = field(init=False)

    def __post_init__(self) -> None:
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / "llamacpp.log"
        self.state_path = logs / "llamacpp.json"
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # папку создадут вручную — покажем это в статусе

    # --- что есть на диске ----------------------------------------------
    @property
    def url(self) -> str:
        """Адрес OpenAI-совместимого API этого сервера."""
        return f"http://127.0.0.1:{self.port}/v1"

    def binary_path(self) -> Optional[Path]:
        """Найденный llama-server: путь из config.yaml, рядом с проектом или в PATH."""
        if self.binary:
            path = Path(self.binary)
            if not path.is_absolute():
                path = self.root / path
            return path if path.is_file() else None
        for candidate in SEARCH_PATHS:
            path = self.root / candidate
            if path.is_file():
                return path
        for name in ("llama-server", "llama-server.exe"):
            found = shutil.which(name)
            if found:
                return Path(found)
        return None

    def models(self) -> List[Dict[str, Any]]:
        """GGUF-модели из папки models (включая подпапки)."""
        if not self.models_dir.is_dir():
            return []
        found: List[Dict[str, Any]] = []
        for path in sorted(self.models_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".gguf":
                continue
            if any(part.startswith(".") for part in path.relative_to(self.models_dir).parts):
                continue
            size = path.stat().st_size
            found.append({
                "name": path.relative_to(self.models_dir).as_posix(),
                "size": size,
                "size_text": human_size(size),
            })
        return found

    def model_path(self, name: str) -> Path:
        """Проверяет имя модели: только .gguf внутри папки models."""
        name = (name or "").strip().replace("\\", "/")
        if not name or name.startswith("/") or ".." in name.split("/"):
            raise LlamaCppError("Выберите модель .gguf из папки models")
        if not name.lower().endswith(".gguf"):
            raise LlamaCppError("Модель должна быть файлом .gguf")
        path = (self.models_dir / name).resolve()
        try:
            path.relative_to(self.models_dir.resolve())
        except ValueError:
            raise LlamaCppError("Модель должна лежать в папке models") from None
        if not path.is_file():
            raise LlamaCppError(f"Файл модели не найден: {name}")
        return path

    # --- состояние ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Что сейчас с сервером: работает, грузит модель или выключен."""
        state = self._read_state()
        pid = int(state.get("pid") or 0)
        alive = _alive(pid)
        responding = self._responding()
        if alive and responding:
            mode = "running"
        elif alive:
            mode = "starting"
        elif responding:
            mode = "external"     # кто-то уже держит порт — например, свой запуск
        else:
            mode = "off"

        error = ""
        if not alive and not responding and pid and state.get("model"):
            error = "llama-server завершился — смотрите лог"

        binary = self.binary_path()
        return {
            "state": mode,
            "found": binary is not None,
            "binary": str(binary) if binary else "",
            "models_dir": str(self.models_dir.resolve()),
            "models": self.models(),
            "port": self.port,
            "url": self.url,
            "pid": pid if alive else 0,
            "model": state.get("model", "") if alive else "",
            "alias": state.get("alias", "") if alive else "",
            "started": state.get("started", "") if alive else "",
            "args": list(self.args),
            "can_start": binary is not None and mode == "off",
            "can_stop": alive,
            "error": error,
            "install_url": INSTALL_URL,
            "log": self.log_tail(),
        }

    def start(self, model: str) -> Dict[str, Any]:
        """Запускает llama-server с выбранной моделью. Возвращает статус.

        Ждать загрузки модели не нужно: интерфейс опрашивает состояние и сам
        покажет «загружает модель…», когда сервер ответит.
        """
        binary = self.binary_path()
        if binary is None:
            raise LlamaCppError(
                "llama-server не найден. Скачайте сборку llama.cpp "
                f"({INSTALL_URL}), положите llama-server.exe в папку "
                "llama.cpp/ рядом с проектом или укажите путь в config.yaml "
                "(llamacpp.binary)."
            )
        path = self.model_path(model)
        state = self.status()
        if state["state"] in ("running", "starting"):
            raise LlamaCppError("llama-server уже запущен — сначала остановите его")
        if state["state"] == "external":
            raise LlamaCppError(
                f"Порт {self.port} уже занят. Закройте программу, которая его "
                "занимает, или укажите другой порт в config.yaml (llamacpp.port)."
            )
        if self._port_busy():
            raise LlamaCppError(
                f"Порт {self.port} занят другой программой. Освободите его или "
                "укажите другой порт в config.yaml (llamacpp.port)."
            )

        alias = self._alias(path)
        command = [str(binary), "-m", str(path),
                   "--host", "127.0.0.1", "--port", str(self.port),
                   "--alias", alias, *[str(arg) for arg in self.args]]
        try:
            handle = open(self.log_path, "ab")
        except OSError as exc:
            raise LlamaCppError(f"Не удалось открыть журнал llama-server: {exc}") from exc
        try:
            handle.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M} запуск: {path.name}\n"
                         .encode("utf-8"))
            handle.flush()
            process = subprocess.Popen(
                command, cwd=str(self.root), stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise LlamaCppError(f"Не удалось запустить llama-server: {exc}") from exc
        finally:
            handle.close()

        self._save_state({"pid": process.pid, "model": path.name, "alias": alias,
                          "started": datetime.now().strftime("%Y-%m-%d %H:%M")})
        time.sleep(0.2)      # мгновенно упавший процесс сразу виден в статусе
        return self.status()

    def stop(self) -> Dict[str, Any]:
        """Останавливает запущенный приложением llama-server."""
        state = self._read_state()
        pid = int(state.get("pid") or 0)
        if pid and _alive(pid):
            _kill(pid)
        self._clear_state()
        time.sleep(0.3)
        return self.status()

    def log_tail(self, lines: int = LOG_LINES) -> str:
        """Последние строки журнала: по ним видно, почему сервер не поднялся."""
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - LOG_BYTES))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    # --- служебное ------------------------------------------------------
    def _responding(self) -> bool:
        """Отвечает ли сервер по своему адресу."""
        try:
            response = httpx.get(f"{self.url}/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def _port_busy(self) -> bool:
        return not port_is_free("127.0.0.1", self.port)

    def _alias(self, path: Path) -> str:
        """Имя модели, которое увидит интерфейс: имя файла без .gguf."""
        return path.stem or "model"

    def _read_state(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, data: Dict[str, Any]) -> None:
        try:
            self.state_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass  # без файла состояние не переживёт перезапуск, но сервер работает

    def _clear_state(self) -> None:
        try:
            self.state_path.unlink()
        except OSError:
            pass
