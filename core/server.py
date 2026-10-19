"""Управление самим сервером: адреса в сети, перезапуск и остановка.

Эти функции нужны интерфейсу (⚙ → Сервер): показать адрес для телефона,
перезапустить после правки config.yaml или git pull, остановить на ночь.
Запустить остановленный сервер из браузера нельзя — для этого нужна
команда на компьютере, и об этом сервер честно пишет в ответе.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

# Метка для перезапущенного процесса: он ждёт, пока предыдущий отпустит порт.
SUCCESSOR_ENV = "CATLU_SUCCESSOR"


def local_addresses() -> List[str]:
    """IPv4-адреса этого ПК в сети (без 127.0.0.1). Сначала основной (через
    маршрут), затем остальные, чтобы можно было показать их все."""
    found: List[str] = []
    # 1) Основной адрес — по тому, через какой маршрут уходит трафик.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.5)
        probe.connect(("8.8.8.8", 80))   # никуда не стучимся, просто выбираем маршрут
        address = probe.getsockname()[0]
        probe.close()
        if address and not address.startswith("127."):
            found.append(address)
    except OSError:
        pass
    # 2) Остальные адреса интерфейсов — на случай VPN или двух сетей.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    return found


def local_urls(port: int) -> List[str]:
    """Все адреса, по которым сервер доступен (свой ПК + сеть)."""
    return [f"http://127.0.0.1:{port}"] + [f"http://{address}:{port}" for address in local_addresses()]


def port_is_free(host: str, port: int) -> bool:
    """Свободен ли порт: пробуем подключиться — если никто не слушает, свободен."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.7)
            return probe.connect_ex((probe_host, port)) != 0
    except OSError:
        return True


def wait_until_free(host: str, port: int, timeout: float = 30.0) -> bool:
    """Ждёт, пока предыдущий сервер отпустит порт (перезапуск)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_free(host, port):
            return True
        time.sleep(0.4)
    return False


def spawn_successor(env_extra: Optional[Dict[str, str]] = None) -> int:
    """Запускает копию сервера отдельным процессом. Возвращает его номер.

    Процесс остаётся в той же консоли: Ctrl+C по-прежнему останавливает
    и его. Так подхватываются свежий код после git pull и правки config.yaml.
    """
    env = dict(os.environ)
    env[SUCCESSOR_ENV] = "1"
    env.update(env_extra or {})
    # Вывод преемника: в консоли — как обычно, чтобы всё было видно сразу
    # и Ctrl+C работал; при запуске через файл или трубу — в отдельный журнал,
    # иначе преемник продолжил бы писать в поток, который уже никто не читает.
    handle = None
    if not sys.stdout or not sys.stdout.isatty():
        log_path = ROOT / "logs" / "server.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "a", encoding="utf-8", errors="replace")
        except OSError:
            handle = None
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(ROOT / "app.py")],
        cwd=str(ROOT), env=env, stdout=handle, stderr=handle,
    )
    return process.pid


def start_hint(folder: str) -> str:
    """Команда, которой поднимают сервер вручную — для сообщения после остановки."""
    if os.name == "nt":
        return f'cd "{folder}"\n.\.venv\Scripts\python.exe -X utf8 app.py'
    return f'cd "{folder}"\n.venv/bin/python -X utf8 app.py'


class ServerControl:
    """Состояние текущего процесса сервера: адреса, статус, перезапуск, остановка."""

    def __init__(self, host: str, port: int, started: Optional[float] = None):
        self.host = host
        self.port = port
        self.started = started if started is not None else time.time()
        self.uvicorn = None  # uvicorn.Server: назначается при запуске
        self._stop_thread: Optional[threading.Thread] = None

    def status(self) -> Dict[str, Any]:
        urls = local_urls(self.port)
        return {
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "started": self.started,
            "uptime": max(0, round(time.time() - self.started)),
            "urls": urls,
            "lan_url": next((url for url in urls if not url.startswith("http://127.")), None),
            "python": sys.version.split()[0],
            "folder": str(ROOT),
            "start_command": start_hint(str(ROOT)),
            "can_restart": True,
            "can_start": False,  # запустить остановленный сервер из браузера нельзя
        }

    def restart(self, delay: float = 0.8) -> int:
        """Запускает копию сервера, затем завершает этот процесс. Возвращает номер
        новой копии. Если копию запустить не удалось — ничего не останавливаем."""
        pid = spawn_successor()
        self.shutdown_soon(delay)
        return pid

    def stop(self, delay: float = 0.8) -> None:
        """Останавливает сервер: обратно его поднимают только на компьютере."""
        self.shutdown_soon(delay)

    def shutdown_soon(self, delay: float) -> None:
        """Завершает uvicorn после ответа браузеру — ответ должен успеть дойти."""
        if self._stop_thread is not None and self._stop_thread.is_alive():
            return
        thread = threading.Thread(target=self._shutdown_after, args=(delay,), daemon=True)
        self._stop_thread = thread
        thread.start()

    def _shutdown_after(self, delay: float) -> None:
        time.sleep(delay)
        self.shutdown()

    def shutdown(self) -> None:
        if self.uvicorn is not None:
            self.uvicorn.should_exit = True
        else:
            os.kill(os.getpid(), signal.SIGINT)
