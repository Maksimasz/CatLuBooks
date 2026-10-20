"""Проверки управления самим сервером: адреса, перезапуск, остановка."""
import os
import signal
import socket
import threading
import unittest
from unittest.mock import Mock, patch

from core.server import (ServerControl, local_addresses, local_urls,
                        port_is_free, spawn_successor, start_hint,
                        wait_until_free)


class AddressTests(unittest.TestCase):
    def test_local_urls_contain_loopback_and_lan_port(self):
        urls = local_urls(8000)
        self.assertIn("http://127.0.0.1:8000", urls)
        self.assertTrue(all(url.startswith("http://") and url.endswith(":8000")
                           for url in urls))
        # Адреса сети не должны повторяться.
        self.assertEqual(len(urls), len(set(urls)))

    def test_local_addresses_skip_loopback(self):
        for address in local_addresses():
            self.assertFalse(address.startswith("127."))

    def test_start_hint_contains_folder_and_python_command(self):
        hint = start_hint("/tmp/project")
        self.assertIn("/tmp/project", hint)
        self.assertIn("python", hint)


class PortTests(unittest.TestCase):
    def test_port_is_free_after_socket_closes(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            probe.listen(1)
            port = probe.getsockname()[1]
            self.assertFalse(port_is_free("127.0.0.1", port))
        self.assertTrue(port_is_free("127.0.0.1", port))

    def test_wait_until_free_returns_immediately_for_free_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.assertTrue(wait_until_free("127.0.0.1", port, timeout=1.0))

    def test_wait_until_free_waits_for_busy_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            # Вызываем в отдельном потоке: он должен ждать, пока мы освободим порт.
            result = {}
            def worker():
                result["value"] = wait_until_free("127.0.0.1", port, timeout=5.0)
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            self.assertFalse(thread.join(timeout=1.0))   # ещё не завершился
            probe.close()
            thread.join(timeout=2.0)
            self.assertTrue(result["value"])


class ServerControlTests(unittest.TestCase):
    def setUp(self):
        self.control = ServerControl("0.0.0.0", 8000, started=100.0)

    def test_status_contains_expected_fields(self):
        status = self.control.status()
        for field in ("pid", "host", "port", "uptime", "urls", "lan_url",
                      "python", "folder", "start_command", "can_restart", "can_start"):
            self.assertIn(field, status)
        self.assertEqual(status["port"], 8000)
        self.assertEqual(status["host"], "0.0.0.0")
        self.assertFalse(status["can_start"])    # запустить остановленный из браузера нельзя
        self.assertTrue(status["can_restart"])

    def test_restart_spawns_before_shutdown(self):
        events = []
        with patch("core.server.spawn_successor", side_effect=lambda: events.append("spawn") or 42), \
             patch.object(self.control, "shutdown_soon", side_effect=lambda delay: events.append("shutdown")):
            pid = self.control.restart()
        self.assertEqual(pid, 42)
        self.assertEqual(events, ["spawn", "shutdown"])

    def test_restart_failure_does_not_shutdown(self):
        with patch("core.server.spawn_successor", side_effect=RuntimeError("boom")), \
             patch.object(self.control, "shutdown_soon") as shutdown:
            with self.assertRaises(RuntimeError):
                self.control.restart()
        shutdown.assert_not_called()

    def test_stop_does_not_spawn(self):
        with patch("core.server.spawn_successor") as spawn, \
             patch.object(self.control, "shutdown_soon") as shutdown:
            self.control.stop()
        spawn.assert_not_called()
        shutdown.assert_called_once()

    def test_shutdown_soon_is_deduplicated(self):
        gate = threading.Event()
        started = threading.Event()

        def blocked(delay):
            started.set()
            gate.wait(timeout=2.0)

        with patch.object(self.control, "_shutdown_after", side_effect=blocked):
            self.control.shutdown_soon(0.5)
            started.wait(timeout=2.0)
            first = self.control._stop_thread
            self.assertTrue(first.is_alive())
            self.control.shutdown_soon(0.5)
            self.assertIs(self.control._stop_thread, first)
            gate.set()
            first.join(timeout=2.0)

    def test_shutdown_sets_uvicorn_flag(self):
        server = Mock()
        self.control.uvicorn = server
        self.control.shutdown()
        self.assertTrue(server.should_exit)

    def test_shutdown_falls_back_to_signal(self):
        self.control.uvicorn = None
        with patch("core.server.os.kill") as kill:
            self.control.shutdown()
        kill.assert_called_once_with(os.getpid(), signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
