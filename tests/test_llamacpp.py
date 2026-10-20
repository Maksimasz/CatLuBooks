"""Проверки управляемого llama-server: модели, запуск, остановка, состояние.

Процессы не запускаются по-настоящему: Popen, проверка порта и «жив ли
процесс» подменены, поэтому тесты одинаково идут и на Windows, и на Linux.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.llamacpp import LlamaCppError, LlamaServer

PORT = 8099


class _Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = LlamaServer(root=self.root,
                                  models_dir=self.root / "models",
                                  port=PORT)

    def make_binary(self) -> Path:
        path = self.root / "llama.cpp" / "llama-server.exe"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"exe")
        return path

    def make_model(self, name: str = "qwen2.5-3b.gguf") -> Path:
        path = self.root / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return path


class ModelListTests(_Base):
    def test_lists_gguf_including_subfolders(self):
        (self.root / "models").mkdir(exist_ok=True)
        (self.root / "models" / "Qwen.gguf").write_bytes(b"x" * 2048)
        (self.root / "models" / "sub").mkdir()
        (self.root / "models" / "sub" / "tiny.gguf").write_bytes(b"y")
        (self.root / "models" / "readme.txt").write_text("не модель", encoding="utf-8")

        models = self.server.models()

        self.assertEqual([model["name"] for model in models],
                         ["Qwen.gguf", "sub/tiny.gguf"])
        self.assertEqual(models[0]["size"], 2048)
        self.assertIn("КБ", models[0]["size_text"])

    def test_model_path_accepts_a_file_from_models(self):
        path = self.make_model()

        self.assertEqual(self.server.model_path("qwen2.5-3b.gguf"), path.resolve())

    def test_model_path_rejects_absolute_and_parent_paths(self):
        for name in ("../secret.gguf", "/abs.gguf", "C:/secret.gguf",
                     "book.pdf", ""):
            with self.subTest(name=name), self.assertRaises(LlamaCppError):
                self.server.model_path(name)

    def test_model_path_rejects_missing_file(self):
        with self.assertRaises(LlamaCppError) as ctx:
            self.server.model_path("нет-такой.gguf")

        self.assertIn("не найден", str(ctx.exception))


class StatusTests(_Base):
    def test_off_when_nothing_runs(self):
        with patch("core.llamacpp._alive", return_value=False), \
                patch.object(self.server, "_responding", return_value=False):
            status = self.server.status()

        self.assertEqual(status["state"], "off")
        self.assertFalse(status["can_stop"])
        self.assertEqual(status["port"], PORT)
        self.assertEqual(status["url"], f"http://127.0.0.1:{PORT}/v1")
        self.assertEqual(status["error"], "")

    def test_starting_while_the_process_loads_the_model(self):
        self.server._save_state({"pid": 4242, "model": "qwen.gguf", "alias": "qwen"})

        with patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4242), \
                patch.object(self.server, "_responding", return_value=False):
            status = self.server.status()

        self.assertEqual(status["state"], "starting")
        self.assertEqual(status["pid"], 4242)
        self.assertTrue(status["can_stop"])
        self.assertFalse(status["can_start"])

    def test_running_when_the_process_answers(self):
        self.server._save_state({"pid": 4242, "model": "qwen.gguf", "alias": "qwen"})

        with patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4242), \
                patch.object(self.server, "_responding", return_value=True):
            status = self.server.status()

        self.assertEqual(status["state"], "running")
        self.assertEqual(status["model"], "qwen.gguf")
        self.assertEqual(status["alias"], "qwen")

    def test_external_when_the_port_answers_without_our_process(self):
        # llama-server запущен отдельно: приложение его не трогает.
        with patch("core.llamacpp._alive", return_value=False), \
                patch.object(self.server, "_responding", return_value=True):
            status = self.server.status()

        self.assertEqual(status["state"], "external")
        self.assertFalse(status["can_start"])
        self.assertFalse(status["can_stop"])

    def test_finished_process_is_reported_with_a_hint(self):
        self.server._save_state({"pid": 4242, "model": "qwen.gguf"})

        with patch("core.llamacpp._alive", return_value=False), \
                patch.object(self.server, "_responding", return_value=False):
            status = self.server.status()

        self.assertEqual(status["state"], "off")
        self.assertIn("завершился", status["error"])

    def test_models_dir_is_created(self):
        self.assertTrue((self.root / "models").is_dir())


class StartTests(_Base):
    def test_start_builds_the_command_and_remembers_the_process(self):
        binary = self.make_binary()
        model = self.make_model()
        calls = {}

        class FakeProcess:
            pid = 4242

        def fake_popen(command, **kwargs):
            calls["command"] = command
            calls["kwargs"] = kwargs
            return FakeProcess()

        with patch("core.llamacpp.subprocess.Popen", side_effect=fake_popen), \
                patch("core.llamacpp.port_is_free", return_value=True), \
                patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4242), \
                patch.object(self.server, "_responding", return_value=False), \
                patch("core.llamacpp.time.sleep"):
            status = self.server.start("qwen2.5-3b.gguf")

        command = calls["command"]
        self.assertEqual(command[0], str(binary))
        self.assertEqual(command[command.index("-m") + 1], str(model.resolve()))
        self.assertEqual(command[command.index("--alias") + 1], "qwen2.5-3b")
        self.assertEqual(command[command.index("--port") + 1], str(PORT))
        self.assertEqual(calls["kwargs"]["cwd"], str(self.root))

        state = json.loads(self.server.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["pid"], 4242)
        self.assertEqual(state["alias"], "qwen2.5-3b")
        self.assertEqual(status["state"], "starting")

    def test_extra_args_go_to_the_command(self):
        self.make_binary()
        self.make_model()
        self.server.args = ["-c", "8192", "-ngl", "99"]
        calls = {}

        class FakeProcess:
            pid = 4243

        def fake_popen(command, **kwargs):
            calls["cmd"] = command
            return FakeProcess()

        with patch("core.llamacpp.subprocess.Popen", side_effect=fake_popen), \
                patch("core.llamacpp.port_is_free", return_value=True), \
                patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4243), \
                patch.object(self.server, "_responding", return_value=False), \
                patch("core.llamacpp.time.sleep"):
            self.server.start("qwen2.5-3b.gguf")

        command = calls["cmd"]
        self.assertIn("-ngl", command)
        self.assertEqual(command[command.index("-ngl") + 1], "99")
        self.assertEqual(command[command.index("-c") + 1], "8192")

    def test_start_refuses_when_already_running(self):
        self.make_binary()
        self.make_model()
        self.server._save_state({"pid": 4242, "model": "qwen.gguf"})

        with patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4242), \
                patch.object(self.server, "_responding", return_value=True):
            with self.assertRaises(LlamaCppError) as ctx:
                self.server.start("qwen2.5-3b.gguf")

        self.assertIn("уже запущен", str(ctx.exception))

    def test_start_refuses_a_busy_port(self):
        self.make_binary()
        self.make_model()

        with patch.object(self.server, "_responding", return_value=False), \
                patch("core.llamacpp._alive", return_value=False), \
                patch("core.llamacpp.port_is_free", return_value=False):
            with self.assertRaises(LlamaCppError) as ctx:
                self.server.start("qwen2.5-3b.gguf")

        self.assertIn("занят", str(ctx.exception))

    def test_start_without_binary_explains_where_to_get_it(self):
        self.make_model()

        with patch("core.llamacpp.shutil.which", return_value=None):
            with self.assertRaises(LlamaCppError) as ctx:
                self.server.start("qwen2.5-3b.gguf")

        self.assertIn("llama.cpp/releases", str(ctx.exception))


class StopTests(_Base):
    def test_stop_kills_the_process_and_clears_the_state(self):
        self.server._save_state({"pid": 4242, "model": "qwen.gguf"})
        killed = []

        with patch("core.llamacpp._alive", side_effect=lambda pid: pid == 4242), \
                patch("core.llamacpp._kill", side_effect=killed.append), \
                patch.object(self.server, "_responding", return_value=False), \
                patch("core.llamacpp.time.sleep"):
            status = self.server.stop()

        self.assertEqual(killed, [4242])
        self.assertFalse(self.server.state_path.exists())
        self.assertEqual(status["state"], "off")

    def test_stop_without_a_process_is_harmless(self):
        killed = []

        with patch("core.llamacpp._alive", return_value=False), \
                patch("core.llamacpp._kill", side_effect=killed.append), \
                patch.object(self.server, "_responding", return_value=False), \
                patch("core.llamacpp.time.sleep"):
            status = self.server.stop()

        self.assertEqual(killed, [])
        self.assertEqual(status["state"], "off")


class LogTests(_Base):
    def test_tail_returns_the_last_lines(self):
        self.server.log_path.write_text(
            "\n".join(f"строка {number}" for number in range(100)), encoding="utf-8")

        tail = self.server.log_tail(5).splitlines()

        self.assertEqual(len(tail), 5)
        self.assertEqual(tail[-1], "строка 99")

    def test_tail_without_a_log_is_empty(self):
        self.assertEqual(self.server.log_tail(), "")


class BinaryTests(_Base):
    def test_binary_is_found_next_to_the_project(self):
        binary = self.make_binary()

        self.assertEqual(self.server.binary_path(), binary)

    def test_binary_from_config_wins(self):
        binary = self.make_binary()
        self.server.binary = "llama.cpp/llama-server.exe"

        self.assertEqual(self.server.binary_path(), binary)

    def test_missing_binary_from_config_is_reported(self):
        self.server.binary = "нет/llama-server.exe"

        self.assertIsNone(self.server.binary_path())

    def test_binary_can_come_from_path(self):
        with patch("core.llamacpp.shutil.which",
                   side_effect=lambda name: r"C:\tools\llama-server.exe"):
            self.assertEqual(str(self.server.binary_path()),
                             r"C:\tools\llama-server.exe")


if __name__ == "__main__":
    unittest.main()
