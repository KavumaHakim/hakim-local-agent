"""Model manager tests. No llama-server and no model files required."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.manager import (
    HARD_FLOOR_MB,
    ModelManager,
    ModelManagerError,
    ModelState,
    load_registry,
)


class FakeProcess:
    """Stands in for subprocess.Popen."""

    def __init__(self, pid: int = 4242, exit_code: int | None = None) -> None:
        self.pid = pid
        self.returncode = exit_code
        self._alive = exit_code is None
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.returncode = 0

    def kill(self):
        self.killed = True
        self._alive = False
        self.returncode = -9

    def wait(self, timeout=None):
        self._alive = False
        return self.returncode


class ManagerHarness(ModelManager):
    """Manager with process spawning and health checks replaced.

    Everything above the OS boundary is the real implementation.
    """

    def __init__(self, registry_path, healthy_ports=(), **kwargs):
        super().__init__(registry_path, start_timeout=5, stop_timeout=1, **kwargs)
        self.healthy_ports = set(healthy_ports)
        self.spawned: list[list[str]] = []
        self.spawn_fails = False
        self.never_healthy = False
        # Plenty, unless a test says otherwise. Without this the suite would
        # pass or fail depending on this machine's free memory.
        self.fake_ram: int | None = 8000

    def _available_ram(self):
        return self.fake_ram

    def _healthy(self, port: int) -> bool:
        return port in self.healthy_ports

    def _spawn(self, command):
        self.spawned.append(command)
        if self.spawn_fails:
            raise OSError("cannot start")
        port = int(command[command.index("--port") + 1])
        if not self.never_healthy:
            self.healthy_ports.add(port)
        return FakeProcess()

    def _terminate(self, process, spec):
        process.terminate()
        self.healthy_ports.discard(spec.port)


def write_registry(tmp: Path, *, files_exist=("a.gguf", "b.gguf")) -> Path:
    exe = tmp / "llama-server.exe"
    exe.write_text("x", encoding="utf-8")
    for name in files_exist:
        (tmp / name).write_bytes(b"gguf")

    registry = {
        "server_exe": str(exe),
        "models_dir": str(tmp),
        "default": "fast",
        "max_active": 1,
        "idle_timeout_seconds": 0,
        "models": [
            {"key": "fast", "label": "Fast 2B", "file": "a.gguf", "port": 8080,
             "min_free_mb": 0},
            {"key": "big", "label": "Big 8B", "file": "b.gguf", "port": 8082,
             "min_free_mb": 0},
            {"key": "gone", "label": "Missing", "file": "nope.gguf", "port": 8083,
             "min_free_mb": 0},
        ],
    }
    path = tmp / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.path = write_registry(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_specs(self):
        registry = load_registry(self.path)
        self.assertEqual(sorted(registry["specs"]), ["big", "fast", "gone"])
        self.assertEqual(registry["specs"]["fast"].port, 8080)
        self.assertEqual(registry["default"], "fast")

    def test_missing_file_is_reported_not_fatal(self):
        registry = load_registry(self.path)
        self.assertTrue(registry["specs"]["fast"].available)
        self.assertFalse(registry["specs"]["gone"].available)

    def test_duplicate_ports_rejected(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["models"][1]["port"] = 8080
        self.path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ModelManagerError) as ctx:
            load_registry(self.path)
        self.assertIn("share a port", str(ctx.exception))

    def test_missing_registry_file(self):
        with self.assertRaises(ModelManagerError):
            load_registry(self.tmp / "absent.json")

    def test_invalid_json(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{nope", encoding="utf-8")
        with self.assertRaises(ModelManagerError):
            load_registry(bad)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.path = write_registry(self.tmp)
        self.manager = ManagerHarness(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_starts_on_ensure(self):
        url = self.manager.ensure("fast")
        self.assertEqual(url, "http://127.0.0.1:8080")
        self.assertEqual(self.manager.status("fast").state, ModelState.READY)
        self.assertEqual(len(self.manager.spawned), 1)

    def test_command_carries_expected_flags(self):
        self.manager.ensure("fast")
        command = self.manager.spawned[0]
        self.assertIn("--jinja", command)
        self.assertIn("--port", command)
        self.assertEqual(command[command.index("--port") + 1], "8080")
        self.assertEqual(command[command.index("-np") + 1], "1")

    def test_second_ensure_does_not_respawn(self):
        self.manager.ensure("fast")
        self.manager.ensure("fast")
        self.assertEqual(len(self.manager.spawned), 1)

    def test_switching_stops_the_other_model(self):
        self.manager.ensure("fast")
        self.manager.ensure("big")

        self.assertEqual(self.manager.status("big").state, ModelState.READY)
        self.assertEqual(self.manager.status("fast").state, ModelState.STOPPED)
        self.assertEqual(self.manager.active_key(), "big")

    def test_only_one_model_ready_at_a_time(self):
        self.manager.ensure("fast")
        self.manager.ensure("big")
        ready = [s.spec.key for s in self.manager.statuses()
                 if s.state is ModelState.READY]
        self.assertEqual(ready, ["big"])

    def test_missing_weights_refused(self):
        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("gone")
        self.assertIn("weights not found", str(ctx.exception))
        self.assertEqual(self.manager.spawned, [])

    def test_unknown_key_refused(self):
        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("nope")
        self.assertIn("Unknown model", str(ctx.exception))

    def test_stop_marks_stopped(self):
        self.manager.ensure("fast")
        self.assertTrue(self.manager.stop("fast"))
        self.assertEqual(self.manager.status("fast").state, ModelState.STOPPED)

    def test_failure_to_become_healthy_cleans_up(self):
        self.manager.never_healthy = True
        with self.assertRaises(ModelManagerError):
            self.manager.ensure("fast")
        # FAILED, not STOPPED: it was asked to run and did not.
        self.assertEqual(self.manager.status("fast").state, ModelState.FAILED)
        # The half-started process must not be left behind.
        self.assertNotIn("fast", self.manager._processes)

    def test_spawn_failure_reported(self):
        self.manager.spawn_fails = True
        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("fast")
        self.assertIn("Could not start", str(ctx.exception))


class AdoptionTests(unittest.TestCase):
    """A server already listening on the port is adopted, not duplicated."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.path = write_registry(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_existing_server_is_adopted(self):
        manager = ManagerHarness(self.path, healthy_ports=[8080])
        url = manager.ensure("fast")

        self.assertEqual(url, "http://127.0.0.1:8080")
        self.assertEqual(manager.spawned, [])  # nothing started
        self.assertTrue(manager.status("fast").adopted)

    # Stopping and switching away from an adopted server is covered in
    # tests/test_port_reclaim.py, which fakes the port listener. Testing it
    # here would shell out to netstat and depend on what happens to be
    # running on this machine.


class IdleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.path = write_registry(self.tmp)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["idle_timeout_seconds"] = 60
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.manager = ManagerHarness(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_recently_used_model_is_kept(self):
        self.manager.ensure("fast")
        self.assertEqual(self.manager.unload_idle(), [])
        self.assertEqual(self.manager.status("fast").state, ModelState.READY)

    def test_idle_model_is_unloaded(self):
        self.manager.ensure("fast")
        status = self.manager.status("fast")
        status.last_used -= 120  # pretend two minutes passed

        self.assertEqual(self.manager.unload_idle(), ["fast"])
        self.assertEqual(self.manager.status("fast").state, ModelState.STOPPED)

    def test_disabled_timeout_never_unloads(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["idle_timeout_seconds"] = 0
        self.path.write_text(json.dumps(data), encoding="utf-8")
        manager = ManagerHarness(self.path)
        manager.ensure("fast")
        manager.status("fast").last_used -= 10_000
        self.assertEqual(manager.unload_idle(), [])


class CrashDetectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.path = write_registry(self.tmp)
        self.manager = ManagerHarness(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dead_process_becomes_failed(self):
        self.manager.ensure("fast")
        # Simulate the server dying on its own.
        self.manager._processes["fast"].returncode = 1
        self.manager._processes["fast"]._alive = False
        self.manager.healthy_ports.discard(8080)

        status = self.manager.status("fast")
        self.assertEqual(status.state, ModelState.FAILED)
        self.assertIn("exited with code 1", status.error)

    def test_restart_after_crash(self):
        self.manager.ensure("fast")
        self.manager._processes["fast"].returncode = 1
        self.manager._processes["fast"]._alive = False
        self.manager.healthy_ports.discard(8080)
        self.manager.refresh()

        self.manager.ensure("fast")
        self.assertEqual(self.manager.status("fast").state, ModelState.READY)
        self.assertEqual(len(self.manager.spawned), 2)


if __name__ == "__main__":
    unittest.main()


class RamGuardTests(unittest.TestCase):
    """The guard accounts for mmap: a shortfall means slow, not broken."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        path = write_registry(self.tmp)
        # Give 'fast' a figure well above anything we will simulate.
        data = json.loads(path.read_text(encoding="utf-8"))
        for model in data["models"]:
            model["min_free_mb"] = 2000
        path.write_text(json.dumps(data), encoding="utf-8")
        self.manager = ManagerHarness(path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_plenty_of_ram_starts_without_warning(self):
        self.manager.fake_ram = 8000
        self.manager.ensure("fast")
        self.assertEqual(self.manager.status("fast").state, ModelState.READY)
        self.assertEqual(self.manager.status("fast").warning, "")

    def test_shortfall_starts_anyway_with_a_warning(self):
        # The case that used to be refused: below the model's figure but well
        # above the floor. Measured on this machine, it loads and runs.
        self.manager.fake_ram = 500
        self.manager.ensure("fast")

        status = self.manager.status("fast")
        self.assertEqual(status.state, ModelState.READY)
        self.assertIn("slower generation", status.warning)
        self.assertIn("500 MB", status.warning)

    def test_below_the_hard_floor_is_refused(self):
        self.manager.fake_ram = HARD_FLOOR_MB - 1
        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("fast")
        self.assertIn("floor", str(ctx.exception))
        self.assertEqual(self.manager.spawned, [])

    def test_unknown_ram_never_blocks(self):
        self.manager.fake_ram = None
        self.manager.ensure("fast")
        self.assertEqual(self.manager.status("fast").state, ModelState.READY)
        self.assertEqual(self.manager.status("fast").warning, "")

    def test_warning_clears_on_a_later_healthy_start(self):
        self.manager.fake_ram = 500
        self.manager.ensure("fast")
        self.assertTrue(self.manager.status("fast").warning)

        self.manager.stop("fast")
        self.manager.fake_ram = 8000
        self.manager.ensure("fast")
        self.assertEqual(self.manager.status("fast").warning, "")
