"""Model manager tests. No llama-server and no model files required."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.manager import (
    parse_meminfo,
    parse_ss_pid,
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


class GpuLayerTests(unittest.TestCase):
    """Handing layers to the GPU, and not doing it by accident."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.path = write_registry(self.tmp)

    def gpu_layers_on_the_command_line(self, manager) -> str:
        manager.ensure("fast")
        command = manager.spawned[0]
        return command[command.index("-ngl") + 1]

    def test_nothing_is_offloaded_unless_asked(self):
        """The default has to be zero.

        The CPU build is what the setup script installs, so a non-zero default
        would be asking every fresh install to offload onto hardware it has no
        way of knowing exists.
        """
        manager = ManagerHarness(self.path)
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "0")

    def test_the_flag_is_always_passed_even_at_zero(self):
        """Explicit, so that swapping in an accelerator build changes nothing
        on its own. A build whose default is to offload would otherwise start
        offloading the moment `server_exe` moved."""
        manager = ManagerHarness(self.path)
        manager.ensure("fast")
        self.assertIn("-ngl", manager.spawned[0])

    def test_a_registry_entry_can_ask_for_layers(self):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["models"][0]["gpu_layers"] = 99
        self.path.write_text(json.dumps(raw), encoding="utf-8")

        manager = ManagerHarness(self.path)
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "99")

    def test_the_settings_panel_can_ask_for_layers(self):
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 99})
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "99")

    def test_zero_survives_being_set_deliberately(self):
        """Turning offloading back off is a real edit, not an empty one - so
        it must not be dropped the way a blank field is."""
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["models"][0]["gpu_layers"] = 99
        self.path.write_text(json.dumps(raw), encoding="utf-8")

        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 0})
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "0")

    def test_clearing_the_override_goes_back_to_the_registry(self):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["models"][0]["gpu_layers"] = 32
        self.path.write_text(json.dumps(raw), encoding="utf-8")

        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 0})
        manager.clear_override("fast")
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "32")

    def test_a_nonsense_value_is_clamped_rather_than_refused(self):
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 100_000})
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "999")

    def test_a_negative_value_becomes_none(self):
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": -5})
        self.assertEqual(self.gpu_layers_on_the_command_line(manager), "0")

    def test_the_setting_survives_a_restart(self):
        """It is written to preferences, not held in memory - otherwise the
        one setting you have to restart a model to apply would be lost by
        restarting the application."""
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 99})

        reopened = ManagerHarness(self.path)
        self.assertEqual(self.gpu_layers_on_the_command_line(reopened), "99")

    def test_offloading_raises_what_the_model_wants_free(self):
        """An integrated GPU has no memory of its own, so a weight handed to
        it is still in the same DIMMs. Measured: a 738 MB model peaked at
        944 MB with nothing offloaded and 1,169 MB with all of it."""
        manager = ManagerHarness(self.path)
        # The fixture guards nothing by default, and 0 means "do not guard".
        manager.set_override("fast", {"min_free_mb": 1000})
        plain = manager._ram_wanted(manager.get_spec("fast"))

        manager.set_override("fast", {"gpu_layers": 99})
        offloaded = manager._ram_wanted(manager.get_spec("fast"))
        self.assertGreater(offloaded, plain)

    def test_not_offloading_changes_nothing_about_the_guard(self):
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"min_free_mb": 1000})
        spec = manager.get_spec("fast")
        self.assertEqual(manager._ram_wanted(spec), spec.min_free_mb)

    def test_a_model_with_no_threshold_gains_no_offload_charge(self):
        """0 means "do not guard this one". Adding an offload charge to it
        would invent a threshold nobody asked for."""
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 99, "min_free_mb": 0})
        self.assertEqual(manager._ram_wanted(manager.get_spec("fast")), 0)

    def test_the_warning_reports_the_higher_figure_when_offloading(self):
        """Saying 900 MB while wanting 1,119 MB is worse than not warning."""
        manager = ManagerHarness(self.path)
        manager.set_override("fast", {"gpu_layers": 99, "min_free_mb": 1000})
        manager.fake_ram = 1050
        manager.ensure("fast")

        warning = manager._statuses["fast"].warning
        self.assertIn("1050 MB", warning)
        self.assertNotIn("about 1000 MB", warning)

    def test_the_server_binary_is_reported(self):
        """The panel cannot ask about GPU layers without saying which
        llama-server they would be handed to."""
        manager = ManagerHarness(self.path)
        self.assertEqual(manager.server_exe, self.tmp / "llama-server.exe")


class ServerChoiceTests(unittest.TestCase):
    """Pointing this machine at a different llama-server build."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.path = write_registry(self.tmp)
        self.other = self.tmp / "llama-server-vulkan.exe"
        self.other.write_text("x", encoding="utf-8")

    def test_the_chosen_binary_is_what_gets_run(self):
        manager = ManagerHarness(self.path)
        manager.set_server_exe(str(self.other))
        manager.ensure("fast")
        self.assertEqual(manager.spawned[0][0], str(self.other))

    def test_the_choice_survives_a_restart(self):
        """It goes to preferences, not to models.json: models.json is in
        version control, and a path committed from one laptop is wrong on
        every other machine."""
        ManagerHarness(self.path).set_server_exe(str(self.other))
        self.assertEqual(ManagerHarness(self.path).server_exe, self.other)

    def test_an_empty_path_goes_back_to_searching(self):
        manager = ManagerHarness(self.path)
        original = manager.server_exe
        manager.set_server_exe(str(self.other))
        manager.set_server_exe("")
        self.assertEqual(manager.server_exe, original)

    def test_a_path_with_nothing_at_it_is_refused(self):
        """Refused rather than remembered. A stored path to nothing would turn
        every later start into the same confusing failure."""
        manager = ManagerHarness(self.path)
        original = manager.server_exe
        with self.assertRaises(ModelManagerError):
            manager.set_server_exe(str(self.tmp / "ghost.exe"))
        self.assertEqual(manager.server_exe, original)

    def test_models_are_not_forgotten_when_the_server_changes(self):
        """It rescans, and a rescan that lost the catalogue would be worse
        than the setting is worth."""
        manager = ManagerHarness(self.path)
        before = {spec.key for spec in manager.specs()}
        manager.set_server_exe(str(self.other))
        self.assertEqual({spec.key for spec in manager.specs()}, before)


class PlatformTests(unittest.TestCase):
    """The Linux halves of the process layer, exercised from any machine.

    These parse what `ss` and /proc/meminfo produce. The calls themselves
    cannot be made on Windows, but the parsing is where the bugs live, and
    untested platform code is a promise rather than a feature.
    """

    MEMINFO = (
        "MemTotal:        8123456 kB\n"
        "MemFree:          123456 kB\n"
        "MemAvailable:    1908736 kB\n"
        "Buffers:           45678 kB\n"
    )

    def test_available_memory_is_read_from_meminfo(self):
        self.assertEqual(parse_meminfo(self.MEMINFO), 1864)

    def test_it_reads_available_not_free(self):
        """MemFree excludes the page cache, which the kernel hands back on
        demand - using it would refuse to start a model on a machine with
        plenty of room."""
        self.assertNotEqual(parse_meminfo(self.MEMINFO), 120)

    def test_meminfo_without_the_field_is_unknown_rather_than_zero(self):
        self.assertIsNone(parse_meminfo("MemTotal: 8123456 kB\n"))
        self.assertIsNone(parse_meminfo(""))

    def test_a_malformed_meminfo_line_is_unknown(self):
        self.assertIsNone(parse_meminfo("MemAvailable:    not-a-number kB\n"))

    def test_the_listening_pid_is_read_from_ss(self):
        line = (
            'LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:* '
            'users:(("llama-server",pid=8123,fd=7))'
        )
        self.assertEqual(parse_ss_pid(line), 8123)

    def test_ss_output_with_no_process_column_yields_nothing(self):
        """Without root, ss omits the users:(...) column entirely."""
        self.assertIsNone(parse_ss_pid("LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:*"))
        self.assertIsNone(parse_ss_pid(""))

    def test_the_first_listener_wins_when_several_are_reported(self):
        output = (
            'LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:* users:(("first",pid=11,fd=7))\n'
            'LISTEN 0 4096 [::1]:8080 [::]:* users:(("second",pid=22,fd=8))\n'
        )
        self.assertEqual(parse_ss_pid(output), 11)


class MeminfoStatusTests(unittest.TestCase):
    """The Linux memory probe, parsed without a Linux machine."""

    TEXT = "MemTotal:        8000000 kB\nMemFree:         100000 kB\nMemAvailable:    2000000 kB\n"

    def test_total_available_and_a_derived_load(self):
        from models.manager import parse_meminfo_status

        status = parse_meminfo_status(self.TEXT)
        self.assertEqual(status.total_mb, 7812)
        self.assertEqual(status.available_mb, 1953)
        # Used over total, the way Windows reports dwMemoryLoad.
        self.assertEqual(status.load_percent, 75)

    def test_the_old_parser_still_answers_available_only(self):
        from models.manager import parse_meminfo

        self.assertEqual(parse_meminfo(self.TEXT), 1953)

    def test_missing_or_malformed_lines_give_none(self):
        from models.manager import parse_meminfo_status

        self.assertIsNone(parse_meminfo_status("MemTotal: 8000000 kB\n"))
        self.assertIsNone(parse_meminfo_status("MemTotal: lots\nMemAvailable: 1 kB\n"))
        self.assertIsNone(parse_meminfo_status(""))
