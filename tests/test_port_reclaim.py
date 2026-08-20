"""Reclaiming a port held by a llama-server this manager did not start.

The case that matters: restart the UI, and every server started by the old
process looks foreign. Refusing to touch those made switching impossible.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.manager import ModelManagerError, ModelState
from tests.test_manager import ManagerHarness, write_registry


class ReclaimHarness(ManagerHarness):
    """Adds a fake view of what is listening on each port."""

    def __init__(self, registry_path, **kwargs):
        super().__init__(registry_path, **kwargs)
        # port -> (pid, process name)
        self.listeners: dict[int, tuple[int, str]] = {}
        self.killed: list[int] = []

    def listener_pid(self, port):
        entry = self.listeners.get(port)
        return entry[0] if entry else None

    def process_name(self, pid):
        for _, (listener_pid, name) in self.listeners.items():
            if listener_pid == pid:
                return name
        return ""

    def _run_quiet(self, command):
        if command and command[0] == "taskkill":
            pid = int(command[-1])
            self.killed.append(pid)
            for port, (listener_pid, _) in list(self.listeners.items()):
                if listener_pid == pid:
                    del self.listeners[port]
                    self.healthy_ports.discard(port)
            return ""
        return ""


def harness(tmp: Path) -> ReclaimHarness:
    return ReclaimHarness(write_registry(tmp))


class AdoptedServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.manager = harness(self.tmp)
        # A llama-server left behind on 'fast' by a process that has exited.
        self.manager.healthy_ports.add(8080)
        self.manager.listeners[8080] = (4242, "llama-server.exe")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_leftover_server_is_adopted(self):
        self.manager.ensure("fast")
        self.assertTrue(self.manager.status("fast").adopted)
        self.assertEqual(self.manager.spawned, [])

    def test_switching_away_reclaims_the_port(self):
        self.manager.ensure("fast")          # adopts the leftover
        self.manager.ensure("big")           # must now be able to switch

        self.assertEqual(self.manager.status("big").state, ModelState.READY)
        self.assertEqual(self.manager.status("fast").state, ModelState.STOPPED)
        self.assertIn(4242, self.manager.killed)

    def test_stopping_an_adopted_server_works(self):
        self.manager.ensure("fast")
        self.assertTrue(self.manager.stop("fast"))
        self.assertIn(4242, self.manager.killed)
        self.assertFalse(self.manager.status("fast").adopted)

    def test_a_foreign_process_is_left_alone(self):
        # Something that is not a llama-server holds the port.
        self.manager.listeners[8080] = (777, "nginx.exe")
        self.manager.ensure("fast")

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("big")
        self.assertIn("not a llama-server", str(ctx.exception))
        self.assertNotIn(777, self.manager.killed)

    def test_stop_refuses_a_foreign_process(self):
        self.manager.listeners[8080] = (777, "nginx.exe")
        self.manager.ensure("fast")
        self.assertFalse(self.manager.stop("fast"))
        self.assertEqual(self.manager.killed, [])

    def test_vanished_listener_is_not_an_error(self):
        # The process died between the health check and the kill.
        self.manager.ensure("fast")
        self.manager.listeners.clear()
        self.manager.healthy_ports.discard(8080)
        self.manager.ensure("big")
        self.assertEqual(self.manager.status("big").state, ModelState.READY)


class OwnProcessTests(unittest.TestCase):
    """Servers we started still go through the normal path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.manager = harness(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_our_own_server_is_terminated_not_taskkilled(self):
        self.manager.ensure("fast")
        self.manager.ensure("big")
        # Stopped through Popen.terminate, so no taskkill was needed.
        self.assertEqual(self.manager.killed, [])
        self.assertEqual(self.manager.status("fast").state, ModelState.STOPPED)


if __name__ == "__main__":
    unittest.main()
