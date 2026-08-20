"""Turn queue tests. No model, no HTTP, no database.

The queue is the piece that replaces a property Streamlit gave away free -
that only one turn can run at a time - so it is tested on its own rather than
only through the endpoint that uses it.
"""

from __future__ import annotations

import threading
import time
import unittest

from api.turns import Turn, TurnQueue, TurnQueueFull, TurnRequest, drain


def make_turn(prompt: str = "hello", conversation_id: int = 1) -> Turn:
    return Turn(
        request=TurnRequest(
            conversation_id=conversation_id,
            prompt=prompt,
            user_message_id=1,
            model_key="fast",
        )
    )


class TurnQueueTests(unittest.TestCase):
    def tearDown(self) -> None:
        queue = getattr(self, "queue", None)
        if queue is not None:
            queue.stop(timeout=2)

    def test_runs_a_turn_and_closes_the_stream(self):
        def runner(turn: Turn) -> None:
            turn.emit("token", text="hi")

        self.queue = TurnQueue(runner)
        self.queue.start()
        turn = self.queue.submit(make_turn())

        events = [event for event in drain(turn, timeout=0.05) if event]
        self.assertEqual([e["type"] for e in events], ["token"])
        self.assertEqual(turn.state, "finished")

    def test_turns_run_in_submission_order(self):
        order: list[str] = []

        def runner(turn: Turn) -> None:
            order.append(turn.request.prompt)

        self.queue = TurnQueue(runner)
        self.queue.start()
        turns = [self.queue.submit(make_turn(prompt=str(n))) for n in range(5)]
        for turn in turns:
            list(drain(turn, timeout=0.05))

        self.assertEqual(order, ["0", "1", "2", "3", "4"])

    def test_only_one_turn_runs_at_a_time(self):
        """The whole point of the queue: two cores, one model."""
        concurrent = 0
        peak = 0
        lock = threading.Lock()

        def runner(turn: Turn) -> None:
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.02)
            with lock:
                concurrent -= 1

        self.queue = TurnQueue(runner)
        self.queue.start()
        turns = [self.queue.submit(make_turn()) for _ in range(4)]
        for turn in turns:
            list(drain(turn, timeout=0.05))

        self.assertEqual(peak, 1)

    def test_position_counts_turns_ahead(self):
        release = threading.Event()

        def runner(turn: Turn) -> None:
            release.wait(timeout=2)

        self.queue = TurnQueue(runner)
        self.queue.start()

        first = self.queue.submit(make_turn())
        # Let the worker pick the first one up before measuring the rest.
        deadline = time.monotonic() + 2
        while not self.queue.busy() and time.monotonic() < deadline:
            time.sleep(0.005)

        second = self.queue.submit(make_turn())
        third = self.queue.submit(make_turn())

        self.assertEqual(self.queue.position(first), 0)
        self.assertEqual(self.queue.position(second), 1)
        self.assertEqual(self.queue.position(third), 2)
        self.assertEqual(self.queue.depth(), 2)

        release.set()
        for turn in (first, second, third):
            list(drain(turn, timeout=0.05))

    def test_backlog_is_bounded(self):
        release = threading.Event()

        def runner(turn: Turn) -> None:
            release.wait(timeout=2)

        self.queue = TurnQueue(runner, max_waiting=2)
        self.queue.start()

        self.queue.submit(make_turn())
        deadline = time.monotonic() + 2
        while not self.queue.busy() and time.monotonic() < deadline:
            time.sleep(0.005)

        self.queue.submit(make_turn())
        self.queue.submit(make_turn())
        with self.assertRaises(TurnQueueFull):
            self.queue.submit(make_turn())

        release.set()

    def test_a_runner_that_raises_does_not_kill_the_worker(self):
        """A dead worker would hang every later turn, not just the bad one."""
        def runner(turn: Turn) -> None:
            if turn.request.prompt == "boom":
                raise RuntimeError("tool exploded")
            turn.emit("token", text="fine")

        self.queue = TurnQueue(runner)
        self.queue.start()

        bad = self.queue.submit(make_turn(prompt="boom"))
        events = [event for event in drain(bad, timeout=0.05) if event]
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("tool exploded", events[0]["message"])

        good = self.queue.submit(make_turn(prompt="ok"))
        events = [event for event in drain(good, timeout=0.05) if event]
        self.assertEqual([e["type"] for e in events], ["token"])

    def test_drain_reports_idle_ticks(self):
        """Idle ticks are how the stream sends queue positions and heartbeats."""
        release = threading.Event()

        def runner(turn: Turn) -> None:
            release.wait(timeout=2)
            turn.emit("done", content="ok")

        self.queue = TurnQueue(runner)
        self.queue.start()
        turn = self.queue.submit(make_turn())

        ticks = 0
        collected = []
        for event in drain(turn, timeout=0.01):
            if event is None:
                ticks += 1
                if ticks == 3:
                    release.set()
                continue
            collected.append(event)

        self.assertGreaterEqual(ticks, 3)
        self.assertEqual([e["type"] for e in collected], ["done"])

    def test_submitting_after_stop_is_refused(self):
        self.queue = TurnQueue(lambda turn: None)
        self.queue.start()
        self.queue.stop(timeout=2)
        with self.assertRaises(TurnQueueFull):
            self.queue.submit(make_turn())


if __name__ == "__main__":
    unittest.main()
