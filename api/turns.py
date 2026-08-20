"""Serialised execution of agent turns, with a queue the caller can watch.

Streamlit serialised turns for free: one script run at a time, so a second
request simply could not start. An HTTP API has no such property, and this
machine cannot survive losing it - `models.json` sets `max_active: 1` and the
CPU has two cores, so two turns at once means both models thrash and neither
finishes.

So turns run one at a time on a single worker thread, and everything else
queues behind. A queued caller is told its position rather than left waiting:
a turn here can take minutes (285.7 s measured for one calculator turn), and
an unexplained five-minute silence is indistinguishable from a hang.

The queue is deliberately not a thread pool. Widening it would not make
anything faster - it would only spread the same two cores thinner.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class TurnQueueFull(Exception):
    """The backlog is longer than the queue is willing to hold."""


@dataclass(frozen=True)
class TurnRequest:
    """Everything needed to run one turn, resolved before it is queued."""

    conversation_id: int
    prompt: str
    # The stored id of the user message. History for the model is rebuilt from
    # everything before it, which is exact even when several turns for the same
    # conversation are queued at once.
    user_message_id: int
    model_key: str
    enable_thinking: bool = False
    auto_route: bool = False


@dataclass
class Turn:
    """One queued or running turn, plus the events it has produced."""

    request: TurnRequest
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "queued"  # queued | running | finished
    # SimpleQueue is unbounded and lock-free enough for one producer and one
    # consumer, which is exactly the shape here: the worker thread writes, the
    # streaming response reads.
    events: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)

    def emit(self, type: str, **data: Any) -> None:
        """Publish one event to whoever is streaming this turn."""
        self.events.put({"type": type, **data})

    def close(self) -> None:
        """Signal that no further events will arrive."""
        self.state = "finished"
        self.events.put(None)


# Runs one turn to completion, emitting events into it. Injected so the queue
# can be tested without a model, a registry or a database.
Runner = Callable[[Turn], None]


class TurnQueue:
    """Runs turns one at a time on a background thread.

    Ordinary use is `submit()` from a request handler and `position()` from the
    streaming response while it waits.
    """

    def __init__(self, runner: Runner, *, max_waiting: int = 8) -> None:
        self._runner = runner
        self._max_waiting = max_waiting
        # One lock guards both the backlog and the running turn, so `position`
        # can never report a turn as both queued and running.
        self._condition = threading.Condition(threading.Lock())
        self._waiting: list[Turn] = []
        self._current: Turn | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False

    # --- lifecycle ---

    def start(self) -> None:
        """Start the worker. Idempotent, so a double lifespan call is safe."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(
            target=self._loop, name="agent-turns", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting work and wait briefly for the current turn.

        The wait is short on purpose. A turn in progress can have minutes left
        and shutdown cannot wait for that, so the thread is a daemon and the
        process is allowed to exit out from under it. Nothing is lost that was
        not already going to be lost: the assistant message is only written
        once the turn completes.
        """
        with self._condition:
            self._stopping = True
            self._waiting.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- submission ---

    def submit(self, turn: Turn) -> Turn:
        """Add a turn to the backlog."""
        with self._condition:
            if self._stopping:
                raise TurnQueueFull("The server is shutting down.")
            if len(self._waiting) >= self._max_waiting:
                raise TurnQueueFull(
                    f"{len(self._waiting)} turns are already waiting. On this "
                    f"hardware that is well over an hour of work, so the queue "
                    f"refuses more rather than pretending they will run soon."
                )
            self._waiting.append(turn)
            self._condition.notify()
        return turn

    def position(self, turn: Turn) -> int:
        """How many turns must finish before this one starts.

        0 means nothing is ahead of it - either it is already running, or it is
        next and the worker has simply not picked it up yet. That distinction
        is invisible from outside and lasts microseconds, and counting it as a
        turn ahead would tell the user they are behind a queue of one that does
        not exist.
        """
        with self._condition:
            if self._current is turn:
                return 0
            try:
                index = self._waiting.index(turn)
            except ValueError:
                # Finished, or never submitted; either way nothing is ahead.
                return 0
            # The running turn, if there is one, is also ahead of this.
            return index + (1 if self._current is not None else 0)

    def depth(self) -> int:
        """Turns waiting, not counting the one running."""
        with self._condition:
            return len(self._waiting)

    def busy(self) -> bool:
        with self._condition:
            return self._current is not None

    # --- worker ---

    def _loop(self) -> None:
        while True:
            with self._condition:
                while not self._waiting and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                turn = self._waiting.pop(0)
                turn.state = "running"
                self._current = turn

            try:
                self._runner(turn)
            except Exception as exc:  # noqa: BLE001 - the worker must not die
                # A runner is expected to handle its own errors and report them
                # as events. Reaching here means it did not, and the stream
                # would otherwise hang until the client gave up.
                turn.emit("error", message=f"Turn failed: {exc}", kind="internal")
            finally:
                # close() is safe to call twice: the reader stops at the first
                # sentinel, so a second one is never seen.
                turn.close()
                with self._condition:
                    self._current = None


def drain(turn: Turn, timeout: float = 1.0) -> Iterator[dict[str, Any] | None]:
    """Yield events as they arrive, and `None` once per idle `timeout`.

    The idle ticks are what let a streaming response send queue positions and
    heartbeats without a second thread watching the clock.
    """
    while True:
        try:
            event = turn.events.get(timeout=timeout)
        except queue.Empty:
            yield None
            continue
        if event is None:
            return
        yield event
