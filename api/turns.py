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
import time
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
class Approval:
    """One command waiting on a person to say yes or no."""

    command: str
    reason: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    answered: threading.Event = field(default_factory=threading.Event)
    granted: bool = False


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
    # Set when someone asked for this turn to stop. An Event rather than a
    # bool because it is written from the request thread and read from the
    # worker, and it is the flag itself - not a lock around it - that has to
    # be safe to share.
    stopped: threading.Event = field(default_factory=threading.Event)
    # Approval requests this turn is waiting on, by id. Written from the
    # worker thread and read from the request thread that answers, so the
    # dict itself needs the lock even though each Event does not.
    pending: dict[str, "Approval"] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, type: str, **data: Any) -> None:
        """Publish one event to whoever is streaming this turn."""
        self.events.put({"type": type, **data})

    def ask(self, command: str, reason: str, *, timeout: float) -> bool:
        """Ask whoever is watching whether `command` may run. Blocks.

        Called from the worker thread, in the middle of a tool call. The
        answer arrives on a request thread through `answer`, so this is a
        handshake between two threads with a stream in between.

        Denied by default when the timeout passes. That direction is not
        arbitrary: nobody watching means nobody agreed, and a command that
        runs because a person walked away is exactly what this exists to
        prevent. A stop, likewise, is not a yes.
        """
        approval = Approval(command=command, reason=reason)
        with self.lock:
            self.pending[approval.id] = approval

        self.emit(
            "approval",
            request_id=approval.id,
            command=command,
            reason=reason,
            timeout=timeout,
        )
        try:
            # Waited in slices rather than one long block, so a stop is
            # noticed while the prompt is still on screen. A turn that ignored
            # the stop button for the whole approval timeout would look hung.
            deadline = time.monotonic() + timeout
            while True:
                if self.is_stopped():
                    self.emit(
                        "approval_closed", request_id=approval.id, granted=False
                    )
                    return False
                if approval.answered.wait(min(0.25, max(0.0, deadline - time.monotonic()))):
                    return approval.granted and not self.is_stopped()
                if time.monotonic() >= deadline:
                    self.emit(
                        "approval_closed", request_id=approval.id, granted=False
                    )
                    return False
        finally:
            with self.lock:
                self.pending.pop(approval.id, None)

    def answer(self, request_id: str, granted: bool) -> bool:
        """Record a decision. False when there was nothing waiting for it."""
        with self.lock:
            approval = self.pending.get(request_id)
        if approval is None:
            return False
        approval.granted = granted
        approval.answered.set()
        self.emit("approval_closed", request_id=request_id, granted=granted)
        return True

    def stop(self) -> None:
        """Ask this turn to stop at its next checkpoint."""
        self.stopped.set()

    def is_stopped(self) -> bool:
        """Whether a stop has been asked for. Passed to the agent as a callable."""
        return self.stopped.is_set()

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

    # --- stopping ---

    def answer_approval(self, turn_id: str, request_id: str, granted: bool) -> str:
        """Pass a decision to the turn waiting on it.

        Only the running turn can be waiting: a queued one has not reached a
        tool call yet. "stale" covers a prompt answered twice, or answered
        after it timed out - both ordinary, neither worth an error.
        """
        with self._condition:
            current = self._current
        if current is None or current.id != turn_id:
            return "unknown"
        return "answered" if current.answer(request_id, granted) else "stale"

    def stop_turn(self, turn_id: str) -> str:
        """Ask one turn to stop, wherever it is. Returns what was found.

        Two quite different things share one entry point, because from outside
        they are the same request and which one applies is an accident of
        timing:

        "queued"  - it had not started. It is dropped from the backlog and its
                    stream is closed here, because no worker will ever pick it
                    up to do that.
        "running" - the flag is set and the worker stops at its next
                    checkpoint: the next token, the end of a tool call, or the
                    end of a model round. Nothing is killed. A thread cannot
                    be safely interrupted mid-write, and a half-written
                    conversation would be a worse outcome than a few more
                    seconds of CPU.
        "unknown" - finished, never submitted, or already stopped and gone.
        """
        with self._condition:
            if self._current is not None and self._current.id == turn_id:
                self._current.stop()
                return "running"

            for index, waiting in enumerate(self._waiting):
                if waiting.id == turn_id:
                    self._waiting.pop(index)
                    waiting.stop()
                    break
            else:
                return "unknown"

        # Outside the lock: emitting and closing touch the turn's own queue,
        # not the backlog, and holding the condition while doing it would put
        # the reader's wake-up behind a lock the reader does not need.
        waiting.emit("stopped", state="queued", content="", tools=[], elapsed=0.0)
        waiting.close()
        return "queued"

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
