"""The auxiliary model's session: switch once, do everything, switch back.

This module is the only place in the memory system that loads a language
model, and it is built around one rule from the design: **Mistral and the
auxiliary model are never resident together.**

That rule is not enforced by anything written here. It is enforced by
`ModelManager.ensure()`, which already stops every other chat model when
`max_active` is 1 - the same code path the model picker in the UI uses. Adding
a second loading mechanism here would create exactly the situation the rule
forbids, so this drives the existing manager and does no process handling of
its own.

The sequence for one batch:

    refuse if a turn is running        (responsiveness wins over memory)
    take a batch of pending jobs
    manager.ensure(aux)                -> stops Mistral, starts the aux model
    run every job in the batch         -> one session, many jobs
    manager.stop(aux)                  -> nothing resident afterwards

`manager.ensure` is called once, before the loop, and `manager.stop` once
after. That is section 25: the expensive thing is the switch, not the work, so
forty jobs cost one switch rather than forty.

Everything here is optional. If the auxiliary model is missing, unavailable, or
simply switched off, `run_batch` reports that and the deterministic half of the
system carries on - explicit memories, retrieval, decay and dedupe all work
with no model at all (section 26).
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from typing import Any, Callable

from memory.store import MemoryStore
from memory.types import (
    Candidate,
    Job,
    JobKind,
    JobStatus,
    MemoryStatus,
    MemoryType,
)

# How many jobs one session will take. Enough that the switch is amortised,
# small enough that a batch cannot run for an unbounded time on two cores.
DEFAULT_BATCH = 12

# The aux model is small and the prompts are short, but this machine is slow.
# A job that takes longer than this is not going to finish usefully.
JOB_TIMEOUT_SECONDS = 180.0

# Conversation turns are trimmed before they reach the model: a memory
# extraction prompt does not need a 4,000-character paste.
MAX_TURN_CHARS = 1200

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

EXTRACT_SYSTEM = """You extract durable memories from a conversation.

Return ONLY a JSON array. Each item:
{"content": "<one short sentence about the user>", "type": "<type>", \
"importance": <0-1>, "confidence": <0-1>}

type is one of: fact, preference, event, intention, temporary, irrelevant.

Rules:
- Write about the user in the third person: "User prefers X".
- One fact per item. Keep each under 20 words.
- preference = a durable choice. fact = durable state. event = something that
  happened. intention = something they MIGHT do - use low confidence.
  temporary = true today only. irrelevant = not worth storing.
- Return [] when nothing is worth remembering. Most turns are worth nothing.
- Never store greetings, thanks, small talk, one-off arithmetic, or anything
  about this assistant.
"""

CONSOLIDATE_SYSTEM = """You merge two memories that say related things.

Return ONLY JSON: {"content": "<merged sentence>", "keep": "both"|"merged"}

- "merged" when one sentence can carry both meanings; put it in content.
- "both" when they are genuinely separate facts; content is then ignored.
- Keep the merged sentence under 25 words, third person, about the user.
"""

SUMMARIZE_SYSTEM = """You summarise a conversation so it can be dropped from \
context.

Return ONLY the summary as plain prose, at most 120 words. Record decisions \
made, problems being solved, files and tools involved, and anything the user \
asked for. Omit pleasantries. Write it so someone resuming the conversation \
could continue without the original messages.
"""


class MemoryProcessor:
    """Runs queued memory work in a single auxiliary-model session."""

    def __init__(
        self,
        store: MemoryStore,
        vectors,
        *,
        manager,
        config,
        aux_key: str,
        client_factory: Callable[[Any], Any] | None = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.manager = manager
        self.config = config
        self.aux_key = aux_key
        self.batch_size = max(1, int(batch_size))
        # A seam, so tests can script the model without a llama-server. The
        # real one builds a QwenClient pointed at the aux model's port.
        self._client_factory = client_factory or self._default_client
        # One batch at a time, process-wide. Two batches would each try to own
        # the single model slot.
        self._lock = threading.Lock()
        self._running = False
        self.last_run: dict[str, Any] = {}

    # --- state ---

    @property
    def running(self) -> bool:
        return self._running

    def available(self) -> tuple[bool, str]:
        """Whether the auxiliary model could run right now, and why not."""
        try:
            spec = self.manager.get_spec(self.aux_key)
        except Exception as exc:
            return False, f"No model registered as {self.aux_key!r}: {exc}"
        if getattr(spec, "remote", False):
            return False, (
                f"{spec.label} is a hosted model. Memory processing is "
                f"deliberately local: it reads whole conversations."
            )
        if not spec.available:
            return False, f"{spec.label}: weights not found at {spec.path}."
        return True, ""

    # --- the batch ---

    def run_batch(
        self, *, reason: str = "manual", busy: Callable[[], bool] | None = None
    ) -> dict[str, Any]:
        """Process pending jobs in one auxiliary-model session.

        `busy` is checked before the switch and again between jobs: a user who
        starts typing mid-batch should not wait for the queue to drain and then
        for Mistral to load again. The remaining jobs stay pending.
        """
        if not self._lock.acquire(blocking=False):
            return {"ran": False, "reason": "a memory batch is already running"}

        try:
            self._running = True
            return self._run(reason=reason, busy=busy)
        finally:
            self._running = False
            self._lock.release()

    def _run(self, *, reason: str, busy: Callable[[], bool] | None) -> dict[str, Any]:
        if busy is not None and busy():
            return {"ran": False, "reason": "a turn is running"}

        jobs = self.store.pending_jobs(limit=self.batch_size)
        if not jobs:
            return {"ran": False, "reason": "no pending memory jobs"}

        usable, why = self.available()
        if not usable:
            return {"ran": False, "reason": why, "pending": len(jobs)}

        started = time.time()
        self.store.mark_running([job.id for job in jobs])

        counts = {"extract": 0, "consolidate": 0, "summarize": 0}
        created = 0
        failed = 0
        stopped_early = False

        client = None
        try:
            # THE switch. `ensure` stops every other chat model first, which is
            # what makes "never both" true - and it is the manager's rule, not
            # a second copy of it here.
            url = self.manager.ensure(self.aux_key)
            client = self._client_factory(url)

            for job in jobs:
                if busy is not None and busy():
                    # Put this job and everything after it back, and get out of
                    # the way. Deferred, not failed: `defer_jobs` leaves the
                    # attempt count alone, so being postponed by a busy agent
                    # can never use up a job's retries.
                    self.store.defer_jobs(
                        [item.id for item in jobs[jobs.index(job) :]]
                    )
                    stopped_early = True
                    break

                try:
                    made = self._run_job(client, job)
                    created += made
                    counts[job.kind.value] = counts.get(job.kind.value, 0) + 1
                    self.store.finish_job(
                        job.id,
                        JobStatus.DONE if made or job.kind is not JobKind.EXTRACT
                        else JobStatus.DISCARDED,
                    )
                except Exception as exc:  # noqa: BLE001 - one job must not sink the batch
                    failed += 1
                    self.store.retry_or_fail(job.id, f"{type(exc).__name__}: {exc}")
        finally:
            # Always stop the aux model, on every path. Leaving it resident is
            # the one outcome that breaks the memory ceiling for the next turn.
            try:
                self.manager.stop(self.aux_key)
            except Exception:  # noqa: BLE001
                pass

        # New memories need vectors before they can be retrieved.
        embedded = 0
        try:
            embedded = self.vectors.sync()
        except Exception:  # noqa: BLE001 - the memories are stored either way
            pass

        self.last_run = {
            "ran": True,
            "reason": reason,
            "jobs": len(jobs),
            "by_kind": counts,
            "memories_created": created,
            "embedded": embedded,
            "failed": failed,
            "stopped_early": stopped_early,
            "seconds": round(time.time() - started, 1),
            "model": self.aux_key,
        }
        return dict(self.last_run)

    # --- individual jobs ---

    def _run_job(self, client, job: Job) -> int:
        if job.kind is JobKind.EXTRACT:
            return self._extract(client, job)
        if job.kind is JobKind.CONSOLIDATE:
            return self._consolidate(client, job)
        if job.kind is JobKind.SUMMARIZE:
            return self._summarize(client, job)
        if job.kind is JobKind.CLASSIFY:
            return self._classify(client, job)
        return 0

    def _extract(self, client, job: Job) -> int:
        """Turn a stretch of conversation into memories."""
        turns = job.payload.get("turns") or []
        if not turns:
            return 0

        transcript = "\n".join(
            f"{turn.get('role', 'user')}: {str(turn.get('content', ''))[:MAX_TURN_CHARS]}"
            for turn in turns
        )[:6000]

        reply = self._ask(client, EXTRACT_SYSTEM, transcript)
        items = _json_from(reply)
        if not isinstance(items, list):
            return 0

        stored = 0
        for item in items[:8]:
            candidate = _candidate_from(item, job.conversation_id)
            if candidate is None:
                continue
            try:
                _, created = self.store.add(candidate)
            except ValueError:
                continue
            if created:
                stored += 1
        return stored

    def _classify(self, client, job: Job) -> int:
        """Re-type memories a rule stored without understanding them."""
        memory_id = job.payload.get("memory_id")
        memory = self.store.get(int(memory_id)) if memory_id else None
        if memory is None:
            return 0

        reply = self._ask(client, EXTRACT_SYSTEM, f"user: {memory.content}")
        items = _json_from(reply)
        if not isinstance(items, list) or not items:
            return 0

        first = items[0]
        kind = _type_from(first.get("type"))
        if kind is None:
            return 0
        if kind is MemoryType.IRRELEVANT:
            self.store.update(memory.id, status=MemoryStatus.ARCHIVED)
            return 0
        self.store.update(
            memory.id,
            type=kind,
            importance=_number(first.get("importance"), memory.importance),
            confidence=_number(first.get("confidence"), memory.confidence),
        )
        return 0

    def _consolidate(self, client, job: Job) -> int:
        """Merge two related memories into one sentence, when that is right."""
        first = self.store.get(int(job.payload.get("first_id", 0)))
        second = self.store.get(int(job.payload.get("second_id", 0)))
        if first is None or second is None:
            return 0
        if first.status is not MemoryStatus.ACTIVE:
            return 0
        if second.status is not MemoryStatus.ACTIVE:
            return 0

        reply = self._ask(
            client,
            CONSOLIDATE_SYSTEM,
            f"A: {first.content}\nB: {second.content}",
        )
        payload = _json_from(reply)
        if not isinstance(payload, dict):
            return 0
        if payload.get("keep") != "merged":
            return 0

        content = str(payload.get("content", "")).strip()
        if len(content) < 8:
            return 0

        newer = first if first.updated_at >= second.updated_at else second
        older = second if newer is first else first
        self.store.update(
            newer.id,
            content=content,
            importance=max(first.importance, second.importance),
            confidence=min(1.0, max(first.confidence, second.confidence) + 0.05),
        )
        self.store.update(
            older.id, status=MemoryStatus.SUPERSEDED, superseded_by=newer.id
        )
        self.store.link(newer.id, older.id, "merged_into")
        return 0

    def _summarize(self, client, job: Job) -> int:
        """Compress the old part of a conversation into a stored summary."""
        conversation_id = job.conversation_id
        turns = job.payload.get("turns") or []
        covers_up_to = int(job.payload.get("covers_up_to", 0))
        if conversation_id is None or not turns:
            return 0

        previous = self.store.get_summary(conversation_id)
        head = f"Earlier summary: {previous['summary']}\n\n" if previous else ""
        transcript = head + "\n".join(
            f"{turn.get('role', 'user')}: {str(turn.get('content', ''))[:MAX_TURN_CHARS]}"
            for turn in turns
        )

        summary = self._ask(client, SUMMARIZE_SYSTEM, transcript[:8000]).strip()
        if len(summary) < 20:
            return 0

        self.store.save_summary(
            conversation_id, summary[:4000], covers_up_to, len(turns)
        )
        return 0

    # --- talking to the model ---

    def _ask(self, client, system: str, user: str) -> str:
        message = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
        )
        return str(message.get("content") or "")

    def _default_client(self, url: str):
        """A chat client pointed at the auxiliary model.

        Thinking is forced off: a memory job wants a short JSON answer, and a
        reasoning trace on this hardware would cost minutes per job for
        nothing.
        """
        from models.qwen import QwenClient

        return QwenClient(
            dataclasses.replace(
                self.config,
                qwen_url=url,
                enable_thinking=False,
                request_timeout=JOB_TIMEOUT_SECONDS,
            )
        )


def _json_from(text: str):
    """Pull the JSON out of a small model's reply.

    Small models wrap JSON in prose and fences however firmly they are asked
    not to. Finding the outermost bracketed span is what makes the difference
    between a working extractor and one that fails on every other reply.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    try:
        return json.loads(text)
    except ValueError:
        pass

    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def _candidate_from(item: Any, conversation_id: int | None) -> Candidate | None:
    if not isinstance(item, dict):
        return None
    content = str(item.get("content", "")).strip()
    if len(content) < 8:
        return None

    kind = _type_from(item.get("type")) or MemoryType.UNCERTAIN
    if kind is MemoryType.IRRELEVANT:
        return None

    confidence = _number(item.get("confidence"), 0.5)
    # The model is allowed to say "preference" about a hedged statement; the
    # design says an uncertain intention must not become a permanent fact, so
    # low confidence overrides the label rather than the other way round.
    if confidence < 0.4 and kind in (MemoryType.FACT, MemoryType.PREFERENCE):
        kind = MemoryType.UNCERTAIN

    return Candidate(
        content=content[:400],
        type=kind,
        importance=_number(item.get("importance"), 0.5),
        confidence=confidence,
        source_conversation=conversation_id,
        origin="model",
    )


def _type_from(value: Any) -> MemoryType | None:
    try:
        return MemoryType(str(value).strip().lower())
    except ValueError:
        return None


def _number(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback
