"""The chat endpoint: one POST, one long stream of events.

Server-sent events rather than a WebSocket, because the traffic is almost
entirely one-directional - tokens out - and SSE is a plain HTTP response that
needs no second protocol on either side.

Not `EventSource` on the client, though: that is GET-only and cannot carry a
prompt in a body. The browser reads this with `fetch` and a `ReadableStream`,
which is why the response is a normal POST.

Event types, in the order a healthy turn produces them:

    accepted  turn id, conversation id, and where it landed in the queue
    queued    position, re-sent whenever it changes while waiting
    route     the auto-router chose a different model, and why
    model     a model is loading, then ready
    start     the model is about to generate
    token     one fragment of the answer
    tool      a tool ran; the client clears its streamed text on this
    done      final content, tools used, elapsed seconds
    error     kind, message, and whether escalating could help

A turn that is running is not cancelled when the client disconnects. It
finishes and its answer is stored, so a reload shows it. On a machine where a
turn costs minutes, throwing that away because a tab closed would be worse
than the stray CPU.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.deps import get_runtime
from api.runtime import ModelChoice, Runtime, open_conversation
from api.schemas import ChatRequest
from api.turns import Turn, TurnQueueFull, TurnRequest, drain

router = APIRouter(tags=["chat"])

# Long enough to be quiet, short enough that a dead connection is noticed.
HEARTBEAT_SECONDS = 15.0


def _sse(event: str, payload: dict) -> str:
    """One server-sent event.

    `json.dumps` escapes newlines, so the data field is always a single line
    and can never accidentally terminate the event early.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
def chat(body: ChatRequest, runtime: Runtime = Depends(get_runtime)):
    """Queue a turn and stream it."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Empty prompt.")

    if body.model_key is not None:
        try:
            runtime.manager.get_spec(body.model_key)
        except Exception as exc:  # ModelManagerError, but keep the 400 shape
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    # Decided here, not in the worker: a hosted model has to be agreed to
    # first, and once the turn is running there is nobody to ask.
    choice = runtime.decide_model(
        prompt,
        body.model_key,
        auto_route=body.auto_route,
        conversation_id=body.conversation_id,
    )
    spec = runtime.manager.get_spec(choice.key)

    if spec.remote and choice.routed and not body.confirm_remote:
        # Explicitly picking a cloud model is already a deliberate act; being
        # moved onto one by the router is not, so that is what needs consent.
        # Nothing has been stored or queued at this point, so re-sending with
        # confirm_remote is the whole retry.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "kind": "remote_confirmation_required",
                "model_key": choice.key,
                "label": spec.label,
                "provider": spec.provider,
                "reason": choice.reason,
                "message": (
                    f"Auto-routing chose {spec.label}, which runs on "
                    f"{spec.provider}'s servers. Your prompt, this "
                    f"conversation and any tool results would be sent there."
                ),
            },
        )

    model_key = choice.key
    conversation_id = open_conversation(
        runtime, body.conversation_id, prompt, model_key
    )
    # Stored before the turn is queued, so the message is durable even if the
    # server dies while the turn is still waiting its turn to run.
    user_message_id = runtime.store.add_message(
        conversation_id, "user", prompt, model_key=model_key
    )

    turn = Turn(
        request=TurnRequest(
            conversation_id=conversation_id,
            prompt=prompt,
            user_message_id=user_message_id,
            model_key=model_key,
            enable_thinking=body.enable_thinking,
            auto_route=body.auto_route,
        )
    )

    try:
        runtime.queue.submit(turn)
    except TurnQueueFull as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from None

    return StreamingResponse(
        _stream(runtime, turn, conversation_id, user_message_id, choice),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Nothing proxies this today, but a buffering proxy would hold the
            # whole five-minute turn and deliver it at the end, which looks
            # exactly like the hang streaming exists to prevent.
            "X-Accel-Buffering": "no",
        },
    )


def _stream(
    runtime: Runtime,
    turn: Turn,
    conversation_id: int,
    user_message_id: int,
    choice: ModelChoice,
) -> Iterator[str]:
    """Render one turn's events as SSE.

    A synchronous generator on purpose: Starlette runs it in a worker thread,
    so blocking on the turn's queue costs a thread rather than the event loop.
    """
    position = runtime.queue.position(turn)
    yield _sse(
        "accepted",
        {
            "turn_id": turn.id,
            "conversation_id": conversation_id,
            "user_message_id": user_message_id,
            "position": position,
        },
    )

    # Both decided before queueing, so they are reported here rather than by
    # the worker.
    if choice.fell_back_from is not None:
        yield _sse(
            "fallback",
            {
                "from": choice.fell_back_from,
                "to": choice.key,
                "reason": choice.reason,
            },
        )
    elif choice.routed:
        spec = runtime.manager.get_spec(choice.key)
        yield _sse(
            "route",
            {
                "key": choice.key,
                "label": spec.label,
                "reason": choice.reason,
                "remote": spec.remote,
            },
        )

    reported = position
    if position:
        yield _sse("queued", {"position": position})

    last_beat = time.monotonic()

    for event in drain(turn):
        if event is None:
            # Idle tick: report movement in the queue, and keep the connection
            # warm through the long silence while a model loads.
            current = runtime.queue.position(turn)
            if current != reported:
                reported = current
                yield _sse("queued", {"position": current})
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_SECONDS:
                last_beat = now
                yield ": ping\n\n"
            continue

        last_beat = time.monotonic()
        kind = event.pop("type")
        yield _sse(kind, event)
