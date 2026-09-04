"""Saved conversations: list, read, rename, delete, rewind."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    ConversationDetail,
    ConversationOut,
    MessageOut,
    RenameRequest,
    TruncateOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
def list_conversations(limit: int = 30, runtime: Runtime = Depends(get_runtime)):
    return [
        ConversationOut(**vars(conversation))
        for conversation in runtime.store.list_conversations(limit=limit)
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, runtime: Runtime = Depends(get_runtime)):
    conversation = runtime.store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")

    messages = runtime.store.get_messages(conversation_id)
    strong = runtime.manager.router_strong
    return ConversationDetail(
        **vars(conversation),
        messages=[MessageOut(**vars(message)) for message in messages],
        escalated=any(message.model_key == strong for message in messages),
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: int,
    body: RenameRequest,
    runtime: Runtime = Depends(get_runtime),
):
    if runtime.store.get_conversation(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")
    runtime.store.rename_conversation(conversation_id, body.title.strip())
    return ConversationOut(**vars(runtime.store.get_conversation(conversation_id)))


@router.delete(
    "/{conversation_id}/messages/{message_id}/only", response_model=TruncateOut
)
def delete_one_message(
    conversation_id: int,
    message_id: int,
    runtime: Runtime = Depends(get_runtime),
):
    """Delete a single message, leaving the rest of the conversation.

    The other half of rewinding, and a different intent: rewinding changes
    what was asked and everything downstream was a reply to the old question,
    where this removes one thing and keeps the rest. Both are wanted.

    Refused mid-turn for the same reason rewinding is: a queued turn's
    history is read when it runs, so removing rows underneath it changes what
    it is answering.
    """
    if runtime.store.get_conversation(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")
    if runtime.queue.busy() or runtime.queue.depth():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is still running or waiting. Deleting a message would "
            "change what it is answering - stop it first.",
        )
    if not runtime.store.delete_message(conversation_id, message_id):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No such message in this conversation."
        )
    return TruncateOut(
        removed=1,
        emptied=runtime.store.message_count(conversation_id) == 0,
    )


@router.post(
    "/{conversation_id}/messages/{message_id}/fork", response_model=ConversationOut
)
def fork(
    conversation_id: int,
    message_id: int,
    runtime: Runtime = Depends(get_runtime),
):
    """Copy this conversation up to here into a new one, and return it.

    For trying a second direction without losing the first. Not refused
    mid-turn: nothing existing is touched, so a running turn is unaffected -
    the copy stops at a message that was already stored.
    """
    if runtime.store.get_conversation(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")

    new_id = runtime.store.fork_conversation(conversation_id, message_id)
    if new_id is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No such message in this conversation."
        )
    return ConversationOut(**vars(runtime.store.get_conversation(new_id)))


@router.delete(
    "/{conversation_id}/messages/{message_id}", response_model=TruncateOut
)
def rewind(
    conversation_id: int,
    message_id: int,
    runtime: Runtime = Depends(get_runtime),
):
    """Delete a message and everything after it.

    What editing a question is built on. The old question, the answer it got
    and anything that followed all go, because they were a reply to something
    that is no longer what was asked; leaving them would make a transcript of
    a conversation nobody had.

    Refused while any turn is in flight. A queued turn's history is read when
    it runs and is identified by the id of its own user message - so deleting
    rows underneath it would either change what it is answering or delete the
    question itself.

    It does not reach memory. Anything already extracted from the old messages
    stays in the memory store, which has its own lifecycle and its own way of
    being corrected; silently deleting from it here would be a second,
    invisible deletion nobody asked for.
    """
    if runtime.store.get_conversation(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")

    if runtime.queue.busy() or runtime.queue.depth():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is still running or waiting. Editing a question would "
            "pull the ground out from under it - stop it first.",
        )

    removed = runtime.store.truncate_from(conversation_id, message_id)
    if not removed:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No such message in this conversation."
        )
    return TruncateOut(
        removed=removed,
        emptied=runtime.store.message_count(conversation_id) == 0,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, runtime: Runtime = Depends(get_runtime)):
    """Delete a conversation and its messages.

    The one destructive operation in the whole API. It is here because the
    Streamlit sidebar had it and losing a conversation is a small, obvious
    loss the user asked for - unlike the tools, where delete and rename are
    absent by design.
    """
    if runtime.store.get_conversation(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation.")
    runtime.store.delete_conversation(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
