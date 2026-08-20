"""Saved conversations: list, read, rename, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    ConversationDetail,
    ConversationOut,
    MessageOut,
    RenameRequest,
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
