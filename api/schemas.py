"""Request and response bodies.

Deliberately thin. These describe what crosses the wire; they are not a second
model layer over `ModelSpec`, `Conversation` and `StoredMessage`, which remain
the real ones.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- chat ---


class ChatRequest(BaseModel):
    """One user turn.

    Settings ride with the request rather than living in a server session:
    the client owns its own UI state, and the server stays restartable.
    """

    prompt: str = Field(min_length=1)
    conversation_id: int | None = None
    model_key: str | None = None
    enable_thinking: bool = False
    auto_route: bool = False


class ChatAccepted(BaseModel):
    """Returned in the stream's first event, before any model work starts."""

    turn_id: str
    conversation_id: int
    user_message_id: int
    position: int


# --- conversations ---


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    tools: list[dict[str, Any]] = []
    elapsed: float | None = None
    model_key: str | None = None
    created_at: str = ""


class ConversationOut(BaseModel):
    id: int
    title: str
    model_key: str | None = None
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []
    # Whether this conversation has already needed the strong model, which is
    # what stops the router sending it back down.
    escalated: bool = False


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


# --- models ---


class ModelOut(BaseModel):
    key: str
    label: str
    description: str = ""
    port: int
    url: str
    context: int
    threads: int
    min_free_mb: int
    # False when the GGUF is not on disk. The UI greys these out rather than
    # hiding them, so a missing file looks like a missing file.
    available: bool
    state: str
    pid: int | None = None
    error: str = ""
    warning: str = ""
    adopted: bool = False


class ModelsOut(BaseModel):
    models: list[ModelOut]
    default_key: str
    active_key: str | None = None
    router_fast: str
    router_strong: str
    max_active: int
    idle_timeout_seconds: int
    available_ram_mb: int | None = None


# --- tools ---


class ToolOut(BaseModel):
    name: str
    category: str
    description: str = ""
    parameters: dict[str, Any] = {}


class DisabledToolOut(BaseModel):
    category: str
    reason: str


class ToolsOut(BaseModel):
    """The tool roster, and why the missing ones are missing.

    Disabled tools are reported with their reason because that reason names the
    environment variable that would turn them on, which is the only way to
    discover it from the UI.
    """

    tools: list[ToolOut]
    disabled: list[DisabledToolOut]
    workspace: str


# --- health ---


class HealthOut(BaseModel):
    ok: bool
    busy: bool
    queue_depth: int
    workspace: str
    db_path: str
