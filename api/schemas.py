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
    # Agreement to send this turn to a hosted provider, when the auto-router
    # chose one rather than the user. Requests that need it and do not carry it
    # are refused with 409 and the details, before anything is stored or run -
    # the agent loop cannot pause mid-turn to ask.
    confirm_remote: bool = False


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
    # "local" for a llama-server on this machine, otherwise the hosted provider.
    provider: str = "local"
    remote: bool = False
    # Remote only: whether the key is present, and whether it is usable right
    # now. A model can have a key and still be unusable with no network, and
    # the UI needs to say which of the two is wrong.
    has_key: bool = True
    usable: bool = True


class ModelsOut(BaseModel):
    models: list[ModelOut]
    default_key: str
    active_key: str | None = None
    router_fast: str
    router_strong: str
    max_active: int
    idle_timeout_seconds: int
    available_ram_mb: int | None = None
    # Whether hosted models can be reached at all. Cached, so this is cheap.
    online: bool = False


# --- tools ---


class ToolOut(BaseModel):
    name: str
    category: str
    description: str = ""
    parameters: dict[str, Any] = {}


class DisabledToolOut(BaseModel):
    category: str
    reason: str


class SwitchOut(BaseModel):
    """One tool switch the UI can flip."""

    id: str
    label: str
    enabled: bool
    # True when this is on because the environment says so, rather than
    # because someone flipped it here. Shown so the UI can say which switches
    # will survive a restart.
    from_env: bool
    # The switch this one is the sharp end of, if any: unrestricted Python is
    # meaningless without the Python tool.
    depends_on: str | None = None
    # The same text the disabled list carries: what the tool can do and what
    # it cannot protect you from.
    risk: str = ''


class ToggleRequest(BaseModel):
    enabled: bool


class ToolsOut(BaseModel):
    """The tool roster, why the missing ones are missing, and the switches.

    Disabled tools are reported with their reason because that reason states
    the real boundary - and, for anything switched on from here, it is the
    only place that boundary is written down.
    """

    tools: list[ToolOut]
    disabled: list[DisabledToolOut]
    switches: list[SwitchOut] = []
    workspace: str


# --- health ---


class HealthOut(BaseModel):
    ok: bool
    busy: bool
    queue_depth: int
    workspace: str
    db_path: str
