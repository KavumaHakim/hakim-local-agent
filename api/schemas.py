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

    # No min_length: a turn may be an attachment with nothing typed. The route
    # refuses a request that is empty *and* has no attachments, which is the
    # condition that actually matters and one the schema cannot see.
    prompt: str = ""
    conversation_id: int | None = None
    model_key: str | None = None
    enable_thinking: bool = False
    auto_route: bool = False
    # Agreement to send this turn to a hosted provider, when the auto-router
    # chose one rather than the user. Requests that need it and do not carry it
    # are refused with 409 and the details, before anything is stored or run -
    # the agent loop cannot pause mid-turn to ask.
    confirm_remote: bool = False
    # Workspace-relative paths from POST /uploads. Named in the prompt so the
    # model knows the file exists and can reach it with ocr_image.
    attachments: list[str] = []


class SpeechStatusOut(BaseModel):
    """Whether dictation can run, and on what."""

    available: bool
    # The model's short name ("base.en"), for the microphone's tooltip.
    model: str = ""
    detail: str = ""


class SpeechOut(BaseModel):
    """One transcribed clip.

    Goes to the message box, never straight to the agent: whisper invents
    words when it hears no speech, so this is always read before it is sent.
    """

    text: str
    bytes_received: int = 0


class UploadOut(BaseModel):
    """An image stored in the workspace, ready to be named in a prompt."""

    # Workspace-relative, because that is the only form ocr_image accepts.
    path: str
    name: str
    size: int
    # False when OCR is switched off or its server is not running, which are
    # different problems with different fixes - hence the hint rather than a
    # bare boolean.
    ocr_ready: bool
    hint: str = ""


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
    # Layers on the GPU. 0 everywhere unless somebody has both an accelerator
    # build and a reason.
    gpu_layers: int = 0
    min_free_mb: int
    # False when the GGUF is not on disk. The UI greys these out rather than
    # hiding them, so a missing file looks like a missing file.
    available: bool
    state: str
    pid: int | None = None
    error: str = ""
    warning: str = ""
    adopted: bool = False
    # "chat" drives the agent loop; "ocr" is a vision backend that runs
    # alongside one and never appears in the chat model picker.
    role: str = "chat"
    # "local" for a llama-server on this machine, otherwise the hosted provider.
    provider: str = "local"
    remote: bool = False
    # Remote only: whether the key is present, and whether it is usable right
    # now. A model can have a key and still be unusable with no network, and
    # the UI needs to say which of the two is wrong.
    has_key: bool = True
    usable: bool = True
    # True when this model was found in the models folder rather than declared
    # in models.json. Shown, because a discovered model's context and RAM
    # figures are inferred from its header and may want retuning.
    discovered: bool = False
    # Hidden models stay in the catalogue so they can be un-hidden, but are
    # not offered to the model picker or the router.
    hidden: bool = False
    # Set when a value came from the settings panel rather than the registry.
    customised: bool = False
    file_mb: int = 0
    # From the GGUF header: what the model was trained for, against what it is
    # actually being run at. A large gap is the interesting case.
    training_context: int = 0
    kv_cache_mb: int = 0
    notes: list[str] = []


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
    # Where to drop a .gguf file for it to be picked up.
    models_dir: str = ""
    # Which llama-server is being run. Read-only here: it is a property of one
    # computer, so setup.py owns it. Shown because `gpu_layers` is unreadable
    # without knowing whether this build has a GPU backend in it at all.
    server_exe: str = ""
    # True when several chat models exist and no primary has been chosen yet.
    # The UI shows the first-launch picker on this; a single-model install
    # never sets it, because one model is not a choice.
    setup_required: bool = False


# --- model settings ---


class ModelPrimaryRequest(BaseModel):
    key: str = Field(min_length=1)


class ServerExeRequest(BaseModel):
    """Which llama-server this machine should run.

    Empty means "go back to searching": the configured path, then
    vendor/llama, then PATH.
    """

    path: str = Field(default="", max_length=4096)


class ModelRouterRequest(BaseModel):
    fast: str = ""
    strong: str = ""


class ModelOverrideRequest(BaseModel):
    """Retune one model. Every field is optional; only what is sent changes.

    Deliberately no `file`, `port` or `role`: those decide what a model *is*
    and where it runs, and getting them wrong from a settings panel produces a
    model that will not start for reasons the panel cannot explain.
    """

    label: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=200)
    context: int | None = Field(default=None, ge=512, le=131_072)
    threads: int | None = Field(default=None, ge=1, le=64)
    gpu_layers: int | None = Field(default=None, ge=0, le=999)
    min_free_mb: int | None = Field(default=None, ge=0, le=128_000)


class ModelHideRequest(BaseModel):
    hidden: bool = True


class RescanOut(BaseModel):
    """What a rescan of the models folder found."""

    success: bool = True
    models_dir: str
    # Keys that were not in the catalogue before this scan.
    added: list[str] = []
    total: int = 0
    models: ModelsOut


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


class OcrBackendRequest(BaseModel):
    """Choose between Tesseract and the GLM-OCR model."""

    backend: str = Field(pattern="^(tesseract|model)$")


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
    # Which reader ocr_image uses: "tesseract" or "model". They behave
    # differently enough that the UI names the active one rather than showing
    # a single "OCR" switch that means two things.
    ocr_backend: str = "model"
    # Whether that backend could run right now, and why not.
    ocr_ready: bool = False
    ocr_hint: str = ""


# --- document search (RAG) ---


class RagSectionOut(BaseModel):
    """One section of a document, as the outline reports it."""

    section: str
    first_page: int | None = None
    last_page: int | None = None
    chunks: int = 0
    characters: int = 0


class RagFigureOut(BaseModel):
    """One raster figure pulled out of a document."""

    page: int | None = None
    path: str
    # Empty when nothing near the picture looked like a caption. Honest: this
    # only claims a caption when the text says it is one.
    caption: str = ""


class RagOutlineResult(BaseModel):
    """What one document is made of, rather than what matches a question."""

    success: bool = True
    document: str
    path: str
    pages: int | None = None
    chunks: int = 0
    count: int = 0
    sections: list[RagSectionOut] = []
    figures: list[RagFigureOut] = []
    note: str = ""


class RagIndexRequest(BaseModel):
    """Index a file, or a folder of them."""

    path: str = Field(min_length=1)
    # Directories only. A folder of notes is normally worth walking; a folder
    # that happens to sit above a source tree is not, hence the switch.
    recursive: bool = True
    # Re-embed even when size and modification time say nothing changed. The
    # escape hatch for a file whose timestamp lies.
    force: bool = False


class RagIndexedDocument(BaseModel):
    document: str
    chunks: int


class RagFailure(BaseModel):
    document: str
    error: str


class RagIndexResult(BaseModel):
    success: bool = True
    indexed: list[RagIndexedDocument] = []
    # Names only: an unchanged file has nothing else worth reporting.
    skipped: list[str] = []
    # Per-file failures. An unreadable PDF in a folder of 400 is reported
    # here rather than failing the whole run.
    failed: list[RagFailure] = []
    documents_total: int = 0
    chunks_total: int = 0


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Cosine similarity. None uses the configured threshold.
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    # Narrow the search before it runs. A document by name or id, and a
    # section by any part of its heading.
    document: str | None = Field(default=None, max_length=500)
    section: str | None = Field(default=None, max_length=300)


class RagHit(BaseModel):
    document: str
    path: str
    chunk_id: str
    score: float
    text: str
    # How it was found: "semantic", "keyword" or "both". The score means
    # something different for each.
    match: str = "semantic"
    # Absent for formats that have no pages, rather than null.
    page: int | None = None


class RagSearchResult(BaseModel):
    success: bool = True
    query: str
    count: int
    results: list[RagHit] = []
    # Set when there is something the caller should know: an empty index, or
    # nothing above the threshold.
    note: str = ""
    truncated: str = ""
    # What the search was narrowed to, when it was.
    scope: str = ""


class RagDocumentOut(BaseModel):
    id: int
    document: str
    path: str
    chunks: int
    pages: int | None = None
    size_bytes: int
    indexed_at: str


class RagDocumentsOut(BaseModel):
    success: bool = True
    count: int
    chunks_total: int
    documents: list[RagDocumentOut] = []


class RagRemoveResult(BaseModel):
    success: bool = True
    document: str
    removed_chunks: int


class RagStatsOut(BaseModel):
    success: bool = True
    documents: int
    chunks: int
    model: str
    dimension: int
    chunk_tokens: int
    overlap_tokens: int
    store: str
    vector_bytes: int
    # Whether the embedding model is resident right now. The answer should be
    # False most of the time.
    embedder_loaded: bool = False


# --- memory ---


class MemoryOut(BaseModel):
    id: int
    type: str
    content: str
    importance: float
    confidence: float
    status: str
    created_at: str = ""
    subject: str = ""
    superseded_by: int | None = None
    # Present on search results, absent on a plain listing.
    score: float | None = None
    similarity: float | None = None


class MemoryListOut(BaseModel):
    success: bool = True
    query: str = ""
    count: int = 0
    memories: list[MemoryOut] = []
    # Set when nothing was relevant, which is a real answer rather than a
    # failure - the UI shows it instead of an empty box.
    note: str = ""


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)


class MemoryRememberRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    type: str = "fact"
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    subject: str = ""


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    type: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: str | None = None
    subject: str | None = None


class MemoryProcessorOut(BaseModel):
    configured: bool = False
    available: bool = False
    reason: str = ""
    running: bool = False
    last_run: dict[str, Any] = {}


class MemoryStatsOut(BaseModel):
    success: bool = True
    total: int = 0
    active: int = 0
    archived: int = 0
    superseded: int = 0
    deleted: int = 0
    pending_jobs: int = 0
    # Whether semantic retrieval is possible; False falls back to substring.
    embeddings: bool = False
    processor: MemoryProcessorOut = MemoryProcessorOut()

    # Per-type counts arrive as type_fact, type_preference and so on.
    model_config = {"extra": "allow"}


class MemoryConsolidateResult(BaseModel):
    success: bool = True
    merged: int = 0
    superseded: int = 0
    needs_model: int = 0
    queued: int = 0
    note: str = ""


class MemoryProcessResult(BaseModel):
    """What one auxiliary-model batch did, or why it did not run."""

    ran: bool = False
    reason: str = ""
    jobs: int = 0
    by_kind: dict[str, int] = {}
    memories_created: int = 0
    embedded: int = 0
    failed: int = 0
    stopped_early: bool = False
    seconds: float = 0.0
    model: str = ""
    pending: int = 0


class StopTurnOut(BaseModel):
    """What asking a turn to stop actually did."""

    # "queued"  - it never started, and has been dropped
    # "running" - it was asked to stop and will at its next checkpoint
    # "unknown" - already finished, or never here
    state: str
    message: str


class TruncateOut(BaseModel):
    """What rewinding a conversation removed."""

    removed: int
    # True when nothing is left, so the next question is effectively the
    # conversation's first - which is what its title should come from.
    emptied: bool


# --- model hub ---


class HubModelOut(BaseModel):
    """One repository in a search result."""

    id: str
    downloads: int = 0
    likes: int = 0
    # Needs terms accepted on the website; this app holds no credentials.
    gated: bool = False
    tags: list[str] = []


class HubSearchOut(BaseModel):
    query: str
    models: list[HubModelOut] = []


class HubFileOut(BaseModel):
    """One GGUF, and what it would cost to run."""

    path: str
    size_bytes: int
    # The Q4_K_M-ish part of the name, for grouping.
    quantisation: str = ""
    # Estimated from the file size alone, by the same arithmetic discovery
    # uses on a downloaded file. The whole point of the listing: being told a
    # 4.8 GB file wants 6.2 GB free before spending an hour on it.
    needs_ram_mb: int = 0


class HubFilesOut(BaseModel):
    repo: str
    files: list[HubFileOut] = []
    # None where it cannot be determined, which is not the same as zero.
    free_ram_mb: int | None = None


class DownloadRequest(BaseModel):
    repo: str = Field(min_length=1)
    path: str = Field(min_length=1)
    # Sent so the disk check can happen before any bytes move.
    size_bytes: int = 0


class DownloadOut(BaseModel):
    """One model being fetched, and how far it has got."""

    id: str
    repo: str
    path: str
    name: str
    # running | done | failed | cancelled
    state: str
    error: str = ""
    seen_bytes: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    bytes_per_second: int = 0
    # None until there is enough of a rate to extrapolate from.
    seconds_left: int | None = None


class DownloadsOut(BaseModel):
    downloads: list[DownloadOut] = []


# --- workspace ---


class WorkspaceRequest(BaseModel):
    """The folder to hand the file tools."""

    path: str = Field(min_length=1)


class WorkspaceOut(BaseModel):
    """Which folder the file tools may reach, and what that means here."""

    path: str
    # What AGENT_WORKSPACE says, which a restart returns to.
    default: str
    # True when nothing has been chosen here, so the environment is still in
    # charge. The UI says so, because that is the difference between a choice
    # that survives a restart and one that does not.
    from_env: bool
    # Folders chosen in this process, most recent first.
    recent: list[str] = []
    # True when the workspace is the project's own directory. Worth naming:
    # it is the default, unrestricted Python refuses to run there, and it is
    # rarely the folder someone actually wants to work in.
    is_project: bool = False
    # Which of the switched-on tools can write into it right now. Changing the
    # workspace changes what those tools reach, so the answer belongs beside
    # the path rather than three panels away.
    writable: bool = False
    # Tools that would act on the new folder, by label. Empty when only the
    # read-only ones are on.
    active_tools: list[str] = []


class DirectoryEntryOut(BaseModel):
    name: str
    path: str


class DirectoryOut(BaseModel):
    """One level of the filesystem, for picking a folder without typing it.

    A browser cannot tell a page the real path of a chosen folder - the
    directory picker gives relative names and nothing else - so the walking is
    done here. Sub-directory names only: this lists where you could go, never
    what is in a file.
    """

    path: str
    # None at a drive root, which is where walking up stops.
    parent: str | None = None
    entries: list[DirectoryEntryOut] = []
    # Drives on Windows, plus home. The starting points for a walk.
    roots: list[DirectoryEntryOut] = []
    # Set when the folder could be listed only in part, or not at all.
    note: str = ""


# --- health ---


class HealthOut(BaseModel):
    ok: bool
    busy: bool
    queue_depth: int
    workspace: str
    db_path: str
