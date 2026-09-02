"""Central configuration for the local agent.

Everything is read from environment variables with local defaults, so no
model file paths or machine-specific paths are baked into the code.
The llama.cpp servers are started separately; this app only talks to them
over HTTP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The project directory. Used as the default workspace root so the agent has
# something safe and non-empty to look at out of the box.
PROJECT_ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path | None = None) -> None:
    """Read `.env` into the environment, without overriding what is already set.

    Hand-rolled rather than adding python-dotenv: the format needed here is
    KEY=value, and a dependency for that is not worth it.

    Real environment variables win, so exporting a key in the shell overrides
    the file rather than being silently ignored - which is the behaviour people
    expect when they are trying to test one key quickly.
    """
    path = path or (PROJECT_ROOT / ".env")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class Config:
    # --- Qwen3 server (llama.cpp, OpenAI-compatible API) ---
    qwen_url: str = "http://127.0.0.1:8080"
    # llama-server accepts any model name unless started with --alias.
    qwen_model: str = "qwen3"
    temperature: float = 0.7
    top_p: float = 0.8
    # -1 lets the server decide based on its context size.
    max_tokens: int = -1
    # Qwen3 thinks by default. Turning this off makes tool loops much faster on
    # CPU; it is sent as chat_template_kwargs.enable_thinking=false and is only
    # included in the request when explicitly disabled.
    enable_thinking: bool = True

    # --- GLM-OCR server (llama.cpp, separate process/port) ---
    ocr_url: str = "http://127.0.0.1:8081"
    ocr_model: str = "glm-ocr"
    # OFF by default: the transport is not implemented because the GLM-OCR
    # server's request format has not been verified. See tools/ocr_tool.py.
    ocr_enabled: bool = False
    ocr_max_image_bytes: int = 10_000_000
    # Which reader the ocr_image tool uses.
    #
    #   "tesseract" - a ~50 MB binary, under a second a page, transcribes
    #                 text line by line. Needs Tesseract installed.
    #   "model"     - the GLM-OCR vision model: ~1.4 GB and ~30 s a page, but
    #                 it understands tables, columns and handwriting.
    #
    # "model" is the default only because it is what was here first and its
    # weights are already on disk: changing the default would silently break a
    # working setup for anyone without Tesseract installed. On this hardware
    # Tesseract is usually the better trade - set OCR_BACKEND=tesseract once
    # you have it. Neither is strictly better, which is why it is a switch.
    ocr_backend: str = "model"
    # Empty means "find it": PATH first, then the Windows installer's own
    # locations. Set it to a full path when Tesseract is somewhere unusual.
    tesseract_cmd: str = ""
    tesseract_lang: str = "eng"
    # Page segmentation mode. 3 is fully automatic; 6 ("a single uniform
    # block") is what to try when 3 scrambles a simple image.
    tesseract_psm: int = 3
    # Shared by both backends, but they are worlds apart: Tesseract should
    # finish in under a second, the model takes tens of seconds.
    ocr_timeout: float = 120.0
    # --- speech to text ---
    #
    # No enabled flag, unlike OCR. There is nothing to turn off: whisper runs
    # only while a clip is being transcribed, holds no RAM between times, and
    # the microphone is hidden outright when no build is installed. A switch
    # would be a switch for nothing.
    #
    # Empty means "find it": vendor/whisper first, then PATH.
    whisper_cmd: str = ""
    # Empty means "find one": whisper/, then weights/, smallest first.
    whisper_model: str = ""
    # Four, like the model servers, and for the same measured reason - this
    # machine has four hardware threads and nothing gains from oversubscribing
    # them.
    whisper_threads: int = 4

    # Reading a reply aloud. Empty means "find one": tts/, then TTS/, then
    # voices/, taking the first .onnx with its .json beside it.
    piper_voice: str = ""
    # Unlike whisper, the voice is kept loaded between utterances - 7 s of it
    # is loading, and paying that before every spoken reply is 7 s of silence
    # for a one-line answer. 300 s matches the model manager's own idle
    # timeout, and the same sweeper gives the 175 MB back.
    piper_idle_seconds: float = 300.0

    ocr_allowed_extensions: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    )

    # --- HTTP ---
    # Local CPU inference is slow: an 8B model on a 2-core laptop runs at well
    # under 1 token/sec, so a single tool round can take minutes. Raise
    # AGENT_REQUEST_TIMEOUT further if you see QwenTimeoutError.
    request_timeout: float = 1200.0
    connect_timeout: float = 10.0

    # --- Conversation ---
    # Number of non-system messages kept in history. The system prompt is
    # always preserved on top of this. Set to 0 to disable trimming.
    max_history_messages: int = 60

    # --- Tool results in the model's context ---
    #
    # A tool result goes straight into the context, and several of them are
    # unbounded: a page of OCR, a file read, a long directory listing. On a
    # 4096-token model the system prompt and tool schemas already cost about
    # 1,080 tokens, so one dense page overflows the window and the turn fails
    # outright - which is what "I keep hitting the context window" looks like.
    #
    # The cap is a share of the model's own context rather than a fixed number,
    # so a bigger model is allowed bigger results without editing anything.
    model_context: int = 4096
    tool_result_share: float = 0.25
    # And the share the conversation itself may take. `max_history_messages`
    # counts messages, which says nothing about size: sixty short exchanges fit
    # a 4096-token window comfortably and three pages of OCR do not. Both
    # limits apply, and whichever bites first wins.
    history_share: float = 0.5

    # --- Agent loop ---
    max_iterations: int = 8

    # --- Chat history and memory ---
    # SQLite file holding past conversations and stored facts. Kept under
    # data/ so generated state never sits beside source. AGENT_DB_PATH moves it.
    db_path: Path = field(default=PROJECT_ROOT / "data" / "chat_history.db")

    # --- Filesystem tool ---
    # The only directory the agent may read. Defaults to this project.
    workspace: Path = field(default=PROJECT_ROOT)
    # Refuse to read files larger than this.
    max_read_bytes: int = 200_000
    # Writing is OFF by default. Even enabled it can only create files and
    # directories - there is no delete, rename or move anywhere in the tool.
    file_writes_enabled: bool = False
    max_write_bytes: int = 200_000

    # --- Python tool ---
    # OFF by default and it should stay off unless you have read the security
    # note at the top of tools/python_tool.py. The restrictions there raise the
    # cost of an escape; they are not a sandbox.
    python_tool_enabled: bool = False
    python_timeout: float = 10.0
    python_max_output_chars: int = 4000
    # A SECOND opt-in, on top of python_tool_enabled. Runs script files as
    # plain CPython: imports, packages, filesystem. Arbitrary execution, and
    # refused when the workspace is the project itself.
    python_unrestricted: bool = False

    # --- Terminal tool ---
    # OFF by default. Read the security note at the top of tools/shell_tool.py
    # before enabling: it is an allowlist of read-only commands run without a
    # shell, not a sandbox.
    shell_tool_enabled: bool = False
    shell_timeout: float = 30.0
    shell_max_output_chars: int = 4000
    # Extra executables to allow, comma-separated in AGENT_SHELL_EXTRA.
    # Anything added here runs with no sub-command restrictions.
    shell_extra_commands: tuple[str, ...] = ()

    # --- Git tool ---
    # OFF by default. Reading is safe; committing is a second opt-in. Nothing
    # in the tool can reach a remote or discard uncommitted work.
    git_tool_enabled: bool = False
    git_timeout: float = 30.0
    git_allow_writes: bool = False

    # --- Memory tool ---
    # OFF by default. Stores durable facts in the same SQLite file as the
    # chat history. Not injected into the prompt - see tools/memory_tool.py.
    memory_tool_enabled: bool = False

    # --- Document search (RAG) ---
    # OFF by default, like every other optional tool. The cost of it being on
    # is not risk here but memory: searching starts a second Python process
    # holding torch and the embedding model, and this machine has 8 GB.
    #
    # That process is short-lived by design. It loads on the first search,
    # and the same sweeper that unloads an idle llama-server stops it after
    # rag_idle_seconds, so the steady state is no embedding model resident.
    rag_enabled: bool = False
    # Where the index lives. Under data/ with the rest of the generated state.
    rag_store: Path = field(default=PROJECT_ROOT / "data" / "rag")
    # BAAI/bge-small-en-v1.5: 384 dimensions, 512-token window, ~130 MB on
    # disk. Changing this makes existing vectors meaningless, which the
    # manager detects and reports as "rebuild the index" rather than
    # silently mixing two coordinate spaces.
    rag_model: str = "BAAI/bge-small-en-v1.5"
    rag_dimension: int = 384
    # Empty means the standard Hugging Face cache. Set it to pin the model
    # inside the project instead.
    rag_model_dir: str = ""
    # Chunking, in tokens. See rag/chunker.py for how tokens are estimated
    # without loading the tokeniser.
    rag_chunk_tokens: int = 500
    rag_overlap_tokens: int = 75
    # Search. min_score is cosine similarity: BGE puts a genuine match well
    # above 0.6 and unrelated text near 0.3, so this mostly filters out
    # "nothing here matches" rather than ranking.
    rag_top_k: int = 5
    rag_min_score: float = 0.3
    # Keyword matching alongside the embeddings, fused by rank.
    #
    # On, because it is the cheapest recall this project has: the chunk text is
    # already in SQLite, FTS5 is already in the standard library, and it costs
    # no model time and no RAM. It exists because embeddings are weakest at
    # exactly what a reference document is full of - a term that has to appear
    # rather than be alluded to, like "E2", a formula, or a surname - and
    # bge-small's noise floor (see memory_min_similarity) leaves little room to
    # tell a weak match from none.
    #
    # Set RAG_HYBRID=0 to measure what it is actually buying on your documents.
    rag_hybrid: bool = True
    # Pull embedded raster figures out of PDFs while indexing.
    #
    # On, because it costs a PNG write per figure and no model time. What it
    # buys is a figure's caption tied to the page it is on, and the picture
    # itself kept somewhere findable - which is the groundwork for anything
    # that can actually look at a chart. Vector artwork is not covered: a chart
    # drawn as lines is not an image as far as the file is concerned.
    rag_figures: bool = True
    # Total characters of retrieved text returned in one tool call.
    rag_context_chars: int = 6000
    # Two cores, shared with llama-server. Taking both makes the model this is
    # meant to be helping crawl.
    rag_threads: int = 2
    # Measured on this machine, embedding 1,750-character chunks: the worker
    # idles at ~417 MB with the model loaded, and each batch slot adds roughly
    # 15 MB of activations - 535 MB at batch 4, 578 at 8, 674 at 16, 821 at 32.
    # Throughput barely moves across that range because two cores are the
    # bottleneck, so the larger batches buy memory pressure and nothing else.
    rag_batch_size: int = 8
    # Seconds of inactivity before the embedding worker is stopped.
    rag_idle_seconds: float = 120.0
    rag_max_file_bytes: int = 20_000_000

    # --- Memory ---
    # The tools are off by default (AGENT_ENABLE_MEMORY=1); the store itself
    # is always there, because the context builder uses it whether or not the
    # model can call the tools.
    #
    # The auxiliary model is what makes memory *intelligent* rather than
    # merely persistent, and it is the one part that costs a model switch.
    # Everything else - storing, retrieving, ranking, decay, dedupe - is
    # ordinary code and runs with whatever model happens to be loaded.
    memory_aux_model: str = "tiny"
    # OFF by default. Turning it on lets the agent stop the chat model and
    # start the auxiliary one when it is idle, which is a visible pause; that
    # should be a choice, not a surprise.
    memory_processing_enabled: bool = False
    # Where the memory vector index lives.
    memory_store: Path = field(default=PROJECT_ROOT / "data" / "memory")
    # How many memories may be retrieved into one turn's context.
    memory_top_k: int = 5
    # Below this final score a memory is not worth its prompt tokens. This is
    # what keeps "what is photosynthesis?" from dragging in personal notes.
    memory_score_floor: float = 0.10
    # The raw-similarity gate. Calibrated for bge-small-en-v1.5, whose noise
    # floor is high: unrelated English sentences score 0.4-0.55, so anything
    # lower retrieves the whole store for every question. Re-measure this if
    # RAG_MODEL changes - see memory/retrieval.py.
    memory_min_similarity: float = 0.55
    # The whole context budget the builder works to, in tokens. Well under the
    # 4096-token window in models.json, leaving room for the answer.
    memory_context_tokens: int = 3000
    # Queue an extraction job every N messages, not every turn.
    memory_extract_every: int = 4
    # Wait for this many queued jobs before a model switch is worth it.
    memory_queue_high_water: int = 6
    # Summarise a conversation once it passes this many messages.
    memory_summarize_after: int = 24
    # Jobs processed in one auxiliary-model session.
    memory_batch_size: int = 12

    # --- HTTP tool ---
    # OFF by default. Loopback only unless you widen it: adding a public host
    # is what turns this from a local-service inspector into a web client.
    http_tool_enabled: bool = False
    http_allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    http_timeout: float = 20.0
    http_max_bytes: int = 100_000
    # GET and HEAD always; the state-changing methods need this.
    http_allow_writes: bool = False

    @property
    def max_tool_result_chars(self) -> int:
        """How much of one tool result may reach the model, in characters.

        A quarter of the context by default. The rest has to hold the system
        prompt, the tool schemas, the conversation so far and the answer, and
        a single result taking more than a quarter of the window is a result
        that should have been summarised by whatever produced it.
        """
        tokens = max(256, int(self.model_context * self.tool_result_share))
        return int(tokens * 3.27)

    @property
    def max_history_chars(self) -> int:
        """How much conversation may be replayed to the model, in characters.

        Half the context by default, leaving the rest for the system prompt,
        the tool schemas and the answer itself.
        """
        tokens = max(512, int(self.model_context * self.history_share))
        return int(tokens * 3.27)

    @classmethod
    def from_env(cls) -> "Config":
        defaults = cls()
        workspace = Path(
            _env_str("AGENT_WORKSPACE", str(defaults.workspace))
        ).expanduser()
        try:
            workspace = workspace.resolve()
        except OSError as exc:
            raise ValueError(f"AGENT_WORKSPACE is not a usable path: {exc}") from None
        if not workspace.is_dir():
            raise ValueError(f"AGENT_WORKSPACE is not a directory: {workspace}")

        return cls(
            qwen_url=_env_str("QWEN_SERVER_URL", defaults.qwen_url).rstrip("/"),
            qwen_model=_env_str("QWEN_MODEL", defaults.qwen_model),
            temperature=_env_float("QWEN_TEMPERATURE", defaults.temperature),
            top_p=_env_float("QWEN_TOP_P", defaults.top_p),
            max_tokens=_env_int("QWEN_MAX_TOKENS", defaults.max_tokens),
            enable_thinking=_env_bool("QWEN_ENABLE_THINKING", defaults.enable_thinking),
            ocr_url=_env_str("OCR_SERVER_URL", defaults.ocr_url).rstrip("/"),
            ocr_model=_env_str("OCR_MODEL", defaults.ocr_model),
            ocr_enabled=_env_bool("OCR_ENABLED", defaults.ocr_enabled),
            ocr_max_image_bytes=_env_int(
                "OCR_MAX_IMAGE_BYTES", defaults.ocr_max_image_bytes
            ),
            ocr_backend=_env_str("OCR_BACKEND", defaults.ocr_backend).lower(),
            tesseract_cmd=_env_str("TESSERACT_CMD", defaults.tesseract_cmd),
            whisper_cmd=_env_str("WHISPER_CMD", defaults.whisper_cmd),
            whisper_model=_env_str("WHISPER_MODEL", defaults.whisper_model),
            whisper_threads=_env_int("WHISPER_THREADS", defaults.whisper_threads),
            piper_voice=_env_str("PIPER_VOICE", defaults.piper_voice),
            piper_idle_seconds=_env_float(
                "PIPER_IDLE_SECONDS", defaults.piper_idle_seconds
            ),
            tesseract_lang=_env_str("TESSERACT_LANG", defaults.tesseract_lang),
            tesseract_psm=_env_int("TESSERACT_PSM", defaults.tesseract_psm),
            ocr_timeout=_env_float("OCR_TIMEOUT", defaults.ocr_timeout),
            request_timeout=_env_float("AGENT_REQUEST_TIMEOUT", defaults.request_timeout),
            connect_timeout=_env_float("AGENT_CONNECT_TIMEOUT", defaults.connect_timeout),
            max_history_messages=_env_int("AGENT_MAX_HISTORY", defaults.max_history_messages),
            max_iterations=_env_int("AGENT_MAX_ITERATIONS", defaults.max_iterations),
            tool_result_share=_env_float(
                "AGENT_TOOL_RESULT_SHARE", defaults.tool_result_share
            ),
            history_share=_env_float("AGENT_HISTORY_SHARE", defaults.history_share),
            workspace=workspace,
            db_path=Path(
                _env_str("AGENT_DB_PATH", str(defaults.db_path))
            ).expanduser(),
            max_read_bytes=_env_int("AGENT_MAX_READ_BYTES", defaults.max_read_bytes),
            file_writes_enabled=_env_bool(
                "AGENT_ENABLE_FILE_WRITES", defaults.file_writes_enabled
            ),
            max_write_bytes=_env_int(
                "AGENT_MAX_WRITE_BYTES", defaults.max_write_bytes
            ),
            python_tool_enabled=_env_bool(
                "AGENT_ENABLE_PYTHON_TOOL", defaults.python_tool_enabled
            ),
            python_timeout=_env_float("AGENT_PYTHON_TIMEOUT", defaults.python_timeout),
            python_max_output_chars=_env_int(
                "AGENT_PYTHON_MAX_OUTPUT", defaults.python_max_output_chars
            ),
            python_unrestricted=_env_bool(
                "AGENT_PYTHON_UNRESTRICTED", defaults.python_unrestricted
            ),
            shell_tool_enabled=_env_bool(
                "AGENT_ENABLE_SHELL_TOOL", defaults.shell_tool_enabled
            ),
            shell_timeout=_env_float("AGENT_SHELL_TIMEOUT", defaults.shell_timeout),
            shell_max_output_chars=_env_int(
                "AGENT_SHELL_MAX_OUTPUT", defaults.shell_max_output_chars
            ),
            shell_extra_commands=tuple(
                part.strip()
                for part in _env_str("AGENT_SHELL_EXTRA", "").split(",")
                if part.strip()
            ),
            git_tool_enabled=_env_bool(
                "AGENT_ENABLE_GIT_TOOL", defaults.git_tool_enabled
            ),
            git_timeout=_env_float("AGENT_GIT_TIMEOUT", defaults.git_timeout),
            git_allow_writes=_env_bool(
                "AGENT_GIT_ALLOW_WRITES", defaults.git_allow_writes
            ),
            memory_tool_enabled=_env_bool(
                "AGENT_ENABLE_MEMORY", defaults.memory_tool_enabled
            ),
            http_tool_enabled=_env_bool(
                "AGENT_ENABLE_HTTP_TOOL", defaults.http_tool_enabled
            ),
            http_allowed_hosts=tuple(
                part.strip().lower()
                for part in _env_str(
                    "AGENT_HTTP_HOSTS", ",".join(defaults.http_allowed_hosts)
                ).split(",")
                if part.strip()
            ),
            http_timeout=_env_float("AGENT_HTTP_TIMEOUT", defaults.http_timeout),
            http_max_bytes=_env_int("AGENT_HTTP_MAX_BYTES", defaults.http_max_bytes),
            http_allow_writes=_env_bool(
                "AGENT_HTTP_ALLOW_WRITES", defaults.http_allow_writes
            ),
            rag_enabled=_env_bool("AGENT_ENABLE_RAG", defaults.rag_enabled),
            rag_store=Path(
                _env_str("RAG_STORE", str(defaults.rag_store))
            ).expanduser(),
            rag_model=_env_str("RAG_MODEL", defaults.rag_model),
            rag_dimension=_env_int("RAG_DIMENSION", defaults.rag_dimension),
            rag_model_dir=_env_str("RAG_MODEL_DIR", defaults.rag_model_dir),
            rag_chunk_tokens=_env_int("RAG_CHUNK_TOKENS", defaults.rag_chunk_tokens),
            rag_overlap_tokens=_env_int(
                "RAG_CHUNK_OVERLAP", defaults.rag_overlap_tokens
            ),
            rag_top_k=_env_int("RAG_TOP_K", defaults.rag_top_k),
            rag_min_score=_env_float("RAG_MIN_SCORE", defaults.rag_min_score),
            rag_hybrid=_env_bool("RAG_HYBRID", defaults.rag_hybrid),
            rag_figures=_env_bool("RAG_FIGURES", defaults.rag_figures),
            rag_context_chars=_env_int(
                "RAG_CONTEXT_CHARS", defaults.rag_context_chars
            ),
            rag_threads=_env_int("RAG_THREADS", defaults.rag_threads),
            rag_batch_size=_env_int("RAG_BATCH_SIZE", defaults.rag_batch_size),
            rag_idle_seconds=_env_float(
                "RAG_IDLE_SECONDS", defaults.rag_idle_seconds
            ),
            rag_max_file_bytes=_env_int(
                "RAG_MAX_FILE_BYTES", defaults.rag_max_file_bytes
            ),
            memory_aux_model=_env_str("MEMORY_AUX_MODEL", defaults.memory_aux_model),
            memory_processing_enabled=_env_bool(
                "AGENT_ENABLE_MEMORY_PROCESSING", defaults.memory_processing_enabled
            ),
            memory_store=Path(
                _env_str("MEMORY_STORE", str(defaults.memory_store))
            ).expanduser(),
            memory_top_k=_env_int("MEMORY_TOP_K", defaults.memory_top_k),
            memory_score_floor=_env_float(
                "MEMORY_SCORE_FLOOR", defaults.memory_score_floor
            ),
            memory_min_similarity=_env_float(
                "MEMORY_MIN_SIMILARITY", defaults.memory_min_similarity
            ),
            memory_context_tokens=_env_int(
                "MEMORY_CONTEXT_TOKENS", defaults.memory_context_tokens
            ),
            memory_extract_every=_env_int(
                "MEMORY_EXTRACT_EVERY", defaults.memory_extract_every
            ),
            memory_queue_high_water=_env_int(
                "MEMORY_QUEUE_HIGH_WATER", defaults.memory_queue_high_water
            ),
            memory_summarize_after=_env_int(
                "MEMORY_SUMMARIZE_AFTER", defaults.memory_summarize_after
            ),
            memory_batch_size=_env_int(
                "MEMORY_BATCH_SIZE", defaults.memory_batch_size
            ),
        )


# Characters per token, measured on this machine against a real turn: 906
# tokens for the system prompt, the tool schemas and a short question. Used to
# turn a token budget into a character budget without loading a tokeniser.
CHARS_PER_TOKEN = 3.27


def load_config() -> Config:
    """Build the configuration from the environment.

    `.env` is read first so API keys land in the environment before anything
    looks for them.
    """
    load_env_file()
    return Config.from_env()
