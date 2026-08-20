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

    # --- HTTP tool ---
    # OFF by default. Loopback only unless you widen it: adding a public host
    # is what turns this from a local-service inspector into a web client.
    http_tool_enabled: bool = False
    http_allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    http_timeout: float = 20.0
    http_max_bytes: int = 100_000
    # GET and HEAD always; the state-changing methods need this.
    http_allow_writes: bool = False

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
            request_timeout=_env_float("AGENT_REQUEST_TIMEOUT", defaults.request_timeout),
            connect_timeout=_env_float("AGENT_CONNECT_TIMEOUT", defaults.connect_timeout),
            max_history_messages=_env_int("AGENT_MAX_HISTORY", defaults.max_history_messages),
            max_iterations=_env_int("AGENT_MAX_ITERATIONS", defaults.max_iterations),
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
        )


def load_config() -> Config:
    """Build the configuration from the environment.

    `.env` is read first so API keys land in the environment before anything
    looks for them.
    """
    load_env_file()
    return Config.from_env()
