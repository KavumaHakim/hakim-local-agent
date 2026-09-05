"""Assembles the tools available to the agent."""

from __future__ import annotations

from dataclasses import dataclass

from config import Config
from tools.base import ToolRegistry
from tools.calculator import CALCULATOR_TOOL
from tools.filesystem import WorkspaceFiles
from tools.ocr_tool import OcrClient
from tools.python_tool import build_python_file_tool, build_python_tool
from tools.git_tool import build_git_tools
from tools.http_tool import build_http_tool
from tools.results import ResultStore, build_result_tool
from tools.skills import SkillLibrary
from tools.shell_tool import ApprovalCheck, build_shell_tool


@dataclass(frozen=True)
class DisabledTool:
    """A tool that exists but is not offered to the model, and why."""

    category: str
    reason: str


# Said when a tool is switched on but the packages it needs are not there.
# A fresh clone installs the small dependency set by default, so this is what
# someone sees after turning on memory or document search without having run
# the setup script with --with-rag.
MISSING_DEPS = (
    "switched on, but its dependencies are not installed. They are optional "
    "and large (numpy, torch, sentence-transformers), so they are a separate "
    "install: `pip install -r requirements-rag.txt`, or re-run the setup "
    "script with --with-rag."
)


def _installed(category: str) -> bool:
    """Whether an optional tool's dependencies are importable.

    Checked rather than assumed, because the alternative is an ImportError out
    of `build_default_registry` - which is called by /api/tools, so a missing
    package would turn the whole roster endpoint into a 500 instead of one
    tool reporting itself unavailable.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    if category == "documents":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
    return True


def build_default_registry(
    config: Config, *, approve: ApprovalCheck | None = None
) -> tuple[ToolRegistry, list[DisabledTool]]:
    """Build the registry, plus the list of tools deliberately left out.

    Disabled tools are not registered at all: sending the model a definition it
    is only going to fail on wastes both context and a whole round-trip.
    """
    workspace = WorkspaceFiles(
        config.workspace,
        max_read_bytes=config.max_read_bytes,
        max_write_bytes=config.max_write_bytes,
    )

    registry = ToolRegistry([CALCULATOR_TOOL])
    for tool in workspace.tools():
        registry.register(tool)

    # Always registered, never free: `read_result` is the other half of
    # truncating a result, and the lens only opens its group once something
    # has actually been set aside.
    registry.register(build_result_tool(ResultStore(config.results_dir)))

    # Only when there is something to offer. An index of nothing is a tool
    # definition the model pays for on every request and can never use.
    library = SkillLibrary(config.skills_dir)
    if len(library):
        registry.register(library.tool())

    disabled: list[DisabledTool] = []

    if config.file_writes_enabled:
        for tool in workspace.write_tools():
            registry.register(tool)
    else:
        disabled.append(
            DisabledTool(
                category="file writes",
                reason=(
                    "off by default - read-only means the agent can diagnose "
                    "but never fix. Set AGENT_ENABLE_FILE_WRITES=1 to allow "
                    "creating files and directories. There is no delete or "
                    "rename either way."
                ),
            )
        )

    if config.python_tool_enabled:
        registry.register(
            build_python_tool(
                timeout=config.python_timeout,
                max_output_chars=config.python_max_output_chars,
            )
        )
        registry.register(
            build_python_file_tool(
                workspace=workspace,
                timeout=config.python_timeout,
                max_output_chars=config.python_max_output_chars,
                unrestricted=config.python_unrestricted,
                approve=approve,
            )
        )
    else:
        disabled.append(
            DisabledTool(
                category="python",
                reason=(
                    "off by default - restricted, but not a real sandbox. "
                    "Set AGENT_ENABLE_PYTHON_TOOL=1 after reading the note in "
                    "tools/python_tool.py."
                ),
            )
        )

    if config.shell_tool_enabled:
        registry.register(
            build_shell_tool(
                config.workspace,
                timeout=config.shell_timeout,
                max_output_chars=config.shell_max_output_chars,
                extra_commands=config.shell_extra_commands,
                # None wherever there is no one to ask - the CLI, a test.
                # Commands needing approval are refused there rather than
                # run, so the gate cannot be bypassed by the caller.
                approve=approve,
            )
        )
    else:
        disabled.append(
            DisabledTool(
                category="terminal",
                reason=(
                    "off by default - an allowlist of read-only commands run "
                    "without a shell, not a sandbox. Set "
                    "AGENT_ENABLE_SHELL_TOOL=1 after reading the note in "
                    "tools/shell_tool.py."
                ),
            )
        )

    if config.http_tool_enabled:
        registry.register(
            build_http_tool(
                allowed_hosts=config.http_allowed_hosts,
                timeout=config.http_timeout,
                max_bytes=config.http_max_bytes,
                allow_writes=config.http_allow_writes,
                approve=approve,
            )
        )
    else:
        disabled.append(
            DisabledTool(
                category="http",
                reason=(
                    "off by default - loopback hosts only unless widened. "
                    "Set AGENT_ENABLE_HTTP_TOOL=1 to let the agent inspect "
                    "local services."
                ),
            )
        )

    if config.git_tool_enabled:
        for tool in build_git_tools(
            config.workspace,
            timeout=config.git_timeout,
            allow_writes=config.git_allow_writes,
        ):
            registry.register(tool)
    else:
        disabled.append(
            DisabledTool(
                category="git",
                reason=(
                    "off by default - structured status, log, diff and "
                    "branches. Set AGENT_ENABLE_GIT_TOOL=1, and "
                    "AGENT_GIT_ALLOW_WRITES=1 for commits. No push, and "
                    "nothing that discards uncommitted work."
                ),
            )
        )

    if config.memory_tool_enabled and _installed("memory"):
        # Imported here so the CLI does not pull in numpy when memory is
        # off, exactly as document search does.
        from memory.manager import shared_manager
        from tools.memory_tool import build_memory_tools

        memory = shared_manager(
            config.db_path,
            store_dir=config.memory_store,
            dimension=config.rag_dimension,
            top_k=config.memory_top_k,
            score_floor=config.memory_score_floor,
            min_similarity=config.memory_min_similarity,
            context_tokens=config.memory_context_tokens,
            summarize_after=config.memory_summarize_after,
            extract_every=config.memory_extract_every,
            queue_high_water=config.memory_queue_high_water,
        )
        for tool in build_memory_tools(memory):
            registry.register(tool)
    elif config.memory_tool_enabled:
        disabled.append(DisabledTool(category="memory", reason=MISSING_DEPS))
    else:
        disabled.append(
            DisabledTool(
                category="memory",
                reason=(
                    "off by default - durable facts, preferences and "
                    "events that survive across conversations, retrieved "
                    "by meaning. Set AGENT_ENABLE_MEMORY=1. Storing and "
                    "retrieving need no model; AGENT_ENABLE_MEMORY_PROCESSING=1 "
                    "additionally lets the agent briefly swap the chat model "
                    "for a small one to extract and tidy memories while idle."
                ),
            )
        )

    if config.rag_enabled and _installed("documents"):
        # Imported here, not at the top: this module pulls in numpy and
        # the whole rag package, and the CLI is supposed to run on
        # `requests` alone when document search is off.
        from tools.document_search import build_document_tools

        for tool in build_document_tools(
            config.rag_store,
            model=config.rag_model,
            model_dir=config.rag_model_dir or None,
            dimension=config.rag_dimension,
            chunk_tokens=config.rag_chunk_tokens,
            overlap_tokens=config.rag_overlap_tokens,
            top_k=config.rag_top_k,
            min_score=config.rag_min_score,
            hybrid=config.rag_hybrid,
            figures=config.rag_figures,
            max_file_bytes=config.rag_max_file_bytes,
            context_budget=config.rag_context_chars,
            threads=config.rag_threads,
            batch_size=config.rag_batch_size,
            idle_seconds=config.rag_idle_seconds,
        ):
            registry.register(tool)
    elif config.rag_enabled:
        disabled.append(DisabledTool(category="documents", reason=MISSING_DEPS))
    else:
        disabled.append(
            DisabledTool(
                category="documents",
                reason=(
                    "off by default - semantic search over your own files. "
                    "Set AGENT_ENABLE_RAG=1. Searching starts a second Python "
                    "process holding the BGE embedding model (~400 MB), which "
                    "is stopped again once it has been idle. Index files with "
                    "`python -m rag index <path>` first; the model itself only "
                    "reads the index, it cannot change it."
                ),
            )
        )

    if config.ocr_enabled:
        registry.register(OcrClient(config, workspace).tool())
    else:
        disabled.append(
            DisabledTool(
                category="ocr",
                reason=(
                    "off by default - reads text out of images. Set "
                    "OCR_ENABLED=1. Two backends, chosen with OCR_BACKEND: "
                    "'tesseract' (default) is a ~50 MB binary that transcribes "
                    "a page in under a second and needs Tesseract installed; "
                    "'model' is the GLM-OCR vision model, which understands "
                    "tables and handwriting but costs ~1.4 GB and ~30 s a page "
                    "and needs its server running with both the model and its "
                    "mmproj projector file."
                ),
            )
        )

    return registry, disabled
