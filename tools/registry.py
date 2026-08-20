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
from tools.memory_tool import build_memory_tools
from tools.shell_tool import build_shell_tool


@dataclass(frozen=True)
class DisabledTool:
    """A tool that exists but is not offered to the model, and why."""

    category: str
    reason: str


def build_default_registry(config: Config) -> tuple[ToolRegistry, list[DisabledTool]]:
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

    if config.memory_tool_enabled:
        for tool in build_memory_tools(config.db_path):
            registry.register(tool)
    else:
        disabled.append(
            DisabledTool(
                category="memory",
                reason=(
                    "off by default - durable facts across conversations. "
                    "Set AGENT_ENABLE_MEMORY=1."
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
                    "off by default. GLM-OCR-Q8_0.gguf holds only the language "
                    "half; the vision projector (mmproj) is a separate file "
                    "and is not present, so the OCR server cannot start. "
                    "Set OCR_ENABLED=1 once you have it."
                ),
            )
        )

    return registry, disabled
