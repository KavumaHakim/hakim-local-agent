"""Choosing the folder the file tools may reach.

The workspace is the jail: `WorkspaceFiles` resolves every model-supplied path
and refuses anything landing outside it, the terminal tool runs with it as its
working directory, git works on it, and uploads land inside it. Until now it
could only be chosen before startup, with `AGENT_WORKSPACE`.

This is the same deliberate loosening the tool switches were, and the same
answer to it. What the jail *does* has not moved an inch - one directory,
resolved paths, no escape, no delete or rename anywhere in the filesystem
tools. What has moved is that choosing which directory is now a click rather
than a restart, which matters because a local agent is useless if it can only
ever look at its own source.

Two guards apply to the choice itself, in `runtime.resolve_workspace`: a drive
root is refused, because a jail around the whole disk is not a jail, and so are
the operating system's own directories. Neither is a security boundary - this
API can hand a model a shell when the switch is on - they stop the two mistakes
that make the jail meaningless.

The choice lives in memory, like every other override, so a restart returns to
whatever `AGENT_WORKSPACE` says.

Browsing is done here rather than in the browser because it has to be: a
directory picker in a web page reports names, never the absolute path the tools
need. The listing is sub-directory names only - where you could go, never what
is inside a file.
"""

from __future__ import annotations

import os
import string
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import get_runtime
from api.runtime import Runtime, WorkspaceError
from config import PROJECT_ROOT
from api.schemas import (
    DirectoryEntryOut,
    DirectoryOut,
    WorkspaceOut,
    WorkspaceRequest,
)

router = APIRouter(tags=["workspace"])

# Directories in one listing. A folder with more than this is not one anybody
# is picking by eye, and sending 20,000 names to the browser helps nobody.
MAX_ENTRIES = 500

# Which switched-on tools act on the workspace, and whether they can change it.
# Read-only reach is not worth naming - the filesystem tools are always on and
# always read - so this is the list of things that would surprise someone.
WORKSPACE_TOOLS: tuple[tuple[str, str, bool], ...] = (
    ("file_writes_enabled", "File writes", True),
    ("shell_tool_enabled", "Terminal", False),
    ("python_unrestricted", "Unrestricted Python", True),
    ("git_tool_enabled", "Git", False),
    ("git_allow_writes", "Git commits", True),
)


def _snapshot(runtime: Runtime) -> WorkspaceOut:
    config = runtime.effective_config()
    workspace = Path(config.workspace)

    active = [
        label for field, label, _ in WORKSPACE_TOOLS if getattr(config, field, False)
    ]
    writable = any(
        getattr(config, field, False) for field, _, writes in WORKSPACE_TOOLS if writes
    )

    return WorkspaceOut(
        path=str(workspace),
        default=str(runtime.default_workspace),
        from_env=workspace == runtime.default_workspace,
        recent=[str(path) for path in runtime.recent_workspaces],
        is_project=workspace == PROJECT_ROOT,
        writable=writable,
        active_tools=active,
    )


@router.get("/workspace", response_model=WorkspaceOut)
def get_workspace(runtime: Runtime = Depends(get_runtime)):
    """Where the file tools are pointed, and what is pointed there."""
    return _snapshot(runtime)


@router.post("/workspace", response_model=WorkspaceOut)
def set_workspace(body: WorkspaceRequest, runtime: Runtime = Depends(get_runtime)):
    """Point the file tools at another folder, from the next turn on.

    Refused mid-turn for the reason the tool switches are: the registry is
    built when a turn starts, so moving the jail underneath a running one would
    either do nothing or move it halfway through, and the second is worse than
    waiting.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. The workspace applies from the next turn, so "
            "this would not affect it anyway.",
        )
    try:
        runtime.set_workspace(body.path)
    except WorkspaceError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _snapshot(runtime)


@router.delete("/workspace", response_model=WorkspaceOut)
def reset_workspace(runtime: Runtime = Depends(get_runtime)):
    """Go back to the folder AGENT_WORKSPACE names."""
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. The workspace applies from the next turn.",
        )
    runtime.reset_workspace()
    return _snapshot(runtime)


def _roots() -> list[DirectoryEntryOut]:
    """The starting points for a walk: home, and the drives that exist."""
    found: list[DirectoryEntryOut] = []
    home = Path.home()
    if home.is_dir():
        found.append(DirectoryEntryOut(name="Home", path=str(home)))

    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            # A floppy or an empty card reader answers this slowly, but only
            # the ones Windows actually has mounted are tried at all.
            try:
                if drive.is_dir():
                    found.append(
                        DirectoryEntryOut(name=f"{letter}:", path=str(drive))
                    )
            except OSError:
                continue
    else:
        found.append(DirectoryEntryOut(name="/", path="/"))
    return found


@router.get("/workspace/browse", response_model=DirectoryOut)
def browse(
    path: str = Query("", description="Folder to list. Empty means the workspace."),
    runtime: Runtime = Depends(get_runtime),
):
    """List the sub-directories of one folder, so a workspace can be picked.

    Not confined to the current workspace, and that is the point: this is how
    the *next* one is found. It reveals directory names, which the picker
    cannot work without, and nothing else.
    """
    target = Path(path).expanduser() if path.strip() else runtime.workspace
    try:
        target = target.resolve()
    except OSError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"That is not a usable path: {exc}"
        ) from None

    if not target.is_dir():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"There is no such folder: {target}"
        )

    entries: list[DirectoryEntryOut] = []
    note = ""
    try:
        with os.scandir(target) as scan:
            for item in scan:
                try:
                    if not item.is_dir():
                        continue
                except OSError:
                    # A junction or a dead symlink. Skipping it is better than
                    # failing the whole listing over one entry.
                    continue
                entries.append(DirectoryEntryOut(name=item.name, path=item.path))
    except PermissionError:
        note = "Windows will not let this folder be listed."
    except OSError as exc:
        note = f"That folder could not be read: {exc}"

    entries.sort(key=lambda entry: entry.name.lower())
    if len(entries) > MAX_ENTRIES:
        note = (
            f"Showing the first {MAX_ENTRIES} of {len(entries)} folders. "
            f"Type the path instead if the one you want is not here."
        )
        entries = entries[:MAX_ENTRIES]

    return DirectoryOut(
        path=str(target),
        parent=None if target.parent == target else str(target.parent),
        entries=entries,
        roots=_roots(),
        note=note,
    )
