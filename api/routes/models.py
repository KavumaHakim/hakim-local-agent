"""Model lifecycle: what exists, what is loaded, load it, unload it.

Loading is an explicit request, never a side effect of rendering. On this
machine it costs up to 130 s and evicts whatever else was resident, so it
happens when someone asks for it.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    ModelHideRequest,
    ModelOut,
    ModelOverrideRequest,
    ModelPrimaryRequest,
    ModelRouterRequest,
    ModelsOut,
    RescanOut,
)
from models.manager import ModelManagerError, ModelState, available_ram_mb

router = APIRouter(prefix="/models", tags=["models"])


def _header(spec):
    """The model's GGUF header, or None for a remote or missing file.

    Cached on path and mtime: a snapshot is built on every page load, and
    re-reading five headers each time would be work for nothing. Keying on
    mtime means a replaced file is still re-read.
    """
    if spec.remote or not spec.path:
        return None
    try:
        stamp = spec.path.stat().st_mtime_ns
    except OSError:
        return None
    return _read_header(str(spec.path), stamp)


@lru_cache(maxsize=64)
def _read_header(path: str, mtime_ns: int):
    from models.gguf import read_metadata

    return read_metadata(path)


def _file_mb(spec) -> int:
    if spec.remote or not spec.path:
        return 0
    try:
        return int(spec.path.stat().st_size / (1024 * 1024))
    except OSError:
        return 0


def _notes(spec, header) -> list[str]:
    """What someone choosing this model should know before they do.

    Only things that are true and actionable: weights that are missing, or a
    context far below what the model was trained for - which is the common
    surprise, because the cap is a RAM decision the model knows nothing about.
    """
    notes: list[str] = []
    if not spec.remote and not spec.available:
        if spec.mmproj is not None and not spec.mmproj.is_file():
            notes.append(
                f"Its vision projector is missing: expected {spec.mmproj.name} "
                f"beside the model."
            )
        else:
            notes.append(f"Weights not found at {spec.path}.")

    if header is not None and header.training_context > spec.context * 2:
        notes.append(
            f"Running at {spec.context:,} of the {header.training_context:,} "
            f"tokens it was trained for; the full context would need "
            f"{header.kv_cache_mb(header.training_context):,} MB of KV cache."
        )
    return notes


def _snapshot(runtime: Runtime) -> ModelsOut:
    manager = runtime.manager
    # Reconciling state probes every registered port, so it happens once here
    # and everything else is derived from the result. Calling statuses() and
    # then active_key() would do the same work three times over.
    entries = manager.statuses()
    active = next(
        (
            entry.spec.key
            for entry in entries
            if entry.state is ModelState.READY
            and not entry.spec.remote
            and entry.spec.role == "chat"
        ),
        None,
    )
    online = runtime.connectivity.online()

    models = []
    for entry in entries:
        spec = entry.spec
        models.append(
            ModelOut(
                key=spec.key,
                label=spec.label,
                description=spec.description,
                port=spec.port,
                url=spec.url,
                context=spec.context,
                threads=spec.threads,
                gpu_layers=spec.gpu_layers,
                min_free_mb=spec.min_free_mb,
                available=spec.available,
                state=entry.state.value,
                pid=entry.pid,
                error=entry.error,
                warning=entry.warning,
                adopted=entry.adopted,
                role=spec.role,
                provider=spec.provider,
                remote=spec.remote,
                has_key=spec.has_key if spec.remote else True,
                # A local model is usable if its weights exist; a hosted one
                # needs both a key and a network.
                usable=(
                    spec.available and online if spec.remote else spec.available
                ),
                discovered=manager.is_discovered(spec.key),
                hidden=manager.is_hidden(spec.key),
                customised=bool(manager.preferences.for_key(spec.key)),
                file_mb=_file_mb(spec),
                # Straight from the GGUF header, so the settings panel can say
                # "trained for 262,144, running at 4,096" rather than showing a
                # context that looks arbitrary.
                training_context=_header(spec).training_context if _header(spec) else 0,
                kv_cache_mb=(
                    _header(spec).kv_cache_mb(spec.context) if _header(spec) else 0
                ),
                notes=_notes(spec, _header(spec)),
            )
        )
    return ModelsOut(
        models=models,
        default_key=manager.default_key,
        active_key=active,
        router_fast=manager.router_fast,
        router_strong=manager.router_strong,
        max_active=manager.max_active,
        idle_timeout_seconds=int(manager.idle_timeout),
        available_ram_mb=available_ram_mb(),
        online=online,
        models_dir=str(manager.models_dir),
        # Resolved, because the registry stores it relative to the project and
        # the panel shows it to a person: '..\LLAMA CP\llama-server.exe' in the
        # middle of an absolute path is legible to a filesystem, not a reader.
        server_exe=str(manager.server_exe.resolve()),
        setup_required=manager.setup_required,
    )


@router.get("", response_model=ModelsOut)
def list_models(runtime: Runtime = Depends(get_runtime)):
    """Every registered model and its current state.

    The snapshot reconciles first, so a server started outside the agent - or
    one that died - is reflected rather than reported from a stale table.
    """
    return _snapshot(runtime)


@router.post("/{key}/load", response_model=ModelsOut)
def load_model(key: str, runtime: Runtime = Depends(get_runtime)):
    """Start a model, stopping the current one first.

    This blocks for as long as the load takes - minutes for the 8B. The client
    is expected to show that rather than time out.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Switching models now would unload the model "
            "generating it.",
        )
    try:
        runtime.manager.ensure(key)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _snapshot(runtime)


@router.post("/{key}/unload", response_model=ModelsOut)
def unload_model(key: str, runtime: Runtime = Depends(get_runtime)):
    """Stop a model and give back its RAM."""
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT, "A turn is running on this model."
        )
    try:
        runtime.manager.get_spec(key)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    runtime.manager.stop(key)
    return _snapshot(runtime)


# --- settings ---------------------------------------------------------------
#
# Everything below changes the catalogue rather than what is running. None of
# it starts or stops a process: a retuned context reaches llama-server on its
# command line, so it applies the next time that model starts, and saying so is
# better than restarting a model underneath someone mid-conversation.


@router.post("/rescan", response_model=RescanOut)
def rescan(runtime: Runtime = Depends(get_runtime)):
    """Re-read the models folder. Call it after dropping a file in.

    Cheap - it stats files and reads GGUF headers, never a tensor - so it is
    also safe to call speculatively when the settings panel opens.
    """
    try:
        added = runtime.manager.rescan()
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    snapshot = _snapshot(runtime)
    return RescanOut(
        models_dir=str(runtime.manager.models_dir),
        added=added,
        total=len(snapshot.models),
        models=snapshot,
    )


@router.post("/primary", response_model=ModelsOut)
def set_primary(body: ModelPrimaryRequest, runtime: Runtime = Depends(get_runtime)):
    """Choose the model everything defaults to.

    This is what the first-launch picker calls, and what the settings panel
    calls later. It is remembered in data/models.local.json, so it survives a
    restart without touching the shipped registry.
    """
    try:
        runtime.manager.set_primary(body.key)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _snapshot(runtime)


@router.post("/router", response_model=ModelsOut)
def set_router(body: ModelRouterRequest, runtime: Runtime = Depends(get_runtime)):
    """Point the auto-router's cheap and capable ends at chosen models."""
    if not body.fast and not body.strong:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Give at least one of fast or strong."
        )
    try:
        runtime.manager.set_router(fast=body.fast, strong=body.strong)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _snapshot(runtime)


@router.patch("/{key}", response_model=ModelsOut)
def override(
    key: str,
    body: ModelOverrideRequest,
    runtime: Runtime = Depends(get_runtime),
):
    """Retune one model: its label, context, threads, GPU layers or RAM floor.

    Applies from that model's next start, because llama-server is told all of
    them on the command line.
    """
    values = body.model_dump(exclude_none=True)
    if not values:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Give at least one field to change."
        )
    try:
        applied = runtime.manager.set_override(key, values)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    if not applied:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "None of those fields can be changed."
        )
    return _snapshot(runtime)


@router.delete("/{key}/override", response_model=ModelsOut)
def clear_override(key: str, runtime: Runtime = Depends(get_runtime)):
    """Put a model back to its registry or auto-detected values."""
    try:
        runtime.manager.get_spec(key)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    runtime.manager.clear_override(key)
    return _snapshot(runtime)


@router.post("/{key}/hidden", response_model=ModelsOut)
def set_hidden(
    key: str,
    body: ModelHideRequest,
    runtime: Runtime = Depends(get_runtime),
):
    """Stop offering a model, without deleting its file.

    For the case the folder has more in it than you want in the picker. The
    entry stays in the catalogue so it can be brought back.
    """
    try:
        runtime.manager.set_hidden(key, body.hidden)
    except ModelManagerError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _snapshot(runtime)
