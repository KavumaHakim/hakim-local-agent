"""Model lifecycle: what exists, what is loaded, load it, unload it.

Loading is an explicit request, never a side effect of rendering. On this
machine it costs up to 130 s and evicts whatever else was resident, so it
happens when someone asks for it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import ModelOut, ModelsOut
from models.manager import ModelManagerError, available_ram_mb

router = APIRouter(prefix="/models", tags=["models"])


def _snapshot(runtime: Runtime) -> ModelsOut:
    manager = runtime.manager
    models = []
    for entry in manager.statuses():
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
                min_free_mb=spec.min_free_mb,
                available=spec.available,
                state=entry.state.value,
                pid=entry.pid,
                error=entry.error,
                warning=entry.warning,
                adopted=entry.adopted,
            )
        )
    return ModelsOut(
        models=models,
        default_key=manager.default_key,
        active_key=manager.active_key(),
        router_fast=manager.router_fast,
        router_strong=manager.router_strong,
        max_active=manager.max_active,
        idle_timeout_seconds=int(manager.idle_timeout),
        available_ram_mb=available_ram_mb(),
    )


@router.get("", response_model=ModelsOut)
def list_models(runtime: Runtime = Depends(get_runtime)):
    """Every registered model and its current state.

    `refresh()` first so a server started outside the agent - or one that died
    - is reflected rather than reported from a stale table.
    """
    runtime.manager.refresh()
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
