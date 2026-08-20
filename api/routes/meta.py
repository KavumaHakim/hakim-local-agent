"""The tool roster and a health check.

Both are read-only views of how the process was configured. Neither can change
anything: the tool flags are environment variables read at startup, and
exposing a switch for them over HTTP would defeat the point of having them be
deliberate.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import DisabledToolOut, HealthOut, ToolOut, ToolsOut

router = APIRouter(tags=["meta"])


@router.get("/tools", response_model=ToolsOut)
def list_tools(runtime: Runtime = Depends(get_runtime)):
    """Which tools the model is offered, and why the others are withheld.

    The reasons are included because each one names the environment variable
    that would enable it - otherwise there is no way to find that out from the
    interface.
    """
    registry, disabled = runtime.registry_for(runtime.config)
    tools = [
        ToolOut(
            name=tool.name,
            category=category,
            description=tool.description,
            parameters=tool.parameters,
        )
        for category, entries in registry.categories().items()
        for tool in entries
    ]
    return ToolsOut(
        tools=tools,
        disabled=[DisabledToolOut(**vars(item)) for item in disabled],
        workspace=str(runtime.config.workspace),
    )


@router.get("/health", response_model=HealthOut)
def health(runtime: Runtime = Depends(get_runtime)):
    """Whether the API is up, and whether it is mid-turn.

    Says nothing about the model servers - those have their own state and are
    reported by /models, which can be slow when it reconciles processes.
    """
    return HealthOut(
        ok=True,
        busy=runtime.queue.busy(),
        queue_depth=runtime.queue.depth(),
        workspace=str(runtime.config.workspace),
        db_path=str(runtime.config.db_path),
    )
