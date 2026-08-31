"""The tool roster, its switches, and a health check.

The switches are a deliberate loosening of the original design, and it is
worth being honest about which one.

Every risky tool is off unless an environment variable says otherwise, and the
point of that was that turning one on is a considered act taken before the
process starts. Flipping them from a browser makes it one click. What has not
changed is everything underneath: the allowlists, the workspace jail, the
absence of delete and rename, and the fact that the Python and terminal tools
are not sandboxes. What now carries more weight is the loopback binding - this
API can hand a model a shell, so it must never be reachable from anywhere but
this machine.

Overrides live in memory only. A restart returns to what the environment says,
so a switch cannot quietly become the permanent state of the system.
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import FLAGS_BY_ID, TOOL_FLAGS, Runtime
from models.manager import ModelManagerError, ModelState

from api.schemas import (
    DisabledToolOut,
    OcrBackendRequest,
    HealthOut,
    SwitchOut,
    ToggleRequest,
    ToolOut,
    ToolsOut,
)

router = APIRouter(tags=["meta"])

# The registry key of the vision backend the ocr switch starts and stops.
OCR_MODEL_KEY = "ocr"


def _ocr_running(runtime: Runtime) -> bool:
    """Whether the GLM-OCR server is up, without raising if it is not listed."""
    try:
        return runtime.manager.status(OCR_MODEL_KEY).state is ModelState.READY
    except (ModelManagerError, KeyError):
        return False


def _ocr_workable(runtime: Runtime, config) -> bool:
    """Whether OCR would actually work right now.

    The two backends need different things, and reporting the flag alone would
    show a switch that is on while every attempt to use it fails. Tesseract
    needs a binary on disk; the model needs a server answering.
    """
    if config.ocr_backend == "tesseract":
        from tools.tesseract import TesseractBackend

        return TesseractBackend(command=config.tesseract_cmd).available()
    return _ocr_running(runtime)


def _ocr_status(runtime: Runtime, config) -> tuple[bool, str]:
    """Whether the selected OCR backend can run, and what to do if not.

    The two need entirely different things - a binary on disk against a server
    answering - so the hint names the specific missing thing rather than
    saying "OCR is unavailable".
    """
    if config.ocr_backend == "tesseract":
        from tools.tesseract import TesseractBackend

        backend = TesseractBackend(command=config.tesseract_cmd)
        if backend.available():
            return True, ""
        return False, backend.missing_message()

    if _ocr_running(runtime):
        return True, ""
    return False, (
        f"The GLM-OCR server is not running on {config.ocr_url}. Start it, or "
        f"switch to Tesseract, which needs no server and no GPU."
    )


@router.post("/ocr-backend", response_model=ToolsOut)
def set_ocr_backend(
    body: OcrBackendRequest, runtime: Runtime = Depends(get_runtime)
):
    """Choose which reader ocr_image uses.

    Refused mid-turn for the same reason the tool switches are: the registry
    is built when a turn starts, and changing the tool's description underneath
    one would either do nothing or change what the model was told halfway
    through.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. The OCR backend applies from the next turn.",
        )

    if body.backend == "tesseract":
        from tools.tesseract import TesseractBackend

        backend = TesseractBackend(
            command=runtime.effective_config().tesseract_cmd
        )
        if not backend.available():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, backend.missing_message()
            )
    runtime.set_ocr_backend(body.backend)
    return _tools_snapshot(runtime)


def _tools_snapshot(runtime: Runtime) -> ToolsOut:
    config = runtime.effective_config()
    registry, disabled = runtime.registry_for(config)

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

    # A tool drops out of the disabled list the moment it is enabled, which
    # would take its explanation with it - and a switch that is ON is when the
    # warning matters most. So the reasons are harvested from a registry built
    # with everything off, which yields all of them whatever is currently on.
    _, every_reason = runtime.registry_for(
        dataclasses.replace(config, **{flag.field: False for flag in TOOL_FLAGS})
    )
    reasons = {item.category: item.reason for item in every_reason}
    overrides = runtime.overrides

    switches = []
    for flag in TOOL_FLAGS:
        # Sub-flags have no category of their own; they inherit the warning
        # from the tool they are the sharp end of.
        risk = reasons.get(flag.id.replace("_", " "), "")
        if not risk and flag.depends_on:
            risk = reasons.get(flag.depends_on.replace("_", " "), "")

        # For OCR, "on" means the tool is enabled *and* its server answers.
        # Reporting the flag alone would show a switch that is on while every
        # attempt to use it fails.
        enabled = bool(getattr(config, flag.field))
        if flag.id == OCR_MODEL_KEY:
            enabled = enabled and _ocr_workable(runtime, config)

        switches.append(
            SwitchOut(
                id=flag.id,
                label=flag.label,
                enabled=enabled,
                from_env=flag.field not in overrides
                and bool(getattr(runtime.config, flag.field)),
                depends_on=flag.depends_on,
                risk=risk,
            )
        )

    ready, hint = _ocr_status(runtime, config)
    return ToolsOut(
        tools=tools,
        disabled=[DisabledToolOut(**vars(item)) for item in disabled],
        switches=switches,
        workspace=str(config.workspace),
        ocr_backend=config.ocr_backend,
        ocr_ready=ready,
        ocr_hint=hint,
    )


@router.get("/tools", response_model=ToolsOut)
def list_tools(runtime: Runtime = Depends(get_runtime)):
    """Which tools the model is offered, and why the others are withheld."""
    return _tools_snapshot(runtime)


@router.post("/tools/{flag_id}", response_model=ToolsOut)
def set_tool(
    flag_id: str,
    body: ToggleRequest,
    runtime: Runtime = Depends(get_runtime),
):
    """Turn one tool on or off for this process.

    Refused mid-turn: the registry is built when a turn starts, so changing it
    underneath one would either do nothing or change what the model is offered
    halfway through, and neither is worth the confusion.
    """
    if flag_id not in FLAGS_BY_ID:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown tool switch {flag_id!r}. "
            f"Available: {', '.join(FLAGS_BY_ID)}.",
        )

    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Tool changes apply from the next turn, so this "
            "would not affect it anyway.",
        )

    # OCR is the one switch that owns a process as well as a flag. The tool is
    # useless without the GLM-OCR server and the server is dead weight without
    # the tool, so one toggle does both rather than leaving someone to discover
    # the second half from an error message.
    if flag_id == "ocr":
        config = runtime.effective_config()
        if config.ocr_backend == "tesseract":
            # Nothing to start: Tesseract is a binary invoked per image, not a
            # server. Refusing early is better than switching the tool on and
            # having it fail on the first attachment.
            from tools.tesseract import TesseractBackend

            backend = TesseractBackend(command=config.tesseract_cmd)
            if body.enabled and not backend.available():
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, backend.missing_message()
                )
        else:
            try:
                if body.enabled:
                    runtime.ensure_model(OCR_MODEL_KEY)
                else:
                    runtime.manager.stop(OCR_MODEL_KEY)
            except ModelManagerError as exc:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, str(exc)
                ) from None

    flag = FLAGS_BY_ID[flag_id]
    if body.enabled and flag.depends_on:
        parent = FLAGS_BY_ID[flag.depends_on]
        if not getattr(runtime.effective_config(), parent.field):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Turn on {parent.label} first: {flag.label} is its sharp end, "
                f"not a tool of its own.",
            )

    runtime.set_override(flag_id, body.enabled)
    return _tools_snapshot(runtime)


@router.get("/health", response_model=HealthOut)
def health(runtime: Runtime = Depends(get_runtime)):
    """Whether the API is up, and whether it is mid-turn.

    Says nothing about the model servers - those have their own state and are
    reported by /models, which can be slow when it reconciles processes.
    """
    # The effective workspace, not the one the environment chose: the folder
    # the tools would actually reach right now is the only useful answer, and
    # since it can be moved from the UI the two are no longer the same thing.
    return HealthOut(
        ok=True,
        busy=runtime.queue.busy(),
        queue_depth=runtime.queue.depth(),
        workspace=str(runtime.workspace),
        db_path=str(runtime.config.db_path),
    )
