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
import threading

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import FLAGS_BY_ID, TOOL_FLAGS, Runtime
from models.manager import ModelManagerError, ModelState, memory_status

from api.schemas import (
    DisabledToolOut,
    OcrBackendRequest,
    HealthOut,
    McpOut,
    McpServerOut,
    ResourcesOut,
    ShutdownOut,
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


@router.get("/mcp", response_model=McpOut)
def list_mcp(runtime: Runtime = Depends(get_runtime)):
    """The configured servers and how many tools each has cached."""
    return _mcp_snapshot(runtime)


@router.post("/mcp/refresh", response_model=McpOut)
def refresh_mcp(runtime: Runtime = Depends(get_runtime)):
    """Ask every server what it offers, and remember.

    The expensive operation, which is why it is a request rather than
    something that happens on its own: every server is started, questioned
    and stopped. Refused mid-turn, because the registry a running turn is
    using was built from the cache this replaces.
    """
    if runtime.queue.busy() or runtime.queue.depth():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running or waiting. Refreshing rebuilds the tool "
            "roster underneath it - let it finish first.",
        )
    report = runtime.mcp.refresh()
    return _mcp_snapshot(runtime, errors=report.get("errors", {}))


def _mcp_snapshot(runtime: Runtime, errors: dict | None = None) -> McpOut:
    config = runtime.effective_config()
    counts = runtime.mcp._read_cache()
    errors = errors or {}
    return McpOut(
        servers=[
            McpServerOut(
                name=spec.name,
                command=" ".join([spec.command, *spec.args])[:200],
                trusted=spec.trusted,
                tools=len(counts.get(spec.name, [])),
                error=errors.get(spec.name, ""),
            )
            for spec in runtime.mcp.servers
        ],
        configured=config.mcp_config.is_file(),
        config_path=str(config.mcp_config),
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


# --- resources and shutdown -------------------------------------------------

# Where Vite serves the UI in development, as start.bat launches it. In a
# --build-web install the API serves the UI itself and nothing is on this
# port, which is fine: the check finds nothing and the response says so.
UI_DEV_PORT = 5173

# Long enough for the response to leave before the process it came from
# goes; short enough that the person who pressed the button sees it happen.
EXIT_DELAY_SECONDS = 0.5


@router.get("/resources", response_model=ResourcesOut)
def resources(runtime: Runtime = Depends(get_runtime)):
    """RAM and the resident model, for the header.

    Takes no lock and reconciles nothing, on purpose: this is polled, and it
    has to keep answering during a model load - the one time everything
    behind the manager's lock is waiting.
    """
    memory = memory_status()
    resident = runtime.manager.resident()
    return ResourcesOut(
        total_mb=memory.total_mb if memory else None,
        available_mb=memory.available_mb if memory else None,
        load_percent=memory.load_percent if memory else None,
        resident_key=resident.spec.key if resident else None,
        resident_label=resident.spec.label if resident else None,
        resident_state=resident.state.value if resident else None,
    )


@router.post("/shutdown", response_model=ShutdownOut)
def shutdown(runtime: Runtime = Depends(get_runtime)):
    """Unload every model, stop the UI dev server, and exit.

    Refused while a turn is running, like load and unload are: the person
    can stop the turn first, and a shutdown that discarded two minutes of
    generation because a button was under the cursor would be the wrong
    kind of safe.

    The order matters. Models are unloaded here, synchronously, so the
    response can name them. The UI server and this process go on a short
    timer: in development the browser reaches this API *through* Vite, so
    stopping Vite before answering would mean never answering. The exit is
    a SIGINT to ourselves, so uvicorn runs the same lifespan shutdown that
    Ctrl+C does - which unloads the embedding worker and anything else that
    was started along the way.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Stop it first, then shut down.",
        )

    stopped = runtime.manager.stop_all()
    ui_pid = runtime.manager.port_holder(UI_DEV_PORT, "node")

    def finish() -> None:
        if ui_pid is not None:
            runtime.manager.kill_pid(ui_pid)
        runtime.exit_process()

    threading.Timer(EXIT_DELAY_SECONDS, finish).start()

    if ui_pid is not None:
        note = "Models unloaded. The API and the web server are stopping."
    else:
        note = (
            "Models unloaded. The API is stopping. No dev server was found on "
            f"port {UI_DEV_PORT}, so nothing else needed stopping."
        )
    return ShutdownOut(stopped_models=stopped, stopped_ui=ui_pid is not None, note=note)
