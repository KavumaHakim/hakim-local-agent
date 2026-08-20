"""The FastAPI application.

Run it with a single worker, bound to loopback:

    .venv\\Scripts\\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Both of those are load-bearing.

**One worker.** The ModelManager owns `llama-server.exe` child processes. Two
workers would mean two managers, each believing it owns the same ports, each
stopping the other's model.

**Loopback only.** With the tool flags on, this API can write files, run
allowlisted commands and execute Python. That is fine as a local tool and
unacceptable on a network interface, so the host is stated explicitly rather
than left to a default that someone could reasonably "fix" later.

There is deliberately **no CORS middleware**. In development Vite proxies
`/api` so the browser sees one origin; in production this app serves the built
front end itself, which is also one origin. Adding permissive CORS would let
any page you happen to have open drive your agent, and nothing here needs it.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import chat, conversations, meta, models
from api.runtime import Runtime

# The built React app, when there is one. Absent during development.
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

# How often to look for models to unload. Well under the idle timeout so the
# sweep is not itself the thing that makes unloading late.
SWEEP_SECONDS = 30.0


def _sweeper(runtime: Runtime, stop: threading.Event) -> None:
    """Unload models left idle, the way Streamlit's reruns used to.

    `unload_idle` was previously called on every rerun, so it happened
    naturally whenever anyone touched the page. Nothing reruns here, so
    without this the idle timeout in models.json would silently never fire.

    The busy check matters more than it looks: the idle timeout is 300 s and a
    single turn has been measured at 285.7 s. `ensure()` stamps `last_used` at
    the *start* of a turn, so a slow turn can cross the timeout while it is
    still generating, and an unguarded sweep would unload the model mid-answer.
    """
    while not stop.wait(SWEEP_SECONDS):
        if runtime.queue.busy():
            continue
        try:
            runtime.manager.unload_idle()
        except Exception:  # noqa: BLE001 - a sweep failure must not kill the thread
            pass


def _lifespan_for(supplied: Runtime | None):
    """Build the lifespan handler, optionally around a supplied runtime.

    Tests pass their own runtime - temporary database, scripted client - and
    still exercise the real startup and shutdown path rather than a second one
    written to be convenient.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = supplied if supplied is not None else Runtime()
        app.state.runtime = runtime
        runtime.queue.start()

        stop = threading.Event()
        sweeper = threading.Thread(
            target=_sweeper, args=(runtime, stop), name="model-sweeper", daemon=True
        )
        sweeper.start()

        try:
            yield
        finally:
            stop.set()
            runtime.queue.stop()
            # Without this, every restart leaks a llama-server holding
            # gigabytes. It is also why `--reload` is a bad idea here: the
            # reloader kills the worker in a way that does not always reach
            # this.
            runtime.manager.stop_all()

    return lifespan


def create_app(runtime: Runtime | None = None) -> FastAPI:
    """Build the application. Pass a runtime to supply your own."""
    app = FastAPI(
        title="Hakim AI System",
        description="Local agent over llama.cpp. Nothing leaves this machine.",
        version="2.0.0",
        lifespan=_lifespan_for(runtime),
    )

    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(meta.router, prefix="/api")

    _mount_web(app)
    return app


def _mount_web(app: FastAPI) -> None:
    """Serve the built front end, when there is one.

    Production only. In development Vite serves the app and proxies /api back
    here, so this is simply not mounted.
    """
    if not WEB_DIST.is_dir():
        return

    assets = WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        """Serve a real file if it exists, otherwise the app's entry point.

        Registered after the routers so every /api route wins first. Unknown
        paths fall back to index.html rather than 404 because they belong to
        the front-end router, not to this one.
        """
        candidate = (WEB_DIST / path).resolve()
        if path and candidate.is_file() and WEB_DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


app = create_app()
