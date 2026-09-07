"""Adding, switching on and removing MCP servers.

The endpoints behind the Servers pane. Until now the only way to add one was
to edit `mcp.json` by hand and restart, which is fine for the person who wrote
this and no use to anyone else.

**These endpoints write a file that says which programs the agent may start.**
That is the most consequential thing this API does, and it is worth being
exact about why it is nonetheless reasonable:

* It is not an escalation. Editing `mcp.json` in a text editor already does
  this, and anyone who can reach this API can already switch on the terminal
  tool. What changes is that the door has a label on it.
* **The model cannot reach it.** `tools/http_tool.py` refuses writes to the
  agent's own API outright rather than asking, because the approval prompt
  shows a method and a url and never a body - so "POST /api/mcp/servers" is a
  question nobody could answer. The only way through here is a person in the
  interface.
* Nothing is written except one entry in `mcpServers`. `edit_config` reads the
  file, mutates that one key and writes the whole document back, so an `env`
  block holding an API key survives a toggle. That file is git-ignored
  precisely because it holds those, and there is no second copy.

Switching a catalogue entry on is the one place a command line is composed
rather than typed, and it is composed from `tools/mcp_catalog.py` - a fixed
table in the repository, never from anything in the request. The request names
an entry; it does not describe one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    McpCatalogItem,
    McpEnvNeed,
    McpOut,
    McpServerIn,
    McpServerOut,
    McpServerUpdate,
)
from tools import mcp_catalog
from tools.mcp_client import (
    McpConfigError,
    McpError,
    check_url,
    check_name,
    edit_config,
    load_servers,
)

router = APIRouter(tags=["mcp"])


@router.get("/mcp", response_model=McpOut)
def list_mcp(runtime: Runtime = Depends(get_runtime)):
    """The configured servers, and the catalogue of ones on offer."""
    return _snapshot(runtime)


@router.post("/mcp/refresh", response_model=McpOut)
def refresh_mcp(runtime: Runtime = Depends(get_runtime)):
    """Ask every server what it offers, and remember.

    The expensive operation, which is why it is a request rather than
    something that happens on its own: every server is started, questioned
    and stopped. Refused mid-turn, because the registry a running turn is
    using was built from the cache this replaces.
    """
    _refuse_mid_turn(runtime)
    report = runtime.mcp.refresh()
    return _snapshot(runtime, errors=report.get("errors", {}))


@router.post("/mcp/servers", response_model=McpOut)
def add_server(body: McpServerIn, runtime: Runtime = Depends(get_runtime)):
    """Add a server, either from the catalogue or described outright.

    A catalogue entry is named, not described: the command line comes from the
    table in the repository, so switching on "fetch" cannot be talked into
    running something else by the request that asks for it.
    """
    _refuse_mid_turn(runtime)
    config = runtime.effective_config()

    if body.catalog:
        entry = mcp_catalog.entry(body.catalog)
        if entry is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"There is no catalogue entry called {body.catalog!r}.",
            )
        name = entry.name
        written = entry.resolve(config.workspace)
    else:
        try:
            name = check_name(body.name or "")
        except McpConfigError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
        if body.url and body.command:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "A server is reached one way or the other: a command to start "
                "it here, or a url to reach it over HTTP. Not both.",
            )
        if body.url:
            try:
                written = {"url": check_url(body.url, name)}
            except McpError as exc:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, str(exc)
                ) from None
            headers = {
                str(k): str(v)
                for k, v in (body.headers or {}).items()
                if str(k).strip() and str(v).strip()
            }
            if headers:
                written["headers"] = headers
        elif (body.command or "").strip():
            written = {
                "command": body.command.strip(),
                "args": [a for a in (body.args or []) if str(a).strip()],
            }
        else:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "A server needs either a command to start it, e.g. 'npx', or "
                "a url to reach it over HTTP.",
            )

    # Blank values are dropped rather than written: an empty string is not a
    # credential, and writing one would make "set" true for something unusable.
    supplied = {
        str(k): str(v)
        for k, v in (body.env or {}).items()
        if str(k).strip() and str(v).strip()
    }
    if supplied:
        written["env"] = supplied

    existing = {s.name for s in load_servers(config.mcp_config)}
    if name in existing and not body.replace:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A server called {name!r} is already configured. Remove it "
            f"first, or send replace=true.",
        )

    if body.trusted:
        written["trusted"] = True

    def change(servers: dict) -> None:
        # Preserved rather than dropped: someone may have added an env block
        # by hand, and replacing the entry should not silently delete their
        # API key.
        previous = servers.get(name)
        if isinstance(previous, dict) and isinstance(previous.get("env"), dict):
            written.setdefault("env", previous["env"])
        servers[name] = written

    _apply(config.mcp_config, change)
    return _snapshot(runtime)


@router.patch("/mcp/servers/{name}", response_model=McpOut)
def update_server(
    name: str, body: McpServerUpdate, runtime: Runtime = Depends(get_runtime)
):
    """Switch a server on or off, or trust it.

    Off writes `"enabled": false` rather than deleting the entry, so switching
    it back on does not mean typing the command line again.
    """
    _refuse_mid_turn(runtime)
    config = runtime.effective_config()
    known = {s.name for s in load_servers(config.mcp_config)}
    if name not in known:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No server called {name!r} is configured."
        )

    def change(servers: dict) -> None:
        entry = servers.get(name)
        if not isinstance(entry, dict):
            return
        if body.enabled is not None:
            entry["enabled"] = bool(body.enabled)
        if body.trusted is not None:
            entry["trusted"] = bool(body.trusted)

    _apply(config.mcp_config, change)
    return _snapshot(runtime)


@router.delete("/mcp/servers/{name}", response_model=McpOut)
def remove_server(name: str, runtime: Runtime = Depends(get_runtime)):
    """Remove a server's entry entirely, env block and all."""
    _refuse_mid_turn(runtime)
    config = runtime.effective_config()

    def change(servers: dict) -> None:
        servers.pop(name, None)

    _apply(config.mcp_config, change)
    return _snapshot(runtime)


# --- shared ---


def _refuse_mid_turn(runtime: Runtime) -> None:
    """Every write here rebuilds the roster a running turn is holding."""
    if runtime.queue.busy() or runtime.queue.depth():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running or waiting. Changing the servers rebuilds the "
            "tool roster underneath it - let it finish first.",
        )


def _apply(path, change) -> None:
    try:
        edit_config(path, change)
    except McpConfigError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, str(exc)
        ) from None


def _snapshot(runtime: Runtime, errors: dict | None = None) -> McpOut:
    """What the config says now, and what the cache holds for it.

    Read from the file rather than from `runtime.mcp`, for two reasons. The
    manager drops disabled servers, so one switched off would vanish from the
    panel rather than appear switched off. And the manager's list is from
    startup: someone who has just added a server should see it before they
    press Refresh, not after. Refreshing is what reloads the manager itself.
    """
    config = runtime.effective_config()
    counts = runtime.mcp.cached_tools()
    errors = errors or {}
    configured = {s.name for s in load_servers(config.mcp_config)}

    return McpOut(
        servers=[
            McpServerOut(
                name=spec.name,
                command=spec.display[:200],
                transport="http" if spec.remote else "stdio",
                trusted=spec.trusted,
                enabled=spec.enabled,
                tools=len(counts.get(spec.name, [])),
                error=errors.get(spec.name, ""),
                from_catalog=mcp_catalog.entry(spec.name) is not None,
                # Names only. The values stay in the file.
                env_set=sorted(k for k, v in spec.env.items() if v),
            )
            for spec in sorted(load_servers(config.mcp_config), key=lambda s: s.name)
        ],
        catalog=[
            McpCatalogItem(
                name=entry.name,
                title=entry.title,
                summary=entry.summary,
                package=entry.package,
                runtime=entry.runtime,
                caution=entry.caution,
                unmaintained=entry.unmaintained,
                needs=[
                    McpEnvNeed(
                        variable=need.variable, label=need.label, hint=need.hint
                    )
                    for need in entry.needs
                ],
                added=entry.name in configured,
            )
            for entry in mcp_catalog.CATALOG
        ],
        configured=config.mcp_config.is_file(),
        config_path=str(config.mcp_config),
    )
