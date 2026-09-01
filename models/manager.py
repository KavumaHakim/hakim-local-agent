"""Starts, stops and switches llama-server processes.

One model at a time by default, because 8 GB of RAM will not hold two. The
agent asks for a model by logical key ("fast", "reasoning") and gets back a
base URL; which process is alive, and on what port, is this module's problem.

Two deliberate choices worth knowing about:

* It runs in-process, not as a separate service. A local single-user agent
  does not need another daemon and another port to keep alive; every caller
  already runs Python. Wrapping this class in an HTTP API later is easy, and
  the class is written so that stays true.

* If a healthy server is already listening on a model's port, the manager
  adopts it instead of starting a rival. That keeps a CLI and a Streamlit
  session from fighting over the same port without needing a cross-process
  lock file, and it means a server you started by hand still works.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import signal
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from models.preferences import ModelPreferences

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PROJECT_ROOT / "models.json"
# Where dropped-in GGUF files are looked for. Not "models/": that is the
# Python package this module lives in, and a data directory of the same
# name would shadow it on the import path.
DEFAULT_MODELS_DIRNAME = "weights"
# Generated state lives under data/, as everything generated here does.
DEFAULT_PREFERENCES_DIR = PROJECT_ROOT / "data"

# Below this there is not enough room for the process itself, whatever the
# model. Above it, a shortfall against a model's own figure only means paging,
# so it warns rather than refuses - see ModelManager._check_ram.
HARD_FLOOR_MB = 250


class ModelState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class ModelManagerError(Exception):
    """A model could not be started, stopped or switched to."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: Path
    port: int
    context: int
    threads: int
    min_free_mb: int
    description: str = ""

    # --- what this model is for ---
    #
    # "chat" drives the agent loop and joins the one-at-a-time rotation.
    # "ocr" is a llama-server too, but a vision backend that has to run
    # *alongside* a chat model rather than instead of one - and GLM-OCR cannot
    # call tools at all, so it could never drive the loop. It therefore sits
    # outside the rotation in both directions: it never evicts a chat model and
    # is never evicted by one.
    role: str = "chat"
    # Vision projector, for a model that needs one. GLM-OCR's language half has
    # no vision tensors, so without this it loads and then cannot see.
    mmproj: Path | None = None

    # --- remote models ---
    #
    # "local" means a llama-server this manager starts and owns. Anything else
    # is a hosted API: no process, no port, no RAM threshold, and nothing to
    # load. They are in the same registry because everything above the client
    # - the agent loop, the tools, the history - is identical either way.
    provider: str = "local"
    # The provider's own name for the model, which is not our key.
    model: str = ""
    # Name of the environment variable holding the key. The key itself is never
    # stored on the spec, so it cannot be serialised out to the UI by accident.
    api_key_env: str = ""
    base_url: str = ""

    @property
    def remote(self) -> bool:
        return self.provider != "local"

    @property
    def url(self) -> str:
        if self.remote:
            return self.base_url
        return f"http://127.0.0.1:{self.port}"

    @property
    def has_key(self) -> bool:
        """Whether the API key for this provider is present."""
        if not self.api_key_env:
            return False
        return bool(os.environ.get(self.api_key_env, "").strip())

    @property
    def available(self) -> bool:
        """Whether this model could be used at all.

        For a local model that means the weights are on disk; for a remote one,
        that a key exists. Neither says anything about whether it will work -
        a remote model also needs the internet, which is checked separately
        because it changes minute to minute.

        A model with an mmproj needs both files. Checking only the language
        half is the trap here: it loads happily and then reports no vision,
        which looks like a broken tool rather than a missing file.
        """
        if self.remote:
            return self.has_key
        if not self.path.is_file():
            return False
        if self.mmproj is not None and not self.mmproj.is_file():
            return False
        return True


@dataclass
class ModelStatus:
    spec: ModelSpec
    state: ModelState = ModelState.STOPPED
    pid: int | None = None
    last_used: float = 0.0
    error: str = ""
    # Set when the model started with less RAM than it would like.
    warning: str = ""
    # True when the process was already running and we merely attached to it.
    adopted: bool = False


def available_ram_mb() -> int | None:
    """Free physical RAM in MB, or None if it cannot be determined.

    The platform call on each side rather than a psutil dependency for one
    number: Windows has GlobalMemoryStatusEx, Linux has /proc/meminfo.

    "Available" rather than "free" on both. Linux's MemFree excludes the page
    cache, which the kernel hands back on demand, so a machine with plenty of
    room can report almost none - and this number decides whether a model is
    allowed to start.
    """
    if os.name == "nt":
        return _windows_ram_mb()
    return _proc_meminfo_ram_mb()


def _windows_ram_mb() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys // (1024 * 1024))
    except (AttributeError, OSError):
        return None


def _proc_meminfo_ram_mb() -> int | None:
    """MemAvailable from /proc/meminfo, in MB. None where there is no procfs."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            return parse_meminfo(handle.read())
    except OSError:
        return None


def parse_meminfo(text: str) -> int | None:
    """MemAvailable, in MB, from the contents of /proc/meminfo.

    Split from the file read so the Linux path can be tested from any machine.
    Untested platform code is a promise rather than a feature.
    """
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            try:
                # "MemAvailable:    1908736 kB"
                return int(line.split()[1]) // 1024
            except (ValueError, IndexError):
                return None
    return None


def parse_ss_pid(output: str) -> int | None:
    """The pid out of `ss -lntp` output, or None.

    A line looks like:

        LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:* users:(("llama-server",pid=8123,fd=7))

    Split out from the call so the parsing is testable without Linux.
    """
    marker = "pid="
    for line in output.splitlines():
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        digits = ""
        for character in tail:
            if character.isdigit():
                digits += character
            else:
                break
        if digits:
            try:
                return int(digits)
            except ValueError:
                continue
    return None


def load_registry(
    path: Path | None = None,
    *,
    preferences_dir: Path | None = None,
    discover_models: bool = True,
) -> dict[str, Any]:
    """Build the model catalogue from three layers, in increasing priority.

      1. `models.json` - curated, hand-tuned, in version control. Optional:
         a fresh install with no registry at all still works.
      2. the models folder - every .gguf that no curated entry already claims,
         sized from its GGUF header by `models.discovery`.
      3. `data/models.local.json` - the user's own choices from the settings
         panel, which win over both.

    Layering rather than replacing is what makes discovery safe to add. A
    measured `min_free_mb` in models.json is better than anything inferred
    here, so a curated entry is never overwritten by a scan of the same file.
    """
    path = path or DEFAULT_REGISTRY
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # No registry is a supported state, not an error: drop a model in the
        # folder and the app finds it. Everything below has a default.
        raw = {}
    except ValueError as exc:
        raise ModelManagerError(f"{path} is not valid JSON: {exc}") from None

    # Relative paths resolve against models.json's own directory, not the
    # working directory, so the project folder can be renamed or moved and
    # nothing has to be edited. An absolute path still wins, for a weights
    # directory kept on another drive.
    here = path.parent

    def anchor(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (here / candidate)

    def find_server(value: str) -> Path:
        """Where llama-server is, preferring the registry but not insisting.

        A fresh clone has whatever path the last person committed, which is
        almost certainly wrong on someone else's machine. So a configured path
        that does not exist falls back to PATH - if `llama-server` is
        installed the way its own instructions suggest, the checkout works
        without editing anything.

        A configured path that DOES exist always wins: someone who wrote one
        down meant it, including when a different build is also on PATH.
        """
        if value:
            configured = anchor(value)
            if configured.is_file():
                return configured

        # What the setup script downloads, before PATH: someone who ran it
        # meant to use that build, and a stale llama-server left on PATH from
        # some other project should not quietly take precedence over it.
        for name in ("llama-server.exe", "llama-server"):
            vendored = PROJECT_ROOT / "vendor" / "llama"
            for found in sorted(vendored.rglob(name)) if vendored.is_dir() else []:
                if found.is_file():
                    return found

        for name in ("llama-server", "llama-server.exe"):
            found = shutil.which(name)
            if found:
                return Path(found)
        # Nothing found. Return the configured path anyway so the error names
        # what was actually looked for rather than something invented here.
        return anchor(value) if value else Path("llama-server")

    # Defaults to weights/ rather than the registry's own folder. `models/`
    # would be the obvious name but it is the Python package, and a data
    # directory sharing it would shadow the import.
    models_dir = anchor(raw.get("models_dir", "") or DEFAULT_MODELS_DIRNAME)
    specs: dict[str, ModelSpec] = {}
    for entry in raw.get("models", []):
        try:
            key = entry["key"]
            provider = entry.get("provider", "local")
            if provider == "local":
                mmproj = entry.get("mmproj")
                specs[key] = ModelSpec(
                    key=key,
                    label=entry.get("label", key),
                    path=(models_dir / entry["file"]).resolve(),
                    port=int(entry["port"]),
                    context=int(entry.get("context", 4096)),
                    threads=int(entry.get("threads", 4)),
                    min_free_mb=int(entry.get("min_free_mb", 0)),
                    description=entry.get("description", ""),
                    role=entry.get("role", "chat"),
                    mmproj=(models_dir / mmproj).resolve() if mmproj else None,
                )
            else:
                # A hosted model has no file, no port and no RAM threshold, so
                # those fields are placeholders and nothing may read them: the
                # `remote` property is what everything branches on.
                specs[key] = ModelSpec(
                    key=key,
                    label=entry.get("label", key),
                    path=Path(),
                    port=0,
                    context=int(entry.get("context", 0)),
                    threads=0,
                    min_free_mb=0,
                    description=entry.get("description", ""),
                    provider=provider,
                    model=entry["model"],
                    api_key_env=entry.get("api_key_env", ""),
                    base_url=entry.get("base_url", "").rstrip("/"),
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelManagerError(f"Bad model entry in {path}: {exc}") from None

    curated_keys = set(specs)

    # --- layer 2: whatever is in the models folder ---
    discovered: list[Any] = []
    if discover_models:
        from models.discovery import discover

        discovered = discover(
            models_dir,
            taken_keys=curated_keys,
            taken_ports=[s.port for s in specs.values() if not s.remote],
            known_files=[
                name
                for spec in specs.values()
                if not spec.remote
                for name in (
                    spec.path.name,
                    spec.mmproj.name if spec.mmproj else "",
                )
                if name
            ],
        )
        for item in discovered:
            specs[item.key] = ModelSpec(
                key=item.key,
                label=item.label,
                path=item.file.resolve(),
                port=item.port,
                context=item.context,
                threads=item.threads,
                min_free_mb=item.min_free_mb,
                description=item.description,
                role=item.role,
                mmproj=item.mmproj.resolve() if item.mmproj else None,
            )

    # --- layer 3: the user's own settings ---
    # Preferences live beside the registry that named them, not at a fixed
    # path. In production that is the project's data/ directory; in a test that
    # passes a temporary registry it is a temporary directory, which is what
    # keeps the suite from reading and rewriting the real user's settings.
    if preferences_dir is None:
        preferences_dir = (
            DEFAULT_PREFERENCES_DIR if path == DEFAULT_REGISTRY else path.parent
        )
    preferences = ModelPreferences.load(preferences_dir)

    # A remembered path wins over everything the search would guess at. It is
    # only ever there because somebody typed it in, and a person who has said
    # where their llama-server is should not be second-guessed by a stale one
    # on PATH. A path that has since gone is ignored rather than honoured, so
    # deleting the binary falls back to the search instead of failing.
    chosen_server = find_server(raw.get("server_exe", ""))
    if preferences.server_exe:
        remembered = Path(preferences.server_exe).expanduser()
        if remembered.is_file():
            chosen_server = remembered
    for key, values in preferences.overrides.items():
        spec = specs.get(key)
        if spec is None:
            continue
        specs[key] = dataclasses.replace(
            spec,
            **{
                name: value
                for name, value in values.items()
                if name in ("label", "context", "threads", "min_free_mb", "description")
            },
        )

    # Hidden models stay in the catalogue but are not offered. Removing them
    # outright would make un-hiding impossible from the UI, and would silently
    # break a `primary` that points at one.
    offered = {
        key: spec for key, spec in specs.items() if not preferences.is_hidden(key)
    }

    # Remote models all report port 0, so only the local ones are checked.
    ports = [spec.port for spec in specs.values() if not spec.remote]
    if len(set(ports)) != len(ports):
        raise ModelManagerError(
            "Two models share a port. Give one of them a different `port` in "
            "models.json."
        )

    router = raw.get("router") or {}
    chat_keys = [
        key for key, spec in offered.items() if spec.role == "chat" and spec.available
    ]

    # The primary, in priority order: what the user chose, what the registry
    # ships, then the first usable chat model. The last is what makes a fresh
    # install with one dropped-in model work without being asked anything.
    fallback = ""
    for candidate in (preferences.primary, raw.get("default", "")):
        if candidate and candidate in offered:
            fallback = candidate
            break
    if not fallback:
        fallback = chat_keys[0] if chat_keys else (next(iter(specs), ""))

    if not specs:
        raise ModelManagerError(
            f"No models. Put a .gguf file in {models_dir}, or add an entry to "
            f"{path}."
        )

    # True when there is a real choice to make and nobody has made it. One
    # model is not a choice, so a single-model install is never interrupted.
    setup_required = not preferences.setup_complete and len(chat_keys) > 1

    def routed(name: str) -> str:
        chosen = getattr(preferences, f"router_{name}", "") or router.get(name, "")
        return chosen if chosen in offered else fallback

    return {
        "server_exe": chosen_server,
        "specs": specs,
        "offered": offered,
        "default": fallback,
        "max_active": int(raw.get("max_active", 1)),
        "idle_timeout": float(raw.get("idle_timeout_seconds", 0)),
        # Which model the router treats as cheap and which as capable.
        "router_fast": routed("fast"),
        "router_strong": routed("strong"),
        "models_dir": models_dir,
        "preferences": preferences,
        "discovered": [item.key for item in discovered],
        "curated": sorted(curated_keys),
        "setup_required": setup_required,
        "registry_path": path,
    }


class ModelManager:
    """Owns the llama-server processes for every registered model."""

    def __init__(
        self,
        registry_path: Path | None = None,
        *,
        start_timeout: float = 600.0,
        stop_timeout: float = 10.0,
        preferences_dir: Path | None = None,
    ) -> None:
        self._registry_path = registry_path
        self._preferences_dir = preferences_dir
        registry = load_registry(registry_path, preferences_dir=preferences_dir)
        self._apply(registry)
        self._start_timeout = start_timeout
        self._stop_timeout = stop_timeout

        self._statuses = {
            key: ModelStatus(spec=spec) for key, spec in self._specs.items()
        }
        self._processes: dict[str, subprocess.Popen] = {}
        # Every transition is serialised: two threads must never race to start
        # a second model while one is coming up.
        self._lock = threading.RLock()
        self._session = requests.Session()

    def _apply(self, registry: dict[str, Any]) -> None:
        """Adopt a freshly loaded registry.

        Shared by construction and `rescan`, so the two cannot drift: a field
        added to the registry has exactly one place to be picked up.
        """
        self._server_exe: Path = registry["server_exe"]
        self._specs: dict[str, ModelSpec] = registry["specs"]
        self._offered: dict[str, ModelSpec] = registry["offered"]
        self._default: str = registry["default"]
        self._max_active: int = registry["max_active"]
        self._idle_timeout: float = registry["idle_timeout"]
        self.router_fast: str = registry["router_fast"]
        self.router_strong: str = registry["router_strong"]
        self._models_dir: Path = registry["models_dir"]
        self._preferences: ModelPreferences = registry["preferences"]
        self._discovered: list[str] = registry["discovered"]
        self._curated: list[str] = registry["curated"]
        self._setup_required: bool = registry["setup_required"]

    # --- the models folder ---

    @property
    def models_dir(self) -> Path:
        """Where dropped-in GGUF files are looked for."""
        return self._models_dir

    @property
    def setup_required(self) -> bool:
        """Whether a primary model still has to be chosen.

        False when there is only one candidate: a single-model install has no
        choice to offer and should not be interrupted to make one.
        """
        return self._setup_required

    def is_discovered(self, key: str) -> bool:
        """Whether this model was found on disk rather than declared."""
        return key in self._discovered

    def is_hidden(self, key: str) -> bool:
        return self._preferences.is_hidden(key)

    def rescan(self) -> list[str]:
        """Re-read the models folder and the preferences file.

        Returns the keys that are new. Running processes survive: keys are
        derived from the filename and are stable, so a model loaded before a
        rescan is still loaded, and still addressable, after one.
        """
        with self._lock:
            before = set(self._specs)
            registry = load_registry(
                self._registry_path, preferences_dir=self._preferences_dir
            )
            self._apply(registry)

            # Carry over the status of anything still present, so a rescan does
            # not forget which model is running.
            statuses = {}
            for key, spec in self._specs.items():
                existing = self._statuses.get(key)
                if existing is not None:
                    existing.spec = spec
                    statuses[key] = existing
                else:
                    statuses[key] = ModelStatus(spec=spec)
            self._statuses = statuses

            # A model whose file was deleted while it was running would leave
            # an orphan process that nothing can address any more.
            for key in list(self._processes):
                if key not in self._specs:
                    self.stop(key)
            return sorted(set(self._specs) - before)

    # --- user settings ---

    @property
    def preferences(self) -> ModelPreferences:
        return self._preferences

    def set_primary(self, key: str) -> None:
        """Choose the model everything defaults to, and remember it."""
        spec = self.get_spec(key)
        if spec.role != "chat":
            raise ModelManagerError(
                f"{spec.label} is a {spec.role} backend, not a chat model, so "
                f"it cannot be the primary."
            )
        with self._lock:
            self._preferences.set_primary(key)
            self._preferences.save()
            self._default = key
            self._setup_required = False
            if self.router_fast not in self._offered:
                self.router_fast = key

    def set_router(self, *, fast: str = "", strong: str = "") -> None:
        """Point the auto-router's cheap and capable ends at chosen models."""
        for candidate in (fast, strong):
            if candidate:
                self.get_spec(candidate)
        with self._lock:
            self._preferences.set_router(fast=fast, strong=strong)
            self._preferences.save()
            if fast:
                self.router_fast = fast
            if strong:
                self.router_strong = strong

    def set_override(self, key: str, values: dict[str, Any]) -> dict[str, Any]:
        """Retune one model from the settings panel, and persist it.

        Takes effect on the model's next start: llama-server is given its
        context and thread count on the command line, so a running process
        keeps the values it was started with.
        """
        self.get_spec(key)
        with self._lock:
            applied = self._preferences.override(key, values)
            if applied:
                self._preferences.save()
        if applied:
            self.rescan()
        return applied

    def clear_override(self, key: str) -> bool:
        """Put a model back to its registry or discovered values."""
        with self._lock:
            cleared = self._preferences.clear_override(key)
            if cleared:
                self._preferences.save()
        if cleared:
            self.rescan()
        return cleared

    def set_hidden(self, key: str, hidden: bool) -> None:
        """Stop offering a model, without deleting its file."""
        self.get_spec(key)
        if hidden and key == self._default:
            raise ModelManagerError(
                "That is the primary model. Choose a different primary first."
            )
        with self._lock:
            self._preferences.hide(key, hidden)
            self._preferences.save()
        self.rescan()

    # --- introspection ---

    @property
    def default_key(self) -> str:
        return self._default

    @property
    def max_active(self) -> int:
        """How many models may be resident at once. 1 on this machine."""
        return self._max_active

    @property
    def idle_timeout(self) -> float:
        """Seconds of disuse before a model is unloaded. 0 disables it."""
        return self._idle_timeout

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_spec(self, key: str) -> ModelSpec:
        spec = self._specs.get(key)
        if spec is None:
            raise ModelManagerError(
                f"Unknown model {key!r}. Available: {', '.join(self._specs)}."
            )
        return spec

    def status(self, key: str) -> ModelStatus:
        self.refresh()
        return self._statuses[self.get_spec(key).key]

    def statuses(self) -> list[ModelStatus]:
        self.refresh()
        return [self._statuses[key] for key in self._specs]

    def active_key(self) -> str | None:
        """The key of the local model currently resident, if any.

        Remote models are skipped even though they report READY: this answers
        "what is holding RAM right now", and a hosted model holds none. Without
        the filter, adding a cloud model to the registry would make it look
        like something was always loaded.
        """
        self.refresh()
        for key, status in self._statuses.items():
            if status.spec.remote or status.spec.role != "chat":
                continue
            if status.state is ModelState.READY:
                return key
        return None

    def refresh(self) -> None:
        """Reconcile recorded state with reality.

        A process can die on its own, and a server can appear because someone
        started it by hand; both show up here rather than as a confusing
        failure later.
        """
        with self._lock:
            for key, status in self._statuses.items():
                if status.spec.remote:
                    # Nothing to reconcile: there is no process and no port.
                    # A hosted model is usable when its key exists; whether the
                    # network is up is a separate question, checked where it is
                    # cheap to cache rather than on every status read.
                    status.state = (
                        ModelState.READY
                        if status.spec.available
                        else ModelState.STOPPED
                    )
                    status.error = (
                        ""
                        if status.spec.available
                        else f"{status.spec.api_key_env} is not set."
                    )
                    continue

                process = self._processes.get(key)
                if process is not None and process.poll() is not None:
                    # Ours, and it exited.
                    self._processes.pop(key, None)
                    status.pid = None
                    if status.state is not ModelState.STOPPING:
                        status.state = ModelState.FAILED
                        status.error = (
                            f"llama-server exited with code {process.returncode}."
                        )
                    else:
                        status.state = ModelState.STOPPED
                    continue

                if status.state in (ModelState.STARTING, ModelState.STOPPING):
                    continue

                healthy = self._healthy(status.spec.port)
                if healthy and status.state is not ModelState.READY:
                    status.state = ModelState.READY
                    status.adopted = process is None
                    status.error = ""
                elif not healthy and status.state is ModelState.READY:
                    status.state = ModelState.STOPPED
                    status.pid = None
                    status.adopted = False

    # --- lifecycle ---

    def ensure(self, key: str | None = None) -> str:
        """Make `key` the active model and return its base URL.

        Starts it if needed, stopping other models first when only one may run
        at a time. Safe to call before every request: if the model is already
        READY this is just a health check.
        """
        key = key or self._default
        spec = self.get_spec(key)

        if spec.remote:
            # Nothing to start, and deliberately nothing to stop either: a
            # hosted model uses no RAM, so it does not join the one-at-a-time
            # rotation. The local model stays resident, which means switching
            # back to it later costs nothing instead of another cold load.
            if not spec.available:
                raise ModelManagerError(
                    f"{spec.label}: {spec.api_key_env} is not set. Put it in "
                    f".env or export it, then restart the API."
                )
            with self._lock:
                self._statuses[key].state = ModelState.READY
                self._statuses[key].last_used = time.time()
            return spec.url

        with self._lock:
            self.refresh()
            status = self._statuses[key]

            if status.state is ModelState.READY:
                status.last_used = time.time()
                return spec.url

            if not spec.available:
                raise ModelManagerError(
                    f"{spec.label}: weights not found at {spec.path}."
                )

            # Only chat models compete for the single slot. An OCR backend is
            # meant to run alongside one, not instead of it.
            if self._max_active <= 1 and spec.role == "chat":
                self._stop_others(key)

            warning = self._check_ram(spec)
            self._start(spec)
            status.warning = warning
            status.last_used = time.time()
            return spec.url

    def switch_to(self, key: str) -> str:
        """Explicit alias for ensure(); reads better at call sites."""
        return self.ensure(key)

    def stop(self, key: str) -> bool:
        """Stop a model we started. Returns False if it was not ours."""
        with self._lock:
            spec = self.get_spec(key)
            if spec.remote:
                # There is no process and no RAM to reclaim. Reporting success
                # would put a "stopped" model in the UI that the next refresh
                # flips straight back to ready.
                return False

            status = self._statuses[key]
            process = self._processes.get(key)

            if process is None:
                if status.state is ModelState.READY and status.adopted:
                    # Ours by port, even if another process started it.
                    if not self._stop_foreign(spec):
                        return False
                status.state = ModelState.STOPPED
                status.pid = None
                status.adopted = False
                return True

            status.state = ModelState.STOPPING
            self._terminate(process, spec)
            self._processes.pop(key, None)
            status.state = ModelState.STOPPED
            status.pid = None
            status.adopted = False
            return True

    def stop_all(self) -> None:
        with self._lock:
            for key in list(self._processes):
                self.stop(key)

    def unload_idle(self) -> list[str]:
        """Stop models untouched for longer than the idle timeout.

        Call this periodically. Returns the keys that were unloaded.
        """
        if self._idle_timeout <= 0:
            return []

        unloaded: list[str] = []
        with self._lock:
            now = time.time()
            for key, status in self._statuses.items():
                if status.state is not ModelState.READY or key not in self._processes:
                    continue
                if status.last_used and now - status.last_used > self._idle_timeout:
                    if self.stop(key):
                        unloaded.append(key)
        return unloaded

    # --- internals ---

    def _stop_others(self, keep: str) -> None:
        for key, status in self._statuses.items():
            if key == keep:
                continue
            # Remote models hold no RAM and OCR is deliberately outside the
            # rotation, so neither is ever stopped to make room.
            if status.spec.remote or status.spec.role != "chat":
                continue
            if key in self._processes:
                self.stop(key)
            elif status.state is ModelState.READY and status.adopted:
                # Refusing outright used to deadlock the common case: restart
                # the UI and every server it started earlier looks foreign, so
                # switching became impossible until you killed things by hand.
                # These ports are declared ours in models.json, so a
                # llama-server sitting on one is part of this system - but
                # anything else is left alone.
                if not self._stop_foreign(status.spec):
                    raise ModelManagerError(
                        f"Port {status.spec.port} is held by a process that is "
                        f"not a llama-server, so {status.spec.label} cannot be "
                        f"started. Free that port and retry."
                    )
                status.state = ModelState.STOPPED
                status.adopted = False
                status.pid = None

    def _available_ram(self) -> int | None:
        """Free RAM in MB. A seam: tests override this so the suite does not
        depend on how much memory this machine happens to have spare."""
        return available_ram_mb()

    def _check_ram(self, spec: ModelSpec) -> str:
        """Decide whether starting `spec` is reckless, risky, or fine.

        Measured on this machine, which is why this is not a simple
        "free RAM must exceed model size" test:

          * llama.cpp maps the weights, and pages fault in lazily. Starting
            Qwen3.5 2B XS (704 MB) with 503 MB available took 9 seconds and
            left available RAM unchanged.
          * Committed memory then climbs towards the model's size as inference
            touches the weights - the same process later showed 738 MB private.
          * That turn still completed correctly in 89 seconds.

        So a shortfall at load time predicts *slow*, not *broken*: the pages it
        cannot keep resident get re-read from disk on every token. Refusing
        outright would block work that runs, so a shortfall is a warning and
        only a genuinely tiny margin is fatal.

        Returns a warning string, empty when there is nothing to say.
        """
        free = self._available_ram()
        if free is None:
            return ""

        if free < HARD_FLOOR_MB:
            raise ModelManagerError(
                f"Only {free} MB of RAM is available, below the {HARD_FLOOR_MB} MB "
                f"floor. Starting {spec.label} now would leave nothing for the "
                f"rest of the system. Close something first."
            )

        if spec.min_free_mb > 0 and free < spec.min_free_mb:
            return (
                f"{spec.label} runs best with about {spec.min_free_mb} MB "
                f"available and there is {free} MB. It will still start, but "
                f"the weights cannot all stay resident, so expect slower "
                f"generation while pages are read back from disk."
            )
        return ""

    def _start(self, spec: ModelSpec) -> None:
        if not self._server_exe.is_file():
            raise ModelManagerError(
                f"llama-server not found: {self._server_exe}. Install it and "
                f"put it on PATH, or set \"server_exe\" in models.json to "
                f"where it actually is."
            )

        status = self._statuses[spec.key]
        status.state = ModelState.STARTING
        status.error = ""
        status.warning = ""

        command = [
            str(self._server_exe),
            "-m", str(spec.path),
            "--jinja",
            "-c", str(spec.context),
            "-t", str(spec.threads),
            "-np", "1",
            # No --cache-reuse here, and that is a measured decision rather
            # than an omission. The theory was that replaying history without
            # tool calls - they are display metadata and are not stored as
            # conversation - would make the second turn's prompt diverge from
            # the cached tokens and force a reprocess. Measured, it does not:
            # turn two reprocesses 50 tokens of 982 with the flag and 50
            # without, because the divergence is near the end of the prompt
            # where there is almost nothing left to redo. Ordinary prefix
            # caching already covers this, so the flag would be cargo.
            "--host", "127.0.0.1",
            "--port", str(spec.port),
        ]
        if spec.mmproj is not None:
            # Without this the language half loads and reports no vision, which
            # reads as a broken tool rather than a missing argument.
            command += ["--mmproj", str(spec.mmproj)]

        try:
            process = self._spawn(command)
        except OSError as exc:
            status.state = ModelState.FAILED
            status.error = str(exc)
            raise ModelManagerError(f"Could not start {spec.label}: {exc}") from None

        self._processes[spec.key] = process
        status.pid = process.pid
        status.adopted = False

        if self._wait_healthy(spec, process):
            status.state = ModelState.READY
            return

        # Never leave a half-started process behind.
        self._terminate(process, spec)
        self._processes.pop(spec.key, None)
        status.state = ModelState.FAILED
        status.pid = None
        raise ModelManagerError(
            status.error
            or f"{spec.label} did not become healthy within "
            f"{self._start_timeout:.0f}s."
        )

    # --- adopting and reclaiming ports ---

    def _run_quiet(self, command: list[str]) -> str:
        """Run a short console command and return stdout, or "" on failure."""
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout or ""

    def listener_pid(self, port: int) -> int | None:
        """PID of whatever is listening on `port`, or None.

        Used to decide whether a server already on one of our ports is a
        llama-server worth adopting, or something else that must be left
        alone. Both halves of that question need a pid.
        """
        if os.name == "nt":
            return self._listener_pid_windows(port)
        return self._listener_pid_posix(port)

    def _listener_pid_windows(self, port: int) -> int | None:
        for line in self._run_quiet(["netstat", "-ano", "-p", "TCP"]).splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3].upper() == "LISTENING":
                if parts[1].rsplit(":", 1)[-1] == str(port):
                    try:
                        return int(parts[4])
                    except ValueError:
                        return None
        return None

    def _listener_pid_posix(self, port: int) -> int | None:
        """`ss` first, then `lsof`.

        `ss` is part of iproute2 and present on essentially every modern
        Linux; `lsof` covers the rest and macOS. Parsing /proc/net/tcp by hand
        would avoid both, but it means matching socket inodes against every
        process's file descriptors, which is a great deal of code for a
        question two standard tools answer directly.
        """
        pid = parse_ss_pid(self._run_quiet(["ss", "-lntpH", f"sport = :{port}"]))
        if pid is not None:
            return pid

        out = self._run_quiet(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"]
        )
        for line in out.splitlines():
            try:
                return int(line.strip())
            except ValueError:
                continue
        return None

    def process_name(self, pid: int) -> str:
        """Image name for a pid, lowercased. Empty when unknown."""
        if os.name == "nt":
            out = self._run_quiet(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"]
            ).strip()
            if not out or "No tasks" in out:
                return ""
            return out.splitlines()[0].split(",")[0].strip('"').lower()

        # procfs knows without spawning anything.
        try:
            with open(f"/proc/{int(pid)}/comm", "r", encoding="utf-8") as handle:
                return handle.read().strip().lower()
        except OSError:
            pass
        return self._run_quiet(["ps", "-p", str(pid), "-o", "comm="]).strip().lower()

    def _stop_foreign(self, spec: ModelSpec) -> bool:
        """Stop a llama-server on one of our ports that we did not start.

        Returns False when the port is held by something else, which is left
        alone: the identity check is what keeps this from killing whatever
        happens to be on the port.
        """
        pid = self.listener_pid(spec.port)
        if pid is None:
            return True  # nothing there after all

        if "llama-server" not in self.process_name(pid):
            return False

        self._kill(pid)
        deadline = time.time() + 10
        while time.time() < deadline and self._healthy(spec.port):
            time.sleep(0.5)
        return not self._healthy(spec.port)

    def _kill(self, pid: int) -> None:
        """Stop a process we did not start, by pid.

        Only ever called after `process_name` has confirmed it is a
        llama-server, which is what keeps this from killing whatever happens
        to be holding the port.
        """
        if os.name == "nt":
            self._run_quiet(["taskkill", "/F", "/PID", str(pid)])
            return
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError, AttributeError):
            pass

    def _spawn(self, command: list[str]) -> subprocess.Popen:
        """Launch llama-server. Overridden in tests to avoid real processes."""
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _wait_healthy(self, spec: ModelSpec, process: subprocess.Popen) -> bool:
        """Poll /health until ready, the process dies, or we run out of time.

        Loading a 5 GB model on a machine that has to page it in from disk
        takes minutes, so the default timeout is generous.
        """
        deadline = time.time() + self._start_timeout
        status = self._statuses[spec.key]

        while time.time() < deadline:
            if process.poll() is not None:
                status.error = (
                    f"llama-server exited with code {process.returncode} while "
                    f"loading {spec.label}."
                )
                return False
            if self._healthy(spec.port):
                return True
            time.sleep(1.0)

        status.error = (
            f"{spec.label} did not report healthy within {self._start_timeout:.0f}s."
        )
        return False

    def _healthy(self, port: int) -> bool:
        """Whether a llama-server is answering on `port`.

        The socket is checked before the HTTP request. An HTTP call to a port
        with nothing behind it can sit for the whole timeout, and this runs
        once per registered model every time state is reconciled - which made
        /api/models an 18-second request with three dead ports in the registry.
        A refused connection comes back in about a millisecond.

        The HTTP timeout stays generous on purpose: a server that is mid-
        generation can be slow to answer /health, and calling it dead would
        make the manager try to restart a model that is working.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return False

        try:
            response = self._session.get(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def _terminate(self, process: subprocess.Popen, spec: ModelSpec) -> None:
        """Ask the process to stop, then insist.

        Windows has no true graceful signal for a non-console child, so
        terminate() is already abrupt. That is acceptable for an inference
        server, which holds no state worth flushing; kill() is the backstop
        for a process wedged in a blocking read.
        """
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=self._stop_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass

        # The port takes a moment to free up after the process goes.
        deadline = time.time() + 5
        while time.time() < deadline and self._healthy(spec.port):
            time.sleep(0.5)
