"""The user's own model settings, kept apart from the shipped registry.

Two files, and the split is the point.

`models.json` is **curated**: hand-tuned entries, measured RAM thresholds, the
reasoning behind them in comments, and it is in version control. Editing it is
a deliberate act.

`data/models.local.json` is **yours**: which model is primary, anything you
retuned in the settings panel, and which discovered models you hid. It is
generated, git-ignored, and safe to delete - doing so returns the system to
first-launch state rather than breaking it.

Keeping them apart is what makes the settings panel safe to use. A UI that
wrote back into `models.json` would rewrite the comments explaining why those
numbers are what they are, and a `git pull` would then conflict with a
preference. Here, an upgrade can change the shipped registry and a preference
still applies on top.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILENAME = "models.local.json"

# Bumped if the file's shape changes incompatibly. An older file is ignored
# rather than misread - the cost is re-choosing a primary model, which is one
# click, against silently applying a setting that no longer means what it did.
SCHEMA_VERSION = 1

# What a per-model override may change. Deliberately not `file`, `port` or
# `role`: those decide what the model *is* and where it runs, and getting them
# wrong from a settings panel produces a model that cannot start for reasons
# the panel cannot explain.
OVERRIDABLE = frozenset({"label", "context", "threads", "min_free_mb", "description"})

# Guard rails for anything typed into the settings panel. A context of zero
# means "the model's trained maximum" to llama.cpp, which on a 262k model is
# how you fill a machine.
LIMITS: dict[str, tuple[int, int]] = {
    "context": (512, 131_072),
    "threads": (1, 64),
    "min_free_mb": (0, 128_000),
}


@dataclass
class ModelPreferences:
    """Everything the user chose, and how to persist it."""

    path: Path
    primary: str = ""
    router_fast: str = ""
    router_strong: str = ""
    # key -> {field: value}, filtered to OVERRIDABLE on the way in.
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Discovered models the user does not want offered. Curated entries are
    # hidden the same way, so this is the one list either kind honours.
    hidden: list[str] = field(default_factory=list)
    # False until a primary has been chosen, which is what the first-launch
    # prompt keys off. Separate from `primary` being empty so that choosing
    # and then clearing is not mistaken for a fresh install.
    setup_complete: bool = False
    # Where this machine's llama-server is, when it is somewhere the search
    # would not find on its own. Kept here rather than in models.json because
    # it is a property of one computer, and models.json is in version control -
    # a path committed from someone's laptop is wrong on every other machine.
    server_exe: str = ""

    # --- loading and saving ---

    @classmethod
    def load(cls, directory: str | Path) -> "ModelPreferences":
        """Read preferences from `directory`. A missing file is not an error."""
        path = Path(directory).expanduser() / FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path=path)
        except (OSError, ValueError):
            # A corrupt preferences file must not stop the application
            # starting. Falling back to first-launch state costs one choice.
            return cls(path=path)

        if not isinstance(raw, dict) or int(raw.get("version", 0)) != SCHEMA_VERSION:
            return cls(path=path)

        overrides = {}
        for key, values in (raw.get("overrides") or {}).items():
            if isinstance(values, dict):
                cleaned = _clean_override(values)
                if cleaned:
                    overrides[str(key)] = cleaned

        router = raw.get("router") or {}
        return cls(
            path=path,
            primary=str(raw.get("primary", "") or ""),
            router_fast=str(router.get("fast", "") or ""),
            router_strong=str(router.get("strong", "") or ""),
            overrides=overrides,
            hidden=[str(key) for key in (raw.get("hidden") or []) if key],
            setup_complete=bool(raw.get("setup_complete", False)),
            server_exe=str(raw.get("server_exe", "") or ""),
        )

    def save(self) -> None:
        """Write the file, atomically.

        Written beside the target and renamed over it, so a crash part-way
        cannot leave a half-written file where a good one used to be - the same
        rule the filesystem tool follows.
        """
        payload = {
            "version": SCHEMA_VERSION,
            "_comment": (
                "Your model settings. Generated - safe to delete, which "
                "returns the app to first-launch state. The shipped registry "
                "is models.json; this layers on top of it."
            ),
            "primary": self.primary,
            "server_exe": self.server_exe,
            "router": {"fast": self.router_fast, "strong": self.router_strong},
            "overrides": self.overrides,
            "hidden": self.hidden,
            "setup_complete": self.setup_complete,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    # --- changing them ---

    def set_server_exe(self, path: str) -> None:
        """Remember where llama-server is on this machine. "" forgets it."""
        self.server_exe = str(path or "")
        self.save()

    def set_primary(self, key: str) -> None:
        """Choose the model everything defaults to.

        Also seeds the router's cheap end, because a router pointing at a model
        the user has never chosen is a surprising default - and leaves the
        strong end alone, since that is a separate decision.
        """
        self.primary = key
        if not self.router_fast:
            self.router_fast = key
        self.setup_complete = True

    def set_router(self, *, fast: str = "", strong: str = "") -> None:
        if fast:
            self.router_fast = fast
        if strong:
            self.router_strong = strong

    def override(self, key: str, values: dict[str, Any]) -> dict[str, Any]:
        """Retune one model. Returns what was actually applied."""
        cleaned = _clean_override(values)
        if not cleaned:
            return {}
        self.overrides.setdefault(key, {}).update(cleaned)
        return cleaned

    def clear_override(self, key: str) -> bool:
        """Put one model back to its registry or discovered values."""
        return self.overrides.pop(key, None) is not None

    def hide(self, key: str, hidden: bool = True) -> None:
        """Stop offering a model, without deleting its file."""
        if hidden and key not in self.hidden:
            self.hidden.append(key)
        elif not hidden and key in self.hidden:
            self.hidden.remove(key)

    def is_hidden(self, key: str) -> bool:
        return key in self.hidden

    def for_key(self, key: str) -> dict[str, Any]:
        return dict(self.overrides.get(key, {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "router": {"fast": self.router_fast, "strong": self.router_strong},
            "overrides": {key: dict(v) for key, v in self.overrides.items()},
            "hidden": list(self.hidden),
            "setup_complete": self.setup_complete,
        }


def _clean_override(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields that may be overridden, clamped to sane ranges."""
    cleaned: dict[str, Any] = {}
    for name, value in values.items():
        if name not in OVERRIDABLE or value is None:
            continue
        if name in LIMITS:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            low, high = LIMITS[name]
            cleaned[name] = max(low, min(number, high))
        else:
            text = str(value).strip()
            if text:
                cleaned[name] = text[:200]
    return cleaned
