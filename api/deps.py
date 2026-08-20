"""Shared request dependencies."""

from __future__ import annotations

from fastapi import Request

from api.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    """The process-wide runtime, built once by the lifespan handler."""
    return request.app.state.runtime
