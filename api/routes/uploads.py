"""Image upload, so a file can be dropped in and read with OCR.

The OCR tool takes a *path inside the workspace* and resolves it through the
same jail as the filesystem tools. A browser upload is bytes, so this is the
bridge: bytes in, a workspace-relative path out, which is exactly what
`ocr_image` wants.

Uploads therefore land inside `config.workspace`, not beside the database.
If `AGENT_WORKSPACE` points elsewhere, they follow it - anywhere else and the
agent could not read its own attachment.

Nothing here trusts the client. The name is rebuilt rather than sanitised, the
extension must be one the OCR tool accepts, and the size cap is the one the
OCR tool already enforces, so an upload that would be refused later is refused
here instead.
"""

from __future__ import annotations

import re
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import UploadOut

router = APIRouter(tags=["uploads"])

UPLOAD_DIRNAME = "uploads"

# Read in chunks so a large file cannot be materialised in memory before the
# size is known - the cap has to be enforced while reading, not after.
CHUNK = 64 * 1024


def _safe_stem(name: str) -> str:
    """A filename built from the client's, rather than trusted from it.

    Only the basename is considered, and everything outside a conservative
    allowlist becomes an underscore - so "../../.ssh/config" cannot survive as
    anything but a flat, harmless string.
    """
    stem = Path(name or "upload").name
    stem = Path(stem).stem[:60]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "upload"


def _ocr_server_reachable(url: str) -> bool:
    """Whether something is listening where the OCR server should be.

    A TCP probe, not a request: this only answers "is it up", and the tool
    itself checks that the server actually has vision before sending an image.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


@router.post("/uploads", response_model=UploadOut)
async def upload_image(
    file: UploadFile = File(...),
    runtime: Runtime = Depends(get_runtime),
):
    """Store an image in the workspace and return the path OCR wants."""
    config = runtime.effective_config()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ocr_allowed_extensions:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{suffix or 'That file'} is not an image the OCR tool reads. "
            f"Allowed: {', '.join(config.ocr_allowed_extensions)}.",
        )

    directory = Path(config.workspace) / UPLOAD_DIRNAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Could not create {directory}: {exc}",
        ) from None

    # A short prefix keeps two uploads of "scan.png" apart, and means an upload
    # can never overwrite an existing file.
    target = directory / f"{uuid.uuid4().hex[:8]}-{_safe_stem(file.filename)}{suffix}"

    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > config.ocr_max_image_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"Larger than the {config.ocr_max_image_bytes:,}-byte "
                        f"limit the OCR tool accepts.",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not save: {exc}"
        ) from None

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That file was empty.")

    # Relative to the workspace, because that is the only form the OCR tool
    # accepts - it resolves paths against the jail, not the filesystem root.
    relative = target.relative_to(Path(config.workspace)).as_posix()

    enabled = config.ocr_enabled
    reachable = _ocr_server_reachable(config.ocr_url) if enabled else False
    if not enabled:
        hint = (
            "The OCR tool is off, so the agent cannot read this yet. Turn on "
            "OCR in the sidebar."
        )
    elif not reachable:
        hint = (
            f"OCR is on but nothing is listening on {config.ocr_url}. Start "
            f"the GLM-OCR server first - it runs separately from the chat "
            f"models and needs both the model and its mmproj file."
        )
    else:
        hint = ""

    return UploadOut(
        path=relative,
        name=file.filename or target.name,
        size=written,
        ocr_ready=enabled and reachable,
        hint=hint,
    )
