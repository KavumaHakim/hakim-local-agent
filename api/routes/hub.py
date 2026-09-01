"""Finding and fetching models from Hugging Face, from inside the app.

Choosing a model is the one decision this project deliberately leaves to the
person using it, because it depends on their RAM, their language and what they
want the agent for. That is a reason to help with the choice, not to leave
somebody alone with a browser and a hundred near-identical repositories.

So: search, see what a file would cost in RAM *before* downloading it, and
download it into `weights/` where discovery picks it up on its own.

Downloads run on their own thread rather than the turn queue. A model is
gigabytes and a domestic connection is slow, and blocking every conversation
for an hour to fetch one would be a strange way to treat a chat application.
They are polled, exactly as turns are.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    DownloadOut,
    DownloadRequest,
    DownloadsOut,
    HubFilesOut,
    HubSearchOut,
)
from models.hub import HubError, estimate_ram_mb, files, search
from models.manager import available_ram_mb

router = APIRouter(prefix="/hub", tags=["hub"])


def _fail(exc: HubError) -> HTTPException:
    """Hugging Face's problems are the user's problems, said plainly."""
    return HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))


@router.get("/search", response_model=HubSearchOut)
def search_models(
    q: str = Query("", description="What to look for."),
    limit: int = Query(20, ge=1, le=50),
):
    """Repositories carrying GGUF files, most downloaded first.

    By downloads rather than relevance on purpose: for any given model there
    are dozens of re-uploads, and the popular one is overwhelmingly the one
    that is complete, correctly converted, and still there in a month.
    """
    try:
        found = search(q, limit=limit)
    except HubError as exc:
        raise _fail(exc) from None
    return HubSearchOut(query=q, models=[model.as_dict() for model in found])


@router.get("/files", response_model=HubFilesOut)
def list_files(repo: str = Query(..., description="owner/name")):
    """The GGUF files in one repository, with what each would cost.

    The RAM figure is the point of this endpoint. It is the same arithmetic
    `models.discovery` applies to a downloaded file, minus the parts needing
    the GGUF header - so someone can be told a 4.8 GB file wants 6.2 GB free
    before spending an hour finding out.
    """
    try:
        found = files(repo)
    except HubError as exc:
        raise _fail(exc) from None

    return HubFilesOut(
        repo=repo,
        files=[item.as_dict() for item in found],
        free_ram_mb=available_ram_mb(),
    )


@router.post("/download", response_model=DownloadOut)
def start_download(body: DownloadRequest, runtime: Runtime = Depends(get_runtime)):
    """Fetch one GGUF into the models folder.

    Refused before any bytes move when the disk cannot take it or the file is
    already there - both are better answers than discovering it an hour in.
    """
    try:
        download = runtime.downloads.start(
            body.repo, body.path, size_bytes=body.size_bytes
        )
    except HubError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return DownloadOut(**download.as_dict())


@router.get("/downloads", response_model=DownloadsOut)
def list_downloads(runtime: Runtime = Depends(get_runtime)):
    """Every download this process has run, newest first."""
    return DownloadsOut(downloads=[DownloadOut(**item) for item in runtime.downloads.all()])


@router.post("/downloads/{download_id}/cancel", response_model=DownloadsOut)
def cancel_download(download_id: str, runtime: Runtime = Depends(get_runtime)):
    """Stop one, and delete the part-file it was writing.

    Not a 404 when it has already finished: by the time anyone clicks, it may
    have, and that is the outcome they asked for.
    """
    runtime.downloads.cancel(download_id)
    return DownloadsOut(downloads=[DownloadOut(**item) for item in runtime.downloads.all()])
