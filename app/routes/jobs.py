from __future__ import annotations

import asyncio
import logging
import mimetypes
import uuid
from pathlib import Path as FilePath
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.backends.base import ComputeBackend
from app.deps import (
    get_backend,
    get_default_snakemake_args,
    get_store,
    provide_background_tasks,
    require_job,
)
from app.models import (
    TERMINAL_STATUSES,
    JobCleanedResponse,
    JobCreate,
    JobListResponse,
    JobOutputsResponse,
    JobResponse,
    JobStatus,
    OutputFile,
    WorkflowFileInfo,
)
from app.store import JobRecord, JobStore
from app.tasks import ExecuteJobParams, execute_job, try_cleanup_job

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

_OUTPUTS_AVAILABLE_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RUNNING}
)


def _require_outputs_available(record: JobRecord, job_id: str) -> str:
    """Raise 409/500 if outputs are not accessible."""
    if record.status not in _OUTPUTS_AVAILABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job {job_id} is in state {record.status}, outputs not available yet"
            ),
        )
    if not record.work_dir:
        raise HTTPException(status_code=500, detail="work_dir not set for job")
    return record.work_dir


def _validate_output_path(path: str) -> str:
    """Normalize and validate a user-supplied output file path."""
    normalized = PurePosixPath(path).as_posix()
    parts = normalized.split("/")
    if ".." in parts or any(p.startswith(".") for p in parts):
        raise HTTPException(status_code=400, detail="Invalid path")
    return normalized


def _build_outputs_response(
    files_raw: list[WorkflowFileInfo],
) -> JobOutputsResponse:
    files = [OutputFile(path=f["path"], size=f["size"]) for f in files_raw]
    return JobOutputsResponse(files=files)


_RESPONSE_FIELDS = tuple(JobResponse.model_fields.keys())


def _build_job_response(record: JobRecord) -> JobResponse:
    """Build a JobResponse from a JobRecord using _RESPONSE_FIELDS."""
    return JobResponse(**{f: getattr(record, f) for f in _RESPONSE_FIELDS})


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    body: JobCreate,
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
    default_snakemake_args: Annotated[list[str], Depends(get_default_snakemake_args)],
) -> JobResponse:
    """Submit a new Snakemake job for execution."""
    source = body.workflow
    is_url = source.startswith(("http://", "https://"))
    if not is_url:
        p = FilePath(source)
        if not p.exists() or not p.is_dir():
            raise HTTPException(
                status_code=422,
                detail=f"Local path does not exist or is not a directory: {source}",
            )

    # Merge default snakemake args (from config) with per-job args. User args
    # take precedence over defaults
    snakemake_args = (default_snakemake_args + (body.snakemake_args or [])) or None

    job_id = str(uuid.uuid4())
    record = store.create_job(job_id, workflow=source, configfile=body.configfile)
    store.persist(record)

    task = asyncio.create_task(
        execute_job(
            store,
            backend,
            job_id,
            source,
            ExecuteJobParams(
                configfile=body.configfile,
                git_ref=body.git_ref,
                snakemake_args=snakemake_args,
                extra_files=body.extra_files,
                cache_key=body.cache_key,
                cache_dirs=body.cache_dirs,
            ),
        ),
        name=f"execute-{job_id}",
    )
    record.task = task

    return _build_job_response(record)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    store: Annotated[JobStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobListResponse:
    """List all known jobs, newest first, with support for pagination."""
    all_records = list(store.all_jobs().values())
    all_records.sort(key=lambda r: r.created_at, reverse=True)
    total = len(all_records)
    page = [_build_job_response(r) for r in all_records[offset : offset + limit]]
    return JobListResponse(jobs=page, total=total)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
) -> JobResponse:
    """Get the current status of a job."""
    record = require_job(store, job_id)
    return _build_job_response(record)


# Two streaming patterns coexist in this module by design:
# Running jobs stream logs via SSE (EventSourceResponse), the log is still
# being produced, so SSE gives a live stream with reconnect semantics.
# Completed jobs serve files via chunked HTTP (StreamingResponse), the data
# is static, so a plain download is all what is needed.
@router.get("/jobs/{job_id}/logs")
async def stream_logs(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
) -> EventSourceResponse:
    """Stream job logs via Server Sent Events."""
    require_job(store, job_id)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        sent_index = 0
        while True:
            new_lines, sent_index = store.get_logs(job_id, sent_index)
            for line in new_lines:
                yield {"data": line}

            record = store.get_job(job_id)
            if record is None or record.status in TERMINAL_STATUSES:
                # Flush any lines written between the last poll and
                # job completion. Falls back to the on-disk log if
                # nothing is left in memory (e.g. after a restart).
                final_lines, sent_index = store.get_logs_with_disk_fallback(
                    job_id, sent_index
                )
                for line in final_lines:
                    yield {"data": line}
                yield {
                    "event": "done",
                    "data": record.status if record else "UNKNOWN",
                }
                return

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}/outputs", response_model=JobOutputsResponse)
async def list_outputs(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
) -> JobOutputsResponse:
    """List workflow files (excluding hidden files/dirs)."""
    record = require_job(store, job_id)
    work_dir = _require_outputs_available(record, job_id)

    # Return cached file list if available
    if record.workflow_files is not None:
        return _build_outputs_response(record.workflow_files)

    files_raw = await backend.list_workflow_files(job_id, work_dir)
    store.cache_workflow_files(job_id, files_raw)
    return _build_outputs_response(files_raw)


@router.get("/jobs/{job_id}/outputs/{path:path}")
async def download_output(
    job_id: Annotated[str, Path()],
    path: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
) -> StreamingResponse:
    """Download a specific output file."""
    record = require_job(store, job_id)
    work_dir = _require_outputs_available(record, job_id)

    normalized = _validate_output_path(path)

    content_type = mimetypes.guess_type(normalized)[0] or "application/octet-stream"
    filename = normalized.rsplit("/", 1)[-1]

    # Verify file exists in the cached file listing if available
    record_files = record.workflow_files
    if record_files is not None:
        known_paths = {f["path"] for f in record_files}
        if normalized not in known_paths:
            raise HTTPException(status_code=404, detail="File not found")

    async def _safe_stream() -> AsyncGenerator[bytes, None]:
        # Catch FileNotFoundError to avoid crashing mid-stream (headers already sent).
        try:
            async for chunk in backend.stream_file(job_id, work_dir, normalized):
                yield chunk
        except FileNotFoundError:
            logger.warning(
                "File not found during streaming: %s (job %s)", normalized, job_id
            )

    return StreamingResponse(
        _safe_stream(),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
    background_tasks: Annotated[
        set[asyncio.Task[None]], Depends(provide_background_tasks)
    ],
) -> JobResponse:
    """Cancel a running job. Sets status to CANCELLED but keeps the record."""
    record = require_job(store, job_id)

    if record.status in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already in terminal state {record.status}",
        )

    store.cancel_job(job_id)

    # Run remote cleanup in background, don't block the response
    if record.work_dir:
        task = asyncio.create_task(try_cleanup_job(backend, job_id, record.work_dir))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    return _build_job_response(record)


@router.delete("/jobs/{job_id}", response_model=JobCleanedResponse)
async def delete_job(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
    background_tasks: Annotated[
        set[asyncio.Task[None]], Depends(provide_background_tasks)
    ],
) -> JobCleanedResponse:
    """Remove a job entirely: kill processes, cancel SLURM jobs, remove files."""
    record = require_job(store, job_id)

    # No need to kill the remote process first (unlike cancel), since the record
    # is about to be deleted
    if record.status not in TERMINAL_STATUSES:
        store.cancel_job(job_id)

    work_dir = record.work_dir
    store.delete_job(job_id)

    # Run remote cleanup in background, don't block the response
    if work_dir:
        task = asyncio.create_task(try_cleanup_job(backend, job_id, work_dir))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    return JobCleanedResponse(job_id=job_id)
