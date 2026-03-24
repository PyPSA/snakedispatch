"""Snkmt workflow endpoints: expose per-job snakemake DB data via REST.

Queries the snkmt SQLite database (synced from the job's work directory)
and maps results to response models. See app/snkmt.py for the query layer.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path as FilePath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path

from app import snkmt
from app.backends.base import ComputeBackend
from app.deps import get_backend, get_store, require_job
from app.models import (
    JobStatus,
    SnkmtErrorResponse,
    SnkmtFileResponse,
    SnkmtJobResponse,
    SnkmtRuleResponse,
    SnkmtWorkflowResponse,
    WorkflowFileInfo,
)
from app.store import JobStore

logger = logging.getLogger(__name__)

router = APIRouter()


# Job statuses where snkmt data is not yet available (before workflow execution starts).
_PRE_EXECUTION_STATUSES = frozenset({JobStatus.PENDING, JobStatus.SETUP})


def _require_snkmt(
    store: JobStore, job_id: str
) -> tuple[FilePath, str | None, list[WorkflowFileInfo] | None]:
    """Return the snkmt DB path, work_dir, and cached workflow_files."""
    record = require_job(store, job_id)
    if record.status in _PRE_EXECUTION_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job {job_id} is in state {record.status},"
                " workflow data not available yet"
            ),
        )
    snkmt_db = store.get_snkmt_db_path(job_id)
    if snkmt_db is None:
        raise HTTPException(status_code=500, detail="Persistence not configured")
    if not snkmt_db.exists() and record.status == JobStatus.RUNNING:
        # Not synced yet: 409 (retry later) rather than 404.
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still running; snkmt database not yet synced",
        )
    if not snkmt_db.exists():
        raise HTTPException(
            status_code=404,
            detail=f"snkmt database not available for job {job_id}",
        )
    return snkmt_db, record.work_dir, record.workflow_files


async def _run_snkmt_query[T](job_id: str, query: Callable[[], T]) -> T:
    """Run query in a thread and map sqlite3.DatabaseError to HTTP 502."""
    try:
        return await asyncio.to_thread(query)
    except sqlite3.DatabaseError as exc:
        logger.warning("snkmt DB query failed for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"snkmt database is corrupt for job {job_id}",
        ) from exc


def _strip_work_dir(path: str, work_dir: str | None) -> str:
    """Strip work_dir prefix from a file path, returning a relative path."""
    if not work_dir:
        return path
    prefix = work_dir.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _filter_empty_log_files(
    jobs: list[SnkmtJobResponse],
    workflow_files: list[WorkflowFileInfo] | None,
) -> None:
    """Remove LOG files not in the (size-filtered) file listing."""
    if not workflow_files:
        return
    valid_paths = {wf["path"] for wf in workflow_files}
    for job in jobs:
        job.files = [
            f for f in job.files if f.file_type != "LOG" or f.path in valid_paths
        ]


def _build_snkmt_job_response(
    j: snkmt.JobRow,
    files_by_job: dict[int, list[snkmt.FileRow]],
    work_dir: str | None = None,
) -> SnkmtJobResponse:
    """Map a JobRow to SnkmtJobResponse."""
    job_files = files_by_job.get(j["id"], [])
    return SnkmtJobResponse(
        snakemake_id=j["snakemake_id"],
        rule=j["rule_name"] or "",
        status=j["status"] or "",
        wildcards=snkmt.safe_json_loads(j["wildcards"]),
        resources=snkmt.safe_json_loads(j["resources"]),
        shellcmd=j["shellcmd"],
        threads=j["threads"],
        started_at=j["started_at"],
        completed_at=j["completed_at"],
        files=[
            SnkmtFileResponse(
                path=_strip_work_dir(f["path"], work_dir),
                file_type=f["file_type"],
            )
            for f in job_files
        ],
    )


def _build_snkmt_rule_response(
    r: snkmt.RuleRow, jobs: list[SnkmtJobResponse]
) -> SnkmtRuleResponse:
    """Map a RuleRow + pre-built job list to SnkmtRuleResponse."""
    return SnkmtRuleResponse(
        name=r["name"],
        total_job_count=r["total_job_count"] or 0,
        jobs_finished=r["jobs_finished"] or 0,
        jobs=jobs,
    )


def _build_snkmt_error_response(e: snkmt.ErrorRow) -> SnkmtErrorResponse:
    """Map an ErrorRow to SnkmtErrorResponse."""
    return SnkmtErrorResponse(
        timestamp=e["timestamp"],
        exception=e["exception"],
        rule=e["rule_name"],
        traceback=e["traceback"],
    )


def _build_snkmt_workflow_response(
    data: snkmt.WorkflowData, work_dir: str | None = None
) -> SnkmtWorkflowResponse:
    """Assemble a full SnkmtWorkflowResponse from WorkflowData.

    Groups jobs by rule, builds nested rule responses, and maps errors.
    """
    jobs_by_rule: dict[str, list[SnkmtJobResponse]] = {}
    for j in data.jobs:
        rule_name = j["rule_name"] or ""
        jobs_by_rule.setdefault(rule_name, []).append(
            _build_snkmt_job_response(j, data.files_by_job, work_dir)
        )

    if data.workflow is None or data.workflow_id is None:
        msg = "workflow and workflow_id must not be None"
        raise RuntimeError(msg)
    return SnkmtWorkflowResponse(
        workflow_id=data.workflow_id,
        status=data.workflow["status"] or "",
        started_at=data.workflow["started_at"],
        completed_at=data.workflow["completed_at"],
        total_job_count=data.workflow["total_job_count"] or 0,
        jobs_finished=data.workflow["jobs_finished"] or 0,
        rulegraph=data.rulegraph,
        rules=[
            _build_snkmt_rule_response(r, jobs_by_rule.get(r["name"], []))
            for r in data.rules
        ],
        errors=[_build_snkmt_error_response(e) for e in data.errors],
    )


@router.get("/jobs/{job_id}/workflow", response_model=SnkmtWorkflowResponse)
async def get_workflow(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
) -> SnkmtWorkflowResponse:
    """Full workflow overview: rules with nested jobs+files, errors, rulegraph."""
    snkmt_db, work_dir, workflow_files = _require_snkmt(store, job_id)

    data = await _run_snkmt_query(job_id, lambda: snkmt.fetch_workflow_data(snkmt_db))

    if data.workflow is None or data.workflow_id is None:
        raise HTTPException(status_code=404, detail="Workflow not found in snkmt DB")

    response = _build_snkmt_workflow_response(data, work_dir)
    all_jobs = [job for rule in response.rules for job in rule.jobs]
    backend.resolve_job_logs(all_jobs, workflow_files)
    _filter_empty_log_files(all_jobs, workflow_files)
    return response


@router.get("/jobs/{job_id}/workflow/jobs", response_model=list[SnkmtJobResponse])
async def get_workflow_jobs(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
) -> list[SnkmtJobResponse]:
    """List all snakemake jobs with timing, status, and files."""
    snkmt_db, work_dir, workflow_files = _require_snkmt(store, job_id)

    result = await _run_snkmt_query(job_id, lambda: snkmt.fetch_workflow_jobs(snkmt_db))
    jobs = [
        _build_snkmt_job_response(j, result.files_by_job, work_dir) for j in result.jobs
    ]
    backend.resolve_job_logs(jobs, workflow_files)
    _filter_empty_log_files(jobs, workflow_files)
    return jobs


@router.get(
    "/jobs/{job_id}/workflow/rules/{rule_name}",
    response_model=SnkmtRuleResponse,
)
async def get_workflow_rule(
    job_id: Annotated[str, Path()],
    rule_name: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
    backend: Annotated[ComputeBackend, Depends(get_backend)],
) -> SnkmtRuleResponse:
    """Get a single rule with its jobs and files."""
    snkmt_db, work_dir, workflow_files = _require_snkmt(store, job_id)

    result = await _run_snkmt_query(
        job_id, lambda: snkmt.fetch_rule_jobs(snkmt_db, rule_name)
    )

    if result.rule is None:
        raise HTTPException(
            status_code=404, detail=f"Rule '{rule_name}' not found in workflow"
        )

    rule_jobs = [
        _build_snkmt_job_response(j, result.files_by_job, work_dir) for j in result.jobs
    ]
    backend.resolve_job_logs(rule_jobs, workflow_files)
    _filter_empty_log_files(rule_jobs, workflow_files)

    return _build_snkmt_rule_response(result.rule, rule_jobs)


@router.get(
    "/jobs/{job_id}/workflow/jobs/{snakemake_job_id}/files",
    response_model=list[SnkmtFileResponse],
)
async def get_workflow_job_files(
    job_id: Annotated[str, Path()],
    snakemake_job_id: Annotated[
        int, Path()
    ],  # snakemake_id column, not DB primary key or UUID
    store: Annotated[JobStore, Depends(get_store)],
) -> list[SnkmtFileResponse]:
    """Files for a specific snakemake job."""
    snkmt_db, work_dir, _workflow_files = _require_snkmt(store, job_id)

    files_raw = await _run_snkmt_query(
        job_id, lambda: snkmt.fetch_job_files(snkmt_db, snakemake_job_id)
    )
    return [
        SnkmtFileResponse(
            path=_strip_work_dir(f["path"], work_dir),
            file_type=f["file_type"],
        )
        for f in files_raw
    ]


@router.get(
    "/jobs/{job_id}/workflow/rulegraph",
    response_model=dict[str, Any],  # schema owned by snakemake, passed through as-is
)
async def get_workflow_rulegraph(
    job_id: Annotated[str, Path()],
    store: Annotated[JobStore, Depends(get_store)],
) -> dict[str, Any]:
    """Return the rulegraph JSON for the workflow."""
    snkmt_db, _work_dir, _workflow_files = _require_snkmt(store, job_id)

    rulegraph = await _run_snkmt_query(job_id, lambda: snkmt.fetch_rulegraph(snkmt_db))

    if rulegraph is None:
        raise HTTPException(status_code=404, detail="Rulegraph not available")
    return rulegraph
