from __future__ import annotations

import enum
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


def _ensure_utc(v: str | datetime | None) -> datetime | None:
    """Treat naive datetimes as UTC."""
    if v is None:
        return None
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


class JobStatus(enum.StrEnum):
    PENDING = "PENDING"
    SETUP = "SETUP"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.ERROR,
        JobStatus.CANCELLED,
    }
)


class JobCreate(BaseModel):
    """POST /jobs request body."""

    workflow: str = Field(
        ...,
        description="Git URL (https://...repo.git) to clone, or local path "
        "for a directory mounted into the container.",
    )
    git_ref: str | None = Field(
        None,
        description="Git branch, tag, or commit hash to check out. "
        "Only used when workflow is a Git URL.",
    )
    configfile: str | None = Field(
        None,
        description="Path to Snakemake config file, relative to repo root",
    )
    snakemake_args: list[str] | None = Field(
        None,
        description="Extra arguments passed to the snakemake command.",
    )
    extra_files: dict[str, str] | None = Field(
        None,
        description="Files to write into working dir before execution. "
        "Keys are relative paths, values are file contents.",
    )
    cache_key: str | None = Field(
        None,
        description="Cache identifier. If set together with cache_dirs, "
        "restores cached directories before run and saves them back after success.",
    )
    cache_dirs: list[str] | None = Field(
        None,
        description="Relative paths within the work directory to cache, "
        "e.g. ['data', 'resources']. Requires cache_key.",
    )
    env_vars: dict[str, str] | None = Field(
        None,
        description="Environment variables to merge into the Snakemake process "
        "environment. Only keys listed in the server's allowed_env_vars config are accepted.",
    )

    @field_validator("cache_key")
    @classmethod
    def _validate_cache_key(cls, v: str | None) -> str | None:
        if v is not None and (not v or "/" in v or "\\" in v or ".." in v):
            msg = "cache_key must not contain '/', '\\', or '..'"
            raise ValueError(msg)
        return v

    @field_validator("extra_files")
    @classmethod
    def _validate_extra_files(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None:
            for key in v:
                parts = PurePosixPath(key).parts
                if ".." in parts or any(p.startswith("/") for p in parts):
                    msg = f"extra_files key is not a safe relative path: {key!r}"
                    raise ValueError(msg)
        return v

    @field_validator("cache_dirs")
    @classmethod
    def _validate_cache_dirs(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for entry in v:
                parts = PurePosixPath(entry).parts
                if ".." in parts or any(p.startswith("/") for p in parts):
                    msg = f"cache_dirs entry is not a safe relative path: {entry!r}"
                    raise ValueError(msg)
        return v


class JobResponse(BaseModel):
    """Unified job response for all endpoints."""

    job_id: str
    status: JobStatus
    workflow: str | None = None
    configfile: str | None = None
    git_ref: str | None = None
    git_sha: str | None = None
    exit_code: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_job_count: int | None = None
    jobs_finished: int | None = None


class WorkflowFileInfo(TypedDict):
    """Typed shape returned by ComputeBackend.list_workflow_files."""

    path: str
    size: int


class OutputFile(BaseModel):
    path: str
    size: int


class JobOutputsResponse(BaseModel):
    """GET /jobs/{job_id}/outputs response body."""

    files: list[OutputFile]


class JobCleanedResponse(BaseModel):
    """DELETE /jobs/{job_id} response body."""

    job_id: str
    status: Literal["cleaned"] = "cleaned"


class JobListResponse(BaseModel):
    """GET /jobs response body."""

    jobs: list[JobResponse]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    backend: bool


class SnkmtFileResponse(BaseModel):
    path: str
    file_type: str


class SnkmtJobResponse(BaseModel):
    snakemake_id: int
    rule: str
    status: str
    wildcards: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    shellcmd: str | None = None
    threads: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    files: list[SnkmtFileResponse] = []
    log: str | None = None

    @field_validator("started_at", "completed_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: str | datetime | None) -> datetime | None:
        return _ensure_utc(v)


class SnkmtRuleResponse(BaseModel):
    name: str
    total_job_count: int = 0
    jobs_finished: int = 0
    jobs: list[SnkmtJobResponse] = []


class SnkmtErrorResponse(BaseModel):
    timestamp: datetime | None = None
    exception: str | None = None
    rule: str | None = None
    traceback: str | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: str | datetime | None) -> datetime | None:
        return _ensure_utc(v)


class SnkmtWorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_job_count: int = 0
    jobs_finished: int = 0
    rulegraph: dict[str, Any] | None = None
    rules: list[SnkmtRuleResponse]
    errors: list[SnkmtErrorResponse]

    @field_validator("started_at", "completed_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: str | datetime | None) -> datetime | None:
        return _ensure_utc(v)
