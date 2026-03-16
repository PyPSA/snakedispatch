from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TypedDict, cast

from fastapi import HTTPException, Request

from app.backends.base import ComputeBackend
from app.config import Settings
from app.store import JobRecord, JobStore


class HealthCache(TypedDict):
    """In-memory cache for the /health/cached endpoint."""

    backend_ok: bool | None
    checked_at: float


@dataclass
class AppState:
    """Typed container for all shared runtime state attached to the FastAPI app."""

    store: JobStore
    backend: ComputeBackend
    settings: Settings
    health_cache: HealthCache
    default_snakemake_args: list[str]
    background_tasks: set[asyncio.Task[None]] = field(default_factory=set)


def app_state(request: Request) -> AppState:
    return cast("AppState", request.app.state.app)


def get_store(request: Request) -> JobStore:
    return app_state(request).store


def get_backend(request: Request) -> ComputeBackend:
    return app_state(request).backend


def get_default_snakemake_args(request: Request) -> list[str]:
    return app_state(request).default_snakemake_args


def provide_background_tasks(request: Request) -> set[asyncio.Task[None]]:
    """Return the shared background-task set."""
    return app_state(request).background_tasks


def provide_health_cache(request: Request) -> HealthCache:
    """Return the shared health-cache dict."""
    return app_state(request).health_cache


def get_settings(request: Request) -> Settings:
    return app_state(request).settings


def require_job(store: JobStore, job_id: str) -> JobRecord:
    """Return the JobRecord or raise 404."""
    record = store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return record
