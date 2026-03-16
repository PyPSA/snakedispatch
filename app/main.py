"""Application factory: create the FastAPI app with backend, store, and routers."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from app.backends import create_backend
from app.config import Settings, build_settings, load_config
from app.deps import AppState
from app.routes.health import router as health_router
from app.routes.jobs import router as jobs_router
from app.routes.snkmt import router as snkmt_router
from app.store import JobStore
from app.tasks import gc_loop, sync_job_data_loop

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = build_settings()
    backend_config, settings_overrides = load_config(settings.SNAKEDISPATCH_CONFIG)
    if settings_overrides:
        # Re-create settings with YAML as init defaults but env vars still win
        settings = Settings(**settings_overrides)
    store = JobStore(data_dir=Path(settings.DATA_DIR))
    backend = create_backend(backend_config)

    # Scoped to this app instance to avoid shared state across test instances
    app.state.app = AppState(
        store=store,
        backend=backend,
        settings=settings,
        health_cache={"backend_ok": None, "checked_at": 0.0},
        default_snakemake_args=backend_config.default_snakemake_args,
    )

    store.restore_from_disk()

    gc_task = asyncio.create_task(
        gc_loop(store, backend, settings.AUTO_CLEANUP_AFTER_HOURS)
    )
    sync_task = asyncio.create_task(
        sync_job_data_loop(
            store,
            backend,
            backend_config.snkmt_db_sync_interval,
        )
    )

    yield

    gc_task.cancel()
    sync_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await gc_task
    with contextlib.suppress(asyncio.CancelledError):
        await sync_task


app = FastAPI(
    title="snakedispatch",
    description="Dispatch Snakemake workflows to compute backends",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(jobs_router)
app.include_router(health_router)
app.include_router(snkmt_router)
