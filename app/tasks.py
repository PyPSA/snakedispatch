from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from app import snkmt
from app.models import TERMINAL_STATUSES, JobStatus, WorkflowFileInfo
from app.utils import enforce_error_limit

if TYPE_CHECKING:
    from pathlib import Path

    from app.backends.base import ComputeBackend
    from app.store import JobRecord, JobStore


@dataclass
class ExecuteJobParams:
    """Per-job execution parameters (separate from store/backend)."""

    configfile: str | None = None
    git_ref: str | None = None
    snakemake_args: list[str] | None = None
    extra_files: dict[str, str] | None = None
    cache_key: str | None = None
    cache_dirs: list[str] | None = None


logger = logging.getLogger(__name__)

_LOOP_ERROR_THRESHOLD = 5


async def _sync_snkmt_counts(
    backend: ComputeBackend,
    job_id: str,
    work_dir: str,
    snkmt_db_path: Path,
    record: JobRecord,
) -> None:
    """Sync snkmt.db from backend and mutate record counts."""
    await backend.sync_snkmt_db(job_id, work_dir, snkmt_db_path)
    counts = snkmt.fetch_workflow_counts(snkmt_db_path)
    if counts:
        record.total_job_count, record.jobs_finished = counts


def _flush_logs(store: JobStore, job_id: str) -> None:
    """Flush buffered log lines to disk, swallowing I/O errors."""
    try:
        store.flush_logs_to_disk(job_id)
    except OSError:
        logger.warning("Log flush failed for %s", job_id, exc_info=True)


async def _try_sync_snkmt(
    backend: ComputeBackend,
    job_id: str,
    record: JobRecord,
    snkmt_db_path: Path,
) -> None:
    """Sync snkmt.db and update job counts, swallowing non-fatal errors."""
    if not record.work_dir:
        return
    try:
        await _sync_snkmt_counts(
            backend, job_id, record.work_dir, snkmt_db_path, record
        )
    except Exception:
        logger.warning("snkmt sync failed for %s", job_id, exc_info=True)


async def _flush_and_sync(
    store: JobStore,
    backend: ComputeBackend,
    job_id: str,
    record: JobRecord,
) -> None:
    """Flush buffered logs and sync snkmt counts for a running job."""
    _flush_logs(store, job_id)
    snkmt_db_path = store.get_snkmt_db_path(job_id)
    if snkmt_db_path:
        await _try_sync_snkmt(backend, job_id, record, snkmt_db_path)


async def _finalize_job(
    store: JobStore,
    backend: ComputeBackend,
    record: JobRecord,
    *,
    skip_snkmt_sync: bool = False,
) -> None:
    """Final sync of all job data at terminal state, then clear in-memory logs."""
    if not skip_snkmt_sync and record.work_dir:
        await _flush_and_sync(store, backend, record.job_id, record)
    else:
        _flush_logs(store, record.job_id)
    # Persist job metadata + clear in-memory state
    store.persist(record)
    store.clear_job_logs(record.job_id)


async def _try_cache_workflow_files(
    backend: ComputeBackend,
    job_id: str,
    work_dir: str,
) -> list[WorkflowFileInfo] | None:
    """List workflow files from the backend, swallowing errors."""
    try:
        return await backend.list_workflow_files(job_id, work_dir)
    except Exception:
        logger.warning("Failed to cache workflow files for %s", job_id, exc_info=True)
        return None


async def _try_save_cache(
    backend: ComputeBackend,
    job_id: str,
    work_dir: str,
    cache_key: str,
    cache_dirs: list[str],
) -> None:
    """Save workflow cache to the backend, swallowing non-fatal errors."""
    try:
        await backend.save_cache(job_id, work_dir, cache_key, cache_dirs)
    except Exception:
        logger.warning(
            "Cache save failed for job %s (key=%s)", job_id, cache_key, exc_info=True
        )


async def try_cleanup_job(backend: ComputeBackend, job_id: str, work_dir: str) -> None:
    """Run backend.cleanup, swallowing errors."""
    try:
        await backend.cleanup(job_id, work_dir)
    except Exception:
        logger.warning("Background cleanup failed for job %s", job_id, exc_info=True)


async def _restore_cache(
    backend: ComputeBackend,
    store: JobStore,
    job_id: str,
    work_dir: str,
    params: ExecuteJobParams,
) -> None:
    """Restore cache if configured, logging success."""
    if not (params.cache_key and params.cache_dirs):
        return
    restored = await backend.restore_cache(
        job_id, work_dir, params.cache_key, params.cache_dirs
    )
    if restored:
        store.push_log(job_id, f"Cache restored (key={params.cache_key})")


async def _collect_results(
    backend: ComputeBackend,
    store: JobStore,
    job_id: str,
    work_dir: str,
    exit_code: int,
    params: ExecuteJobParams,
) -> None:
    """Cache file listing, mark finished and save build cache on success."""
    workflow_files = await _try_cache_workflow_files(backend, job_id, work_dir)
    if workflow_files is not None:
        store.cache_workflow_files(job_id, workflow_files)
    store.mark_finished(job_id, exit_code)
    if exit_code == 0 and params.cache_key and params.cache_dirs:
        await _try_save_cache(
            backend, job_id, work_dir, params.cache_key, params.cache_dirs
        )


async def execute_job(
    store: JobStore,
    backend: ComputeBackend,
    job_id: str,
    workflow: str,
    params: ExecuteJobParams | None = None,
) -> None:
    """Background task: prepare, launch, and monitor a Snakemake workflow."""
    params = params or ExecuteJobParams()
    record = store.get_job(job_id)
    if record is None:
        msg = f"execute_job: job {job_id} not found in store"
        raise RuntimeError(msg)
    try:
        work_dir, git_ref, git_sha = await backend.prepare(
            job_id, workflow, params.git_ref
        )
        store.mark_setup(job_id, work_dir, git_ref, git_sha)
        await backend.setup(job_id, work_dir, params.extra_files)
        await _restore_cache(backend, store, job_id, work_dir, params)

        store.mark_running(job_id)
        await backend.launch(job_id, work_dir, params.configfile, params.snakemake_args)

        def log_callback(line: str) -> None:
            store.push_log(job_id, line)

        exit_code = await backend.monitor(job_id, work_dir, log_callback, byte_offset=0)
        await _collect_results(backend, store, job_id, work_dir, exit_code, params)
        logger.info(
            "Job %s finished with status %s (exit %d)", job_id, record.status, exit_code
        )

    except asyncio.CancelledError:
        if record.status not in TERMINAL_STATUSES:
            logger.info("Job %s was cancelled", job_id)
            store._mark_cancelled(job_id)
        raise

    except Exception as exc:
        logger.exception("Job %s failed with exception", job_id)
        store.mark_error(job_id, str(exc))
        raise

    finally:
        await _finalize_job(
            store,
            backend,
            record,
            skip_snkmt_sync=record.status in {JobStatus.CANCELLED, JobStatus.ERROR},
        )


async def sync_job_data_loop(
    store: JobStore,
    backend: ComputeBackend,
    sync_interval: float,
) -> None:
    """Periodically flush logs and sync snkmt counts for RUNNING jobs."""
    consecutive_errors = 0
    while True:
        await asyncio.sleep(sync_interval)
        try:
            jobs = store.all_jobs()
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            enforce_error_limit(
                consecutive_errors,
                "sync_job_data_loop",
                exc,
                threshold=_LOOP_ERROR_THRESHOLD,
            )
            continue
        for job_id, record in jobs.items():
            if record.status != JobStatus.RUNNING:
                continue
            if not record.work_dir:
                continue

            try:
                exit_code = await backend.check_job_status(job_id, record.work_dir)
                if exit_code is not None:
                    logger.warning(
                        "Recovering stuck job %s (exit code %d)",
                        job_id,
                        exit_code,
                    )
                    store.mark_finished(job_id, exit_code)
                    store.persist(record)
                    if record.task and not record.task.done():
                        record.task.cancel()
                    continue
            except Exception:
                logger.warning(
                    "Failed to check job %s",
                    job_id,
                    exc_info=True,
                )

            await _flush_and_sync(store, backend, job_id, record)


async def gc_loop(store: JobStore, backend: ComputeBackend, max_age_hours: int) -> None:
    """Periodically remove terminal jobs older than max_age_hours."""
    consecutive_errors = 0
    while True:
        await asyncio.sleep(timedelta(hours=1).total_seconds())
        try:
            now = datetime.now(UTC)
            for job_id, record in list(store.all_jobs().items()):
                if record.status not in TERMINAL_STATUSES:
                    continue
                completed_at = record.completed_at or record.created_at
                age_hours = (now - completed_at) / timedelta(hours=1)
                if age_hours > max_age_hours:
                    logger.info(
                        "GC: cleaning stale job %s (age %.1fh since completion)",
                        job_id,
                        age_hours,
                    )
                    if record.work_dir:
                        await try_cleanup_job(backend, job_id, record.work_dir)
                    store.delete_job(job_id)
            consecutive_errors = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            consecutive_errors += 1
            enforce_error_limit(
                consecutive_errors, "GC loop", exc, threshold=_LOOP_ERROR_THRESHOLD
            )
