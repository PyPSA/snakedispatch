from __future__ import annotations

import json
import logging
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.models import TERMINAL_STATUSES, JobStatus, WorkflowFileInfo

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

logger = logging.getLogger(__name__)

MAX_IN_MEMORY_LINES = 10_000

_PERSIST_FIELDS = (
    "job_id",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "exit_code",
    "work_dir",
    "workflow",
    "configfile",
    "git_ref",
    "git_sha",
    "workflow_files",
    "total_job_count",
    "jobs_finished",
    "error",
)

_DATETIME_FIELDS = frozenset({"created_at", "started_at", "completed_at"})


@dataclass
class JobRecord:
    """In memory representation of a single job's state."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    exit_code: int | None = None
    work_dir: str | None = None
    workflow: str | None = None
    configfile: str | None = None
    git_ref: str | None = None
    git_sha: str | None = None
    workflow_files: list[WorkflowFileInfo] | None = None
    total_job_count: int | None = None
    jobs_finished: int | None = None
    error: str | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_IN_MEMORY_LINES))
    _unflushed_logs: list[str] = field(default_factory=list)
    total_log_lines: int = 0
    task: asyncio.Task[None] | None = None


class JobStore:
    """In-memory job store with optional disk persistence."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self.data_dir = data_dir

    def _job_dir(self, job_id: str) -> Path | None:
        if self.data_dir is None:
            return None
        return self.data_dir / "jobs" / job_id

    def get_snkmt_db_path(self, job_id: str) -> Path | None:
        """Return data_dir/jobs/<job_id>/snkmt.db, or None."""
        job_dir = self._job_dir(job_id)
        return job_dir / "snkmt.db" if job_dir is not None else None

    def get_log_path(self, job_id: str) -> Path | None:
        """Return data_dir/jobs/<job_id>/output.log, or None."""
        job_dir = self._job_dir(job_id)
        return job_dir / "output.log" if job_dir is not None else None

    def persist(self, record: JobRecord) -> None:
        """Atomically write job metadata to disk as JSON (via .json.tmp rename)."""
        job_dir = self._job_dir(record.job_id)
        if job_dir is None:
            return
        job_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for key in _PERSIST_FIELDS:
            value = getattr(record, key)
            if key in _DATETIME_FIELDS and value is not None:
                value = value.isoformat()
            data[key] = value
        target = job_dir.joinpath("job.json")
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(target)

    def flush_logs_to_disk(self, job_id: str) -> None:
        """Batch write unflushed log lines to disk."""
        record = self._jobs.get(job_id)
        if record is None or not record._unflushed_logs:
            return
        job_dir = self._job_dir(job_id)
        if job_dir is None:
            return
        job_dir.mkdir(parents=True, exist_ok=True)
        with (job_dir / "output.log").open("a", encoding="utf-8") as f:
            for line in record._unflushed_logs:
                f.write(line + "\n")
        record._unflushed_logs.clear()

    def restore_from_disk(self) -> None:
        """Reload all jobs from `data_dir/jobs/*/job.json`."""
        if self.data_dir is None:
            return
        jobs_root = self.data_dir / "jobs"
        if not jobs_root.is_dir():
            return
        try:
            job_dirs = sorted(jobs_root.iterdir())
        except OSError as exc:
            logger.warning("Failed to scan jobs directory %s: %s", jobs_root, exc)
            return
        for job_dir in job_dirs:
            json_path = job_dir / "job.json"
            if not json_path.is_file():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for dt_field in _DATETIME_FIELDS:
                    if data.get(dt_field) is not None:
                        data[dt_field] = datetime.fromisoformat(data[dt_field])
                data["status"] = JobStatus(data["status"])
                kwargs = {k: data[k] for k in _PERSIST_FIELDS}
                record = JobRecord(**kwargs)
                self._jobs[record.job_id] = record
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping corrupt job file: %s: %s", json_path, exc)

    def mark_setup(
        self, job_id: str, work_dir: str, git_ref: str | None, git_sha: str | None
    ) -> None:
        """Transition to SETUP, record work_dir and git metadata. Persists."""
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("mark_setup called for unknown job %s", job_id)
            return
        record.status = JobStatus.SETUP
        record.work_dir = work_dir
        record.git_ref = git_ref
        record.git_sha = git_sha
        self.persist(record)

    def mark_running(self, job_id: str) -> None:
        """Transition to RUNNING, record start time. Persists."""
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("mark_running called for unknown job %s", job_id)
            return
        record.status = JobStatus.RUNNING
        record.started_at = datetime.now(UTC)
        self.persist(record)

    def mark_finished(self, job_id: str, exit_code: int) -> None:
        """Transition to COMPLETED/FAILED based on exit code. Does not persist."""
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("mark_finished called for unknown job %s", job_id)
            return
        record.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
        record.exit_code = exit_code
        record.completed_at = datetime.now(UTC)

    def mark_error(self, job_id: str, error_message: str) -> None:
        """Transition to ERROR, record message, append to log. Does not persist."""
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("mark_error called for unknown job %s", job_id)
            return
        record.status = JobStatus.ERROR
        record.completed_at = datetime.now(UTC)
        record.error = error_message
        self.push_log(job_id, error_message)

    def create_job(
        self,
        job_id: str,
        workflow: str | None = None,
        configfile: str | None = None,
    ) -> JobRecord:
        """Create a new PENDING job and return its record."""
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
            workflow=workflow,
            configfile=configfile,
        )
        self._jobs[job_id] = record
        return record

    def cache_workflow_files(self, job_id: str, files: list[WorkflowFileInfo]) -> None:
        """Cache the workflow file listing on the job record."""
        record = self._jobs.get(job_id)
        if record is not None:
            record.workflow_files = files

    def get_job(self, job_id: str) -> JobRecord | None:
        """Return the JobRecord for job_id, or None."""
        return self._jobs.get(job_id)

    def push_log(self, job_id: str, line: str) -> None:
        """Append a log line to the in-memory buffer."""
        record = self._jobs.get(job_id)
        if record is not None:
            record.logs.append(line)
            record._unflushed_logs.append(line)
            record.total_log_lines += 1

    def get_logs(self, job_id: str, offset: int = 0) -> tuple[list[str], int]:
        """Return (lines, new_offset) from the given offset."""
        record = self._jobs.get(job_id)
        if record is None:
            return [], offset
        # in-memory buffer covers [buf_start, total_log_lines)
        buf_start = record.total_log_lines - len(record.logs)
        if offset < buf_start:
            # requested lines were already evicted — skip to what's available
            logger.warning(
                "Log lines %d–%d evicted for job %s, skipping to %d",
                offset,
                buf_start - 1,
                job_id,
                buf_start,
            )
            return list(record.logs), record.total_log_lines
        lines = list(record.logs)[offset - buf_start :]
        return lines, offset + len(lines)

    def get_logs_with_disk_fallback(
        self, job_id: str, offset: int = 0
    ) -> tuple[list[str], int]:
        """Like get_logs, but falls back to on-disk log if no in-memory lines."""
        lines, new_offset = self.get_logs(job_id, offset)
        if not lines:
            log_path = self.get_log_path(job_id)
            if log_path and log_path.exists():
                try:
                    disk_lines = log_path.read_text(encoding="utf-8").splitlines()
                    return disk_lines, len(disk_lines)
                except OSError:
                    logger.warning(
                        "Failed to read log file for job %s", job_id, exc_info=True
                    )
        return lines, new_offset

    def _mark_cancelled(self, job_id: str) -> None:
        """Set CANCELLED status if not already terminal. Does not cancel the task."""
        record = self._jobs.get(job_id)
        if record is None or record.status in TERMINAL_STATUSES:
            return
        record.status = JobStatus.CANCELLED
        record.completed_at = datetime.now(UTC)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a job: set CANCELLED status and cancel the asyncio task."""
        record = self._jobs.get(job_id)
        if record is None or record.status in TERMINAL_STATUSES:
            return
        self._mark_cancelled(job_id)
        if record.task and not record.task.done():
            record.task.cancel()

    def delete_job(self, job_id: str) -> None:
        """Remove job from store, cancel task, delete on-disk data."""
        record = self._jobs.pop(job_id, None)
        if record and record.task and not record.task.done():
            record.task.cancel()
        job_dir = self._job_dir(job_id)
        if job_dir is not None and job_dir.is_dir():
            try:
                shutil.rmtree(job_dir)
            except OSError:
                logger.warning(
                    "Failed to remove job directory %s", job_dir, exc_info=True
                )

    def clear_job_logs(self, job_id: str) -> None:
        """Drop in-memory log buffers; the disk log is retained."""
        record = self._jobs.get(job_id)
        if record is not None:
            record.logs.clear()
            record._unflushed_logs.clear()

    def all_jobs(self) -> dict[str, JobRecord]:
        """Return a shallow copy of all jobs."""
        return dict(self._jobs)
