from __future__ import annotations

import json
from datetime import UTC, datetime

from app.models import JobStatus
from app.store import JobStore


class TestJobStoreInMemory:
    """Verify existing in-memory behaviour still works."""

    def test_create_and_get(self):
        store = JobStore()
        rec = store.create_job("j1")
        assert rec.job_id == "j1"
        assert rec.status == JobStatus.PENDING
        assert store.get_job("j1") is rec

    def test_get_missing(self):
        store = JobStore()
        assert store.get_job("nope") is None

    def test_direct_status_mutation(self):
        store = JobStore()
        record = store.create_job("j1")
        record.status = JobStatus.RUNNING
        assert store.get_job("j1").status == JobStatus.RUNNING

    def test_delete_job(self):
        store = JobStore()
        store.create_job("j1")
        store.delete_job("j1")
        assert store.get_job("j1") is None


class TestPersist:
    def test_persist_creates_json(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j1")
        rec.work_dir = "/scratch/j1"
        store.persist(rec)

        job_json = tmp_path / "jobs" / "j1" / "job.json"
        assert job_json.exists()
        data = json.loads(job_json.read_text())
        assert data["job_id"] == "j1"
        assert data["status"] == "PENDING"
        assert data["work_dir"] == "/scratch/j1"
        # datetimes serialised as isoformat strings
        assert isinstance(data["created_at"], str)
        datetime.fromisoformat(data["created_at"])

    def test_persist_none_datetimes(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j1")
        store.persist(rec)

        data = json.loads((tmp_path / "jobs" / "j1" / "job.json").read_text())
        assert data["started_at"] is None
        assert data["completed_at"] is None

    def test_persist_noop_without_data_dir(self):
        store = JobStore()
        rec = store.create_job("j1")
        # Should not raise
        store.persist(rec)


class TestFlushLogsToDisk:
    def test_flushes_buffered_lines(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("j1")
        store.push_log("j1", "line one")
        store.push_log("j1", "line two")
        store.flush_logs_to_disk("j1")

        log_path = tmp_path / "jobs" / "j1" / "output.log"
        lines = log_path.read_text().splitlines()
        assert lines == ["line one", "line two"]

        # Buffer should be cleared after flush
        record = store.get_job("j1")
        assert record._unflushed_logs == []

    def test_noop_without_data_dir(self):
        store = JobStore()
        store.create_job("j1")
        store.push_log("j1", "hello")
        store.flush_logs_to_disk("j1")  # should not raise

    def test_noop_without_unflushed_logs(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("j1")
        store.flush_logs_to_disk("j1")  # nothing to flush
        log_path = tmp_path / "jobs" / "j1" / "output.log"
        assert not log_path.exists()


class TestGetLogsWithDiskFallback:
    def test_disk_fallback_at_offset_zero(self, tmp_path):
        """offset=0 with empty buffer activates the disk fallback."""
        store = JobStore(data_dir=tmp_path)
        store.create_job("j1")
        log_path = store.get_log_path("j1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("disk line 1\ndisk line 2\n")

        lines, new_offset = store.get_logs_with_disk_fallback("j1", 0)

        assert lines == ["disk line 1", "disk line 2"]
        assert new_offset == 2

    def test_nonzero_offset_with_empty_buffer_falls_back_to_disk(self, tmp_path):
        """Non-zero offset with empty buffer still falls back to disk."""
        store = JobStore(data_dir=tmp_path)
        store.create_job("j1")
        log_path = store.get_log_path("j1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("disk line 1\ndisk line 2\n")

        lines, new_offset = store.get_logs_with_disk_fallback("j1", 5)

        assert lines == ["disk line 1", "disk line 2"]
        assert new_offset == 2


class TestRestoreFromDisk:
    def test_round_trip(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j1")
        rec.status = "COMPLETED"
        rec.started_at = datetime(2025, 1, 1, tzinfo=UTC)
        rec.completed_at = datetime(2025, 1, 2, tzinfo=UTC)
        rec.exit_code = 0
        rec.work_dir = "/scratch/j1"
        rec.workflow = "wf"
        rec.git_ref = "main"
        rec.git_sha = "abc123"
        rec.workflow_files = [{"path": "a.txt", "size": 10}]
        store.persist(rec)

        store2 = JobStore(data_dir=tmp_path)
        store2.restore_from_disk()
        restored = store2.get_job("j1")
        assert restored is not None
        assert restored.status == "COMPLETED"
        assert restored.exit_code == 0
        assert restored.work_dir == "/scratch/j1"
        assert restored.started_at == datetime(2025, 1, 1, tzinfo=UTC)
        assert restored.completed_at == datetime(2025, 1, 2, tzinfo=UTC)
        assert restored.git_sha == "abc123"
        assert restored.workflow_files == [{"path": "a.txt", "size": 10}]

    def test_configfile_survives_round_trip(self, tmp_path):
        """configfile is in _PERSIST_FIELDS and survives persist + restore."""
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j-cf")
        rec.status = "COMPLETED"
        rec.configfile = "config/prod.yaml"
        store.persist(rec)

        store2 = JobStore(data_dir=tmp_path)
        store2.restore_from_disk()
        restored = store2.get_job("j-cf")
        assert restored is not None
        assert restored.configfile == "config/prod.yaml"

    def test_skips_corrupt_files(self, tmp_path):
        jobs_dir = tmp_path / "jobs" / "bad"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job.json").write_text("NOT JSON")

        store = JobStore(data_dir=tmp_path)
        store.restore_from_disk()
        assert store.all_jobs() == {}

    def test_skips_dirs_without_json(self, tmp_path):
        (tmp_path / "jobs" / "empty").mkdir(parents=True)
        store = JobStore(data_dir=tmp_path)
        store.restore_from_disk()
        assert store.all_jobs() == {}

    def test_noop_without_data_dir(self):
        store = JobStore()
        store.restore_from_disk()

    def test_noop_missing_jobs_dir(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.restore_from_disk()
        assert store.all_jobs() == {}

    def test_cancel_after_restore_sets_completed_at(self, tmp_path):
        """After restore_from_disk, task=None; cancel_job must set completed_at."""
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j-restart")
        store.mark_running("j-restart")
        store.persist(rec)

        store2 = JobStore(data_dir=tmp_path)
        store2.restore_from_disk()
        restored = store2.get_job("j-restart")
        assert restored is not None
        assert restored.task is None

        store2.cancel_job("j-restart")

        cancelled = store2.get_job("j-restart")
        assert cancelled.status == JobStatus.CANCELLED
        assert cancelled.completed_at is not None

    def test_handles_oserror_scanning_jobs_dir(self, tmp_path):
        """OSError while iterating jobs_root is logged and results in no jobs loaded."""
        from pathlib import Path
        from unittest.mock import patch

        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir(parents=True)

        store = JobStore(data_dir=tmp_path)
        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            store.restore_from_disk()
        assert store.all_jobs() == {}


class TestDeleteJobWithDisk:
    def test_removes_disk_dir(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        rec = store.create_job("j1")
        store.persist(rec)
        store.push_log("j1", "log line")
        store.flush_logs_to_disk("j1")
        job_dir = tmp_path / "jobs" / "j1"
        assert job_dir.exists()

        store.delete_job("j1")
        assert store.get_job("j1") is None
        assert not job_dir.exists()

    def test_delete_without_disk_dir(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("j1")
        # No persist, so no dir on disk
        store.delete_job("j1")
        assert store.get_job("j1") is None


class TestStateTransitionMethods:
    def test_mark_setup(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        store.mark_setup("j1", "/work/j1", "main", "abc123")
        rec = store.get_job("j1")
        assert rec.status == JobStatus.SETUP
        assert rec.work_dir == "/work/j1"
        assert rec.git_ref == "main"
        assert rec.git_sha == "abc123"

    def test_mark_running(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        store.mark_running("j1")
        rec = store.get_job("j1")
        assert rec.status == JobStatus.RUNNING
        assert rec.started_at is not None

    def test_mark_finished_zero_exit(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        store.mark_finished("j1", exit_code=0)
        rec = store.get_job("j1")
        assert rec.status == JobStatus.COMPLETED
        assert rec.exit_code == 0
        assert rec.completed_at is not None

    def test_mark_finished_nonzero_exit(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        store.mark_finished("j1", exit_code=1)
        rec = store.get_job("j1")
        assert rec.status == JobStatus.FAILED
        assert rec.exit_code == 1

    def test_mark_finished_sets_workflow_files(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        files = [{"path": "out.csv", "size": 42}]
        store.cache_workflow_files("j1", files)
        store.mark_finished("j1", exit_code=0)
        assert store.get_job("j1").workflow_files == files

    def test_mark_error(self):
        store = JobStore(data_dir=None)
        store.create_job("j1")
        store.mark_error("j1", "something went wrong")
        rec = store.get_job("j1")
        assert rec.status == JobStatus.ERROR
        assert rec.completed_at is not None
        assert "something went wrong" in rec.logs

    def test_cancel_job_with_no_task_sets_completed_at(self):
        # When task=None (e.g. after server restart), cancel_job sets completed_at
        # directly because the CancelledError handler in execute_job will never run.
        store = JobStore(data_dir=None)
        rec = store.create_job("j1")
        assert rec.task is None
        store.cancel_job("j1")
        assert rec.status == JobStatus.CANCELLED
        assert rec.completed_at is not None

    def test_state_methods_noop_for_unknown_job(self):
        store = JobStore(data_dir=None)
        store.mark_setup("unknown", "/work", None, None)
        store.mark_running("unknown")
        store.mark_finished("unknown", 0)
        store.mark_error("unknown", "err")


def test_persist_includes_expected_fields(tmp_path):
    """persist() writes all semantically important fields to job.json."""
    store = JobStore(data_dir=tmp_path)
    rec = store.create_job("j1")
    rec.work_dir = "/scratch/j1"
    rec.git_ref = "main"
    rec.git_sha = "abc123"
    store.persist(rec)
    data = json.loads((tmp_path / "jobs" / "j1" / "job.json").read_text())
    for field_name in (
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
    ):
        assert field_name in data, (
            f"Expected field '{field_name}' missing from persisted job.json"
        )
