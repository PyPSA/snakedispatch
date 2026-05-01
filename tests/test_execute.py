from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import JobStatus
from app.store import JobStore
from app.tasks import (
    ExecuteJobParams,
    _finalize_job,
    _sync_snkmt_counts,
    _try_cache_workflow_files,
    _try_save_cache,
    execute_job,
    gc_loop,
    sync_job_data_loop,
)


class TestExecuteJob:
    async def test_successful_job(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-success")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-success",
            workflow="https://github.com/org/repo.git@v1.0",
            params=ExecuteJobParams(
                configfile="config.yaml",
                snakemake_args=["--profile", "slurm"],
                extra_files={"gurobi.lic": "TOKENSERVER=example.com"},
            ),
        )

        record = store.get_job("test-success")
        assert record.status == "COMPLETED"
        assert record.exit_code == 0
        assert record.work_dir is not None
        assert record.started_at is not None
        assert record.completed_at is not None
        assert list(record.logs) == []  # logs cleared after persist

        mock_backend.prepare.assert_called_once_with(
            "test-success", "https://github.com/org/repo.git@v1.0", None
        )
        mock_backend.setup.assert_called_once()
        mock_backend.launch.assert_called_once_with(
            "test-success",
            "/scratch/test/jobs/test-job-id",
            "config.yaml",
            ["--profile", "slurm"],
            None,
        )
        mock_backend.monitor.assert_called_once()

    async def test_failed_job_nonzero_exit(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-fail")
        mock_backend.monitor = AsyncMock(return_value=1)

        await execute_job(
            store,
            mock_backend,
            "test-fail",
            "https://example.com/repo.git@main",
        )

        record = store.get_job("test-fail")
        assert record.status == "FAILED"
        assert record.exit_code == 1

    async def test_exception_during_clone(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-clone-fail")
        mock_backend.prepare = AsyncMock(side_effect=RuntimeError("clone failed"))

        with pytest.raises(RuntimeError, match="clone failed"):
            await execute_job(
                store,
                mock_backend,
                "test-clone-fail",
                "https://example.com/repo.git@main",
            )

        record = store.get_job("test-clone-fail")
        assert record.status == "ERROR"
        assert record.completed_at is not None
        assert list(record.logs) == []  # logs cleared after persist
        # Error message persisted to disk log
        log_file = tmp_path / "jobs" / "test-clone-fail" / "output.log"
        assert log_file.exists()
        assert "clone failed" in log_file.read_text()

    async def test_exception_during_launch(self, mock_backend, tmp_path):
        mock_backend.launch = AsyncMock(side_effect=RuntimeError(".pid never appeared"))
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-launch-fail")

        with pytest.raises(RuntimeError, match=".pid never appeared"):
            await execute_job(
                store,
                mock_backend,
                "test-launch-fail",
                "https://example.com/repo.git@main",
            )

        record = store.get_job("test-launch-fail")
        assert record.status == "ERROR"

    async def test_minimal_args(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-minimal")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-minimal",
            "https://example.com/repo.git",
        )

        record = store.get_job("test-minimal")
        assert record.status == "COMPLETED"
        mock_backend.launch.assert_called_once_with(
            "test-minimal",
            "/scratch/test/jobs/test-job-id",
            None,
            None,
            None,
        )

    async def test_cache_restore_and_save(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-cache")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-cache",
            "https://example.com/repo.git@main",
            ExecuteJobParams(cache_key="pypsa-eur", cache_dirs=["data", "resources"]),
        )

        record = store.get_job("test-cache")
        assert record.status == "COMPLETED"
        mock_backend.restore_cache.assert_called_once_with(
            "test-cache",
            "/scratch/test/jobs/test-job-id",
            "pypsa-eur",
            ["data", "resources"],
        )
        mock_backend.save_cache.assert_called_once_with(
            "test-cache",
            "/scratch/test/jobs/test-job-id",
            "pypsa-eur",
            ["data", "resources"],
        )

    async def test_cache_not_saved_on_failure(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-cache-fail")
        mock_backend.monitor = AsyncMock(return_value=1)

        await execute_job(
            store,
            mock_backend,
            "test-cache-fail",
            "https://example.com/repo.git@main",
            ExecuteJobParams(cache_key="pypsa-eur", cache_dirs=["data"]),
        )

        record = store.get_job("test-cache-fail")
        assert record.status == "FAILED"
        mock_backend.restore_cache.assert_called_once()
        mock_backend.save_cache.assert_not_called()

    async def test_no_cache_without_key(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-no-cache")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-no-cache",
            "https://example.com/repo.git",
        )

        assert store.get_job("test-no-cache").status == "COMPLETED"
        mock_backend.restore_cache.assert_not_called()
        mock_backend.save_cache.assert_not_called()

    async def test_cache_save_failure_non_fatal(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-cache-err")
        mock_backend.monitor = AsyncMock(return_value=0)
        mock_backend.save_cache = AsyncMock(side_effect=RuntimeError("rsync failed"))

        await execute_job(
            store,
            mock_backend,
            "test-cache-err",
            "https://example.com/repo.git@main",
            ExecuteJobParams(cache_key="pypsa-eur", cache_dirs=["data"]),
        )

        record = store.get_job("test-cache-err")
        assert record.status == "COMPLETED"
        assert record.exit_code == 0

    @pytest.mark.parametrize(
        ("cache_key", "cache_dirs"),
        [("pypsa-eur", None), (None, ["data"])],
        ids=["key_only", "dirs_only"],
    )
    async def test_incomplete_cache_config_skips_restore_and_save(
        self, mock_backend, tmp_path, cache_key, cache_dirs
    ):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-incomplete-cache")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-incomplete-cache",
            "https://example.com/repo.git",
            ExecuteJobParams(cache_key=cache_key, cache_dirs=cache_dirs),
        )

        mock_backend.restore_cache.assert_not_called()
        mock_backend.save_cache.assert_not_called()

    async def test_local_path_source(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-local")
        mock_backend.monitor = AsyncMock(return_value=0)

        await execute_job(
            store,
            mock_backend,
            "test-local",
            "/mnt/workflows/pypsa-eur",
        )

        record = store.get_job("test-local")
        assert record.status == "COMPLETED"
        mock_backend.prepare.assert_called_once_with(
            "test-local", "/mnt/workflows/pypsa-eur", None
        )

    async def test_cancelled_error_sets_cancelled_status(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-cancel-err")
        mock_backend.monitor = AsyncMock(side_effect=asyncio.CancelledError())

        with contextlib.suppress(asyncio.CancelledError):
            await execute_job(
                store,
                mock_backend,
                "test-cancel-err",
                "https://example.com/repo.git@main",
            )

        record = store.get_job("test-cancel-err")
        assert record.status == JobStatus.CANCELLED
        assert record.completed_at is not None

    async def test_cancelled_error_already_cancelled_in_store(
        self, mock_backend, tmp_path
    ):
        store = JobStore(data_dir=tmp_path)
        store.create_job("test-cancel-pre")
        mock_backend.monitor = AsyncMock(side_effect=asyncio.CancelledError())

        # Simulate cancel_job being called before monitor returns
        async def _cancel_then_raise(*_args, **_kwargs):
            store.cancel_job("test-cancel-pre")
            raise asyncio.CancelledError()

        mock_backend.monitor = AsyncMock(side_effect=_cancel_then_raise)

        with contextlib.suppress(asyncio.CancelledError):
            await execute_job(
                store,
                mock_backend,
                "test-cancel-pre",
                "https://example.com/repo.git@main",
            )

        record = store.get_job("test-cancel-pre")
        assert record.status == JobStatus.CANCELLED


async def _start_failing_loop(store, coro_fn, *args, **kwargs) -> asyncio.Task:
    """Inject a failing all_jobs and run the loop until it stops."""
    _real_sleep = asyncio.sleep

    def _failing_all_jobs():
        raise RuntimeError("store exploded")

    store.all_jobs = _failing_all_jobs  # type: ignore[method-assign]

    async def _instant_sleep(*a, **kw):
        await _real_sleep(0)

    with patch("asyncio.sleep", side_effect=_instant_sleep):
        task = asyncio.create_task(coro_fn(*args, **kwargs))
        for _ in range(30):
            await _real_sleep(0)
    return task


async def _run_one_iteration(coro_fn, *args, **kwargs) -> asyncio.Task:
    """Run one iteration of an infinite-loop task, then cancel it."""
    _real_sleep = asyncio.sleep
    store = args[0]  # store is the first arg for both gc_loop and sync_job_data_loop
    iter_count = 0
    _real_all_jobs = store.all_jobs

    def _counting_all_jobs():
        nonlocal iter_count
        iter_count += 1
        return _real_all_jobs()

    async def _sleep_zero(*a, **kw):
        await _real_sleep(0)

    store.all_jobs = _counting_all_jobs
    try:
        with patch("asyncio.sleep", side_effect=_sleep_zero):
            task = asyncio.create_task(coro_fn(*args, **kwargs))
            # Wait for iter_count >= 2: iter 1 start (1) + iter 2 start (2)
            while iter_count < 2:
                await _real_sleep(0)
            await _real_sleep(0)  # Let the current body settle before cancelling
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        store.all_jobs = _real_all_jobs
    return task


class TestSyncJobData:
    async def test_flushes_logs_for_running_jobs(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("sync-job")
        store.mark_setup("sync-job", "/some/work/dir", None, None)
        store.mark_running("sync-job")
        store.push_log("sync-job", "test log line")

        log_path = store.get_log_path("sync-job")
        assert not log_path.exists()

        await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        assert log_path.exists()
        assert "test log line" in log_path.read_text()

    async def test_syncs_snkmt_db_for_running_jobs(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("sync-snkmt")
        store.mark_setup("sync-snkmt", "/some/work/dir", None, None)
        store.mark_running("sync-snkmt")

        await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        mock_backend.sync_snkmt_db.assert_called()

    async def test_skips_non_running_jobs(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("sync-skip")
        store.mark_finished("sync-skip", 0)
        store.push_log("sync-skip", "some line")

        await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        mock_backend.sync_snkmt_db.assert_not_called()

    async def test_flush_logs_failure_is_swallowed(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("sync-flush-err")
        store.mark_setup("sync-flush-err", "/some/work/dir", None, None)
        store.mark_running("sync-flush-err")

        with patch.object(
            store, "flush_logs_to_disk", side_effect=OSError("disk full")
        ):
            await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        # Loop should have continued, snkmt sync still attempted
        mock_backend.sync_snkmt_db.assert_called()

    async def test_recovers_stuck_job(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("stuck-job")
        store.mark_setup("stuck-job", "/some/work/dir", None, None)
        store.mark_running("stuck-job")

        record = store.get_job("stuck-job")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        record.task = mock_task

        # Backend reports job finished with exit code 1
        mock_backend.check_job_status = AsyncMock(return_value=1)

        await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        assert record.status == JobStatus.FAILED
        assert record.exit_code == 1
        mock_task.cancel.assert_called_once()

    async def test_check_job_status_failure_is_swallowed(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("check-err")
        store.mark_setup("check-err", "/some/work/dir", None, None)
        store.mark_running("check-err")

        mock_backend.check_job_status = AsyncMock(
            side_effect=RuntimeError("ssh failed")
        )

        # Should not crash — error is swallowed
        await _run_one_iteration(sync_job_data_loop, store, mock_backend, 0)

        # Job should still be running (not recovered)
        record = store.get_job("check-err")
        assert record.status == JobStatus.RUNNING

    async def test_five_consecutive_errors_stops_loop(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        task = await _start_failing_loop(
            store, sync_job_data_loop, store, mock_backend, 0
        )
        with pytest.raises(RuntimeError, match="stopped after"):
            await task


class TestGcLoop:
    async def test_removes_old_jobs(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        old = store.create_job("gc-old")
        store.mark_finished("gc-old", 0)
        old.work_dir = "/work/old"
        old.completed_at = datetime.now(UTC) - timedelta(hours=25)

        recent = store.create_job("gc-recent")
        store.mark_finished("gc-recent", 0)
        recent.completed_at = datetime.now(UTC) - timedelta(hours=1)

        await _run_one_iteration(gc_loop, store, mock_backend, max_age_hours=24)

        assert store.get_job("gc-old") is None
        assert store.get_job("gc-recent") is not None
        mock_backend.cleanup.assert_called_once_with("gc-old", "/work/old")

    async def test_cleanup_failure_does_not_prevent_job_removal(
        self, mock_backend, tmp_path
    ):
        store = JobStore(data_dir=tmp_path)
        old = store.create_job("gc-cleanup-err")
        store.mark_finished("gc-cleanup-err", 0)
        old.work_dir = "/work/old"
        old.completed_at = datetime.now(UTC) - timedelta(hours=25)

        mock_backend.cleanup = AsyncMock(side_effect=RuntimeError("ssh failed"))

        await _run_one_iteration(gc_loop, store, mock_backend, max_age_hours=24)

        # Job should still be removed despite cleanup failure
        assert store.get_job("gc-cleanup-err") is None

    async def test_keeps_recent_jobs(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        recent = store.create_job("gc-keep")
        store.mark_finished("gc-keep", 0)
        recent.completed_at = datetime.now(UTC) - timedelta(hours=10)

        await _run_one_iteration(gc_loop, store, mock_backend, max_age_hours=24)

        assert store.get_job("gc-keep") is not None
        mock_backend.cleanup.assert_not_called()

    async def test_running_jobs_not_gc_even_when_old(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        running = store.create_job("gc-running")
        store.mark_setup("gc-running", "/work/running", None, None)
        store.mark_running("gc-running")
        running.created_at = datetime.now(UTC) - timedelta(hours=200)

        await _run_one_iteration(gc_loop, store, mock_backend, max_age_hours=24)

        assert store.get_job("gc-running") is not None
        mock_backend.cleanup.assert_not_called()

    async def test_five_consecutive_errors_stops_loop(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        task = await _start_failing_loop(
            store, gc_loop, store, mock_backend, max_age_hours=24
        )
        with pytest.raises(RuntimeError, match="stopped after"):
            await task


class TestConcurrentJobs:
    async def test_concurrent_jobs_no_state_interference(self, tmp_path):
        store = JobStore(data_dir=tmp_path)
        job_ids = ["job-a", "job-b", "job-c"]
        for jid in job_ids:
            store.create_job(jid)

        backend = AsyncMock()
        backend.setup = AsyncMock()
        backend.launch = AsyncMock()
        backend.list_workflow_files = AsyncMock(return_value=[])
        backend.sync_snkmt_db = AsyncMock()

        async def fake_prepare(job_id, workflow, git_ref=None):
            work_dir = f"/scratch/{job_id}"
            await asyncio.sleep(0)  # Yield to event loop to allow task interleaving
            return work_dir, "main", f"sha-{job_id}"

        async def fake_monitor(job_id, work_dir, log_callback, byte_offset=0):
            log_callback(f"log from {job_id}")
            await asyncio.sleep(0)  # Yield to event loop to allow task interleaving
            return 0

        backend.prepare = AsyncMock(side_effect=fake_prepare)
        backend.monitor = AsyncMock(side_effect=fake_monitor)

        tasks = [
            asyncio.create_task(
                execute_job(store, backend, jid, f"https://example.com/{jid}.git")
            )
            for jid in job_ids
        ]
        await asyncio.gather(*tasks)

        for jid in job_ids:
            record = store.get_job(jid)
            assert record.status == JobStatus.COMPLETED
            assert record.exit_code == 0
            assert record.work_dir == f"/scratch/{jid}"
            assert record.git_sha == f"sha-{jid}"


class TestFinalizeJob:
    async def test_flushes_logs_to_disk(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-flush")
        store.mark_finished("finalize-flush", 0)
        record = store.get_job("finalize-flush")
        store.push_log("finalize-flush", "important log line")

        await _finalize_job(store, mock_backend, record)

        log_path = store.get_log_path("finalize-flush")
        assert log_path.exists()
        assert "important log line" in log_path.read_text()

    async def test_clears_in_memory_logs_after_flush(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-clear")
        store.mark_finished("finalize-clear", 0)
        record = store.get_job("finalize-clear")
        store.push_log("finalize-clear", "line 1")
        store.push_log("finalize-clear", "line 2")

        await _finalize_job(store, mock_backend, record)

        assert list(record.logs) == []

    async def test_calls_snkmt_sync_when_work_dir_set(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-sync")
        store.mark_setup("finalize-sync", "/work/finalize-sync", None, None)
        store.mark_finished("finalize-sync", 0)
        record = store.get_job("finalize-sync")

        await _finalize_job(store, mock_backend, record)

        mock_backend.sync_snkmt_db.assert_called_once()

    async def test_skips_snkmt_sync_when_cancelled(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-cancel")
        store.mark_setup("finalize-cancel", "/work/finalize-cancel", None, None)
        store.cancel_job("finalize-cancel")
        record = store.get_job("finalize-cancel")

        await _finalize_job(store, mock_backend, record, skip_snkmt_sync=True)

        mock_backend.sync_snkmt_db.assert_not_called()

    async def test_persists_record(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-persist")
        store.mark_finished("finalize-persist", 0)
        record = store.get_job("finalize-persist")

        await _finalize_job(store, mock_backend, record)

        job_file = tmp_path / "jobs" / "finalize-persist" / "job.json"
        assert job_file.exists()

    async def test_sync_error_is_swallowed(self, mock_backend, tmp_path):
        store = JobStore(data_dir=tmp_path)
        store.create_job("finalize-sync-err")
        store.mark_setup("finalize-sync-err", "/work/dir", None, None)
        store.mark_finished("finalize-sync-err", 0)
        record = store.get_job("finalize-sync-err")
        mock_backend.sync_snkmt_db = AsyncMock(side_effect=RuntimeError("sync failed"))

        await _finalize_job(store, mock_backend, record)

        job_file = tmp_path / "jobs" / "finalize-sync-err" / "job.json"
        assert job_file.exists()


class TestSyncSnkmtCounts:
    async def test_no_update_when_snkmt_db_has_no_workflow(self, tmp_path):
        import sqlite3 as _sqlite3

        snkmt_path = tmp_path / "snkmt.db"
        # Create schema-only DB with no rows
        conn = _sqlite3.connect(str(snkmt_path))
        conn.execute(
            "CREATE TABLE workflows ("
            "id TEXT PRIMARY KEY, "
            "total_job_count INTEGER, "
            "jobs_finished INTEGER)"
        )
        conn.commit()
        conn.close()

        store = JobStore(data_dir=tmp_path)
        store.create_job("empty-db-test")
        store.mark_setup("empty-db-test", str(tmp_path), None, None)
        record = store.get_job("empty-db-test")
        assert record.total_job_count is None

        mock_backend = AsyncMock()
        mock_backend.sync_snkmt_db = AsyncMock(return_value=None)

        await _sync_snkmt_counts(
            mock_backend, "empty-db-test", str(tmp_path), snkmt_path, record
        )

        assert record.total_job_count is None
        assert record.jobs_finished is None

    async def test_updates_count_fields_from_real_db(self, tmp_path):
        from tests.conftest import create_snkmt_db

        snkmt_path = tmp_path / "snkmt.db"
        create_snkmt_db(snkmt_path)

        store = JobStore(data_dir=tmp_path)
        store.create_job("count-test")
        store.mark_setup("count-test", str(tmp_path), None, None)
        record = store.get_job("count-test")
        assert record.total_job_count is None
        assert record.jobs_finished is None

        mock_backend = AsyncMock()
        # sync_snkmt_db does nothing, snkmt.db already exists at snkmt_path
        mock_backend.sync_snkmt_db = AsyncMock(return_value=None)

        await _sync_snkmt_counts(
            mock_backend, "count-test", str(tmp_path), snkmt_path, record
        )

        assert record.total_job_count == 3
        assert record.jobs_finished == 3


class TestTryCacheWorkflowFiles:
    @pytest.mark.parametrize("exc", [RuntimeError("ssh failed"), OSError("refused")])
    async def test_swallows_errors(self, mock_backend, exc):
        mock_backend.list_workflow_files = AsyncMock(side_effect=exc)
        result = await _try_cache_workflow_files(mock_backend, "test-job", "/work/dir")
        assert result is None


class TestTrySaveCache:
    @pytest.mark.parametrize(
        "exc", [RuntimeError("rsync failed"), OSError("disk full")]
    )
    async def test_swallows_errors(self, mock_backend, exc):
        mock_backend.save_cache = AsyncMock(side_effect=exc)
        await _try_save_cache(mock_backend, "test-job", "/work/dir", "key", ["data"])
