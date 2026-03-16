from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

from app.main import app, create_backend, lifespan
from tests.conftest import SAMPLE_WORKFLOW_ID, create_snkmt_db


def _setup_snkmt(store, job_id: str) -> None:
    """Create snkmt.db in the job's local directory."""
    db_path = store.get_snkmt_db_path(job_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_snkmt_db(db_path)


class TestCreateJob:
    async def test_returns_201(self, async_client, store):
        response = await async_client.post(
            "/jobs",
            json={
                "workflow": "https://github.com/org/repo.git",
                "git_ref": "v1.0.0",
                "configfile": "config/config.yaml",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"
        assert "created_at" in data

    async def test_url_without_ref(self, async_client, store):
        response = await async_client.post(
            "/jobs",
            json={"workflow": "https://github.com/org/repo.git"},
        )
        assert response.status_code == 201

    async def test_invalid_local_path_returns_422(self, async_client):
        response = await async_client.post(
            "/jobs",
            json={"workflow": "/nonexistent/path/to/workflow"},
        )
        assert response.status_code == 422

    async def test_spawns_background_task(self, async_client, store):
        response = await async_client.post(
            "/jobs",
            json={
                "workflow": "https://github.com/org/repo.git",
                "git_ref": "main",
            },
        )
        assert response.status_code == 201
        job_id = response.json()["job_id"]
        record = store.get_job(job_id)
        assert record is not None
        assert record.task is not None

    async def test_with_extra_files(self, async_client, store):
        response = await async_client.post(
            "/jobs",
            json={
                "workflow": "https://github.com/org/repo.git",
                "git_ref": "v2.0",
                "extra_files": {"gurobi.lic": "TOKENSERVER=license.example.com"},
            },
        )
        assert response.status_code == 201

    async def test_local_path_source(self, async_client, store, tmp_path):
        response = await async_client.post(
            "/jobs",
            json={"workflow": str(tmp_path)},
        )
        assert response.status_code == 201


class TestListJobs:
    async def test_empty_store(self, async_client):
        response = await async_client.get("/jobs")
        assert response.status_code == 200
        assert response.json() == {"jobs": [], "total": 0}

    async def test_multiple_jobs_newest_first(self, async_client, store):
        r1 = store.create_job("job-old")
        store.mark_finished("job-old", 0)
        r1.created_at = datetime.fromisoformat("2025-01-01T00:00:00+00:00")

        r2 = store.create_job("job-new")
        store.mark_running("job-new")
        r2.created_at = datetime.fromisoformat("2025-06-01T00:00:00+00:00")

        response = await async_client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == "job-new"
        assert data["jobs"][1]["job_id"] == "job-old"

    async def test_includes_all_fields(self, async_client, store):
        r = store.create_job("job-full")
        store.mark_finished("job-full", 0)
        r.started_at = datetime.fromisoformat("2025-02-02T10:30:05+00:00")
        r.completed_at = datetime.fromisoformat("2025-02-02T11:00:00+00:00")

        response = await async_client.get("/jobs")
        data = response.json()
        job = data["jobs"][0]
        assert job["status"] == "COMPLETED"
        assert job["exit_code"] == 0
        assert job["started_at"] is not None
        assert job["completed_at"] is not None


class TestGetJob:
    async def test_returns_status(self, async_client, store):
        job_id = "test-job-123"
        record = store.create_job(job_id)
        store.mark_running(job_id)
        record.created_at = datetime.fromisoformat("2025-02-02T10:30:00+00:00")
        record.started_at = datetime.fromisoformat("2025-02-02T10:30:05+00:00")

        response = await async_client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RUNNING"
        assert data["exit_code"] is None

    async def test_completed_job(self, async_client, store):
        job_id = "test-job-done"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record = store.get_job(job_id)
        record.completed_at = datetime.fromisoformat("2025-02-02T11:45:30+00:00")

        response = await async_client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert data["exit_code"] == 0

    async def test_nonexistent_returns_404(self, async_client):
        response = await async_client.get("/jobs/nonexistent")
        assert response.status_code == 404


class TestListOutputs:
    async def test_list_outputs(self, async_client, store, mock_backend):
        job_id = "test-job-456"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/test-job-456"

        response = await async_client.get(f"/jobs/{job_id}/outputs")
        assert response.status_code == 200
        data = response.json()
        assert len(data["files"]) == 1
        assert data["files"][0]["path"] == "results/output.csv"
        assert data["files"][0]["size"] == 1234

    async def test_before_running_returns_409(self, async_client, store):
        job_id = "test-job-early"
        store.create_job(job_id)
        store.mark_setup(job_id, "/tmp", None, None)

        response = await async_client.get(f"/jobs/{job_id}/outputs")
        assert response.status_code == 409


class TestDownloadOutput:
    async def test_rejects_non_results_path(self, async_client, store):
        job_id = "test-job-dl"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/dl"

        response = await async_client.get(f"/jobs/{job_id}/outputs/.snakemake/log")
        assert response.status_code == 400

    async def test_rejects_path_traversal(self, async_client, store):
        job_id = "test-job-traversal"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/traversal"

        response = await async_client.get(
            f"/jobs/{job_id}/outputs/results/%2e%2e/%2e%2e/etc/passwd"
        )
        assert response.status_code == 400

    async def test_returns_404_for_file_not_in_cached_listing(
        self, async_client, store
    ):
        job_id = "test-job-cached-404"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/cached"
        record.workflow_files = [{"path": "results/output.txt", "size": 100}]

        response = await async_client.get(
            f"/jobs/{job_id}/outputs/results/nonexistent.txt"
        )
        assert response.status_code == 404


class TestDeleteJob:
    async def test_cleans_up(self, async_client, store, mock_backend):
        job_id = "test-job-789"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/test-job-789"

        response = await async_client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cleaned"

        # Let background cleanup task run
        await asyncio.sleep(0)

        mock_backend.cleanup.assert_called_once_with(
            job_id, "/scratch/test/jobs/test-job-789"
        )
        assert store.get_job(job_id) is None

    async def test_delete_running_job(self, async_client, store, mock_backend):
        from unittest.mock import MagicMock

        job_id = "test-job-delete-running"
        record = store.create_job(job_id)
        store.mark_running(job_id)
        record.work_dir = "/scratch/test/jobs/test-job-delete-running"
        mock_task = MagicMock()
        mock_task.done.return_value = False
        record.task = mock_task

        response = await async_client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cleaned"

        mock_task.cancel.assert_called()
        assert store.get_job(job_id) is None

    async def test_nonexistent_returns_404(self, async_client):
        response = await async_client.delete("/jobs/nonexistent")
        assert response.status_code == 404


class TestCancelJob:
    async def test_cancel_running_job(self, async_client, store, mock_backend):
        job_id = "test-cancel-run"
        record = store.create_job(job_id)
        store.mark_running(job_id)
        record.work_dir = "/scratch/test/jobs/test-cancel-run"

        response = await async_client.post(f"/jobs/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "CANCELLED"
        assert data["job_id"] == job_id

    async def test_cancel_completed_job_returns_409(self, async_client, store):
        job_id = "test-cancel-done"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)

        response = await async_client.post(f"/jobs/{job_id}/cancel")
        assert response.status_code == 409

    async def test_cancel_nonexistent_returns_404(self, async_client):
        response = await async_client.post("/jobs/nonexistent/cancel")
        assert response.status_code == 404

    async def test_cancel_calls_backend_cleanup(
        self, async_client, store, mock_backend
    ):
        job_id = "test-cancel-cleanup"
        record = store.create_job(job_id)
        store.mark_running(job_id)
        record.work_dir = "/scratch/test/jobs/cleanup-test"

        await async_client.post(f"/jobs/{job_id}/cancel")

        # Let background cleanup task run
        await asyncio.sleep(0)

        mock_backend.cleanup.assert_called_once_with(
            job_id, "/scratch/test/jobs/cleanup-test"
        )


class TestSseStreaming:
    async def test_returns_event_stream(self, async_client, store):
        job_id = "test-sse"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)

        async with async_client.stream("GET", f"/jobs/{job_id}/logs") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    async def test_streams_log_lines(self, async_client, store):
        job_id = "test-sse-logs"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        store.push_log(job_id, "hello from snakemake")
        store.push_log(job_id, "second line")

        async with async_client.stream("GET", f"/jobs/{job_id}/logs") as response:
            content = await response.aread()

        assert b"hello from snakemake" in content
        assert b"second line" in content

    async def test_nonexistent_job_returns_404(self, async_client):
        response = await async_client.get("/jobs/nonexistent/logs")
        assert response.status_code == 404

    async def test_disk_fallback_when_memory_empty(self, async_client, store, tmp_path):
        job_id = "test-sse-disk"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        # Write logs to disk but don't push to in-memory buffer
        log_path = store.get_log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("disk line 1\ndisk line 2\n")

        async with async_client.stream("GET", f"/jobs/{job_id}/logs") as response:
            content = await response.aread()

        assert b"disk line 1" in content
        assert b"disk line 2" in content

    async def test_live_transition_streams_all_lines(self, async_client, store):
        job_id = "test-sse-live"
        store.create_job(job_id)
        store.mark_running(job_id)
        store.push_log(job_id, "early line")

        # Event-based coordination: finish only after the SSE loop has consumed the
        # early line, avoiding timing fragility from asyncio.sleep.
        logs_consumed = asyncio.Event()
        original_get_logs = store.get_logs

        def _get_logs_and_signal(
            job_id_arg: str, offset: int = 0
        ) -> tuple[list[str], int]:
            lines, new_offset = original_get_logs(job_id_arg, offset)
            if lines and job_id_arg == job_id:
                logs_consumed.set()
            return lines, new_offset

        store.get_logs = _get_logs_and_signal  # type: ignore[method-assign]

        async def finish_when_consumed() -> None:
            await asyncio.wait_for(logs_consumed.wait(), timeout=5.0)
            store.push_log(job_id, "late line")
            store.mark_finished(job_id, 0)

        finish_task = asyncio.create_task(finish_when_consumed())
        try:
            async with async_client.stream("GET", f"/jobs/{job_id}/logs") as response:
                content = await response.aread()
        finally:
            store.get_logs = original_get_logs  # type: ignore[method-assign]

        await finish_task
        assert b"early line" in content
        assert b"late line" in content
        assert b"done" in content

    async def test_log_eviction_streams_available(self, async_client, store):
        job_id = "test-sse-evict"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        # Push more lines than MAX_IN_MEMORY_LINES to trigger eviction
        from app.store import MAX_IN_MEMORY_LINES

        for i in range(MAX_IN_MEMORY_LINES + 100):
            store.push_log(job_id, f"line-{i}")

        # Request from offset 0, evicted lines are skipped
        lines, new_offset = store.get_logs(job_id, 0)
        assert len(lines) == MAX_IN_MEMORY_LINES
        assert new_offset == MAX_IN_MEMORY_LINES + 100


class TestHealth:
    async def test_returns_status(self, async_client, mock_backend):
        mock_backend.check_connectivity = AsyncMock(return_value=True)
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["backend"] is True
        assert "redis" not in data


class TestHealthCached:
    async def test_returns_200_on_first_call(self, async_client, mock_backend):
        mock_backend.check_connectivity = AsyncMock(return_value=True)
        response = await async_client.get("/health/cached")
        assert response.status_code == 200
        data = response.json()
        assert data["backend"] is True
        assert data["status"] == "ok"
        mock_backend.check_connectivity.assert_called_once()

    async def test_second_call_within_ttl_uses_cache(
        self, async_client, mock_backend, settings
    ):
        settings.HEALTH_CACHE_TTL_SECONDS = 300

        mock_backend.check_connectivity = AsyncMock(return_value=True)
        await async_client.get("/health/cached")
        await async_client.get("/health/cached")
        mock_backend.check_connectivity.assert_called_once()

    async def test_call_after_ttl_rechecks(self, async_client, mock_backend, settings):
        settings.HEALTH_CACHE_TTL_SECONDS = 300

        mock_backend.check_connectivity = AsyncMock(return_value=True)
        await async_client.get("/health/cached")
        assert mock_backend.check_connectivity.call_count == 1

        # Simulate TTL expiry by backdating checked_at
        app.state.app.health_cache["checked_at"] -= 301

        await async_client.get("/health/cached")
        assert mock_backend.check_connectivity.call_count == 2


class TestWorkflowEndpoint:
    async def test_returns_workflow(self, async_client, store):
        job_id = "test-wf-job"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == SAMPLE_WORKFLOW_ID
        assert data["status"] == "finished"
        assert data["total_job_count"] == 3
        assert len(data["rules"]) == 2
        assert len(data["errors"]) == 1
        assert data["rulegraph"] is not None

    async def test_rules_contain_nested_jobs(self, async_client, store):
        job_id = "test-wf-nested"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        data = response.json()
        rules_by_name = {r["name"]: r for r in data["rules"]}

        # rule_a has 2 jobs, rule_b has 1
        assert len(rules_by_name["rule_a"]["jobs"]) == 2
        assert len(rules_by_name["rule_b"]["jobs"]) == 1

    async def test_jobs_contain_nested_files(self, async_client, store):
        job_id = "test-wf-files"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        data = response.json()
        rules_by_name = {r["name"]: r for r in data["rules"]}

        # rule_a job 0 (snakemake_id=0) has 2 files (INPUT + OUTPUT)
        rule_a_jobs = rules_by_name["rule_a"]["jobs"]
        job_0 = next(j for j in rule_a_jobs if j["snakemake_id"] == 0)
        assert len(job_0["files"]) == 2
        file_types = {f["file_type"] for f in job_0["files"]}
        assert file_types == {"INPUT", "OUTPUT"}

        # rule_a job 1 (snakemake_id=1) has no files
        job_1 = next(j for j in rule_a_jobs if j["snakemake_id"] == 1)
        assert len(job_1["files"]) == 0

        # rule_b job (snakemake_id=2) has 1 file
        rule_b_jobs = rules_by_name["rule_b"]["jobs"]
        assert len(rule_b_jobs[0]["files"]) == 1

    async def test_404_when_no_snkmt_db(self, async_client, store):
        job_id = "test-no-wf"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 404

    async def test_409_for_pending_job(self, async_client, store):
        job_id = "test-pending"
        store.create_job(job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 409

    async def test_409_for_running_job_with_unsynced_db(self, async_client, store):
        job_id = "test-running-unsynced"
        store.create_job(job_id)
        store.mark_running(job_id)
        # Do NOT create snkmt.db, simulates the not-yet-synced state

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 409

    async def test_502_for_corrupt_snkmt_db(self, async_client, store):
        job_id = "test-corrupt-db"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        db_path = store.get_snkmt_db_path(job_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"not a sqlite database")

        response = await async_client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 502
        assert "corrupt" in response.json()["detail"]

    async def test_500_when_persistence_not_configured(
        self, async_client, mock_backend, settings
    ):
        from httpx import ASGITransport, AsyncClient

        from app.backends.local import LocalConfig
        from app.deps import AppState
        from app.store import JobStore

        no_persist_store = JobStore()  # data_dir=None
        job_id = "test-no-persist"
        rec = no_persist_store.create_job(job_id)
        rec.status = "COMPLETED"

        backend_config = LocalConfig()
        app.state.app = AppState(
            store=no_persist_store,
            backend=mock_backend,
            settings=settings,
            health_cache={"backend_ok": None, "checked_at": 0.0},
            default_snakemake_args=backend_config.default_snakemake_args,
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/jobs/{job_id}/workflow")
        assert response.status_code == 500
        assert "Persistence not configured" in response.json()["detail"]


class TestWorkflowJobs:
    async def test_list_jobs(self, async_client, store):
        job_id = "test-wf-jobs"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert all("snakemake_id" in j for j in data)
        assert all("rule" in j for j in data)

    async def test_jobs_include_files(self, async_client, store):
        job_id = "test-wf-jobs-files"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/jobs")
        data = response.json()
        # Job with snakemake_id=0 has 2 files
        job_0 = next(j for j in data if j["snakemake_id"] == 0)
        assert len(job_0["files"]) == 2
        assert all("path" in f and "file_type" in f for f in job_0["files"])


class TestWorkflowRule:
    async def test_returns_rule_with_jobs(self, async_client, store):
        job_id = "test-wf-rule"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/rules/rule_a")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "rule_a"
        assert data["total_job_count"] == 2
        assert data["jobs_finished"] == 2
        assert len(data["jobs"]) == 2
        # Jobs should have files
        job_0 = next(j for j in data["jobs"] if j["snakemake_id"] == 0)
        assert len(job_0["files"]) == 2

    async def test_404_for_unknown_rule(self, async_client, store):
        job_id = "test-wf-rule-404"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/rules/nonexistent")
        assert response.status_code == 404


class TestWorkflowRulegraph:
    async def test_return_rulegraph(self, async_client, store):
        job_id = "test-wf-rg"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/rulegraph")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data


class TestWorkflowJobFiles:
    async def test_returns_files_for_job(self, async_client, store):
        job_id = "test-job-files-ep"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        # snakemake_id=0 maps to internal job id=1 which has INPUT + OUTPUT files
        response = await async_client.get(f"/jobs/{job_id}/workflow/jobs/0/files")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        file_types = {f["file_type"] for f in data}
        assert file_types == {"INPUT", "OUTPUT"}
        assert all("path" in f for f in data)

    async def test_returns_empty_for_unknown_job(self, async_client, store):
        job_id = "test-job-files-empty"
        store.create_job(job_id)
        store.mark_finished(job_id, 0)
        _setup_snkmt(store, job_id)

        response = await async_client.get(f"/jobs/{job_id}/workflow/jobs/999/files")
        assert response.status_code == 200
        assert response.json() == []


class TestListJobsPagination:
    async def test_pagination_limit(self, async_client, store):
        for i in range(5):
            store.create_job(f"page-job-{i}")

        response = await async_client.get("/jobs?limit=2")
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5

    async def test_pagination_offset(self, async_client, store):
        for i in range(5):
            store.create_job(f"offset-job-{i}")

        response = await async_client.get("/jobs?limit=2&offset=3")
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5

    async def test_pagination_beyond_total(self, async_client, store):
        store.create_job("only-job")

        response = await async_client.get("/jobs?offset=10")
        data = response.json()
        assert len(data["jobs"]) == 0
        assert data["total"] == 1


class TestDownloadOutputSuccess:
    async def test_streams_file_content(self, async_client, store, mock_backend):
        job_id = "test-dl-success"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/dl-success"

        async def mock_stream(jid, wd, path):
            yield b"file content here"

        mock_backend.stream_file = mock_stream

        response = await async_client.get(f"/jobs/{job_id}/outputs/results/output.csv")
        assert response.status_code == 200
        assert response.content == b"file content here"
        assert "attachment" in response.headers["content-disposition"]

    async def test_file_disappears_mid_stream_returns_truncated_response(
        self, async_client, store, mock_backend
    ):
        job_id = "test-dl-disappear"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/dl-disappear"

        async def mock_stream_raises(jid, wd, path):
            yield b"partial"
            raise FileNotFoundError("file gone mid-stream")

        mock_backend.stream_file = mock_stream_raises

        response = await async_client.get(f"/jobs/{job_id}/outputs/results/output.csv")
        # Headers already sent, status 200, body truncated (partial only)
        assert response.status_code == 200
        assert response.content == b"partial"

    async def test_permission_error_mid_stream_propagates(
        self, async_client, store, mock_backend
    ):
        job_id = "test-dl-permission"
        record = store.create_job(job_id)
        store.mark_finished(job_id, 0)
        record.work_dir = "/scratch/test/jobs/dl-permission"

        async def mock_stream_permission_error(jid, wd, path):
            yield b"partial"
            raise PermissionError("permission denied")

        mock_backend.stream_file = mock_stream_permission_error

        import pytest

        with pytest.raises(PermissionError):
            await async_client.get(f"/jobs/{job_id}/outputs/results/output.csv")


class TestHealthDegraded:
    async def test_returns_503_when_backend_down(self, async_client, mock_backend):
        mock_backend.check_connectivity = AsyncMock(return_value=False)
        response = await async_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["backend"] is False
        assert data["status"] == "degraded"

    async def test_cached_returns_503_when_backend_down(
        self, async_client, mock_backend
    ):
        mock_backend.check_connectivity = AsyncMock(return_value=False)
        response = await async_client.get("/health/cached")
        assert response.status_code == 503
        data = response.json()
        assert data["backend"] is False
        assert data["status"] == "degraded"


def _lifespan_env(tmp_path):
    """Context manager that sets up env vars for lifespan tests."""
    import contextlib
    import os

    config_path = tmp_path / "config.yaml"
    config_path.write_text("local:\n  scratch_dir: " + str(tmp_path / "scratch") + "\n")

    @contextlib.contextmanager
    def _env():
        old = os.environ.get("SNAKEDISPATCH_CONFIG")
        os.environ["SNAKEDISPATCH_CONFIG"] = str(config_path)
        os.environ["DATA_DIR"] = str(tmp_path / "data")
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("SNAKEDISPATCH_CONFIG", None)
            else:
                os.environ["SNAKEDISPATCH_CONFIG"] = old
            os.environ.pop("DATA_DIR", None)

    return _env()


class TestLifespan:
    async def test_startup_creates_store_and_backend(self, tmp_path):
        with _lifespan_env(tmp_path):
            async with lifespan(app):
                state = app.state.app
                assert state.store is not None
                assert state.backend is not None
                assert state.settings is not None
                assert state.health_cache is not None
                assert state.background_tasks is not None

                from app.backends.local import LocalBackend

                assert isinstance(state.backend, LocalBackend)

    async def test_lifespan_routes_functional(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        with _lifespan_env(tmp_path):
            async with lifespan(app):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    resp = await client.get("/health")
                    assert resp.status_code in (200, 503)
                    resp = await client.get("/jobs")
                    assert resp.status_code == 200

    def test_create_backend_local(self):
        from app.backends.local import LocalBackend
        from app.config import LocalConfig

        backend = create_backend(LocalConfig(scratch_dir="/tmp/test-scratch"))
        assert isinstance(backend, LocalBackend)

    def test_create_backend_slurm(self):
        from app.backends.slurm_ssh import SlurmSSHBackend
        from app.config import SlurmSSHConfig

        backend = create_backend(
            SlurmSSHConfig(
                host="test.example.com",
                user="testuser",
                pixi_path="/usr/local/bin/pixi",
                scratch_dir="/scratch",
            )
        )
        assert isinstance(backend, SlurmSSHBackend)
