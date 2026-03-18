from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app import snkmt
from app.models import JobStatus
from app.routes.snkmt import (
    _build_snkmt_job_response,
    _require_snkmt,
    _run_snkmt_query,
    _strip_work_dir,
)
from tests.conftest import SAMPLE_WORKFLOW_ID

_EMPTY_WORKFLOWS_SCHEMA = (
    "CREATE TABLE workflows (id TEXT PRIMARY KEY, snakefile TEXT, "
    "started_at TEXT, end_time TEXT, status TEXT, command_line TEXT, "
    "dryrun INTEGER, rulegraph_data TEXT, total_job_count INTEGER, "
    "jobs_finished INTEGER)"
)


def _create_empty_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute(_EMPTY_WORKFLOWS_SCHEMA)
    conn.commit()
    conn.close()
    return path


def _make_job(**overrides):
    base = {
        "id": 1,
        "snakemake_id": 1,
        "rule_name": "rule_a",
        "status": "success",
        "wildcards": None,
        "resources": None,
        "shellcmd": None,
        "threads": 1,
        "started_at": None,
        "completed_at": None,
    }
    base.update(overrides)
    return base


class TestBuildJobResponse:
    def test_basic_fields(self):
        resp = _build_snkmt_job_response(
            _make_job(snakemake_id=42, shellcmd="echo hello", threads=2),
            files_by_job={},
        )
        assert resp.snakemake_id == 42
        assert resp.rule == "rule_a"
        assert resp.status == "success"
        assert resp.threads == 2
        assert resp.files == []

    def test_parses_wildcards_json(self):
        resp = _build_snkmt_job_response(
            _make_job(wildcards=json.dumps({"sample": "A"})), files_by_job={}
        )
        assert resp.wildcards == {"sample": "A"}

    def test_parses_resources_json(self):
        resp = _build_snkmt_job_response(
            _make_job(id=2, snakemake_id=2, resources=json.dumps({"mem_mb": 4096})),
            files_by_job={},
        )
        assert resp.resources == {"mem_mb": 4096}

    def test_attaches_files_from_mapping(self):
        files_by_job = {5: [{"path": "data/input.txt", "file_type": "INPUT"}]}
        resp = _build_snkmt_job_response(
            _make_job(id=5, snakemake_id=5), files_by_job=files_by_job
        )
        assert len(resp.files) == 1
        assert resp.files[0].path == "data/input.txt"

    def test_strips_work_dir_from_file_paths(self):
        files_by_job = {
            1: [
                {"path": "/scratch/jobs/abc/results/output.nc", "file_type": "OUTPUT"},
                {"path": "/scratch/jobs/abc/input/data.csv", "file_type": "INPUT"},
            ]
        }
        resp = _build_snkmt_job_response(
            _make_job(id=1, snakemake_id=1),
            files_by_job=files_by_job,
            work_dir="/scratch/jobs/abc",
        )
        assert resp.files[0].path == "results/output.nc"
        assert resp.files[1].path == "input/data.csv"

    def test_none_rule_name_becomes_empty_string(self):
        resp = _build_snkmt_job_response(_make_job(rule_name=None), files_by_job={})
        assert resp.rule == ""


class TestRequireSnkmt:
    def test_raises_409_for_pending_job(self, store):
        record = store.create_job("job-pending")
        record.status = JobStatus.PENDING

        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store, "job-pending")
        assert exc_info.value.status_code == 409

    def test_raises_409_for_setup_job(self, store):
        record = store.create_job("job-setup")
        record.status = JobStatus.SETUP

        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store, "job-setup")
        assert exc_info.value.status_code == 409

    def test_raises_404_for_unknown_job(self, store):
        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store, "nonexistent")
        assert exc_info.value.status_code == 404

    def test_raises_500_when_persistence_disabled(self):
        store_no_persist = __import__("app.store", fromlist=["JobStore"]).JobStore(
            data_dir=None
        )
        record = store_no_persist.create_job("job-no-persist")
        record.status = JobStatus.COMPLETED

        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store_no_persist, "job-no-persist")
        assert exc_info.value.status_code == 500

    def test_raises_404_when_db_does_not_exist(self, store):
        record = store.create_job("job-no-db")
        record.status = JobStatus.COMPLETED
        # Don't create the snkmt.db, should get 404

        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store, "job-no-db")
        assert exc_info.value.status_code == 404

    def test_raises_409_for_running_job_without_db(self, store):
        record = store.create_job("job-running-no-db")
        store.mark_running("job-running-no-db")
        record.work_dir = "/some/work/dir"
        # Don't create snkmt.db, should get 409 (not yet synced)

        with pytest.raises(HTTPException) as exc_info:
            _require_snkmt(store, "job-running-no-db")
        assert exc_info.value.status_code == 409
        assert "not yet synced" in exc_info.value.detail

    def test_returns_db_path(self, store, snkmt_db):
        record = store.create_job("job-ok")
        record.status = JobStatus.COMPLETED
        # Point the store's job dir to snkmt_db's parent
        job_dir = store.get_snkmt_db_path("job-ok").parent
        job_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(snkmt_db, job_dir / "snkmt.db")

        db_path, work_dir = _require_snkmt(store, "job-ok")
        assert db_path.exists()
        assert work_dir is None


class TestStripWorkDir:
    def test_strips_matching_prefix(self):
        result = _strip_work_dir("/scratch/jobs/abc/output.nc", "/scratch/jobs/abc")
        assert result == "output.nc"

    def test_strips_prefix_with_trailing_slash(self):
        result = _strip_work_dir("/scratch/jobs/abc/output.nc", "/scratch/jobs/abc/")
        assert result == "output.nc"

    def test_returns_original_when_no_match(self):
        result = _strip_work_dir("/other/path/file.txt", "/scratch/jobs/abc")
        assert result == "/other/path/file.txt"

    def test_returns_original_when_work_dir_is_none(self):
        result = _strip_work_dir("/scratch/jobs/abc/output.nc", None)
        assert result == "/scratch/jobs/abc/output.nc"

    def test_returns_original_when_work_dir_is_empty(self):
        result = _strip_work_dir("/scratch/jobs/abc/output.nc", "")
        assert result == "/scratch/jobs/abc/output.nc"

    def test_nested_path(self):
        result = _strip_work_dir(
            "/scratch/jobs/abc/results/sub/file.nc", "/scratch/jobs/abc"
        )
        assert result == "results/sub/file.nc"


class TestGetWorkflow:
    def test_returns_workflow(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            wf = snkmt.get_workflow(conn, SAMPLE_WORKFLOW_ID)
        assert wf is not None
        assert wf["id"] == SAMPLE_WORKFLOW_ID
        assert wf["status"] == "finished"
        assert wf["total_job_count"] == 3
        assert wf["jobs_finished"] == 3

    def test_returns_none_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_workflow(conn, "nonexistent") is None


class TestGetRules:
    def test_returns_rules(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            rules = snkmt.get_rules(conn, SAMPLE_WORKFLOW_ID)
        assert len(rules) == 2
        names = {r["name"] for r in rules}
        assert names == {"rule_a", "rule_b"}

    def test_empty_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_rules(conn, "nonexistent") == []


class TestGetJobs:
    def test_returns_jobs_with_rule_name(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            jobs = snkmt.get_jobs(conn, SAMPLE_WORKFLOW_ID)
        assert len(jobs) == 3
        rule_names = {j["rule_name"] for j in jobs}
        assert rule_names == {"rule_a", "rule_b"}

    def test_empty_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_jobs(conn, "nonexistent") == []


class TestGetErrors:
    def test_returns_errors(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            errors = snkmt.get_errors(conn, SAMPLE_WORKFLOW_ID)
        assert len(errors) == 1
        assert "FileNotFoundError" in errors[0]["exception"]
        assert errors[0]["rule_name"] == "rule_a"

    def test_empty_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_errors(conn, "nonexistent") == []


class TestGetRulegraph:
    def test_returns_rulegraph(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            rg = snkmt.get_rulegraph(conn, SAMPLE_WORKFLOW_ID)
        assert rg is not None
        assert "nodes" in rg
        assert "edges" in rg

    def test_returns_none_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_rulegraph(conn, "nonexistent") is None

    def test_returns_none_for_non_dict_json(self, tmp_path):
        db_path = _create_empty_db(tmp_path / "array_rg.db")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO workflows (id, rulegraph_data) VALUES (?, ?)",
            ("wf-array", json.dumps([1, 2, 3])),
        )
        conn.commit()
        result = snkmt.get_rulegraph(conn, "wf-array")
        conn.close()
        assert result is None


class TestGetAllFiles:
    def test_returns_files_grouped_by_job(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            files = snkmt.get_all_files(conn, SAMPLE_WORKFLOW_ID)
        # Job 1 has 2 files, job 3 has 1 file, job 2 has none
        assert len(files[1]) == 2
        assert len(files[3]) == 1
        assert 2 not in files

    def test_empty_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_all_files(conn, "nonexistent") == {}


class TestGetRuleByName:
    def test_returns_rule(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            rule = snkmt.get_rule_by_name(conn, SAMPLE_WORKFLOW_ID, "rule_a")
        assert rule is not None
        assert rule["name"] == "rule_a"
        assert rule["total_job_count"] == 2

    def test_returns_none_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert (
                snkmt.get_rule_by_name(conn, SAMPLE_WORKFLOW_ID, "nonexistent") is None
            )


class TestGetSingleWorkflowId:
    def test_returns_workflow_id(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            wf_id = snkmt.get_single_workflow_id(conn)
        assert wf_id == SAMPLE_WORKFLOW_ID

    def test_returns_none_for_empty_db(self, tmp_path):
        db = _create_empty_db(tmp_path / "empty.db")
        with snkmt.connect(db) as conn:
            assert snkmt.get_single_workflow_id(conn) is None


class TestGetWorkflowCounts:
    def test_returns_counts(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            result = snkmt.get_workflow_counts(conn, SAMPLE_WORKFLOW_ID)
        assert result is not None
        total, finished = result
        assert total == 3
        assert finished == 3

    def test_returns_none_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_workflow_counts(conn, "nonexistent") is None

    def test_null_columns_coerce_to_zero(self, tmp_path):
        db_path = _create_empty_db(tmp_path / "null_counts.db")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO workflows (id, total_job_count, jobs_finished) "
            "VALUES (?, NULL, NULL)",
            ("wf-null",),
        )
        conn.commit()
        result = snkmt.get_workflow_counts(conn, "wf-null")
        conn.close()
        assert result == (0, 0)


class TestGetJobsByRule:
    def test_returns_jobs_for_rule(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            jobs = snkmt.get_jobs_by_rule(conn, SAMPLE_WORKFLOW_ID, "rule_a")
        assert len(jobs) == 2
        assert all(j["rule_name"] == "rule_a" for j in jobs)

    def test_empty_for_unknown_rule(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_jobs_by_rule(conn, SAMPLE_WORKFLOW_ID, "nonexistent") == []


class TestGetFilesForJobs:
    def test_returns_files_grouped(self, snkmt_db):
        # Job 1 has 2 files, job 3 has 1 file
        with snkmt.connect(snkmt_db) as conn:
            result = snkmt.get_files_for_jobs(conn, {1, 3})
        assert len(result[1]) == 2
        assert len(result[3]) == 1

    def test_empty_for_no_ids(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_files_for_jobs(conn, set()) == {}

    def test_empty_for_unknown_ids(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_files_for_jobs(conn, {999}) == {}


class TestGetJobFilesBySnakemakeId:
    def test_returns_files(self, snkmt_db):
        # snakemake_id=0 maps to internal id=1 which has 2 files
        with snkmt.connect(snkmt_db) as conn:
            files = snkmt.get_job_files_by_snakemake_id(conn, 0)
        assert len(files) == 2
        types = {f["file_type"] for f in files}
        assert types == {"INPUT", "OUTPUT"}

    def test_empty_for_unknown(self, snkmt_db):
        with snkmt.connect(snkmt_db) as conn:
            assert snkmt.get_job_files_by_snakemake_id(conn, 999) == []


class TestRunSnkmtQuery:
    async def test_maps_operational_error_to_502(self):
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=sqlite3.OperationalError("db locked")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _run_snkmt_query("test-job", lambda: None)
        assert exc_info.value.status_code == 502
        assert "corrupt" in exc_info.value.detail

    async def test_returns_fn_result_on_success(self):
        with patch("asyncio.to_thread", new=AsyncMock(return_value={"key": "val"})):
            result = await _run_snkmt_query("test-job", lambda: {"key": "val"})
        assert result == {"key": "val"}


class TestSafeJsonLoads:
    def test_returns_none_for_falsy(self):
        assert snkmt.safe_json_loads(None) is None
        assert snkmt.safe_json_loads("") is None

    def test_parses_valid_dict(self):
        result = snkmt.safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="app.snkmt"):
            result = snkmt.safe_json_loads("not-valid-json{")
        assert result is None
        assert any("Invalid JSON" in r.message for r in caplog.records)

    def test_returns_none_for_non_dict_json(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="app.snkmt"):
            result = snkmt.safe_json_loads("[1, 2, 3]")
        assert result is None


class TestFetchFunctions:
    def test_fetch_workflow_data_returns_data(self, snkmt_db):
        data = snkmt.fetch_workflow_data(snkmt_db)
        assert data.workflow_id == SAMPLE_WORKFLOW_ID
        assert data.workflow is not None
        assert len(data.rules) > 0
        assert len(data.jobs) > 0

    def test_fetch_workflow_data_empty_db_returns_sentinel(self, tmp_path):
        db = _create_empty_db(tmp_path / "empty.db")
        data = snkmt.fetch_workflow_data(db)
        assert data.workflow_id is None
        assert data.workflow is None

    def test_fetch_workflow_jobs_returns_jobs_and_files(self, snkmt_db):
        data = snkmt.fetch_workflow_jobs(snkmt_db)
        assert len(data.jobs) > 0
        assert isinstance(data.files_by_job, dict)

    def test_fetch_workflow_jobs_empty_db_returns_empty(self, tmp_path):
        db = _create_empty_db(tmp_path / "empty.db")
        data = snkmt.fetch_workflow_jobs(db)
        assert data.jobs == []
        assert data.files_by_job == {}

    def test_fetch_rule_jobs_returns_rule_and_jobs(self, snkmt_db):
        data = snkmt.fetch_rule_jobs(snkmt_db, "rule_a")
        assert data.rule is not None
        assert len(data.jobs) > 0

    def test_fetch_rule_jobs_unknown_rule_returns_empty(self, snkmt_db):
        data = snkmt.fetch_rule_jobs(snkmt_db, "nonexistent_rule")
        assert data.rule is None
        assert data.jobs == []

    def test_fetch_rulegraph_returns_dict_or_none(self, snkmt_db):
        result = snkmt.fetch_rulegraph(snkmt_db)
        # May be None if no rulegraph data, or a dict if present
        assert result is None or isinstance(result, dict)

    def test_fetch_workflow_counts_returns_tuple(self, snkmt_db):
        result = snkmt.fetch_workflow_counts(snkmt_db)
        assert result is not None
        total, finished = result
        assert total == 3
        assert finished == 3

    def test_fetch_job_files_returns_files(self, snkmt_db):
        files = snkmt.fetch_job_files(snkmt_db, 0)
        assert len(files) == 2
        types = {f["file_type"] for f in files}
        assert types == {"INPUT", "OUTPUT"}

    def test_fetch_job_files_unknown_id_returns_empty(self, snkmt_db):
        files = snkmt.fetch_job_files(snkmt_db, 999)
        assert files == []
