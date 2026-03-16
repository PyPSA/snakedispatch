"""Unit tests for route helpers in routes_jobs and routes_health."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.models import JobStatus, WorkflowFileInfo
from app.routes.health import _build_health_response
from app.routes.jobs import (
    _build_job_response,
    _build_outputs_response,
    _validate_output_path,
)
from app.store import JobRecord


class TestBuildJobResponse:
    def _make_record(self, **kwargs) -> JobRecord:
        defaults = {
            "job_id": "job-1",
            "status": JobStatus.PENDING,
            "created_at": datetime(2025, 1, 1, tzinfo=UTC),
        }
        defaults.update(kwargs)
        return JobRecord(**defaults)

    def test_basic_fields(self):
        record = self._make_record()
        resp = _build_job_response(record)
        assert resp.job_id == "job-1"
        assert resp.status == JobStatus.PENDING
        assert resp.exit_code is None

    def test_optional_fields_preserved(self):
        record = self._make_record(
            status=JobStatus.COMPLETED,
            exit_code=0,
            git_ref="main",
            git_sha="abc123",
            total_job_count=5,
            jobs_finished=5,
        )
        resp = _build_job_response(record)
        assert resp.exit_code == 0
        assert resp.git_ref == "main"
        assert resp.git_sha == "abc123"
        assert resp.total_job_count == 5
        assert resp.jobs_finished == 5


class TestBuildOutputsResponse:
    def test_maps_files(self):
        files_raw: list[WorkflowFileInfo] = [
            {"path": "results/out.csv", "size": 100},
            {"path": "logs/run.log", "size": 200},
        ]
        resp = _build_outputs_response(files_raw)
        assert len(resp.files) == 2
        paths = {f.path for f in resp.files}
        assert "results/out.csv" in paths
        assert "logs/run.log" in paths

    def test_empty_list(self):
        resp = _build_outputs_response([])
        assert resp.files == []


class TestValidateOutputPath:
    def test_normalizes_valid_path(self):
        assert _validate_output_path("results/output.csv") == "results/output.csv"

    def test_rejects_path_traversal(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_output_path("../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_rejects_hidden_files(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_output_path(".hidden/secret")
        assert exc_info.value.status_code == 400

    def test_accepts_nested_path(self):
        assert (
            _validate_output_path("results/subdir/file.txt")
            == "results/subdir/file.txt"
        )


class TestMakeHealthResponse:
    def test_ok_when_backend_healthy(self):
        result = _build_health_response(True)
        assert result.status == "ok"
        assert result.backend is True

    def test_degraded_when_unhealthy(self):
        result = _build_health_response(False)
        assert result.status == "degraded"
        assert result.backend is False
