from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models import (
    JobCleanedResponse,
    JobCreate,
    JobListResponse,
    JobOutputsResponse,
    JobResponse,
    JobStatus,
    OutputFile,
    SnkmtErrorResponse,
    SnkmtFileResponse,
    SnkmtJobResponse,
    SnkmtRuleResponse,
    SnkmtWorkflowResponse,
    _ensure_utc,
)


class TestJobCreateValidation:
    def test_minimal_valid(self):
        job = JobCreate(workflow="https://github.com/org/repo.git")
        assert job.workflow == "https://github.com/org/repo.git"
        assert job.configfile is None
        assert job.cache_key is None

    def test_cache_key_valid(self):
        job = JobCreate(workflow="https://example.com/repo.git", cache_key="pypsa-eur")
        assert job.cache_key == "pypsa-eur"

    def test_cache_key_with_slash_rejected(self):
        with pytest.raises(ValidationError, match="cache_key"):
            JobCreate(workflow="https://example.com/repo.git", cache_key="foo/bar")

    def test_cache_key_with_dotdot_rejected(self):
        with pytest.raises(ValidationError, match="cache_key"):
            JobCreate(workflow="https://example.com/repo.git", cache_key="..")

    def test_extra_files_valid(self):
        job = JobCreate(
            workflow="https://example.com/repo.git",
            extra_files={"config/prod.yaml": "key: value"},
        )
        assert job.extra_files == {"config/prod.yaml": "key: value"}

    def test_extra_files_path_traversal_rejected(self):
        with pytest.raises(ValidationError, match="not a safe relative path"):
            JobCreate(
                workflow="https://example.com/repo.git",
                extra_files={"../etc/passwd": "data"},
            )

    def test_extra_files_absolute_path_rejected(self):
        with pytest.raises(ValidationError, match="not a safe relative path"):
            JobCreate(
                workflow="https://example.com/repo.git",
                extra_files={"/etc/passwd": "data"},
            )

    def test_cache_dirs_valid(self):
        job = JobCreate(
            workflow="https://example.com/repo.git",
            cache_dirs=["data", "resources"],
        )
        assert job.cache_dirs == ["data", "resources"]

    def test_cache_dirs_path_traversal_rejected(self):
        with pytest.raises(ValidationError, match="not a safe relative path"):
            JobCreate(
                workflow="https://example.com/repo.git",
                cache_dirs=["data", "../secrets"],
            )

    def test_cache_key_with_backslash_rejected(self):
        with pytest.raises(ValidationError, match="cache_key"):
            JobCreate(workflow="https://example.com/repo.git", cache_key="foo\\bar")

    def test_cache_key_empty_string_rejected(self):
        with pytest.raises(ValidationError, match="cache_key"):
            JobCreate(workflow="https://example.com/repo.git", cache_key="")

    def test_snakemake_args_stored(self):
        job = JobCreate(
            workflow="https://example.com/repo.git",
            snakemake_args=["--cores", "4"],
        )
        assert job.snakemake_args == ["--cores", "4"]

    def test_extra_files_absolute_path_rejected_windows_style(self):
        with pytest.raises(ValidationError, match="not a safe relative path"):
            JobCreate(
                workflow="https://example.com/repo.git",
                extra_files={"../secret": "data"},
            )

    def test_cache_dirs_absolute_path_rejected(self):
        with pytest.raises(ValidationError, match="not a safe relative path"):
            JobCreate(
                workflow="https://example.com/repo.git",
                cache_dirs=["/absolute/path"],
            )


class TestEnsureUtc:
    def test_none_returns_none(self):
        assert _ensure_utc(None) is None

    def test_naive_datetime_gets_utc_tzinfo(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _ensure_utc(dt)
        assert result.tzinfo == UTC

    def test_aware_datetime_preserved(self):
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = _ensure_utc(dt)
        assert result == dt

    def test_iso_string_parsed(self):
        result = _ensure_utc("2024-01-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.tzinfo == UTC

    def test_iso_string_with_tz_preserved(self):
        result = _ensure_utc("2024-01-15T10:30:00+00:00")
        assert isinstance(result, datetime)


class TestJobStatus:
    def test_all_statuses_exist(self):
        for name in (
            "PENDING",
            "SETUP",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "ERROR",
            "CANCELLED",
        ):
            assert JobStatus(name).value == name

    def test_is_string_enum(self):
        assert isinstance(JobStatus.PENDING, str)


class TestJobResponse:
    def test_minimal_construction(self):
        now = datetime.now(tz=UTC)
        resp = JobResponse(job_id="job-1", status=JobStatus.PENDING, created_at=now)
        assert resp.job_id == "job-1"
        assert resp.status == JobStatus.PENDING
        assert resp.workflow is None
        assert resp.exit_code is None

    def test_full_construction(self):
        now = datetime.now(tz=UTC)
        resp = JobResponse(
            job_id="job-2",
            status=JobStatus.COMPLETED,
            workflow="https://github.com/org/repo.git",
            git_ref="main",
            git_sha="abc123",
            exit_code=0,
            created_at=now,
            started_at=now,
            completed_at=now,
            total_job_count=10,
            jobs_finished=10,
        )
        assert resp.exit_code == 0
        assert resp.git_sha == "abc123"
        assert resp.total_job_count == 10


class TestOutputFileAndResponses:
    def test_output_file(self):
        f = OutputFile(path="results/output.txt", size=1024)
        assert f.path == "results/output.txt"
        assert f.size == 1024

    def test_job_outputs_response(self):
        resp = JobOutputsResponse(files=[OutputFile(path="a.txt", size=1)])
        assert len(resp.files) == 1

    def test_job_cleaned_response_default_status(self):
        resp = JobCleanedResponse(job_id="job-1")
        assert resp.status == "cleaned"

    def test_job_list_response(self):
        now = datetime.now(tz=UTC)
        job = JobResponse(job_id="job-1", status=JobStatus.RUNNING, created_at=now)
        resp = JobListResponse(jobs=[job], total=1)
        assert resp.total == 1
        assert resp.jobs[0].job_id == "job-1"


class TestSnkmtModels:
    def test_snkmt_file_response(self):
        f = SnkmtFileResponse(path="results/out.txt", file_type="output")
        assert f.path == "results/out.txt"

    def test_snkmt_job_response_datetime_coercion(self):
        resp = SnkmtJobResponse(
            snakemake_id=1,
            rule="rule_a",
            status="success",
            threads=4,
            started_at="2024-01-15T10:00:00",
        )
        assert resp.started_at.tzinfo == UTC

    def test_snkmt_job_response_none_times(self):
        resp = SnkmtJobResponse(
            snakemake_id=2, rule="rule_b", status="running", threads=1
        )
        assert resp.started_at is None
        assert resp.completed_at is None

    def test_snkmt_rule_response(self):
        rule = SnkmtRuleResponse(name="rule_a", total_job_count=5, jobs_finished=3)
        assert rule.jobs == []

    def test_snkmt_error_response_timestamp_coercion(self):
        err = SnkmtErrorResponse(
            timestamp="2024-01-15T10:00:00",
            exception="KeyError",
        )
        assert err.timestamp.tzinfo == UTC
        assert err.rule is None

    def test_snkmt_workflow_response(self):
        now = datetime.now(tz=UTC)
        wf = SnkmtWorkflowResponse(
            workflow_id="wf-1",
            status="running",
            started_at=now,
            total_job_count=10,
            jobs_finished=5,
            rules=[],
            errors=[],
        )
        assert wf.workflow_id == "wf-1"
        assert wf.rulegraph is None
        assert wf.errors == []
