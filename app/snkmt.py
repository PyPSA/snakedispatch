"""Query layer for the snkmt SQLite database.

Schema defined by snakemake-logger-plugin-snkmt (pinned in app/utils.py):
https://github.com/cademirch/snakemake-logger-plugin-snkmt
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowRow(TypedDict):
    id: str
    snakefile: str | None
    started_at: (
        str | None
    )  # e.g. "2025-01-01T12:00:00+00:00", coerced to datetime by Pydantic
    completed_at: (
        str | None
    )  # e.g. "2025-01-01T12:00:00+00:00", coerced to datetime by Pydantic
    status: str | None
    command_line: str | None
    dryrun: int
    rulegraph_data: str | None
    total_job_count: int | None
    jobs_finished: int | None


class RuleRow(TypedDict):
    id: int
    name: str
    workflow_id: str
    total_job_count: int | None
    jobs_finished: int | None


class JobRow(TypedDict):
    id: int
    snakemake_id: int
    workflow_id: str
    rule_id: int | None
    wildcards: str | None  # JSON-encoded dict, callers call json.loads()
    resources: str | None  # JSON-encoded dict, callers call json.loads()
    shellcmd: str | None
    threads: int | None
    priority: float
    status: str | None
    started_at: (
        str | None
    )  # e.g. "2025-01-01T12:00:00+00:00"; coerced to datetime by Pydantic
    completed_at: (
        str | None
    )  # e.g. "2025-01-01T12:00:00+00:00"; coerced to datetime by Pydantic
    group_id: str | None
    rule_name: str | None


class FileRow(TypedDict):
    id: int
    path: str
    file_type: str
    job_id: int


class ErrorRow(TypedDict):
    id: int
    timestamp: str | None
    exception: str | None
    location: str | None
    traceback: str | None
    file: str | None
    line: int | None
    rule_id: int | None
    workflow_id: str
    rule_name: str | None


def _rename_end_time_column(row: dict[str, Any]) -> dict[str, Any]:
    """Rename DB column end_time → completed_at to match our naming."""
    row = dict(row)
    row["completed_at"] = row.pop("end_time", None)
    return row


def safe_json_loads(value: str | None) -> dict[str, Any] | None:
    """Parse a JSON encoded dict from a DB field, or None on failure."""
    if not value:
        return None
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid JSON in DB field: %r", value[:100])
        return None
    if not isinstance(result, dict):
        logger.warning(
            "Expected JSON dict in DB field, got %s: %r",
            type(result).__name__,
            value[:100],
        )
        return None
    return result


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open db_path as a read-only SQLite connection."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for sharing a single read only connection across queries."""
    conn = _open_ro(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_workflow(conn: sqlite3.Connection, workflow_id: str) -> WorkflowRow | None:
    """Return the workflow row for workflow_id or None if not found."""
    row = conn.execute(
        "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
    ).fetchone()
    if row is None:
        return None
    return cast("WorkflowRow", _rename_end_time_column(dict(row)))


def get_rules(conn: sqlite3.Connection, workflow_id: str) -> list[RuleRow]:
    """Return all rules for a workflow."""
    rows = conn.execute(
        "SELECT * FROM rules WHERE workflow_id = ?", (workflow_id,)
    ).fetchall()
    return [cast("RuleRow", dict(r)) for r in rows]


def get_jobs(conn: sqlite3.Connection, workflow_id: str) -> list[JobRow]:
    """Return all snakemake jobs for a workflow, with rule name joined."""
    rows = conn.execute(
        "SELECT j.*, r.name AS rule_name "
        "FROM jobs j LEFT JOIN rules r ON j.rule_id = r.id "
        "WHERE j.workflow_id = ?",
        (workflow_id,),
    ).fetchall()
    return [cast("JobRow", _rename_end_time_column(dict(r))) for r in rows]


def get_job_files_by_snakemake_id(
    conn: sqlite3.Connection, snakemake_id: int
) -> list[FileRow]:
    """Return all files for the job identified by its Snakemake integer ID."""
    rows = conn.execute(
        "SELECT f.* FROM files f "
        "JOIN jobs j ON f.job_id = j.id "
        "WHERE j.snakemake_id = ?",
        (snakemake_id,),
    ).fetchall()
    return [cast("FileRow", dict(r)) for r in rows]


def get_errors(conn: sqlite3.Connection, workflow_id: str) -> list[ErrorRow]:
    """Return all errors for a workflow, with rule name joined."""
    rows = conn.execute(
        "SELECT e.*, r.name AS rule_name "
        "FROM errors e LEFT JOIN rules r ON e.rule_id = r.id "
        "WHERE e.workflow_id = ?",
        (workflow_id,),
    ).fetchall()
    return [cast("ErrorRow", dict(r)) for r in rows]


def get_rulegraph(conn: sqlite3.Connection, workflow_id: str) -> dict[str, Any] | None:
    """Return the rulegraph_data JSON for a workflow, or None."""
    row = conn.execute(
        "SELECT rulegraph_data FROM workflows WHERE id = ?",
        (workflow_id,),
    ).fetchone()
    if row is None or row["rulegraph_data"] is None:
        return None
    try:
        result = json.loads(row["rulegraph_data"])
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Invalid rulegraph JSON for workflow %s", workflow_id, exc_info=True
        )
        return None
    if not isinstance(result, dict):
        logger.warning(
            "Rulegraph JSON for workflow %s is not a dict (got %s), ignoring",
            workflow_id,
            type(result).__name__,
        )
        return None
    return result


def get_all_files(
    conn: sqlite3.Connection, workflow_id: str
) -> dict[int, list[FileRow]]:
    """Return all files grouped by job_id for jobs in this workflow."""
    rows = conn.execute(
        "SELECT f.* FROM files f "
        "JOIN jobs j ON f.job_id = j.id "
        "WHERE j.workflow_id = ?",
        (workflow_id,),
    ).fetchall()
    grouped: dict[int, list[FileRow]] = {}
    for r in rows:
        d = cast("FileRow", dict(r))
        grouped.setdefault(d["job_id"], []).append(d)
    return grouped


def get_files_for_jobs(
    conn: sqlite3.Connection,
    job_ids: set[int],
) -> dict[int, list[FileRow]]:
    """Return files grouped by job_id for a specific set of job IDs."""
    if not job_ids:
        return {}
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"SELECT * FROM files WHERE job_id IN ({placeholders})",  # noqa: S608
        list(job_ids),
    ).fetchall()
    grouped: dict[int, list[FileRow]] = {}
    for r in rows:
        d = cast("FileRow", dict(r))
        grouped.setdefault(d["job_id"], []).append(d)
    return grouped


def get_jobs_by_rule(
    conn: sqlite3.Connection,
    workflow_id: str,
    rule_name: str,
) -> list[JobRow]:
    """Return jobs for a specific rule in a workflow, with rule name joined."""
    rows = conn.execute(
        "SELECT j.*, r.name AS rule_name "
        "FROM jobs j LEFT JOIN rules r ON j.rule_id = r.id "
        "WHERE j.workflow_id = ? AND r.name = ?",  # noqa: S608 — parameterized query
        (workflow_id, rule_name),
    ).fetchall()
    return [cast("JobRow", _rename_end_time_column(dict(r))) for r in rows]


def get_rule_by_name(
    conn: sqlite3.Connection,
    workflow_id: str,
    rule_name: str,
) -> RuleRow | None:
    """Return the rule row for rule_name in a workflow, or None if not found."""
    row = conn.execute(
        "SELECT * FROM rules WHERE workflow_id = ? AND name = ?",
        (workflow_id, rule_name),
    ).fetchone()
    if row is None:
        return None
    return cast("RuleRow", dict(row))


def get_workflow_counts(
    conn: sqlite3.Connection, workflow_id: str
) -> tuple[int, int] | None:
    """Return (total_job_count, jobs_finished), or None if workflow not found."""
    row = conn.execute(
        "SELECT total_job_count, jobs_finished FROM workflows WHERE id = ?",
        (workflow_id,),
    ).fetchone()
    if row is None:
        return None
    return (row["total_job_count"] or 0, row["jobs_finished"] or 0)


def get_single_workflow_id(conn: sqlite3.Connection) -> str | None:
    """Return the workflow ID from a per-job snkmt DB, or None if empty."""
    row = conn.execute("SELECT id FROM workflows LIMIT 1").fetchone()
    if row is None:
        return None
    return row["id"]


class WorkflowData(NamedTuple):
    """Bundle of all workflow data fetched in a single DB connection."""

    workflow_id: str | None
    workflow: WorkflowRow | None
    rules: list[RuleRow]
    jobs: list[JobRow]
    files_by_job: dict[int, list[FileRow]]
    errors: list[ErrorRow]
    rulegraph: dict[str, Any] | None


class WorkflowJobsData(NamedTuple):
    """Jobs and their file mappings for a workflow."""

    jobs: list[JobRow]
    files_by_job: dict[int, list[FileRow]]


class RuleData(NamedTuple):
    """A rule with its jobs and file mappings."""

    rule: RuleRow | None
    jobs: list[JobRow]
    files_by_job: dict[int, list[FileRow]]


_EMPTY_WORKFLOW_DATA = WorkflowData(
    workflow_id=None,
    workflow=None,
    rules=[],
    jobs=[],
    files_by_job={},
    errors=[],
    rulegraph=None,
)


def fetch_workflow_data(db_path: Path) -> WorkflowData:
    """Open one connection and fetch all workflow data in a single DB round trip."""
    with connect(db_path) as conn:
        workflow_id = get_single_workflow_id(conn)
        if workflow_id is None:
            return _EMPTY_WORKFLOW_DATA
        workflow = get_workflow(conn, workflow_id)
        if workflow is None:
            return _EMPTY_WORKFLOW_DATA._replace(workflow_id=workflow_id)
        rules_raw = get_rules(conn, workflow_id)
        jobs_raw = get_jobs(conn, workflow_id)
        files_by_job = get_all_files(conn, workflow_id)
        errors_raw = get_errors(conn, workflow_id)
        rulegraph = get_rulegraph(conn, workflow_id)
    return WorkflowData(
        workflow_id=workflow_id,
        workflow=workflow,
        rules=rules_raw,
        jobs=jobs_raw,
        files_by_job=files_by_job,
        errors=errors_raw,
        rulegraph=rulegraph,
    )


def fetch_workflow_jobs(db_path: Path) -> WorkflowJobsData:
    """Fetch all snakemake jobs and their files for the single workflow in db_path."""
    with connect(db_path) as conn:
        workflow_id = get_single_workflow_id(conn)
        if workflow_id is None:
            return WorkflowJobsData(jobs=[], files_by_job={})
        jobs_raw = get_jobs(conn, workflow_id)
        files_by_job = get_all_files(conn, workflow_id)
    return WorkflowJobsData(jobs=jobs_raw, files_by_job=files_by_job)


def fetch_rule_jobs(db_path: Path, rule_name: str) -> RuleData:
    """Fetch a rule and its jobs/files for the single workflow in db_path."""
    with connect(db_path) as conn:
        workflow_id = get_single_workflow_id(conn)
        if workflow_id is None:
            return RuleData(rule=None, jobs=[], files_by_job={})
        rule_raw = get_rule_by_name(conn, workflow_id, rule_name)
        if rule_raw is None:
            return RuleData(rule=None, jobs=[], files_by_job={})
        jobs_raw = get_jobs_by_rule(conn, workflow_id, rule_name)
        job_ids = {j["id"] for j in jobs_raw}
        files_by_job = get_files_for_jobs(conn, job_ids)
    return RuleData(rule=rule_raw, jobs=jobs_raw, files_by_job=files_by_job)


def fetch_rulegraph(db_path: Path) -> dict[str, Any] | None:
    """Fetch the rulegraph for the single workflow in db_path."""
    with connect(db_path) as conn:
        workflow_id = get_single_workflow_id(conn)
        if workflow_id is None:
            return None
        return get_rulegraph(conn, workflow_id)


def fetch_workflow_counts(db_path: Path) -> tuple[int, int] | None:
    """Fetch (total_job_count, jobs_finished) for the single workflow in db_path.

    Returns None if there is no workflow or counts are unavailable.
    """
    with connect(db_path) as conn:
        workflow_id = get_single_workflow_id(conn)
        if workflow_id is None:
            return None
        return get_workflow_counts(conn, workflow_id)


def fetch_job_files(db_path: Path, snakemake_job_id: int) -> list[FileRow]:
    """Fetch all files for a specific snakemake job in db_path."""
    with connect(db_path) as conn:
        return get_job_files_by_snakemake_id(conn, snakemake_job_id)
