from __future__ import annotations

import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sse_starlette.sse import AppStatus

from app.config import LocalConfig, Settings
from app.deps import AppState
from app.main import app
from app.store import JobStore

SAMPLE_WORKFLOW_ID = "wf-00000000-0000-0000-0000-000000000001"


def create_snkmt_db(db_path) -> None:
    """Create a test snkmt.db with sample data."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE workflows (
            id TEXT PRIMARY KEY,
            snakefile TEXT,
            started_at TEXT,
            end_time TEXT,
            status TEXT,
            command_line TEXT,
            dryrun INTEGER,
            rulegraph_data TEXT,
            total_job_count INTEGER,
            jobs_finished INTEGER
        );
        CREATE TABLE rules (
            id INTEGER PRIMARY KEY,
            name TEXT,
            workflow_id TEXT,
            total_job_count INTEGER,
            jobs_finished INTEGER,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id)
        );
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            snakemake_id INTEGER,
            workflow_id TEXT,
            rule_id INTEGER,
            wildcards TEXT,
            resources TEXT,
            shellcmd TEXT,
            threads INTEGER,
            priority REAL,
            status TEXT,
            started_at TEXT,
            end_time TEXT,
            group_id TEXT,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id),
            FOREIGN KEY (rule_id) REFERENCES rules(id)
        );
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            path TEXT,
            file_type TEXT,
            job_id INTEGER,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        );
        CREATE TABLE errors (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            exception TEXT,
            location TEXT,
            traceback TEXT,
            file TEXT,
            line INTEGER,
            rule_id INTEGER,
            workflow_id TEXT,
            FOREIGN KEY (rule_id) REFERENCES rules(id),
            FOREIGN KEY (workflow_id) REFERENCES workflows(id)
        );
    """)
    rulegraph = json.dumps(
        {"nodes": ["rule_a", "rule_b"], "edges": [["rule_a", "rule_b"]]}
    )
    conn.execute(
        "INSERT INTO workflows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            SAMPLE_WORKFLOW_ID,
            "/path/to/Snakefile",
            "2025-06-01T10:00:00",
            "2025-06-01T11:00:00",
            "finished",
            "snakemake --cores 4",
            0,
            rulegraph,
            3,
            3,
        ),
    )
    conn.execute(
        "INSERT INTO rules VALUES (?, ?, ?, ?, ?)",
        (1, "rule_a", SAMPLE_WORKFLOW_ID, 2, 2),
    )
    conn.execute(
        "INSERT INTO rules VALUES (?, ?, ?, ?, ?)",
        (2, "rule_b", SAMPLE_WORKFLOW_ID, 1, 1),
    )
    conn.execute(
        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            0,
            SAMPLE_WORKFLOW_ID,
            1,
            json.dumps({"sample": "A"}),
            json.dumps({"mem_mb": 1000}),
            "echo hello",
            1,
            0.0,
            "finished",
            "2025-06-01T10:00:05",
            "2025-06-01T10:05:00",
            None,
        ),
    )
    conn.execute(
        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            2,
            1,
            SAMPLE_WORKFLOW_ID,
            1,
            json.dumps({"sample": "B"}),
            json.dumps({"mem_mb": 1000}),
            "echo world",
            1,
            0.0,
            "finished",
            "2025-06-01T10:05:01",
            "2025-06-01T10:10:00",
            None,
        ),
    )
    conn.execute(
        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            3,
            2,
            SAMPLE_WORKFLOW_ID,
            2,
            None,
            None,
            "cat A B > merged",
            2,
            0.0,
            "finished",
            "2025-06-01T10:10:01",
            "2025-06-01T10:15:00",
            None,
        ),
    )
    conn.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?)",
        (1, "data/A.txt", "INPUT", 1),
    )
    conn.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?)",
        (2, "results/A.out", "OUTPUT", 1),
    )
    conn.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?)",
        (3, "results/merged.out", "OUTPUT", 3),
    )
    conn.execute(
        "INSERT INTO errors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "2025-06-01T10:04:00",
            "FileNotFoundError: missing input",
            None,
            "Traceback ...",
            "Snakefile",
            10,
            1,
            SAMPLE_WORKFLOW_ID,
        ),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def snkmt_db(tmp_path):
    """Create a temporary snkmt.db with sample data."""
    db_path = tmp_path / "data" / "snkmt.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_snkmt_db(db_path)
    return db_path


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        DATA_DIR=str(tmp_path / "data"),
    )


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(data_dir=tmp_path)


@pytest.fixture
def mock_backend():
    backend = AsyncMock()
    backend.prepare = AsyncMock(
        return_value=("/scratch/test/jobs/test-job-id", "main", "abc123")
    )
    backend.setup = AsyncMock()
    backend.launch = AsyncMock()
    backend.monitor = AsyncMock(return_value=0)
    backend.list_workflow_files = AsyncMock(
        return_value=[{"path": "results/output.csv", "size": 1234}]
    )

    async def fake_stream():
        yield b"file content chunk 1"
        yield b"file content chunk 2"

    backend.stream_file = MagicMock(side_effect=lambda *a, **kw: fake_stream())
    backend.cleanup = AsyncMock()
    backend.restore_cache = AsyncMock()
    backend.save_cache = AsyncMock()
    backend.check_connectivity = AsyncMock(return_value=True)
    backend.sync_snkmt_db = AsyncMock()
    return backend


@pytest.fixture(autouse=True)
def reset_sse_app_status():
    """Reset sse_starlette AppStatus event to avoid cross-loop errors."""
    AppStatus.should_exit_event = asyncio.Event()


@pytest.fixture
async def async_client(store, mock_backend, settings):
    backend_config = LocalConfig()
    app.state.app = AppState(
        store=store,
        backend=mock_backend,
        settings=settings,
        health_cache={"backend_ok": None, "checked_at": 0.0},
        default_snakemake_args=backend_config.default_snakemake_args,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
