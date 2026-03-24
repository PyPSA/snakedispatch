from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.backends.local import LocalBackend
from app.config import LocalConfig


@pytest.fixture
def local_config(tmp_path) -> LocalConfig:
    return LocalConfig(scratch_dir=str(tmp_path / "scratch"), pixi_path="pixi")


@pytest.fixture
def backend(local_config) -> LocalBackend:
    return LocalBackend(local_config)


class TestWorkDir:
    def test_returns_expected_path(self, backend, local_config):
        result = backend.work_dir("job-123")
        assert result == f"{local_config.scratch_dir}/jobs/job-123"


class TestPrepareLocal:
    async def test_copies_local_directory(self, backend, tmp_path):
        # Create a source directory with files
        src = tmp_path / "workflow"
        src.mkdir()
        (src / "Snakefile").write_text("rule all: pass")
        (src / "config.yaml").write_text("key: value")

        work_dir, git_ref, git_sha = await backend.prepare("job-1", str(src))

        assert Path(work_dir).exists()
        assert (Path(work_dir) / "Snakefile").read_text() == "rule all: pass"
        assert git_ref is None
        assert git_sha is None

    async def test_git_url_inits_bare_repo_and_creates_worktree(
        self, backend, tmp_path
    ):
        """prepare() with https:// URL uses bare repo + worktree flow."""
        call_log: list[tuple[str, ...]] = []

        async def fake_run_git_cmd(*args):
            call_log.append(args)
            if "rev-parse" in args:
                return "abc123def456\n"
            return ""

        with patch.object(backend, "_run_git_cmd", side_effect=fake_run_git_cmd):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-git", "https://github.com/org/repo.git", git_ref="main"
            )

        assert git_sha == "abc123def456"
        assert git_ref == "main"

        cmds = [" ".join(c) for c in call_log]
        assert any("git init --bare" in c for c in cmds)
        assert any("git -C" in c and "fetch" in c for c in cmds)
        assert any("worktree add --detach" in c for c in cmds)
        assert any("rev-parse HEAD" in c for c in cmds)
        assert any("update-ref -d" in c for c in cmds)
        # No ls-remote since git_ref was provided
        assert not any("ls-remote" in c for c in cmds)

    async def test_git_fetch_failure_raises(self, backend, tmp_path):
        """Non-zero git fetch exit code raises RuntimeError."""
        call_count = 0

        async def fake_run_git_cmd(*args):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return ""  # init succeeds
            raise RuntimeError("git command failed (exit 128): git fetch")

        with patch.object(backend, "_run_git_cmd", side_effect=fake_run_git_cmd):
            with pytest.raises(RuntimeError, match="git.*failed"):
                await backend.prepare(
                    "job-fail",
                    "https://github.com/org/missing.git",
                    git_ref="nonexistent",
                )

    async def test_prepare_default_branch_resolution(self, backend, tmp_path):
        """git_ref=None resolves default branch via ls-remote."""
        call_log: list[tuple[str, ...]] = []

        async def fake_run_git_cmd(*args):
            call_log.append(args)
            if "ls-remote" in args:
                return "ref: refs/heads/develop\tHEAD\nabc123\tHEAD\n"
            if "rev-parse" in args:
                return "sha256abc\n"
            return ""

        with patch.object(backend, "_run_git_cmd", side_effect=fake_run_git_cmd):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-default", "https://github.com/org/repo.git"
            )

        assert git_ref == "develop"
        assert git_sha == "sha256abc"
        cmds = [" ".join(c) for c in call_log]
        assert any("ls-remote" in c for c in cmds)

    async def test_prepare_with_pr_ref(self, backend, tmp_path):
        """refs/pull/123/head as git_ref is passed through to fetch."""
        call_log: list[tuple[str, ...]] = []

        async def fake_run_git_cmd(*args):
            call_log.append(args)
            if "rev-parse" in args:
                return "pr-sha-123\n"
            return ""

        with patch.object(backend, "_run_git_cmd", side_effect=fake_run_git_cmd):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-pr",
                "https://github.com/org/repo.git",
                git_ref="refs/pull/123/head",
            )

        assert git_ref == "refs/pull/123/head"
        assert git_sha == "pr-sha-123"
        cmds = [" ".join(c) for c in call_log]
        fetch_cmd = next(c for c in cmds if "fetch" in c)
        assert "refs/pull/123/head" in fetch_cmd

    async def test_two_jobs_same_repo_share_bare_repo(self, backend, tmp_path):
        """Two jobs from the same repo share one bare repo, get separate worktrees."""
        init_calls: list[tuple[str, ...]] = []

        async def fake_run_git_cmd(*args):
            if "init" in args:
                init_calls.append(args)
            if "rev-parse" in args:
                return "sha1\n"
            return ""

        with patch.object(backend, "_run_git_cmd", side_effect=fake_run_git_cmd):
            w1, _, _ = await backend.prepare(
                "job-1", "https://github.com/org/repo.git", git_ref="main"
            )
            w2, _, _ = await backend.prepare(
                "job-2", "https://github.com/org/repo.git", git_ref="main"
            )

        assert w1 != w2
        # Both calls init the same bare repo path
        init_paths = [c[-1] for c in init_calls]
        assert len(init_paths) == 2
        assert init_paths[0] == init_paths[1]


class TestSetup:
    async def test_writes_extra_files(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)

        await backend.setup(
            "job-1",
            str(work_dir),
            extra_files={"sub/config.yaml": "key: value", "license.txt": "MIT"},
        )

        assert (work_dir / "sub" / "config.yaml").read_text() == "key: value"
        assert (work_dir / "license.txt").read_text() == "MIT"

    async def test_noop_when_no_extra_files(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        await backend.setup("job-1", str(work_dir), extra_files=None)


class TestLaunch:
    async def test_writes_run_script(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)

        with patch.object(
            backend, "_poll_until_pid_file", new=AsyncMock(return_value=None)
        ):
            await backend.launch("job-1", str(work_dir), configfile=None)

        script = work_dir / ".run.sh"
        assert script.exists()
        content = script.read_text()
        assert "echo $$ > .pid" in content
        assert "snakemake" in content

    async def test_rejects_path_traversal_configfile(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="must not contain"):
            await backend.launch("job-1", str(work_dir), configfile="../etc/passwd")


class TestListWorkflowFiles:
    async def test_lists_files_excluding_hidden(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        (work_dir / "Snakefile").write_text("rule all: pass")
        (work_dir / "results").mkdir()
        (work_dir / "results" / "output.csv").write_text("a,b,c")
        (work_dir / ".hidden").write_text("secret")
        (work_dir / ".git").mkdir()
        (work_dir / ".git" / "config").write_text("gitconfig")

        files = await backend.list_workflow_files("job-1", str(work_dir))

        paths = {f["path"] for f in files}
        assert "Snakefile" in paths
        assert "results/output.csv" in paths
        assert ".hidden" not in paths
        assert ".git/config" not in paths


class TestStreamFile:
    async def test_streams_file_content(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        (work_dir / "results").mkdir()
        (work_dir / "results" / "output.csv").write_text("a,b,c\n1,2,3")

        chunks = [
            chunk
            async for chunk in backend.stream_file(
                "job-1", str(work_dir), "results/output.csv"
            )
        ]

        assert b"".join(chunks) == b"a,b,c\n1,2,3"


class TestCache:
    async def test_save_and_restore_cache(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        (work_dir / "data").mkdir()
        (work_dir / "data" / "input.csv").write_text("cached data")

        await backend.save_cache("job-1", str(work_dir), "my-cache", ["data"])

        # Verify cache was saved
        cache_dir = tmp_path / "scratch" / "cache" / "my-cache" / "data"
        assert cache_dir.exists()
        assert (cache_dir / "input.csv").read_text() == "cached data"

        # Restore to a new work dir
        work_dir2 = tmp_path / "scratch" / "jobs" / "job-2"
        work_dir2.mkdir(parents=True)
        await backend.restore_cache("job-2", str(work_dir2), "my-cache", ["data"])

        assert (work_dir2 / "data" / "input.csv").read_text() == "cached data"

    async def test_restore_nonexistent_cache_is_noop(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        await backend.restore_cache("job-1", str(work_dir), "no-such-key", ["data"])


class TestMonitor:
    async def test_returns_exit_code_when_exitcode_file_appears(
        self, backend, tmp_path
    ):
        work_dir = tmp_path / "scratch" / "jobs" / "job-mon"
        work_dir.mkdir(parents=True)
        log_path = work_dir / ".stdout.log"
        exit_path = work_dir / ".exitcode"
        log_path.write_text("line1\nline2\n")
        exit_path.write_text("0")

        lines: list[str] = []

        def cb(line: str) -> None:
            lines.append(line)

        code = await backend.monitor("job-mon", str(work_dir), cb)

        assert code == 0
        assert "line1" in lines
        assert "line2" in lines

    async def test_returns_nonzero_exit_code(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-fail"
        work_dir.mkdir(parents=True)
        (work_dir / ".stdout.log").write_text("")
        (work_dir / ".exitcode").write_text("1")

        code = await backend.monitor(
            "job-fail", str(work_dir), lambda _line: asyncio.sleep(0)
        )

        assert code == 1

    async def test_byte_offset_skips_already_seen_data(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-offset"
        work_dir.mkdir(parents=True)
        log_path = work_dir / ".stdout.log"
        log_path.write_text("already-seen\nnew-line\n")
        (work_dir / ".exitcode").write_text("0")

        lines: list[str] = []

        def cb(line: str) -> None:
            lines.append(line)

        # byte_offset = len("already-seen\n") skips first line
        offset = len("already-seen\n")
        await backend.monitor("job-offset", str(work_dir), cb, byte_offset=offset)

        assert "already-seen" not in lines
        assert "new-line" in lines

    async def test_consecutive_errors_raise_after_10(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-err"
        work_dir.mkdir(parents=True)
        # Log file must exist so _drain_log calls asyncio.to_thread
        (work_dir / ".stdout.log").write_text("data")

        with (
            patch(
                "app.backends.local.asyncio.to_thread",
                side_effect=OSError("disk error"),
            ),
            patch("app.backends.local.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="10 consecutive"),
        ):
            await backend.monitor("job-err", str(work_dir), AsyncMock())

    async def test_cancelled_error_propagates(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-cancel"
        work_dir.mkdir(parents=True)

        async def _raise_cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        with patch("app.backends.local.asyncio.sleep", side_effect=_raise_cancelled):
            with pytest.raises(asyncio.CancelledError):
                await backend.monitor("job-cancel", str(work_dir), AsyncMock())


class TestSyncSnkmtDb:
    async def test_copies_db_to_local_path(self, backend, tmp_path):
        import sqlite3

        work_dir = tmp_path / "scratch" / "jobs" / "job-sync"
        work_dir.mkdir(parents=True)
        src_db = work_dir / "snkmt.db"
        conn = sqlite3.connect(str(src_db))
        conn.execute("CREATE TABLE workflows (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO workflows VALUES ('wf-1')")
        conn.commit()
        conn.close()

        local_path = tmp_path / "data" / "jobs" / "job-sync" / "snkmt.db"
        await backend.sync_snkmt_db("job-sync", str(work_dir), local_path)

        assert local_path.exists()
        conn2 = sqlite3.connect(str(local_path))
        rows = conn2.execute("SELECT id FROM workflows").fetchall()
        conn2.close()
        assert rows == [("wf-1",)]

    async def test_noop_when_snkmt_db_missing(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-nosync"
        work_dir.mkdir(parents=True)
        local_path = tmp_path / "data" / "jobs" / "job-nosync" / "snkmt.db"

        await backend.sync_snkmt_db("job-nosync", str(work_dir), local_path)

        assert not local_path.exists()


class TestCheckConnectivity:
    async def test_returns_true(self, backend):
        assert await backend.check_connectivity() is True


class TestCleanup:
    async def test_removes_work_dir(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "job-1"
        work_dir.mkdir(parents=True)
        (work_dir / "Snakefile").write_text("rule all: pass")

        await backend.cleanup("job-1", str(work_dir))

        assert not work_dir.exists()

    async def test_cleanup_nonexistent_dir_is_noop(self, backend, tmp_path):
        work_dir = tmp_path / "scratch" / "jobs" / "nonexistent"
        await backend.cleanup("job-1", str(work_dir))


class TestConfigLoading:
    def test_load_local_config(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("local:\n  scratch_dir: /tmp/test\n  pixi_path: pixi\n")

        backend_config, overrides = load_config(str(config_file))
        assert isinstance(backend_config, LocalConfig)
        assert backend_config.scratch_dir == "/tmp/test"
        assert backend_config.pixi_path == "pixi"
        assert overrides == {}

    def test_load_slurm_ssh_config(self, tmp_path):
        from app.config import SlurmSSHConfig, load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "slurm_ssh:\n"
            "  host: hpc.example.com\n"
            "  user: myuser\n"
            "  pixi_path: /opt/pixi\n"
            "  scratch_dir: /scratch\n"
        )

        backend_config, overrides = load_config(str(config_file))
        assert isinstance(backend_config, SlurmSSHConfig)
        assert backend_config.host == "hpc.example.com"
        assert overrides == {}

    def test_loads_app_settings_from_yaml(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "local:\n  scratch_dir: /tmp/test\n"
            "DATA_DIR: /custom/data\n"
            "AUTO_CLEANUP_AFTER_HOURS: 72\n"
        )

        backend_config, overrides = load_config(str(config_file))
        assert isinstance(backend_config, LocalConfig)
        assert overrides == {"DATA_DIR": "/custom/data", "AUTO_CLEANUP_AFTER_HOURS": 72}

    def test_rejects_unknown_keys(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("local:\n  scratch_dir: /tmp\nunknown_key: value\n")

        with pytest.raises(ValueError, match="Unknown keys"):
            load_config(str(config_file))

    def test_rejects_multiple_backends(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "local:\n  scratch_dir: /tmp\n"
            "slurm_ssh:\n  host: x\n  user: y\n  pixi_path: p\n  scratch_dir: /s\n"
        )

        with pytest.raises(ValueError, match="exactly one backend"):
            load_config(str(config_file))

    def test_rejects_missing_file(self):
        from app.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_rejects_no_backend_key(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("DATA_DIR: /custom/data\n")

        with pytest.raises(ValueError, match="exactly one backend"):
            load_config(str(config_file))

    def test_rejects_empty_config(self, tmp_path):
        from app.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="empty or invalid"):
            load_config(str(config_file))


class TestLocalBackendIntegration:
    """End-to-end test with a real LocalBackend and a fake pixi script."""

    @pytest.mark.skipif(
        not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(),
        reason="bash required",
    )
    async def test_execute_job_full_lifecycle(self, tmp_path):
        """Run execute_job with a real LocalBackend and a no-op pixi script."""
        from app.models import JobStatus
        from app.store import JobStore
        from app.tasks import execute_job

        # Create a fake pixi that just exits 0 (ignores all args)
        fake_pixi = tmp_path / "bin" / "fake_pixi"
        fake_pixi.parent.mkdir(parents=True)
        fake_pixi.write_text("#!/bin/bash\nexit 0\n")
        fake_pixi.chmod(0o755)

        # Create a minimal workflow directory
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()
        (workflow_dir / "Snakefile").write_text("rule all:\n    pass\n")

        config = LocalConfig(
            scratch_dir=str(tmp_path / "scratch"),
            pixi_path=str(fake_pixi),
        )
        backend = LocalBackend(config)
        store = JobStore(data_dir=tmp_path / "data")
        store.create_job("integration-job-1")

        await execute_job(
            store,
            backend,
            "integration-job-1",
            workflow=str(workflow_dir),
        )

        record = store.get_job("integration-job-1")
        assert record is not None
        assert record.status == JobStatus.COMPLETED
        assert record.exit_code == 0
        assert record.work_dir is not None
        assert record.started_at is not None
        assert record.completed_at is not None
        # Snakefile should have been copied to work dir
        assert (Path(record.work_dir) / "Snakefile").exists()


class TestCheckJobStatus:
    async def test_returns_exit_code_when_exitcode_file_exists(self, backend, tmp_path):
        work_dir = tmp_path / "job"
        work_dir.mkdir()
        (work_dir / ".exitcode").write_text("0")
        result = await backend.check_job_status("j1", str(work_dir))
        assert result == 0

    async def test_returns_none_when_pid_alive(self, backend, tmp_path):
        work_dir = tmp_path / "job"
        work_dir.mkdir()
        (work_dir / ".pid").write_text(str(os.getpid()))
        result = await backend.check_job_status("j2", str(work_dir))
        assert result is None  # our own PID is alive

    async def test_returns_negative_one_when_pid_dead(self, backend, tmp_path):
        work_dir = tmp_path / "job"
        work_dir.mkdir()
        (work_dir / ".pid").write_text("999999999")  # non-existent PID
        result = await backend.check_job_status("j3", str(work_dir))
        assert result == -1

    async def test_returns_none_when_nothing_exists(self, backend, tmp_path):
        work_dir = tmp_path / "job"
        work_dir.mkdir()
        result = await backend.check_job_status("j4", str(work_dir))
        assert result is None
