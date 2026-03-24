from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.backends.slurm_ssh import (
    MONITOR_DEAD_SENTINEL,
    MONITOR_LOG_MARKER,
    MONITOR_RUNNING_SENTINEL,
    SlurmSSHBackend,
    _build_rsync_filter,
    _build_status_check_cmd,
    _parse_job_status,
)
from app.config import SlurmSSHConfig
from app.models import SnkmtJobResponse
from app.utils import build_wrapper_script


@pytest.fixture
def slurm_config() -> SlurmSSHConfig:
    return SlurmSSHConfig(
        host="hpc.example.com",
        user="testuser",
        pixi_path="/opt/pixi",
        scratch_dir="/scratch",
    )


@pytest.fixture
def backend(slurm_config) -> SlurmSSHBackend:
    return SlurmSSHBackend(slurm_config)


# ---- Pure function tests ----


class TestBuildRsyncFilters:
    def test_simple_dir(self):
        result = _build_rsync_filter(["data"])
        assert "--include=data/" in result
        assert "--include='data/**'" in result
        assert "--exclude='*'" in result

    def test_nested_dir_includes_parents(self):
        result = _build_rsync_filter(["dir1/dir2"])
        assert "--include=dir1/" in result
        assert "--exclude='*'" in result

    def test_glob_pattern(self):
        result = _build_rsync_filter(["data/*.csv"])
        assert "'data/*.csv'" in result
        assert "--exclude='*'" in result

    def test_multiple_dirs(self):
        result = _build_rsync_filter(["data", "resources"])
        assert "--include=data/" in result
        assert "--include=resources/" in result
        assert result.endswith("--exclude='*'")

    def test_empty_list(self):
        result = _build_rsync_filter([])
        assert result == "--exclude='*'"

    def test_strips_leading_slash(self):
        result = _build_rsync_filter(["/data"])
        assert "--include=data/" in result


class TestBuildStatusCheckCmd:
    def test_contains_exitcode_check(self):
        cmd = _build_status_check_cmd("'/work/dir'")
        assert ".exitcode" in cmd
        assert "'/work/dir'" in cmd

    def test_contains_pid_probe(self):
        cmd = _build_status_check_cmd("'/work/dir'")
        assert ".pid" in cmd
        assert "kill -0" in cmd

    def test_contains_sentinels(self):
        cmd = _build_status_check_cmd("/w")
        assert MONITOR_RUNNING_SENTINEL in cmd
        assert MONITOR_DEAD_SENTINEL in cmd


class TestParseJobStatus:
    def test_running_sentinel_returns_none(self):
        assert _parse_job_status(MONITOR_RUNNING_SENTINEL) is None

    def test_dead_sentinel_returns_minus_one(self):
        assert _parse_job_status(MONITOR_DEAD_SENTINEL) == -1

    def test_zero_exit_code(self):
        assert _parse_job_status("0") == 0

    def test_nonzero_exit_code(self):
        assert _parse_job_status("1") == 1
        assert _parse_job_status("137") == 137

    def test_empty_string_returns_none(self):
        assert _parse_job_status("") is None

    def test_whitespace_returns_none(self):
        assert _parse_job_status("  ") is None

    def test_garbage_returns_none(self):
        assert _parse_job_status("not-a-number") is None

    def test_negative_exit_code(self):
        assert _parse_job_status("-1") == -1


# ---- SlurmSSHBackend unit tests with mocked SSH ----


def _make_mock_conn():
    """Return a mock asyncssh connection."""
    conn = AsyncMock()
    conn.is_closed = MagicMock(return_value=False)
    return conn


class TestWorkDir:
    def test_returns_expected_path(self, backend):
        assert backend.work_dir("job-1") == "/scratch/jobs/job-1"


class TestBuildWrapperScript:
    def test_script_contains_pid_and_snakemake(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        assert "echo $$ > .pid" in script
        assert "snakemake" in script
        assert "/opt/pixi" in script

    def test_configfile_arg_included(self):
        script = build_wrapper_script(
            "/opt/pixi", "/work/snkmt.db", "config/prod.yaml", None
        )
        assert "--configfile config/prod.yaml" in script

    def test_extra_args_included(self):
        script = build_wrapper_script(
            "/opt/pixi", "/work/snkmt.db", None, ["--cores", "4"]
        )
        assert "--cores 4" in script

    def test_no_configfile_no_extra_args(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        assert "--configfile" not in script


class TestMonitorParsing:
    """Test the MARKER-based log/status parsing without real SSH."""

    async def test_parses_exit_code_zero(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = f"log line 1\nlog line 2\n{MONITOR_LOG_MARKER}\n0\n"
        mock_result.exit_status = 0

        lines_received = []

        def log_callback(line: str) -> None:
            lines_received.append(line)

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            exit_code = await backend.monitor(
                "job-1", "/scratch/jobs/job-1", log_callback
            )

        assert exit_code == 0
        assert "log line 1" in lines_received
        assert "log line 2" in lines_received

    async def test_parses_nonzero_exit_code(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = f"{MONITOR_LOG_MARKER}\n1\n"
        mock_result.exit_status = 0

        def log_callback(line: str) -> None:
            pass

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            exit_code = await backend.monitor(
                "job-1", "/scratch/jobs/job-1", log_callback
            )

        assert exit_code == 1

    async def test_running_status_continues_polling(self, backend):
        """RUNNING response keeps polling; next response has exit code."""
        running_result = MagicMock()
        running_result.stdout = f"{MONITOR_LOG_MARKER}\nRUNNING\n"
        running_result.exit_status = 0

        done_result = MagicMock()
        done_result.stdout = f"{MONITOR_LOG_MARKER}\n0\n"
        done_result.exit_status = 0

        call_count = 0

        async def mock_run_ssh(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return running_result if call_count == 1 else done_result

        lines = []

        def log_callback(line: str) -> None:
            lines.append(line)

        with (
            patch.object(backend, "_run_ssh", side_effect=mock_run_ssh),
            patch("app.backends.slurm_ssh.asyncio.sleep", new_callable=AsyncMock),
        ):
            exit_code = await backend.monitor(
                "job-1", "/scratch/jobs/job-1", log_callback
            )

        assert exit_code == 0
        assert call_count == 2


class TestLaunchRejection:
    async def test_rejects_path_traversal_configfile(self, backend):
        with pytest.raises(ValueError, match="must not contain"):
            await backend.launch(
                "job-1", "/scratch/jobs/job-1", configfile="../etc/passwd"
            )


class TestCheckConnectivity:
    async def test_returns_true_when_ssh_succeeds(self, backend):
        ok = MagicMock()
        ok.exit_status = 0

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=ok)):
            result = await backend.check_connectivity()

        assert result is True

    async def test_returns_false_on_os_error(self, backend):
        with patch.object(
            backend, "_run_ssh", side_effect=OSError("connection refused")
        ):
            result = await backend.check_connectivity()

        assert result is False

    async def test_returns_false_on_timeout(self, backend):
        with patch.object(backend, "_run_ssh", side_effect=TimeoutError()):
            result = await backend.check_connectivity()

        assert result is False


class TestListWorkflowFiles:
    async def test_parses_find_output(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = "results/output.txt\t1024\nresults/other.csv\t512\n"

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            files = await backend.list_workflow_files("job-1", "/scratch/jobs/job-1")

        assert len(files) == 2
        assert files[0] == {"path": "results/output.txt", "size": 1024}
        assert files[1] == {"path": "results/other.csv", "size": 512}

    async def test_returns_empty_on_no_output(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            files = await backend.list_workflow_files("job-1", "/scratch/jobs/job-1")

        assert files == []

    async def test_skips_lines_without_tab(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = "bad line\nresults/ok.txt\t100\n"

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            files = await backend.list_workflow_files("job-1", "/scratch/jobs/job-1")

        assert len(files) == 1
        assert files[0]["path"] == "results/ok.txt"

    async def test_handles_none_stdout(self, backend):
        mock_result = MagicMock()
        mock_result.stdout = None

        with patch.object(backend, "_run_ssh", new=AsyncMock(return_value=mock_result)):
            files = await backend.list_workflow_files("job-1", "/scratch/jobs/job-1")

        assert files == []


class TestRestoreCache:
    async def test_skips_when_cache_not_found(self, backend):
        miss = MagicMock()
        miss.exit_status = 1

        calls = []

        async def run_ssh(cmd, **kwargs):
            calls.append(cmd)
            return miss

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            await backend.restore_cache(
                "job-1", "/scratch/jobs/job-1", "key1", ["data"]
            )

        # Only the test -d check should have been called
        assert len(calls) == 1
        assert "test -d" in calls[0]

    async def test_runs_rsync_when_cache_found(self, backend):
        hit = MagicMock()
        hit.exit_status = 0

        commands = []

        async def run_ssh(cmd, **kwargs):
            commands.append(cmd)
            return hit

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            await backend.restore_cache(
                "job-1", "/scratch/jobs/job-1", "key1", ["data"]
            )

        assert any("rsync" in c for c in commands)
        assert any("key1" in c for c in commands)


class TestSaveCache:
    async def test_creates_cache_dir_and_rsyncs(self, backend):
        ok = MagicMock()
        ok.exit_status = 0

        commands = []

        async def run_ssh(cmd, **kwargs):
            commands.append(cmd)
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            await backend.save_cache("job-1", "/scratch/jobs/job-1", "mykey", ["data"])

        assert any("mkdir" in c for c in commands)
        assert any("rsync" in c for c in commands)
        assert any("mykey" in c for c in commands)


class TestPrepareUrl:
    async def test_url_workflow_inits_bare_repo_and_creates_worktree(self, backend):
        ok = MagicMock()
        ok.exit_status = 0
        ok.stdout = ""

        ls_remote_result = MagicMock()
        ls_remote_result.exit_status = 0
        ls_remote_result.stdout = "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n"

        sha_result = MagicMock()
        sha_result.exit_status = 0
        sha_result.stdout = "abc123\n"

        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            if "ls-remote" in cmd:
                return ls_remote_result
            if "rev-parse" in cmd:
                return sha_result
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-1", "https://github.com/org/repo.git"
            )

        assert work_dir == "/scratch/jobs/job-1"
        assert git_sha == "abc123"
        assert git_ref == "main"

        assert any("git init --bare" in cmd for cmd in ssh_calls)
        assert any("fetch" in cmd for cmd in ssh_calls)
        assert any("worktree add --detach" in cmd for cmd in ssh_calls)
        assert any("rev-parse HEAD" in cmd for cmd in ssh_calls)
        assert any("update-ref -d" in cmd for cmd in ssh_calls)

    async def test_url_with_ref_fetches_to_job_scoped_ref(self, backend):
        ok = MagicMock()
        ok.exit_status = 0
        ok.stdout = ""

        sha_result = MagicMock()
        sha_result.exit_status = 0
        sha_result.stdout = "deadbeef\n"

        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            if "rev-parse" in cmd:
                return sha_result
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-1", "https://github.com/org/repo.git", git_ref="v1.0"
            )

        assert git_ref == "v1.0"
        fetch_cmd = next(c for c in ssh_calls if "fetch" in c)
        assert "v1.0" in fetch_cmd
        assert "refs/snakedispatch/job-1" in fetch_cmd
        # No ls-remote when ref is provided
        assert not any("ls-remote" in cmd for cmd in ssh_calls)

    async def test_prepare_resolves_default_branch(self, backend):
        ok = MagicMock()
        ok.exit_status = 0
        ok.stdout = ""

        ls_remote_result = MagicMock()
        ls_remote_result.exit_status = 0
        ls_remote_result.stdout = "ref: refs/heads/develop\tHEAD\nsha1\tHEAD\n"

        sha_result = MagicMock()
        sha_result.exit_status = 0
        sha_result.stdout = "sha256abc\n"

        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            if "ls-remote" in cmd:
                return ls_remote_result
            if "rev-parse" in cmd:
                return sha_result
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-1", "https://github.com/org/repo.git"
            )

        assert git_ref == "develop"
        assert git_sha == "sha256abc"
        assert any("ls-remote" in cmd for cmd in ssh_calls)

    async def test_prepare_pr_ref(self, backend):
        ok = MagicMock()
        ok.exit_status = 0
        ok.stdout = ""

        sha_result = MagicMock()
        sha_result.exit_status = 0
        sha_result.stdout = "pr-sha-123\n"

        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            if "rev-parse" in cmd:
                return sha_result
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-1",
                "https://github.com/org/repo.git",
                git_ref="refs/pull/123/head",
            )

        assert git_ref == "refs/pull/123/head"
        assert git_sha == "pr-sha-123"
        fetch_cmd = next(c for c in ssh_calls if "fetch" in c)
        assert "refs/pull/123/head" in fetch_cmd


class TestMonitorByteOffset:
    async def test_byte_offset_passed_to_tail(self, backend):
        done_result = MagicMock()
        done_result.stdout = f"{MONITOR_LOG_MARKER}\n0\n"
        done_result.exit_status = 0

        received_cmds: list[str] = []

        async def run_ssh(cmd, **kwargs):
            received_cmds.append(cmd)
            return done_result

        def log_callback(line: str) -> None:
            pass

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            await backend.monitor(
                "job-1", "/scratch/jobs/job-1", log_callback, byte_offset=100
            )

        assert received_cmds
        # tail -c +N is 1-based, so offset 100 → "+101"
        assert "+101" in received_cmds[0]


class TestMonitorErrorRecovery:
    async def test_consecutive_errors_raise_after_10(self, backend):
        with (
            patch.object(backend, "_run_ssh", side_effect=OSError("connection lost")),
            patch("app.backends.slurm_ssh.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="10 consecutive"),
        ):
            await backend.monitor("job-1", "/scratch/jobs/job-1", AsyncMock())

    async def test_unexpected_status_logged_not_crashed(self, backend):
        """Unparsable status (not int, not RUNNING) should be treated as running."""
        unknown_result = MagicMock()
        unknown_result.stdout = f"{MONITOR_LOG_MARKER}\nUNKNOWN_STATUS\n"
        unknown_result.exit_status = 0

        done_result = MagicMock()
        done_result.stdout = f"{MONITOR_LOG_MARKER}\n0\n"
        done_result.exit_status = 0

        call_count = 0

        async def run_ssh(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            return unknown_result if call_count == 1 else done_result

        with (
            patch.object(backend, "_run_ssh", side_effect=run_ssh),
            patch("app.backends.slurm_ssh.asyncio.sleep", new_callable=AsyncMock),
        ):
            exit_code = await backend.monitor(
                "job-1", "/scratch/jobs/job-1", AsyncMock()
            )

        assert exit_code == 0


class TestPrepare:
    async def test_prepare_git_url_runs_bare_repo_flow(self, backend):
        ok = MagicMock()
        ok.exit_status = 0
        ok.stdout = ""

        ls_remote_result = MagicMock()
        ls_remote_result.exit_status = 0
        ls_remote_result.stdout = "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n"

        sha_result = MagicMock()
        sha_result.exit_status = 0
        sha_result.stdout = "abc123\n"

        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            if "ls-remote" in cmd:
                return ls_remote_result
            if "rev-parse" in cmd:
                return sha_result
            return ok

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            work_dir, git_ref, git_sha = await backend.prepare(
                "job-1", "https://github.com/org/repo.git"
            )

        assert work_dir == backend.work_dir("job-1")
        assert any("git init --bare" in cmd for cmd in ssh_calls)

    async def test_prepare_local_upload_uses_upload_dir(self, backend, tmp_path):
        src = tmp_path / "workflow"
        src.mkdir()
        (src / "Snakefile").write_text("rule all: pass")

        with patch.object(backend, "_upload_dir", new=AsyncMock()) as mock_upload:
            work_dir, git_ref, git_sha = await backend.prepare("job-2", str(src))

        assert work_dir == backend.work_dir("job-2")
        assert git_ref is None
        assert git_sha is None
        mock_upload.assert_called_once()


class TestRunSsh:
    async def test_retries_once_on_stale_connection(self, backend):
        """_run_ssh closes and retries once when the first attempt raises OSError."""
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.exit_status = 0
        mock_result.stdout = "output"
        mock_conn.run = AsyncMock(return_value=mock_result)

        get_conn_calls = 0

        async def get_conn():
            nonlocal get_conn_calls
            get_conn_calls += 1
            if get_conn_calls == 1:
                raise OSError("stale connection")
            return mock_conn

        with (
            patch.object(backend, "_get_conn", side_effect=get_conn),
            patch.object(backend, "_close_conn", new_callable=AsyncMock) as mock_close,
        ):
            result = await backend._run_ssh("echo hello", check=False)

        mock_close.assert_called_once()
        assert get_conn_calls == 2
        assert result.exit_status == 0


class TestGetConn:
    async def test_proxy_jump_creates_tunnel_and_passes_it_to_conn(self):
        """_get_conn opens a tunnel to the jump host, then passes it as tunnel=."""
        config = SlurmSSHConfig(
            host="hpc.example.com",
            user="testuser",
            pixi_path="/opt/pixi",
            scratch_dir="/scratch",
            proxy_jump="jump@bastion.example.com",
        )
        backend = SlurmSSHBackend(config)

        mock_tunnel = MagicMock()
        mock_conn = MagicMock()
        mock_conn.is_closed = MagicMock(return_value=False)

        connect_calls: list[tuple[tuple, dict]] = []
        call_index = 0

        async def mock_connect(*args, **kwargs):
            nonlocal call_index
            connect_calls.append((args, kwargs))
            call_index += 1
            return mock_tunnel if call_index == 1 else mock_conn

        with patch("app.backends.slurm_ssh.asyncssh.connect", side_effect=mock_connect):
            conn = await backend._get_conn()

        assert conn is mock_conn
        assert len(connect_calls) == 2
        # First call: tunnel to bastion with jump user
        first_args, first_kwargs = connect_calls[0]
        assert first_args == ("bastion.example.com",)
        assert first_kwargs["username"] == "jump"
        # Second call: main connection via tunnel
        _, second_kwargs = connect_calls[1]
        assert second_kwargs["host"] == "hpc.example.com"
        assert second_kwargs["tunnel"] is mock_tunnel


class TestResolveJobLogs:
    def test_matches_log_to_job_by_rule_and_wildcards(self, backend):
        jobs = [
            SnkmtJobResponse(
                snakemake_id=1,
                rule="solve",
                status="finished",
                wildcards={"country": "DE", "year": "2030"},
            ),
        ]
        workflow_files = [
            {"path": "logs/slurm/solve/country=DE,year=2030/12345.out"},
        ]
        backend.resolve_job_logs(jobs, workflow_files)
        assert jobs[0].log == "logs/slurm/solve/country=DE,year=2030/12345.out"

    def test_picks_latest_slurm_id_on_retry(self, backend):
        jobs = [
            SnkmtJobResponse(
                snakemake_id=1,
                rule="solve",
                status="finished",
                wildcards={"a": "1"},
            ),
        ]
        workflow_files = [
            {"path": "logs/slurm/solve/a=1/100.out"},
            {"path": "logs/slurm/solve/a=1/200.out"},
            {"path": "logs/slurm/solve/a=1/150.out"},
        ]
        backend.resolve_job_logs(jobs, workflow_files)
        assert jobs[0].log == "logs/slurm/solve/a=1/200.out"

    def test_no_wildcards_uses_rule_only(self, backend):
        jobs = [
            SnkmtJobResponse(
                snakemake_id=1, rule="all", status="finished", wildcards=None
            ),
        ]
        workflow_files = [{"path": "logs/slurm/all/999.out"}]
        backend.resolve_job_logs(jobs, workflow_files)
        assert jobs[0].log == "logs/slurm/all/999.out"

    def test_no_matching_log_leaves_none(self, backend):
        jobs = [
            SnkmtJobResponse(
                snakemake_id=1, rule="solve", status="finished", wildcards={"a": "1"}
            ),
        ]
        workflow_files = [{"path": "logs/slurm/other_rule/a=1/100.out"}]
        backend.resolve_job_logs(jobs, workflow_files)
        assert jobs[0].log is None

    def test_none_workflow_files(self, backend):
        jobs = [
            SnkmtJobResponse(
                snakemake_id=1, rule="solve", status="finished", wildcards=None
            ),
        ]
        backend.resolve_job_logs(jobs, None)
        assert jobs[0].log is None


class TestCleanupSlurm:
    async def test_cleanup_issues_four_separate_ssh_commands(self, backend):
        ssh_calls: list[str] = []

        async def run_ssh(cmd, **kwargs):
            ssh_calls.append(cmd)
            result = MagicMock()
            result.exit_status = 0
            result.stdout = ""
            return result

        with patch.object(backend, "_run_ssh", side_effect=run_ssh):
            await backend.cleanup("job-1", "/scratch/jobs/job-1")

        assert len(ssh_calls) == 4
        assert any("kill" in cmd and "kill -9" not in cmd for cmd in ssh_calls)
        assert any("kill -9" in cmd for cmd in ssh_calls)
        assert any("scancel" in cmd for cmd in ssh_calls)
        assert any("rm -rf" in cmd for cmd in ssh_calls)
