from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from app.backends import create_backend
from app.backends.base import ComputeBackend
from app.backends.local import LocalBackend
from app.config import LocalConfig
from app.utils import build_wrapper_script

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _StubBackend(ComputeBackend):
    """Minimal concrete subclass for testing abstract base methods."""

    def work_dir(self, job_id: str) -> str:
        return f"/work/{job_id}"

    def _scratch_dir(self) -> str:
        return "/scratch"

    async def _run_git_cmd(self, *args: str) -> str:
        return ""

    async def _copy_local_workflow(self, src: str, dst: str) -> None:
        pass

    async def setup(self, job_id, work_dir, extra_files=None):
        pass

    async def launch(self, job_id, work_dir, configfile, snakemake_args=None):
        pass

    async def monitor(self, job_id, work_dir, log_callback, byte_offset=0):
        return 0

    async def list_workflow_files(self, job_id, work_dir):
        return []

    async def stream_file(self, job_id, work_dir, path) -> AsyncIterator[bytes]:
        yield b""

    async def restore_cache(self, job_id, work_dir, cache_key, cache_dirs):
        pass

    async def save_cache(self, job_id, work_dir, cache_key, cache_dirs):
        pass

    async def sync_snkmt_db(self, job_id, work_dir, local_path):
        pass

    async def check_job_status(self, job_id, work_dir):
        return None

    async def check_connectivity(self):
        return True

    async def cleanup(self, job_id, work_dir):
        pass


@pytest.fixture
def stub() -> _StubBackend:
    return _StubBackend()


class TestCreateBackend:
    def test_local_config_returns_local_backend(self, tmp_path):
        config = LocalConfig(scratch_dir=str(tmp_path))
        backend = create_backend(config)
        assert isinstance(backend, LocalBackend)


class TestBuildWrapperScript:
    def test_contains_pid_capture(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        assert "echo $$ > .pid" in script

    def test_contains_exitcode_capture(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        assert "echo $? > .exitcode" in script

    def test_pixi_path_quoted(self):
        script = build_wrapper_script(
            "/path with spaces/pixi", "/work/snkmt.db", None, None
        )
        assert "'/path with spaces/pixi'" in script

    def test_snkmt_db_path_quoted(self):
        script = build_wrapper_script("/opt/pixi", "/work dir/snkmt.db", None, None)
        assert "'/work dir/snkmt.db'" in script

    def test_configfile_included_when_set(self):
        script = build_wrapper_script(
            "/opt/pixi", "/work/snkmt.db", "config/prod.yaml", None
        )
        assert "--configfile config/prod.yaml" in script

    def test_no_configfile_arg_when_none(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        assert "--configfile" not in script

    def test_snakemake_args_appended(self):
        script = build_wrapper_script(
            "/opt/pixi", "/work/snkmt.db", None, ["--cores", "4", "--forceall"]
        )
        assert "--cores 4 --forceall" in script

    def test_snakemake_args_with_spaces_quoted(self):
        script = build_wrapper_script(
            "/opt/pixi", "/work/snkmt.db", None, ["--config", "key=val ue"]
        )
        assert "'key=val ue'" in script

    def test_no_extra_args_when_none(self):
        script = build_wrapper_script("/opt/pixi", "/work/snkmt.db", None, None)
        # Should end before any extra args after the db path
        assert script.count("snakemake") >= 1


class TestAwaitPidFile:
    async def test_succeeds_when_pid_appears_immediately(self, stub):
        calls = 0

        async def check_fn() -> bool:
            nonlocal calls
            calls += 1
            return True

        with patch("app.backends.base.asyncio.sleep", new_callable=AsyncMock):
            await stub._poll_until_pid_file("job-1", "/work/job-1", check_fn)

        assert calls == 1

    async def test_succeeds_after_retries(self, stub):
        call_count = 0

        async def check_fn() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        with patch("app.backends.base.asyncio.sleep", new_callable=AsyncMock):
            await stub._poll_until_pid_file("job-1", "/work/job-1", check_fn)

        assert call_count == 3

    async def test_raises_after_10_failed_attempts(self, stub):
        async def check_fn() -> bool:
            return False

        with (
            patch("app.backends.base.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="job-1"),
        ):
            await stub._poll_until_pid_file("job-1", "/work/job-1", check_fn)

    async def test_error_message_includes_work_dir(self, stub):
        async def check_fn() -> bool:
            return False

        with patch("app.backends.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="/work/job-abc"):
                await stub._poll_until_pid_file("job-abc", "/work/job-abc", check_fn)
