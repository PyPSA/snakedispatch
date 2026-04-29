from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

try:
    import asyncssh
except ImportError as _err:
    msg = (
        "asyncssh is required for the SlurmSSH backend. "
        "Install it with: uv pip install snakedispatch[slurm]"
    )
    raise ImportError(msg) from _err

from app.backends.base import CHUNK_SIZE, SNKMT_DB_FILENAME, ComputeBackend
from app.config import SlurmSSHConfig
from app.models import JobStatus, SnkmtJobResponse, WorkflowFileInfo
from app.utils import (
    build_wrapper_script,
    enforce_error_limit,
    rename_with_cleanup,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

logger = logging.getLogger(__name__)


# The wrapper script writes "RUNNING" to .exitcode while the process
# is alive, then overwrites it with the numeric exit code on completion.
# The monitor polls .exitcode and uses this sentinel to distinguish
# still running from a real exit code. The check below
# ensures this stays in sync with the JobStatus enum.
MONITOR_RUNNING_SENTINEL = "RUNNING"
if JobStatus.RUNNING.value != MONITOR_RUNNING_SENTINEL:
    msg = (
        "MONITOR_RUNNING_SENTINEL must match JobStatus.RUNNING.value — "
        "update MONITOR_RUNNING_SENTINEL or the enum"
    )
    raise RuntimeError(msg)

# Marker with random string to stay unique
MONITOR_LOG_MARKER = "---SNAKEDISPATCH-LOG-BOUNDARY-7f4e2d1a---"
MONITOR_DEAD_SENTINEL = "DEAD"


def _build_status_check_cmd(work_dir_quoted: str) -> str:
    """Build shell command to check job exit status via .exitcode / PID probe."""
    return (
        f"test -f {work_dir_quoted}/.exitcode && cat {work_dir_quoted}/.exitcode "
        f"|| (kill -0 $(cat {work_dir_quoted}/.pid 2>/dev/null) 2>/dev/null "
        f"&& echo {MONITOR_RUNNING_SENTINEL} "
        f"|| echo {MONITOR_DEAD_SENTINEL})"
    )


def _parse_job_status(status: str) -> int | None:
    """Parse status check output → exit code, -1 for dead, or None for running."""
    if status == MONITOR_RUNNING_SENTINEL:
        return None
    if status == MONITOR_DEAD_SENTINEL:
        return -1
    try:
        return int(status)
    except ValueError:
        logger.warning("Unparsable job status value: %r, treating as running", status)
        return None


def _build_rsync_filter(cache_dirs: list[str]) -> str:
    """Build rsync --include/--exclude args from cache_dirs patterns.

    Supports plain dirs (``data``), nested dirs (``dir1/dir2``),
    and glob patterns (``data/*.csv``).
    """
    includes: list[str] = []
    seen: set[str] = set()

    def _add(arg: str) -> None:
        if arg not in seen:
            seen.add(arg)
            includes.append(arg)

    for pattern in cache_dirs:
        stripped = pattern.strip("/")
        parts = stripped.split("/")
        # rsync skips directories not explicitly included, so for "a/b/c"
        # we must include "a/" and "a/b/" or rsync never reaches "a/b/c/"
        for i in range(len(parts) - 1):
            parent = "/".join(parts[: i + 1]) + "/"
            _add(f"--include={shlex.quote(parent)}")
        # Include the target directory and everything inside it
        _add(f"--include={shlex.quote(stripped + '/')}")
        _add(f"--include={shlex.quote(stripped + '/**')}")
        # Glob patterns (e.g. data/*.csv) need to match as file paths too
        if any(c in stripped for c in "*?["):
            _add(f"--include={shlex.quote(stripped)}")

    includes.append(f"--exclude={shlex.quote('*')}")
    return " ".join(includes)


class SlurmSSHBackend(ComputeBackend):
    """
    Compute backend that connects to an HPC head node via SSH,
    fetches a workflow into a bare repo and creates a worktree,
    runs Snakemake via pixi in a detached process, and monitors via polling.
    """

    def __init__(self, config: SlurmSSHConfig) -> None:
        self._config = config
        self._tunnel: asyncssh.SSHClientConnection | None = None
        self._conn: asyncssh.SSHClientConnection | None = None
        self._lock = asyncio.Lock()

    def work_dir(self, job_id: str) -> str:
        """Canonical work directory for a job on the remote host."""
        return f"{self._config.scratch_dir}/jobs/{job_id}"

    async def _open_tunnel(self) -> asyncssh.SSHClientConnection | None:
        """Open a proxy-jump tunnel if configured, or return None."""
        cfg = self._config
        if not cfg.proxy_jump:
            return None
        jump_parts = cfg.proxy_jump.split("@", 1)
        jump_user = jump_parts[0] if len(jump_parts) == 2 else cfg.user
        jump_host = jump_parts[-1]
        return await asyncio.wait_for(
            asyncssh.connect(
                jump_host,
                username=jump_user,
                client_keys=[cfg.key_path],
                known_hosts=None,
            ),
            timeout=cfg.command_timeout,
        )

    async def _get_conn(self) -> asyncssh.SSHClientConnection:
        """Return a persistent SSH connection, creating one if needed."""
        async with self._lock:
            if self._conn is not None and not self._conn.is_closed():
                return self._conn

            cfg = self._config
            connect_kwargs: dict[str, Any] = {
                "host": cfg.host,
                "username": cfg.user,
                "client_keys": [cfg.key_path],
                "known_hosts": None,
            }

            tunnel = await self._open_tunnel()
            if tunnel is not None:
                connect_kwargs["tunnel"] = tunnel

            try:
                conn = await asyncio.wait_for(
                    asyncssh.connect(**connect_kwargs),
                    timeout=cfg.command_timeout,
                )
            except (TimeoutError, OSError, asyncssh.Error):
                if tunnel is not None:
                    tunnel.close()
                raise

            self._tunnel = tunnel
            self._conn = conn
            return self._conn

    async def _run_ssh(
        self,
        cmd: str,
        *,
        check: bool = True,
        timeout: float | None = None,  # noqa: ASYNC109  # passed to asyncio.wait_for
    ) -> asyncssh.SSHCompletedProcess:
        """Run a command over the persistent SSH connection."""
        if timeout is None:
            timeout = self._config.command_timeout
        try:
            conn = await self._get_conn()
            result = await asyncio.wait_for(conn.run(cmd, check=False), timeout=timeout)
        except (TimeoutError, OSError, asyncssh.Error) as first_exc:
            # Reset and retry once
            logger.debug("SSH command failed on first attempt, retrying: %s", first_exc)
            await self._close_conn()
            try:
                conn = await self._get_conn()
                result = await asyncio.wait_for(
                    conn.run(cmd, check=False), timeout=timeout
                )
            except (TimeoutError, OSError, asyncssh.Error) as retry_exc:
                raise retry_exc from first_exc

        if check and result.exit_status != 0:
            msg = (
                f"SSH command failed (exit {result.exit_status}): {cmd}\n"
                f"stderr: {result.stderr}"
            )
            raise RuntimeError(msg)
        return result

    async def _close_conn(self) -> None:
        """Close the persistent connection."""
        if self._conn and not self._conn.is_closed():
            self._conn.close()
            await self._conn.wait_closed()
        self._conn = None
        if self._tunnel and not self._tunnel.is_closed():
            self._tunnel.close()
            await self._tunnel.wait_closed()
        self._tunnel = None

    async def _upload_dir(self, local_dir: str, remote_dir: str) -> None:
        """Upload a local directory to remote via tar pipe over SSH."""
        conn = await self._get_conn()
        await self._run_ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        async with conn.create_process(
            f"tar xf - -C {shlex.quote(remote_dir)}"
        ) as remote_proc:
            local_proc = await asyncio.create_subprocess_exec(
                "tar",
                "cf",
                "-",
                "--exclude=.git",
                "-C",
                local_dir,
                ".",
                stdout=asyncio.subprocess.PIPE,
            )
            while chunk := await local_proc.stdout.read(CHUNK_SIZE):
                remote_proc.stdin.write(chunk)
                await remote_proc.stdin.drain()
            remote_proc.stdin.write_eof()
            await local_proc.wait()
            if local_proc.returncode != 0:
                msg = f"local tar failed with exit {local_proc.returncode}"
                raise RuntimeError(msg)

    def _scratch_dir(self) -> str:
        return self._config.scratch_dir

    async def _run_git_cmd(self, *args: str) -> str:
        """Run a git command over SSH, returning stdout."""
        quoted = [shlex.quote(a) for a in args]
        # For init, ensure parent dirs exist remotely
        if "init" in args:
            repo_path = shlex.quote(args[-1])
            await self._run_ssh(f"mkdir -p $(dirname {repo_path})")
        result = await self._run_ssh(" ".join(quoted))
        return result.stdout or ""

    async def _copy_local_workflow(self, src: str, dst: str) -> None:
        await self._upload_dir(src, dst)

    async def setup(
        self,
        job_id: str,
        work_dir: str,
        extra_files: dict[str, str] | None = None,
    ) -> None:
        if extra_files:
            conn = await self._get_conn()
            async with conn.start_sftp_client() as sftp:
                for rel_path, content in extra_files.items():
                    full_path = f"{work_dir}/{rel_path}"
                    parent = str(PurePosixPath(full_path).parent)
                    try:
                        await sftp.makedirs(parent)
                    except asyncssh.SFTPError as e:
                        if e.code != asyncssh.FX_FILE_ALREADY_EXISTS:
                            raise
                    async with sftp.open(full_path, "w") as f:
                        await f.write(content)
                    logger.debug("Wrote setup file: %s", full_path)

    async def launch(
        self,
        job_id: str,
        work_dir: str,
        configfile: str | None,
        snakemake_args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        cfg = self._config
        self._validate_configfile(configfile)
        snkmt_db_path = f"{work_dir}/{SNKMT_DB_FILENAME}"
        wrapper_content = build_wrapper_script(
            cfg.pixi_path, snkmt_db_path, configfile, snakemake_args, env_vars
        )

        # Write wrapper script via SFTP, then make executable and launch detached
        conn = await self._get_conn()
        async with conn.start_sftp_client() as sftp:
            script_path = f"{work_dir}/.run.sh"
            async with sftp.open(script_path, "w") as f:
                await f.write(wrapper_content)

        # nohup - survives SSH channel close
        # & - run in background
        # /dev/null redirects - detach stdin/stdout/stderr so nothing blocks
        launch_cmd = (
            f"chmod +x {shlex.quote(work_dir)}/.run.sh && "
            f"cd {shlex.quote(work_dir)} && "
            f"nohup bash .run.sh < /dev/null > /dev/null 2>&1 &"
        )
        async with conn.create_process(launch_cmd) as proc:
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            # Timeout is fine since process is detached and will continue running
            except TimeoutError:
                logger.debug("Launch process timed out (expected for detached process)")

        pid_path = f"{work_dir}/.pid"

        async def _pid_exists() -> bool:
            result = await self._run_ssh(
                f"test -f {shlex.quote(pid_path)} && echo EXISTS",
                check=False,
            )
            return "EXISTS" in (result.stdout or "")

        await self._poll_until_pid_file(job_id, work_dir, _pid_exists)

    async def monitor(
        self,
        job_id: str,
        work_dir: str,
        log_callback: Callable[[str], None],
        byte_offset: int = 0,
    ) -> int:
        offset = byte_offset
        consecutive_errors = 0

        while True:
            try:
                wd = shlex.quote(work_dir)
                cmd = (
                    f"tail -c +{offset + 1} {wd}/.stdout.log 2>/dev/null; "
                    f"echo '{MONITOR_LOG_MARKER}'; "
                    f"{_build_status_check_cmd(wd)}"
                )
                result = await self._run_ssh(cmd, check=False)
                stdout = result.stdout or ""

                # Split on the marker to separate log output from status
                parts = stdout.split(MONITOR_LOG_MARKER, 1)
                new_log_data = parts[0] if len(parts) == 2 else ""
                status_part = parts[1].strip() if len(parts) == 2 else stdout.strip()

                consecutive_errors = 0

                if new_log_data:
                    offset += len(new_log_data.encode("utf-8", errors="replace"))
                    for line in new_log_data.splitlines():
                        log_callback(line)

                exit_code = _parse_job_status(status_part)
                if exit_code == -1:
                    logger.warning(
                        "Job %s: wrapper process died without writing "
                        ".exitcode (likely OOM or SIGKILL)",
                        job_id,
                    )
                    return -1
                if exit_code is not None:
                    return exit_code

            except (TimeoutError, OSError, asyncssh.Error) as exc:
                consecutive_errors += 1
                enforce_error_limit(
                    consecutive_errors, f"Job {job_id}", exc, label="SSH errors"
                )

            await asyncio.sleep(self._config.poll_interval)

    async def check_job_status(self, job_id: str, work_dir: str) -> int | None:
        """Check if a job process has finished without blocking."""
        wd = shlex.quote(work_dir)
        result = await self._run_ssh(_build_status_check_cmd(wd), check=False)
        return _parse_job_status((result.stdout or "").strip())

    async def check_connectivity(self) -> bool:
        """Check SSH connectivity and scratch filesystem health."""
        try:
            scratch = shlex.quote(self._config.scratch_dir)
            result = await self._run_ssh(
                f"stat {scratch} >/dev/null 2>&1", check=False, timeout=15
            )
        except (TimeoutError, OSError, asyncssh.Error):
            return False
        else:
            return result.exit_status == 0

    async def list_workflow_files(
        self,
        job_id: str,
        work_dir: str,
    ) -> list[WorkflowFileInfo]:
        cmd = (
            f"find {shlex.quote(work_dir)} -type f "
            f"-not -path '*/.*' "
            f'-printf "%P\\t%s\\n" 2>/dev/null'
        )
        result = await self._run_ssh(cmd, check=False)
        files: list[WorkflowFileInfo] = []
        for line in (result.stdout or "").strip().splitlines():
            if "\t" not in line:
                continue
            rel_path, size_str = line.split("\t", 1)
            try:
                size = int(size_str)
                if size == 0:
                    continue
                files.append(WorkflowFileInfo(path=rel_path, size=size))
            except ValueError:
                logger.warning("Skipping malformed file listing line: %r", line)
        return files

    async def stream_file(
        self,
        job_id: str,
        work_dir: str,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        full_path = f"{work_dir}/{path}"
        conn = await self._get_conn()
        try:
            async with (
                conn.start_sftp_client() as sftp,
                sftp.open(full_path, "rb") as f,
            ):
                while True:
                    chunk = await f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
        except asyncssh.SFTPNoSuchFile as exc:
            raise FileNotFoundError(full_path) from exc

    async def restore_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> bool:
        cache_base = f"{self._config.scratch_dir}/cache/{cache_key}"
        check = await self._run_ssh(f"test -d {shlex.quote(cache_base)}", check=False)
        if check.exit_status != 0:
            logger.info("No cache found for key=%s", cache_key)
            return False
        filters = _build_rsync_filter(cache_dirs)
        cmd = f"rsync -a {filters} {shlex.quote(cache_base)}/ {shlex.quote(work_dir)}/"
        await self._run_ssh(cmd)
        logger.info("Cache restore done for job %s (key=%s)", job_id, cache_key)
        return True

    async def save_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> None:
        cache_base = f"{self._config.scratch_dir}/cache/{cache_key}"
        await self._run_ssh(f"mkdir -p {shlex.quote(cache_base)}")
        filters = _build_rsync_filter(cache_dirs)
        cmd = f"rsync -a {filters} {shlex.quote(work_dir)}/ {shlex.quote(cache_base)}/"
        await self._run_ssh(cmd)
        logger.info("Cache save done for job %s (key=%s)", job_id, cache_key)

    async def sync_snkmt_db(self, job_id: str, work_dir: str, local_path: Path) -> None:
        """Sync {work_dir}/snkmt.db from remote via sqlite3 .backup + SFTP download."""
        remote_db = f"{work_dir}/{SNKMT_DB_FILENAME}"
        remote_bak = f"{work_dir}/{SNKMT_DB_FILENAME}.bak"
        # Create consistent snapshot on remote (safe while DB is being written)
        result = await self._run_ssh(
            f'sqlite3 {shlex.quote(remote_db)} ".backup {shlex.quote(remote_bak)}"',
            check=False,
        )
        if result.exit_status != 0:
            # DB may not exist yet or be locked, next sync cycle retries
            logger.warning(
                "sqlite3 backup failed (exit %s), skipping download",
                result.exit_status,
            )
            return
        # Download to .tmp first, then atomically rename so readers
        # never see a partially-written file
        conn = await self._get_conn()
        try:
            async with conn.start_sftp_client() as sftp:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = local_path.with_suffix(".tmp")
                await sftp.get(remote_bak, str(tmp))
                rename_with_cleanup(tmp, local_path)
        finally:
            await self._run_ssh(f"rm -f {shlex.quote(remote_bak)}", check=False)

    def resolve_job_logs(
        self,
        jobs: list[SnkmtJobResponse],
        workflow_files: list[WorkflowFileInfo] | None,
    ) -> None:
        """Match slurm log files to jobs via structured path convention.

        Expects logs at: logs/slurm/{rule}/{wildcards_string}/{slurm_id}.out
        If multiple logs exist (retries), picks the latest (highest slurm ID).
        """
        # Index slurm log files by directory prefix
        slurm_logs: dict[str, list[str]] = {}
        if workflow_files:
            for wf in workflow_files:
                p = wf["path"]
                if p.startswith("logs/slurm/") and p.endswith(".out"):
                    dir_prefix = p.rsplit("/", 1)[0]
                    slurm_logs.setdefault(dir_prefix, []).append(p)

        def _slurm_id(path: str) -> int:
            fname = path.rsplit("/", 1)[-1].removesuffix(".out")
            try:
                return int(fname)
            except ValueError:
                return -1

        for job in jobs:
            if job.wildcards:
                wildcards_str = ",".join(
                    f"{k}={v}" for k, v in sorted(job.wildcards.items())
                )
                slurm_dir = f"logs/slurm/{job.rule}/{wildcards_str}"
            else:
                slurm_dir = f"logs/slurm/{job.rule}"

            candidates = slurm_logs.get(slurm_dir, [])
            if candidates:
                job.log = max(candidates, key=_slurm_id)

    async def cleanup(
        self,
        job_id: str,
        work_dir: str,
    ) -> None:
        # 1. Kill the wrapper process if a PID file exists
        pid_file = f"{shlex.quote(work_dir)}/.pid"
        await self._run_ssh(
            f"test -f {pid_file} && kill $(cat {pid_file}) 2>/dev/null || true",
            check=False,
        )
        # Wait before sending SIGKILL in case SIGTERM wasn't enough
        await asyncio.sleep(5)
        await self._run_ssh(
            f"test -f {pid_file} && kill -9 $(cat {pid_file}) 2>/dev/null || true",
            check=False,
        )
        # 2. Cancel only SLURM jobs launched from this work_dir
        await self._run_ssh(
            f"squeue --user={shlex.quote(self._config.user)} "
            f"--noheader --format='%i %Z' 2>/dev/null | "
            f"grep {shlex.quote(work_dir)} | "
            f"awk '{{print $1}}' | "
            f"xargs -r scancel 2>/dev/null || true",
            check=False,
        )
        # 3. Remove the work directory in the background
        await self._run_ssh(
            f"nohup rm -rf {shlex.quote(work_dir)} &",
            check=False,
        )
        logger.info("Cleaned up job %s at %s", job_id, work_dir)
