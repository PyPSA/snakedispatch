from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import shutil
import signal
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from app.backends.base import CHUNK_SIZE, SNKMT_DB_FILENAME, ComputeBackend
from app.config import LocalConfig
from app.models import WorkflowFileInfo
from app.utils import (
    build_wrapper_script,
    enforce_error_limit,
    rename_with_cleanup,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

logger = logging.getLogger(__name__)


class LocalBackend(ComputeBackend):
    """Compute backend that runs workflows on the local machine."""

    def __init__(self, config: LocalConfig) -> None:
        self._config = config

    def work_dir(self, job_id: str) -> str:
        return f"{self._config.scratch_dir}/jobs/{job_id}"

    async def prepare(
        self,
        job_id: str,
        workflow: str,
        git_ref: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        Path(self._config.scratch_dir).mkdir(parents=True, exist_ok=True)
        work_dir = self.work_dir(job_id)
        git_sha: str | None = None

        if workflow.startswith(("http://", "https://")):
            cmd = ["git", "clone", "--depth=1"]
            if git_ref:
                cmd += ["--branch", git_ref]
            cmd += [workflow, work_dir]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                msg = f"git clone failed (exit {proc.returncode}): {stderr.decode()}"
                raise RuntimeError(msg)

            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                work_dir,
                "rev-parse",
                "HEAD",
                "--abbrev-ref",
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            lines = out.decode().strip().splitlines()
            git_sha = lines[0] if lines else None
            git_ref = lines[1] if len(lines) > 1 else None
        else:
            src = Path(workflow)
            dst = Path(work_dir)
            await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)

        return work_dir, git_ref, git_sha

    async def setup(
        self,
        job_id: str,
        work_dir: str,
        extra_files: dict[str, str] | None = None,
    ) -> None:
        if extra_files:
            for rel_path, content in extra_files.items():
                full_path = Path(work_dir) / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(full_path.write_text, content)
                logger.debug("Wrote setup file: %s", full_path)


    async def launch(
        self,
        job_id: str,
        work_dir: str,
        configfile: str | None,
        snakemake_args: list[str] | None = None,
    ) -> None:
        self._validate_configfile(configfile)
        snkmt_db_path = str(Path(work_dir).resolve() / SNKMT_DB_FILENAME)
        wrapper_content = build_wrapper_script(
            self._config.pixi_path, snkmt_db_path, configfile, snakemake_args
        )

        wd = Path(work_dir)
        script_path = wd / ".run.sh"
        await asyncio.to_thread(script_path.write_text, wrapper_content)
        await asyncio.to_thread(script_path.chmod, 0o755)

        proc = await asyncio.create_subprocess_shell(
            "nohup bash "
            f"{shlex.quote(script_path.name)}"
            " < /dev/null > /dev/null 2>&1 &",
            cwd=work_dir,
        )
        await proc.wait()

        pid_path = wd / ".pid"

        async def _pid_exists() -> bool:
            return await asyncio.to_thread(pid_path.exists)

        await self._poll_until_pid_file(job_id, work_dir, _pid_exists)

    async def monitor(
        self,
        job_id: str,
        work_dir: str,
        log_callback: Callable[[str], None],
        byte_offset: int = 0,
    ) -> int:
        wd = Path(work_dir)
        log_path = wd / ".stdout.log"
        exitcode_path = wd / ".exitcode"
        offset = byte_offset
        consecutive_errors = 0

        def _read_from_offset(path: Path, off: int) -> bytes:
            with path.open("rb") as f:
                f.seek(off)
                return f.read()

        async def _drain_log(off: int) -> int:
            """Read new log data from offset, dispatch lines, return new offset."""
            if not log_path.exists():
                return off
            data = await asyncio.to_thread(_read_from_offset, log_path, off)
            if not data:
                return off
            text = data.decode("utf-8", errors="replace")
            for line in text.splitlines():
                log_callback(line)
            return off + len(data)

        while True:
            try:
                offset = await _drain_log(offset)

                if exitcode_path.exists():
                    code_str = (
                        await asyncio.to_thread(
                            exitcode_path.read_text, encoding="utf-8"
                        )
                    ).strip()
                    if code_str:
                        # Capture any log lines written between the last poll and exit
                        offset = await _drain_log(offset)
                        return int(code_str)
                    else:
                        # File exists but is empty
                        # write race, so repoll on the next iteration
                        logger.debug("Job %s: .exitcode is empty, re-polling", job_id)

                consecutive_errors = 0

            except (OSError, ValueError) as exc:
                # Allow for hick ups
                consecutive_errors += 1
                enforce_error_limit(
                    consecutive_errors, f"Job {job_id}", exc, label="I/O errors"
                )

            await asyncio.sleep(self._config.poll_interval)

    async def check_job_status(self, job_id: str, work_dir: str) -> int | None:
        """Check if a job process has finished without blocking."""
        wd = Path(work_dir)
        exitcode_path = wd / ".exitcode"
        if exitcode_path.exists():
            try:
                return int(exitcode_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                return None
        pid_path = wd / ".pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                return None  # still running
            except (ProcessLookupError, PermissionError):
                return -1  # dead without exitcode
            except (ValueError, OSError):
                return None
        return None

    async def list_workflow_files(
        self,
        job_id: str,
        work_dir: str,
    ) -> list[WorkflowFileInfo]:
        wd = Path(work_dir)

        def _scan() -> list[WorkflowFileInfo]:
            files: list[WorkflowFileInfo] = []
            for p in wd.rglob("*"):
                if p.is_file() and not any(
                    part.startswith(".") for part in p.relative_to(wd).parts
                ):
                    rel = str(p.relative_to(wd))
                    files.append(WorkflowFileInfo(path=rel, size=p.stat().st_size))
            return files

        return await asyncio.to_thread(_scan)

    async def stream_file(
        self,
        job_id: str,
        work_dir: str,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        full_path = Path(work_dir) / path
        # FileNotFoundError propagates naturally from open()
        f = await asyncio.to_thread(full_path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(f.read, CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(f.close)

    async def restore_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> bool:
        cache_base = Path(self._config.scratch_dir) / "cache" / cache_key
        if not cache_base.exists():
            logger.info("No cache found for key=%s", cache_key)
            return False
        for cache_dir in cache_dirs:
            src = cache_base / cache_dir
            dst = Path(work_dir) / cache_dir
            if src.exists():
                await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)
        logger.info("Cache restore done for job %s (key=%s)", job_id, cache_key)
        return True

    async def save_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> None:
        cache_base = Path(self._config.scratch_dir) / "cache" / cache_key
        cache_base.mkdir(parents=True, exist_ok=True)
        for cache_dir in cache_dirs:
            src = Path(work_dir) / cache_dir
            dst = cache_base / cache_dir
            if src.exists():
                await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)
        logger.info("Cache save done for job %s (key=%s)", job_id, cache_key)

    async def sync_snkmt_db(self, job_id: str, work_dir: str, local_path: Path) -> None:
        """Sync {work_dir}/snkmt.db to local_path via sqlite3 backup."""
        src = Path(work_dir).resolve() / SNKMT_DB_FILENAME
        if not src.exists():
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)

        def _backup() -> None:
            # Write to .tmp first, then atomically rename so readers
            # never see a partially-written file.
            tmp = local_path.with_suffix(".tmp")
            with (
                contextlib.closing(sqlite3.connect(str(src))) as source_conn,
                contextlib.closing(sqlite3.connect(str(tmp))) as dest_conn,
            ):
                source_conn.backup(dest_conn)
            rename_with_cleanup(tmp, local_path)

        await asyncio.to_thread(_backup)

    async def check_connectivity(self) -> bool:
        """Return True if the scratch dir or its parent exists."""
        try:
            scratch = Path(self._config.scratch_dir)
            return scratch.exists() or scratch.parent.exists()
        except OSError:
            return False

    async def cleanup(
        self,
        job_id: str,
        work_dir: str,
    ) -> None:
        wd = Path(work_dir)
        pid_path = wd / ".pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                await asyncio.sleep(5)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError as exc:
                logger.debug("Process already gone for job %s: %s", job_id, exc)
            except (ValueError, OSError) as exc:
                logger.warning("Could not kill process for job %s: %s", job_id, exc)

        if wd.exists():
            await asyncio.to_thread(shutil.rmtree, wd, ignore_errors=True)
        logger.info("Cleaned up job %s at %s", job_id, work_dir)
