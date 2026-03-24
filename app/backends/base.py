from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from app.utils import bare_repo_dir, parse_default_branch

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from app.models import SnkmtJobResponse, WorkflowFileInfo

logger = logging.getLogger(__name__)


CHUNK_SIZE = 65536  # 64 KB
SNKMT_DB_FILENAME = "snkmt.db"


class ComputeBackend(ABC):
    """Abstract interface for compute backends.

    To add a new backend:
    - subclass ComputeBackend and implement allabstract methods
    - add a config model to app/config.py, register the key in BACKEND_KEYS
    - add a branch to create_backend() in app/backends/__init__.py
    - follow optional dependency pattern import pattern (see SlurmSSHBackend)
    """

    @staticmethod
    def _validate_configfile(configfile: str | None) -> None:
        """Raise ValueError if configfile contains '..' path components."""
        if configfile and ".." in PurePosixPath(configfile).parts:
            msg = f"configfile must not contain '..': {configfile}"
            raise ValueError(msg)

    # total timeout = attempts × interval = 5s
    _PID_POLL_ATTEMPTS = 10
    _PID_POLL_INTERVAL = 0.5  # seconds

    async def _poll_until_pid_file(
        self,
        job_id: str,
        work_dir: str,
        check_fn: Callable[[], Awaitable[bool]],
    ) -> None:
        """Poll for .pid file."""
        for attempt in range(self._PID_POLL_ATTEMPTS):
            await asyncio.sleep(self._PID_POLL_INTERVAL)
            if await check_fn():
                logger.info("Process started, .pid found after attempt %d", attempt + 1)
                return
        msg = f"Job {job_id}: .pid file never appeared in {work_dir}."
        raise RuntimeError(msg)

    @abstractmethod
    def work_dir(self, job_id: str) -> str:
        """Return the canonical work directory path for a job."""
        ...

    @abstractmethod
    async def _run_git_cmd(self, *args: str) -> str:
        """Run a git command and return its stdout as a string.

        Implementations must raise on non-zero exit codes.
        """
        ...

    @abstractmethod
    async def _copy_local_workflow(self, src: str, dst: str) -> None:
        """Copy a local workflow directory to the work directory."""
        ...

    @abstractmethod
    def _scratch_dir(self) -> str:
        """Return the scratch directory path for this backend."""
        ...

    async def prepare(
        self,
        job_id: str,
        workflow: str,
        git_ref: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Prepare workflow in a new working directory.

        If workflow is a URL, set up a bare repo and create a worktree.
        If it is a local path, copy/upload the directory.
        """
        work_dir = self.work_dir(job_id)
        git_sha: str | None = None

        if workflow.startswith(("http://", "https://")):
            repo_dir = bare_repo_dir(self._scratch_dir(), workflow)
            resolved_ref = git_ref

            # 1. Ensure bare repo exists (idempotent)
            await self._run_git_cmd("git", "init", "--bare", repo_dir)

            # 2. Resolve default branch if git_ref not provided
            if resolved_ref is None:
                out = await self._run_git_cmd(
                    "git",
                    "ls-remote",
                    "--symref",
                    workflow,
                    "HEAD",
                )
                resolved_ref = parse_default_branch(out)

            # 3. Fetch to a job-scoped ref
            job_ref = f"refs/snakedispatch/{job_id}"
            await self._run_git_cmd(
                "git",
                "-C",
                repo_dir,
                "fetch",
                workflow,
                f"{resolved_ref}:{job_ref}",
            )

            # 4. Create worktree (detached HEAD)
            await self._run_git_cmd(
                "git",
                "-C",
                repo_dir,
                "worktree",
                "add",
                "--detach",
                work_dir,
                job_ref,
            )

            # 5. Extract SHA
            out = await self._run_git_cmd(
                "git",
                "-C",
                work_dir,
                "rev-parse",
                "HEAD",
            )
            git_sha = out.strip()

            # 6. Clean up temp ref
            await self._run_git_cmd(
                "git",
                "-C",
                repo_dir,
                "update-ref",
                "-d",
                job_ref,
            )

            git_ref = resolved_ref
        else:
            await self._copy_local_workflow(workflow, work_dir)

        return work_dir, git_ref, git_sha

    @abstractmethod
    async def setup(
        self,
        job_id: str,
        work_dir: str,
        extra_files: dict[str, str] | None = None,
    ) -> None:
        """Write extra_files (license files, extra configs) into work_dir."""
        ...

    @abstractmethod
    async def launch(
        self,
        job_id: str,
        work_dir: str,
        configfile: str | None,
        snakemake_args: list[str] | None = None,
    ) -> None:
        """
        Write the .run.sh wrapper script, launch it via nohup/disown,
        and verify .pid file appears within a timeout.
        """
        ...

    @abstractmethod
    async def monitor(
        self,
        job_id: str,
        work_dir: str,
        log_callback: Callable[[str], None],
        byte_offset: int = 0,
    ) -> int:
        """
        Poll .stdout.log and .exitcode until the process completes.
        """
        ...

    @abstractmethod
    async def check_job_status(self, job_id: str, work_dir: str) -> int | None:
        """Check if a job process has finished without blocking.

        Returns:
            Exit code if the process has exited, None if still running.
        """
        ...

    @abstractmethod
    async def list_workflow_files(
        self,
        job_id: str,
        work_dir: str,
    ) -> list[WorkflowFileInfo]:
        """List files under work_dir (excluding hidden files/dirs)."""
        ...

    @abstractmethod
    async def stream_file(
        self,
        job_id: str,
        work_dir: str,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        """Stream file content in chunks. path is relative to work_dir.

        Implementations must be async generator functions (use ``yield``).
        Callers iterate with ``async for chunk in backend.stream_file(...)``.

        Raises FileNotFoundError if the file does not exist. Implementations are
        responsible for translating backend specific exceptions to FileNotFoundError:
        - LocalBackend relies on Python's open() raising it naturally
        - SlurmSSH explicitly translates asyncssh.SFTPNoSuchFile to FileNotFoundError
        """
        ...

    @abstractmethod
    async def restore_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> bool:
        """Restore cached directories into work_dir before launch."""
        ...

    @abstractmethod
    async def save_cache(
        self,
        job_id: str,
        work_dir: str,
        cache_key: str,
        cache_dirs: list[str],
    ) -> None:
        """Save directories from work_dir into cache after successful completion.

        Implementations should be best-effort and failure to save
        cache should not abort the job.
        """
        ...

    @abstractmethod
    async def sync_snkmt_db(self, job_id: str, work_dir: str, local_path: Path) -> None:
        """Ensure a queryable copy of the job's snkmt.db exists at local_path."""
        ...

    @abstractmethod
    async def check_connectivity(self) -> bool:
        """Check whether the backend is reachable. Returns True if healthy."""
        ...

    def resolve_job_logs(  # noqa: B027
        self,
        jobs: list[SnkmtJobResponse],
        workflow_files: list[WorkflowFileInfo] | None,
    ) -> None:
        """Set job.log for each job based on backend-specific log paths."""

    @abstractmethod
    async def cleanup(
        self,
        job_id: str,
        work_dir: str,
    ) -> None:
        """Stop the job and remove its working directory.

        Terminates any running wrapper process, cancels any queued compute jobs,
        and cleans up the working directory on the compute resource.
        """
        ...
