from __future__ import annotations

import contextlib
import logging
import shlex
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def bare_repo_dir(scratch_dir: str, url: str) -> str:
    """Map a workflow URL to a local bare repo path under scratch_dir."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").removesuffix(".git")
    return f"{scratch_dir}/repos/{parsed.hostname}/{path}.git"


def parse_default_branch(ls_remote_output: str) -> str:
    """Extract branch from ``git ls-remote --symref``. Falls back to ``"HEAD"``."""
    for line in ls_remote_output.splitlines():
        if line.startswith("ref: "):
            ref_part = line.split("\t", 1)[0]
            return ref_part.removeprefix("ref: refs/heads/")
    return "HEAD"


def build_wrapper_script(
    pixi_path: str,
    snkmt_db_path: str,
    configfile: str | None,
    snakemake_args: list[str] | None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Build the .run.sh bash wrapper script for a Snakemake workflow run."""
    configfile_arg = f" --configfile {shlex.quote(configfile)}" if configfile else ""
    extra_args = ""
    if snakemake_args:
        extra_args = " " + " ".join(shlex.quote(a) for a in snakemake_args)
    pixi = shlex.quote(pixi_path)
    snkmt = shlex.quote(snkmt_db_path)
    exports = (
        "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env_vars.items()) + "\n"
        if env_vars
        else ""
    )
    return f"""\
#!/bin/bash
{exports}echo $$ > .pid
SNKMT_ARGS=()
if {pixi} run python -c "import snakemake_logger_plugin_snkmt" 2>/dev/null; then
    SNKMT_ARGS=("--logger" "snkmt" "--logger-snkmt-db" {snkmt})
fi
{pixi} run snakemake "${{SNKMT_ARGS[@]}}"{configfile_arg}{extra_args} > .stdout.log 2>&1
echo $? > .exitcode
"""


def rename_with_cleanup(tmp: Path, dest: Path) -> None:
    """Rename tmp to dest, cleaning up tmp on failure."""
    try:
        tmp.rename(dest)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def enforce_error_limit(
    count: int,
    context: str,
    exc: Exception,
    *,
    threshold: int = 10,
    label: str = "errors",
) -> None:
    """Raise RuntimeError if count >= threshold, otherwise log a warning."""
    if count >= threshold:
        msg = f"{context}: stopped after {count} consecutive {label}"
        raise RuntimeError(msg) from exc
    logger.warning(
        "Error during %s (%d/%d %s, will retry): %s",
        context,
        count,
        threshold,
        label,
        exc,
    )
