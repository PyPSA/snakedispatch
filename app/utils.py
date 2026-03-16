from __future__ import annotations

import contextlib
import logging
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SNKMT_PLUGIN_VERSION = "0.1.6"


def build_snkmt_setup_commands(pixi_path: str) -> list[str]:
    """Return shell commands to swap the pypsa logger plugin for snkmt."""
    pixi = shlex.quote(pixi_path)
    return [
        f"{pixi} remove snakemake-logger-plugin-pypsa 2>/dev/null || true",
        f"{pixi} add snakemake-logger-plugin-snkmt=={SNKMT_PLUGIN_VERSION} 2>/dev/null",
    ]


def build_wrapper_script(
    pixi_path: str,
    snkmt_db_path: str,
    configfile: str | None,
    snakemake_args: list[str] | None,
) -> str:
    """Build the .run.sh bash wrapper script for a Snakemake workflow run."""
    configfile_arg = f" --configfile {shlex.quote(configfile)}" if configfile else ""
    extra_args = ""
    if snakemake_args:
        extra_args = " " + " ".join(shlex.quote(a) for a in snakemake_args)
    pixi = shlex.quote(pixi_path)
    snkmt = shlex.quote(snkmt_db_path)
    snk_cmd = (
        f"{pixi} run snakemake --logger snkmt"
        f" --logger-snkmt-db {snkmt}"
        f"{configfile_arg}{extra_args}"
    )
    return f"""\
#!/bin/bash
echo $$ > .pid
{snk_cmd} > .stdout.log 2>&1
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
