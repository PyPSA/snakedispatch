from __future__ import annotations

from typing import TYPE_CHECKING

from app.backends.local import LocalBackend
from app.config import LocalConfig, SlurmSSHConfig

if TYPE_CHECKING:
    from app.backends.base import ComputeBackend


def create_backend(config: SlurmSSHConfig | LocalConfig) -> ComputeBackend:
    """Create the appropriate backend from config."""
    if isinstance(config, LocalConfig):
        return LocalBackend(config)
    from app.backends.slurm_ssh import (  # noqa: PLC0415, I001  # requires [slurm]
        SlurmSSHBackend,
    )

    return SlurmSSHBackend(config)
