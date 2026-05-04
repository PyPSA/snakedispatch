from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

BACKEND_KEYS: frozenset[str] = frozenset({"slurm_ssh", "local"})

_VALID_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_var_names(v: list[str] | None) -> list[str] | None:
    if v is not None:
        for name in v:
            if not _VALID_ENV_VAR_NAME.match(name):
                msg = f"allowed_env_vars entry is not a valid env var name: {name!r}"
                raise ValueError(msg)
    return v


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "", "case_sensitive": True}

    # Path to backend config file
    SNAKEDISPATCH_CONFIG: str = "./config.yaml"

    # Local persistence
    DATA_DIR: str = "/data"

    # Health check cache
    HEALTH_CACHE_TTL_SECONDS: int = 300

    # Cleanup
    AUTO_CLEANUP_AFTER_HOURS: int = 168


class SlurmSSHConfig(BaseModel):
    """Backend config for SSH-based SLURM clusters."""

    host: str
    user: str
    key_path: str = "~/.ssh/id_rsa"
    proxy_jump: str | None = None
    command_timeout: float = 120
    poll_interval: float = 5.0
    pixi_path: str
    scratch_dir: str
    default_snakemake_args: list[str] = []
    snkmt_db_sync_interval: float = 30.0
    allowed_env_vars: list[str] | None = None

    @field_validator("allowed_env_vars")
    @classmethod
    def _validate_allowed_env_vars(cls, v: list[str] | None) -> list[str] | None:
        return _validate_env_var_names(v)


class LocalConfig(BaseModel):
    """Backend config for local execution."""

    scratch_dir: str = "./.snakedispatch"
    pixi_path: str = "pixi"
    poll_interval: float = 5.0
    default_snakemake_args: list[str] = []
    snkmt_db_sync_interval: float = 30.0
    allowed_env_vars: list[str] | None = None

    @field_validator("allowed_env_vars")
    @classmethod
    def _validate_allowed_env_vars(cls, v: list[str] | None) -> list[str] | None:
        return _validate_env_var_names(v)


def load_config(
    path: str,
) -> tuple[SlurmSSHConfig | LocalConfig, dict[str, object]]:
    """Load backend and app configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict) or len(data) == 0:
        msg = f"Config file is empty or invalid: {path}"
        raise ValueError(msg)

    settings_fields = set(Settings.model_fields)

    # Separate backend key(s) from app settings
    backend_keys = {k for k in data if k in BACKEND_KEYS}
    app_keys = {k for k in data if k in settings_fields}
    unknown_keys = set(data) - backend_keys - app_keys

    if unknown_keys:
        msg = f"Unknown keys in config file: {sorted(unknown_keys)}"
        raise ValueError(msg)

    if len(backend_keys) != 1:
        msg = (
            f"Config file must have exactly one backend key "
            f"({', '.join(sorted(BACKEND_KEYS))}), "
            f"got: {sorted(backend_keys) or 'none'}"
        )
        raise ValueError(msg)

    backend_key = next(iter(backend_keys))
    values = data[backend_key] or {}

    if backend_key == "slurm_ssh":
        backend_config = SlurmSSHConfig(**values)
    else:
        backend_config = LocalConfig(**values)

    settings_overrides = {k: data[k] for k in app_keys}

    return backend_config, settings_overrides


def build_settings() -> Settings:
    """Factory function, just needed for tests."""
    return Settings()
