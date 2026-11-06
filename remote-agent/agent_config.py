"""Remote CUDA Agent configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _default_install_dir() -> Path:
    """Return the installation root used for runtime data."""

    configured = os.getenv("REMOTE_CUDA_AGENT_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    # For <install-dir>/.venv/bin/python, the executable's parent directory is
    # the virtual environment and its parent is the installation root.
    executable = Path(sys.argv[0]).expanduser()
    try:
        executable = executable.resolve()
    except OSError:
        executable = executable.absolute()
    if executable.parent.name == "bin":
        # Covers <install-dir>/.venv/bin/python and <install-dir>/.venv/bin/uvicorn.
        if executable.parent.parent.name == ".venv":
            return executable.parent.parent.parent
        if executable.name == "remote-cuda-agent":
            return executable.parent.parent

    # Flat source-tree execution: config.py is directly under remote-agent/.
    return Path(__file__).resolve().parent


INSTALL_DIR = _default_install_dir()
RUNTIME_DIR = INSTALL_DIR / "runtime"


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bind to loopback by default. Network exposure should be controlled by a
    # firewall or an upstream network policy.
    host: str = "127.0.0.1"
    port: int = Field(default=8443, ge=1, le=65535)
    # Tool locations are explicit so builds do not depend on the service PATH.
    cmake_path: Path
    nvcc_path: Path
    # Nsight tools are optional; a profile request fails safely when absent.
    ncu_path: Path | None = None
    nsys_path: Path | None = None
    # Authentication stays enabled by default. Disabling it is intended only
    # for isolated loopback or otherwise protected development deployments.
    auth_enabled: bool = True
    token: str = Field(default="remote-cuda-development-token")
    workspace_root: Path = RUNTIME_DIR / "workspaces"
    database_path: Path = RUNTIME_DIR / "agent.db"
    max_upload_bytes: int = Field(default=512 * 1024 * 1024, ge=1, le=8 * 1024**3)
    max_artifact_bytes: int = Field(default=1024 * 1024 * 1024, ge=1, le=8 * 1024**3)
    max_output_chars: int = Field(default=1_000_000, ge=1_000, le=100_000_000)
    max_running_jobs: int = Field(default=4, ge=1, le=256)
    max_build_parallel: int = Field(default=8, ge=1, le=256)
    default_timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    allowed_origins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_auth_pair(self) -> "AgentSettings":
        if self.auth_enabled and not self.token:
            raise ValueError("token is required when auth_enabled is true")
        return self

    @field_validator("cmake_path", "nvcc_path", "ncu_path", "nsys_path")
    @classmethod
    def validate_tool_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return value
        if not value.is_absolute():
            raise ValueError("tool path must be absolute")
        if not value.is_file() or not os.access(value, os.X_OK):
            raise ValueError(f"tool is not an executable file: {value}")
        return value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean environment value: {value}")


def load_settings(config_path: str | Path | None = None) -> AgentSettings:
    configured_path = config_path or os.getenv("REMOTE_AGENT_CONFIG")
    if configured_path:
        path = Path(configured_path).expanduser()
    else:
        # Normal startup is `cd remote-agent && python main.py`; the default
        # config is therefore the current directory's remote.yaml.
        path = Path.cwd() / "remote.yaml"
    payload: dict[str, object] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            import json

            payload = json.loads(text)
        else:
            import yaml

            payload = yaml.safe_load(text) or {}
    overrides = {
        key: os.environ[value]
        for key, value in {
            "host": "REMOTE_CUDA_AGENT_HOST",
            "token": "REMOTE_CUDA_AGENT_TOKEN",
            "cmake_path": "REMOTE_CUDA_CMAKE_PATH",
            "nvcc_path": "REMOTE_CUDA_NVCC_PATH",
            "ncu_path": "REMOTE_CUDA_NCU_PATH",
            "nsys_path": "REMOTE_CUDA_NSYS_PATH",
            "workspace_root": "REMOTE_CUDA_WORKSPACE_ROOT",
            "database_path": "REMOTE_CUDA_DATABASE",
            "max_upload_bytes": "REMOTE_CUDA_MAX_UPLOAD_BYTES",
            "max_artifact_bytes": "REMOTE_CUDA_MAX_ARTIFACT_BYTES",
            "max_output_chars": "REMOTE_CUDA_MAX_OUTPUT_CHARS",
        }.items()
        if os.environ.get(value)
    }
    for key in ("max_upload_bytes", "max_artifact_bytes", "max_output_chars"):
        if key in overrides:
            overrides[key] = int(overrides[key])
    if os.environ.get("REMOTE_CUDA_AGENT_PORT", "").isdigit():
        overrides["port"] = int(os.environ["REMOTE_CUDA_AGENT_PORT"])
    if os.environ.get("REMOTE_CUDA_AUTH_ENABLED"):
        overrides["auth_enabled"] = _parse_bool(os.environ["REMOTE_CUDA_AUTH_ENABLED"])
    settings = AgentSettings.model_validate({**payload, **overrides})

    # Relative paths in a config file are resolved beside that config file.
    # This keeps runtime data under the installation directory by default.
    config_dir = path.expanduser().resolve().parent
    relative_updates: dict[str, Path] = {}
    for field_name in ("workspace_root", "database_path"):
        value = getattr(settings, field_name)
        if value is not None and not value.is_absolute():
            relative_updates[field_name] = (config_dir / value).resolve()
    if relative_updates:
        settings = settings.model_copy(update=relative_updates)
    return settings
