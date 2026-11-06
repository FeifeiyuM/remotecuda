"""Pydantic models used by the local registries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ServerConfig(BaseModel):
    # Ignore retired fields while existing SQLite records migrate.
    model_config = ConfigDict(extra="ignore")

    server_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    name: str = Field(min_length=1, max_length=200)
    endpoint: str = Field(min_length=8, max_length=2048)
    auth_enabled: bool = True
    token_env: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Z_][A-Z0-9_]*$")
    enabled: bool = True
    priority: int = Field(default=10, ge=0, le=10_000)
    tags: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30, gt=0, le=600)

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, value: str) -> str:
        if not value.lower().startswith("http://"):
            raise ValueError("endpoint must use http")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_auth_pair(self) -> "ServerConfig":
        if self.auth_enabled and not self.token_env:
            raise ValueError("token_env is required when auth_enabled is true")
        return self

    def safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ConnectionStatus(StrEnum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class ServerRuntimeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    health_status: HealthStatus = HealthStatus.UNKNOWN
    capabilities: dict[str, Any] = Field(default_factory=dict)
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    active_jobs: int = 0
    last_heartbeat: str | None = None
    last_error: str | None = None


class WorkspaceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    server_id: str = Field(min_length=1, max_length=128)
    local_path: str = Field(min_length=1)
    remote_path: str | None = None
    status: str = "READY"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    last_sync_at: str | None = None


class JobType(StrEnum):
    BUILD = "BUILD"
    RUN = "RUN"
    BENCHMARK = "BENCHMARK"
    PROFILER = "PROFILER"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)
    remote_job_id: str = Field(min_length=1, max_length=128)
    server_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    idempotency_key: str | None = Field(default=None, max_length=200)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
