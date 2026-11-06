"""Persistent domain models for the remote agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class WorkspaceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    created_at: str = Field(default_factory=utc_now)
    source_path: str = "source"
    build_path: str = "build"
    artifact_path: str = "artifacts"
    log_path: str = "logs"
    status: str = "READY"
    last_sync_at: str | None = None


class JobStatus(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


ACTIVE_STATUSES = {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING}


class JobMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    job_type: str = Field(pattern="^(BUILD|RUN|BENCHMARK|PROFILER)$")
    status: JobStatus = JobStatus.PENDING
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    log_path: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=200)
    device_id: int | None = None
