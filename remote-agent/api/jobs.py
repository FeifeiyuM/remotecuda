"""Execution, benchmark, and job REST API."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.artifact_manager import ArtifactManager
from services.job_manager import JobManager
from storage.database import Database


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str = Field(min_length=1, max_length=512)
    args: list[str] = Field(default_factory=list, max_length=128)
    device_id: int = Field(default=0, ge=0, le=1023)
    timeout_seconds: int = Field(default=600, ge=1, le=86_400)

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        from pathlib import Path

        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("executable must be relative to build directory")
        return value


class BenchmarkRequest(ExecutionRequest):
    benchmark_name: str = Field(default="benchmark", min_length=1, max_length=200)
    iterations: int = Field(default=1, ge=1, le=10_000)
    warmup_iterations: int = Field(default=0, ge=0, le=10_000)


class NcuOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_set: str = Field(default="detailed", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    kernel_name: str | None = Field(default=None, min_length=1, max_length=512)
    launch_count: int | None = Field(default=None, ge=1, le=1_000_000)
    launch_skip: int | None = Field(default=None, ge=0, le=1_000_000)
    target_processes: str = Field(default="all", pattern=r"^(all|application|none)$")

    @field_validator("kernel_name")
    @classmethod
    def validate_kernel_name(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("kernel_name must be non-empty text")
        return value


class NsysOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: list[str] = Field(default_factory=lambda: ["cuda", "nvtx", "osrt"], max_length=16)
    sample_cpu: bool = True
    duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    delay_seconds: float | None = Field(default=None, ge=0, le=86_400)
    cuda_memory_usage: bool = False

    @field_validator("trace")
    @classmethod
    def validate_trace(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in value))
        if not normalized or any(
            not item or not item.isascii() or not item.replace("_", "").isalnum()
            for item in normalized
        ):
            raise ValueError("trace entries must be simple Nsight Systems feature names")
        return normalized


class ProfilerRequest(ExecutionRequest):
    profiler: str = Field(pattern=r"^(ncu|nsys)$")
    profile_name: str = Field(default="profile", min_length=1, max_length=200)
    ncu: NcuOptions | None = None
    nsys: NsysOptions | None = None

    @model_validator(mode="after")
    def validate_profiler_options(self) -> "ProfilerRequest":
        if self.profiler == "ncu" and self.nsys is not None:
            raise ValueError("nsys options require profiler=nsys")
        if self.profiler == "nsys" and self.ncu is not None:
            raise ValueError("ncu options require profiler=ncu")
        return self


def create_router(jobs: JobManager, artifacts: ArtifactManager, database: Database) -> APIRouter:
    router = APIRouter(tags=["jobs"])

    @router.post("/v1/workspaces/{workspace_id}/run", status_code=202)
    async def run(
        workspace_id: str,
        request: ExecutionRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        result = await jobs.submit(
            workspace_id=workspace_id,
            job_type="RUN",
            config=request.model_dump(),
            idempotency_key=idempotency_key,
        )
        database.audit("job.run", workspace_id=workspace_id, job_id=result["job_id"])
        return result

    @router.post("/v1/workspaces/{workspace_id}/benchmark", status_code=202)
    async def benchmark(
        workspace_id: str,
        request: BenchmarkRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        result = await jobs.submit(
            workspace_id=workspace_id,
            job_type="BENCHMARK",
            config=request.model_dump(),
            idempotency_key=idempotency_key,
        )
        database.audit("job.benchmark", workspace_id=workspace_id, job_id=result["job_id"])
        return result

    @router.post("/v1/workspaces/{workspace_id}/profile", status_code=202)
    async def profile(
        workspace_id: str,
        request: ProfilerRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        result = await jobs.submit(
            workspace_id=workspace_id,
            job_type="PROFILER",
            config=request.model_dump(),
            idempotency_key=idempotency_key,
        )
        database.audit("job.profile", workspace_id=workspace_id, job_id=result["job_id"])
        return result

    @router.get("/v1/jobs/{job_id}")
    async def job(job_id: str) -> dict:
        return jobs.get(job_id).model_dump(mode="json")

    @router.get("/v1/jobs/{job_id}/logs")
    async def job_logs(job_id: str) -> dict:
        return jobs.logs(job_id)

    @router.post("/v1/jobs/{job_id}/cancel", status_code=202)
    async def cancel(job_id: str) -> dict:
        result = await jobs.cancel(job_id)
        database.audit("job.cancel", job_id=job_id, result=result)
        return result

    @router.get("/v1/jobs/{job_id}/benchmark-result")
    async def benchmark_result(job_id: str) -> dict:
        job = jobs.get(job_id)
        artifact_path = job.result.get("artifact_path")
        if not artifact_path:
            raise ValueError("job does not contain a benchmark result")
        return artifacts.read_json_result(job.workspace_id, str(artifact_path))

    @router.get("/v1/jobs/{job_id}/profiler-result")
    async def profiler_result(job_id: str) -> dict:
        job = jobs.get(job_id)
        artifact_path = job.result.get("profiler_result_path")
        if not artifact_path:
            raise ValueError("job does not contain a profiler result")
        return artifacts.read_json_result(job.workspace_id, str(artifact_path))

    return router
