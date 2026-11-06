"""Job orchestration across remote agents."""

from __future__ import annotations

import uuid
from typing import Any

from registry.models import JobType
from remote_client.models import BenchmarkConfig, BuildConfig, ExecutionConfig, ProfilerConfig
from services.context import AppContext


class JobServiceError(RuntimeError):
    pass


class JobService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    def _workspace(self, workspace_id: str, expected_server_id: str | None = None):
        workspace = self.context.workspaces.require(workspace_id)
        if expected_server_id and workspace.server_id != expected_server_id:
            raise JobServiceError(
                f"workspace {workspace_id} belongs to {workspace.server_id}, not {expected_server_id}"
            )
        return workspace

    def _key(self, explicit: str | None, job_type: JobType, workspace_id: str) -> str:
        return explicit or f"{job_type.value.lower()}-{workspace_id}-{uuid.uuid4().hex[:12]}"

    def _existing(
        self, idempotency_key: str | None, workspace_id: str, job_type: JobType
    ) -> dict[str, Any] | None:
        existing = self.context.jobs.get_by_idempotency_key(idempotency_key) if idempotency_key else None
        if existing and (existing.workspace_id != workspace_id or existing.job_type != job_type):
            raise JobServiceError("idempotency key already belongs to a different job")
        return existing.model_dump(mode="json") if existing else None

    async def build(
        self,
        workspace_id: str,
        config: dict[str, Any],
        *,
        server_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        workspace = self._workspace(workspace_id, server_id)
        key = self._key(idempotency_key, JobType.BUILD, workspace_id)
        existing = self._existing(key, workspace_id, JobType.BUILD)
        if existing:
            return existing
        response = await self.context.client_for(workspace.server_id).build(
            workspace_id, BuildConfig.model_validate(config), key
        )
        remote_job_id = str(response.get("job_id") or "")
        if not remote_job_id:
            raise JobServiceError("remote agent did not return job_id")
        job = self.context.jobs.create(
            job_id=f"local-{remote_job_id}",
            remote_job_id=remote_job_id,
            server_id=workspace.server_id,
            workspace_id=workspace_id,
            job_type=JobType.BUILD,
            idempotency_key=key,
        )
        return job.model_dump(mode="json")

    async def run(
        self,
        workspace_id: str,
        config: dict[str, Any],
        *,
        server_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        workspace = self._workspace(workspace_id, server_id)
        key = self._key(idempotency_key, JobType.RUN, workspace_id)
        existing = self._existing(key, workspace_id, JobType.RUN)
        if existing:
            return existing
        response = await self.context.client_for(workspace.server_id).run(
            workspace_id, ExecutionConfig.model_validate(config), key
        )
        remote_job_id = str(response.get("job_id") or "")
        if not remote_job_id:
            raise JobServiceError("remote agent did not return job_id")
        job = self.context.jobs.create(
            job_id=f"local-{remote_job_id}",
            remote_job_id=remote_job_id,
            server_id=workspace.server_id,
            workspace_id=workspace_id,
            job_type=JobType.RUN,
            idempotency_key=key,
        )
        return job.model_dump(mode="json")

    async def benchmark(
        self,
        workspace_id: str,
        config: dict[str, Any],
        *,
        server_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        workspace = self._workspace(workspace_id, server_id)
        key = self._key(idempotency_key, JobType.BENCHMARK, workspace_id)
        existing = self._existing(key, workspace_id, JobType.BENCHMARK)
        if existing:
            return existing
        response = await self.context.client_for(workspace.server_id).benchmark(
            workspace_id, BenchmarkConfig.model_validate(config), key
        )
        remote_job_id = str(response.get("job_id") or "")
        if not remote_job_id:
            raise JobServiceError("remote agent did not return job_id")
        job = self.context.jobs.create(
            job_id=f"local-{remote_job_id}",
            remote_job_id=remote_job_id,
            server_id=workspace.server_id,
            workspace_id=workspace_id,
            job_type=JobType.BENCHMARK,
            idempotency_key=key,
        )
        return job.model_dump(mode="json")

    async def profile(
        self,
        workspace_id: str,
        config: dict[str, Any],
        *,
        server_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        workspace = self._workspace(workspace_id, server_id)
        key = self._key(idempotency_key, JobType.PROFILER, workspace_id)
        existing = self._existing(key, workspace_id, JobType.PROFILER)
        if existing:
            return existing
        response = await self.context.client_for(workspace.server_id).profile(
            workspace_id,
            ProfilerConfig.model_validate(config),
            key,
        )
        remote_job_id = str(response.get("job_id") or "")
        if not remote_job_id:
            raise JobServiceError("remote agent did not return job_id")
        job = self.context.jobs.create(
            job_id=f"local-{remote_job_id}",
            remote_job_id=remote_job_id,
            server_id=workspace.server_id,
            workspace_id=workspace_id,
            job_type=JobType.PROFILER,
            idempotency_key=key,
        )
        return job.model_dump(mode="json")

    async def status(self, job_id: str) -> dict[str, Any]:
        job = self.context.jobs.require(job_id)
        remote = await self.context.client_for(job.server_id).job(job.remote_job_id)
        status = str(remote.get("status", job.status.value))
        updated = self.context.jobs.update_status(job_id, status)
        return {"local": updated.model_dump(mode="json"), "remote": remote}

    async def logs(self, job_id: str) -> dict[str, Any]:
        job = self.context.jobs.require(job_id)
        return await self.context.client_for(job.server_id).job_logs(job.remote_job_id)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.context.jobs.require(job_id)
        response = await self.context.client_for(job.server_id).cancel_job(job.remote_job_id)
        if "status" in response:
            self.context.jobs.update_status(job_id, response["status"])
        return response

    async def clean(self, workspace_id: str, server_id: str | None = None) -> dict[str, Any]:
        workspace = self._workspace(workspace_id, server_id)
        return await self.context.client_for(workspace.server_id).clean_build(workspace_id)
