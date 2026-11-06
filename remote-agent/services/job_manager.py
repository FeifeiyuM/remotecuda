"""Persistent asynchronous job manager."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from executor import (
    BuildExecutor,
    CudaExecutor,
    ProcessFailure,
    ProcessRunner,
    ProcessTimeout,
    ProfilerExecutor,
)
from .artifact_manager import ArtifactManager
from .workspace_manager import WorkspaceManager
from storage.database import Database
from storage.models import ACTIVE_STATUSES, JobMetadata, JobStatus, utc_now


class JobManagerError(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        *,
        database: Database,
        workspaces: WorkspaceManager,
        artifacts: ArtifactManager,
        runner: ProcessRunner,
        build_executor: BuildExecutor,
        cuda_executor: CudaExecutor,
        profiler_executor: ProfilerExecutor,
        max_running_jobs: int,
        default_timeout_seconds: int,
    ) -> None:
        self.database = database
        self.workspaces = workspaces
        self.artifacts = artifacts
        self.runner = runner
        self.build_executor = build_executor
        self.cuda_executor = cuda_executor
        self.profiler_executor = profiler_executor
        self.max_running_jobs = max_running_jobs
        self.default_timeout = default_timeout_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._device_locks: dict[int, asyncio.Lock] = {}
        self._workspace_build_locks: dict[str, asyncio.Lock] = {}

    def active_count(self) -> int:
        return len(self.database.active_jobs())

    def _job_payload(self, job: JobMetadata) -> dict[str, Any]:
        return job.model_dump(mode="json")

    async def submit(
        self,
        *,
        workspace_id: str,
        job_type: str,
        config: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if job_type not in {"BUILD", "RUN", "BENCHMARK", "PROFILER"}:
            raise JobManagerError(f"unsupported job type: {job_type}")
        if idempotency_key:
            existing = self.database.get_job_by_idempotency_key(idempotency_key)
            if existing:
                if existing.workspace_id != workspace_id or existing.job_type != job_type:
                    raise JobManagerError("idempotency key already belongs to a different job")
                return self._job_payload(existing)
        if self.active_count() >= self.max_running_jobs:
            raise JobManagerError("maximum active job count reached")
        self.workspaces.metadata(workspace_id)
        job_id = f"remote-{uuid.uuid4().hex[:16]}"
        log_relative = f"logs/{job_id}.log"
        job = JobMetadata(
            job_id=job_id,
            workspace_id=workspace_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            log_path=log_relative,
            idempotency_key=idempotency_key,
            device_id=int(config.get("device_id", 0)) if job_type != "BUILD" else None,
        )
        self.database.create_job(job)
        self.database.audit("job.create", workspace_id=workspace_id, job_id=job_id, job_type=job_type)

        async def operation() -> dict[str, Any]:
            workspace_root = self.workspaces.workspace_root(workspace_id)
            log_path = workspace_root / log_relative
            timeout = int(config.get("timeout_seconds", self.default_timeout))
            if job_type == "BUILD":
                return await self.build_executor.execute(
                    job_id=job_id,
                    workspace_root=workspace_root,
                    config=config,
                    timeout_seconds=timeout,
                    log_path=log_path,
                )
            if job_type == "BENCHMARK":
                method = self.cuda_executor.benchmark
            elif job_type == "PROFILER":
                method = self.profiler_executor.profile
            else:
                method = self.cuda_executor.execute
            return await method(
                job_id=job_id,
                workspace_root=workspace_root,
                config=config,
                timeout_seconds=timeout,
                log_path=log_path,
            )

        task = asyncio.create_task(self._execute(job, operation), name=job_id)
        self._tasks[job_id] = task
        return self._job_payload(job)

    async def _execute(self, job: JobMetadata, operation: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        device_lock: asyncio.Lock | None = None
        workspace_lock: asyncio.Lock | None = None
        gpu_acquired = False
        if job.job_type == "BUILD":
            workspace_lock = self._workspace_build_locks.setdefault(job.workspace_id, asyncio.Lock())
            await workspace_lock.acquire()
        if job.job_type in {"RUN", "BENCHMARK", "PROFILER"}:
            device_id = int(job.device_id or 0)
            device_lock = self._device_locks.setdefault(device_id, asyncio.Lock())
            await device_lock.acquire()
            if not self.database.acquire_gpu(device_id, job.job_id):
                job.status = JobStatus.FAILED
                job.error = f"GPU {device_id} is locked by another job"
                job.finished_at = utc_now()
                self.database.update_job(job)
                device_lock.release()
                return
            gpu_acquired = True
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        self.database.update_job(job)
        try:
            result = await operation()
            if job.job_type == "BUILD":
                result["binaries"] = self.artifacts.collect_binaries(
                    job.workspace_id, Path(result["build_dir"])
                )
            job.result = result
            job.status = JobStatus.SUCCESS
            job.exit_code = int(result.get("exit_code", 0))
        except ProcessTimeout as exc:
            job.status = JobStatus.TIMEOUT
            job.error = str(exc)
        except ProcessFailure as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.error = "job cancelled"
            raise
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.finished_at = utc_now()
            if job.log_path:
                try:
                    self.artifacts.collect_log(
                        job.workspace_id,
                        job.job_id,
                        self.workspaces.workspace_root(job.workspace_id) / job.log_path,
                    )
                except Exception:
                    pass
            self.database.update_job(job)
            if gpu_acquired:
                self.database.release_gpu(int(job.device_id or 0))
            if device_lock and device_lock.locked():
                device_lock.release()
            if workspace_lock and workspace_lock.locked():
                workspace_lock.release()
            self._tasks.pop(job.job_id, None)

    def get(self, job_id: str) -> JobMetadata:
        job = self.database.get_job(job_id)
        if not job:
            raise JobManagerError(f"job not found: {job_id}")
        return job

    def logs(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        log_path = self.workspaces.workspace_root(job.workspace_id) / str(job.log_path or "")
        content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        return {"job_id": job_id, "workspace_id": job.workspace_id, "log_path": job.log_path, "content": content}

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.status not in ACTIVE_STATUSES:
            return {"job_id": job_id, "status": job.status.value, "cancelled": False, "reason": "job already finished"}
        self.runner.terminate(job_id)
        self.runner.terminate(f"{job_id}:configure")
        self.runner.terminate(f"{job_id}:build")
        self.runner.terminate(f"{job_id}:warmup")
        self.runner.terminate(f"{job_id}:iteration")
        self.runner.terminate(f"{job_id}:profile")
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        final = self.get(job_id)
        if final.status in ACTIVE_STATUSES:
            final.status = JobStatus.CANCELLED
            final.finished_at = utc_now()
            final.error = final.error or "job cancelled"
            self.database.update_job(final)
        return {"job_id": job_id, "status": final.status.value, "cancelled": final.status == JobStatus.CANCELLED}

    async def shutdown(self) -> None:
        """Cancel in-process tasks before the persistent database closes."""

        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
