"""GPU and health REST API."""

from __future__ import annotations

from fastapi import APIRouter

from services.gpu_manager import GPUManager
from storage.database import Database
from storage.models import utc_now


def create_router(gpus: GPUManager, database: Database) -> APIRouter:
    router = APIRouter(tags=["gpu"])

    @router.get("/health")
    async def health() -> dict:
        return {
            "status": "HEALTHY",
            "service": "remote-cuda-agent",
            "version": "2.0.0",
            "timestamp": utc_now(),
        }

    @router.get("/v1/capabilities")
    async def capabilities() -> dict:
        payload = await gpus.capabilities()
        payload["active_jobs"] = len(database.active_jobs())
        return payload

    @router.get("/v1/gpus")
    async def list_gpus() -> dict:
        return {"gpus": await gpus.list_gpus()}

    @router.get("/v1/gpus/{device_id}")
    async def gpu_info(device_id: int) -> dict:
        return await gpus.gpu(device_id)

    @router.get("/v1/gpus/{device_id}/processes")
    async def gpu_processes(device_id: int) -> dict:
        return {"processes": await gpus.processes(device_id)}

    @router.get("/v1/gpus/{device_id}/resource-check")
    async def resource_check(device_id: int, min_memory_mb: int = 0) -> dict:
        return await gpus.resource_check(device_id, min_memory_mb)

    return router
