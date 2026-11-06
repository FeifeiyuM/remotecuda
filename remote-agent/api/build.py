"""Build REST API."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from services.build_manager import BuildManager
from services.job_manager import JobManager
from storage.database import Database


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_system: Literal["cmake"] = "cmake"
    build_type: str = Field(default="Release", pattern="^(Debug|Release|RelWithDebInfo|MinSizeRel)$")
    cuda_arch: str | None = Field(default=None, pattern=r"^[0-9]{2,3}$")
    parallel_jobs: int = Field(default=2, ge=1, le=256)
    target: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.+-]+$")
    clean: bool = False


def create_router(builds: BuildManager, jobs: JobManager, database: Database) -> APIRouter:
    router = APIRouter(prefix="/v1/workspaces/{workspace_id}", tags=["build"])

    @router.post("/build", status_code=202)
    async def create_build(
        request: BuildRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        workspace_id = str(http_request.path_params["workspace_id"])
        result = await builds.submit(workspace_id, request.model_dump(), idempotency_key)
        database.audit("job.build", workspace_id=workspace_id, job_id=result["job_id"])
        return result

    @router.delete("/build", status_code=202)
    async def clean_build(http_request: Request) -> dict:
        workspace_id = str(http_request.path_params["workspace_id"])
        result = builds.clean(workspace_id)
        database.audit("build.clean", workspace_id=workspace_id)
        return result

    return router
