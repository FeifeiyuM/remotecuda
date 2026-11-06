"""Remote CUDA Agent FastAPI application entry point."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.artifacts import create_router as artifacts_router
from api.build import create_router as build_router
from api.gpu import create_router as gpu_router
from api.jobs import create_router as jobs_router
from api.sync import create_router as sync_router
from api.workspace import create_router as workspace_router
from agent_config import AgentSettings, load_settings
from executor import BuildExecutor, CudaExecutor, ProcessRunner, ProfilerExecutor
from security import PathSecurityError
from security.auth import BearerAuth
from services.artifact_manager import ArtifactManager
from services.build_manager import BuildManager
from services.gpu_manager import GPUManager
from services.job_manager import JobManager
from services.sync_manager import SyncManager
from services.workspace_manager import WorkspaceManager
from storage.database import Database


LOGGER = logging.getLogger("remote-cuda-agent")


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    if resolved.auth_enabled and resolved.token == "remote-cuda-development-token":
        LOGGER.warning("using development token; set REMOTE_CUDA_AGENT_TOKEN in production")
    database = Database(resolved.database_path)
    workspaces = WorkspaceManager(resolved.workspace_root, database)
    sync = SyncManager(workspaces, resolved.max_upload_bytes)
    gpus = GPUManager(
        cmake_path=resolved.cmake_path,
        nvcc_path=resolved.nvcc_path,
        ncu_path=resolved.ncu_path,
        nsys_path=resolved.nsys_path,
    )
    artifacts = ArtifactManager(workspaces, resolved.max_artifact_bytes)
    runner = ProcessRunner(resolved.max_output_chars)
    builds_executor = BuildExecutor(
        runner,
        resolved.max_build_parallel,
        cmake_path=resolved.cmake_path,
        nvcc_path=resolved.nvcc_path,
    )
    cuda_executor = CudaExecutor(runner, gpus)
    profiler_executor = ProfilerExecutor(
        runner,
        ncu_path=resolved.ncu_path,
        nsys_path=resolved.nsys_path,
    )
    jobs = JobManager(
        database=database,
        workspaces=workspaces,
        artifacts=artifacts,
        runner=runner,
        build_executor=builds_executor,
        cuda_executor=cuda_executor,
        profiler_executor=profiler_executor,
        max_running_jobs=resolved.max_running_jobs,
        default_timeout_seconds=resolved.default_timeout_seconds,
    )
    builds = BuildManager(workspaces, jobs)
    auth = BearerAuth(resolved.token, enabled=resolved.auth_enabled)

    app = FastAPI(
        title="Remote CUDA Agent",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(gpu_router(gpus, database), dependencies=[Depends(auth)])
    app.include_router(workspace_router(workspaces), dependencies=[Depends(auth)])
    app.include_router(sync_router(sync, database), dependencies=[Depends(auth)])
    app.include_router(build_router(builds, jobs, database), dependencies=[Depends(auth)])
    app.include_router(jobs_router(jobs, artifacts, database), dependencies=[Depends(auth)])
    app.include_router(artifacts_router(artifacts, database), dependencies=[Depends(auth)])

    @app.exception_handler(PathSecurityError)
    @app.exception_handler(StarletteHTTPException)
    @app.exception_handler(ValueError)
    @app.exception_handler(FileNotFoundError)
    @app.exception_handler(IndexError)
    async def bad_request(_request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}},
                headers=getattr(exc, "headers", None),
            )
        return JSONResponse(status_code=400, content={"error": {"code": "BAD_REQUEST", "message": str(exc)}})

    @app.exception_handler(RuntimeError)
    async def upstream_failure(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("request failed", exc_info=exc)
        return JSONResponse(
            status_code=409,
            content={"error": {"code": "JOB_OR_RESOURCE_CONFLICT", "message": str(exc)}},
        )

    @app.middleware("http")
    async def request_size_limit(request: Request, call_next: Any) -> Any:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            # Account for base64 encoding plus the small JSON envelope.
            encoded_limit = resolved.max_upload_bytes * 4 // 3 + 64 * 1024
            if int(content_length) > encoded_limit:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "request body exceeds upload limit"}},
                )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await jobs.shutdown()
        database.close()

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    application = create_app(settings)
    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
