"""Artifact REST API."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from services.artifact_manager import ArtifactManager
from storage.database import Database


def create_router(artifacts: ArtifactManager, database: Database) -> APIRouter:
    router = APIRouter(prefix="/v1/workspaces/{workspace_id}/artifacts", tags=["artifacts"])

    @router.get("")
    async def list_artifacts(workspace_id: str) -> dict:
        return {"workspace_id": workspace_id, "artifacts": artifacts.list(workspace_id)}

    @router.get("/download")
    async def download_artifact(workspace_id: str, path: str, http_request: Request) -> FileResponse:
        target = artifacts.download_path(workspace_id, path)
        database.audit("artifact.download", workspace_id=workspace_id, path=path)
        return FileResponse(target, filename=target.name, media_type="application/octet-stream")

    return router
