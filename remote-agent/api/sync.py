"""Incremental synchronization REST API."""

from __future__ import annotations

from fastapi import APIRouter, Request

from services.sync_manager import FileUploadRequest, SyncManager, SyncManifestRequest
from storage.database import Database


def create_router(sync: SyncManager, database: Database) -> APIRouter:
    router = APIRouter(prefix="/v1/workspaces/{workspace_id}/sync", tags=["sync"])

    @router.post("/manifest")
    async def submit_manifest(request: SyncManifestRequest, http_request: Request) -> dict:
        if request.workspace_id != http_request.path_params["workspace_id"]:
            raise ValueError("workspace_id in body and path do not match")
        result = sync.apply_manifest(request)
        database.audit("sync.manifest", workspace_id=request.workspace_id, files=len(request.files))
        return result

    @router.put("/file")
    async def upload_file(request: FileUploadRequest, http_request: Request) -> dict:
        path_workspace = str(http_request.path_params["workspace_id"])
        if request.workspace_id != path_workspace:
            raise ValueError("workspace_id in body and path do not match")
        result = sync.upload(request)
        database.audit("sync.file", workspace_id=path_workspace, path=request.path, size=result["size"])
        return result

    return router
