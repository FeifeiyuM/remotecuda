"""Workspace REST API."""

from __future__ import annotations

from fastapi import APIRouter, Request

from services.workspace_manager import WorkspaceCreateRequest, WorkspaceManager


def create_router(workspaces: WorkspaceManager) -> APIRouter:
    router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])

    @router.post("", status_code=201)
    async def create_workspace(request: WorkspaceCreateRequest) -> dict:
        return workspaces.create(request)

    @router.get("")
    async def list_workspaces() -> dict:
        return {"workspaces": workspaces.list()}

    @router.get("/{workspace_id}")
    async def workspace_info(workspace_id: str) -> dict:
        return workspaces.info(workspace_id)

    @router.delete("/{workspace_id}", status_code=202)
    async def delete_workspace(workspace_id: str, http_request: Request) -> dict:
        workspaces.delete(workspace_id)
        return {"workspace_id": workspace_id, "deleted": True}

    return router
