"""Workspace MCP tools."""

from __future__ import annotations

from typing import Any

from services.workspace_service import WorkspaceService
from tools import get_context, mcp


@mcp.tool()
async def workspace_create(
    server_id: str | None,
    name: str,
    local_path: str,
    requirements: dict[str, Any] | None = None,
    selection_policy: str = "least_loaded",
) -> dict[str, Any]:
    context = get_context()
    if server_id is None:
        await context.monitor.check_all(concurrent=True)
        server_id = context.scheduler.select_server(
            requirements=requirements, selection_policy=selection_policy
        )
    return await WorkspaceService(context).create(server_id, name, local_path)


@mcp.tool()
async def workspace_list(server_id: str | None = None) -> list[dict[str, Any]]:
    return WorkspaceService(get_context()).list(server_id)


@mcp.tool()
async def workspace_info(workspace_id: str) -> dict[str, Any]:
    return await WorkspaceService(get_context()).info(workspace_id)


@mcp.tool()
async def workspace_delete(workspace_id: str) -> dict[str, Any]:
    return await WorkspaceService(get_context()).delete(workspace_id)


@mcp.tool()
async def workspace_migrate(workspace_id: str, target_server_id: str) -> dict[str, Any]:
    return await WorkspaceService(get_context()).migrate(workspace_id, target_server_id)
