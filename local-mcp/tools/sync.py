"""Source synchronization MCP tool."""

from __future__ import annotations

from typing import Any

from services.workspace_service import WorkspaceService
from tools import get_context, mcp


@mcp.tool()
async def workspace_sync(workspace_id: str) -> dict[str, Any]:
    return await WorkspaceService(get_context()).sync(workspace_id)
