"""Artifact MCP tools."""

from __future__ import annotations

from typing import Any

from services.artifact_service import ArtifactService
from tools import get_context, mcp


@mcp.tool()
async def artifact_list(workspace_id: str) -> dict[str, Any]:
    return await ArtifactService(get_context()).list(workspace_id)


@mcp.tool()
async def artifact_download(workspace_id: str, artifact_path: str) -> dict[str, Any]:
    return await ArtifactService(get_context()).download(workspace_id, artifact_path)
