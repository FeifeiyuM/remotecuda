"""Build MCP tools."""

from __future__ import annotations

from typing import Any

from remote_client.models import BuildConfig
from services.job_service import JobService
from tools import get_context, mcp


@mcp.tool()
async def cuda_build(
    workspace_id: str,
    build_config: dict[str, Any],
    server_id: str | None = None,
) -> dict[str, Any]:
    BuildConfig.model_validate(build_config)
    return await JobService(get_context()).build(workspace_id, build_config, server_id=server_id)


@mcp.tool()
async def cuda_build_status(job_id: str) -> dict[str, Any]:
    return await JobService(get_context()).status(job_id)


@mcp.tool()
async def cuda_build_logs(job_id: str) -> dict[str, Any]:
    return await JobService(get_context()).logs(job_id)


@mcp.tool()
async def cuda_clean(workspace_id: str, server_id: str | None = None) -> dict[str, Any]:
    return await JobService(get_context()).clean(workspace_id, server_id)
