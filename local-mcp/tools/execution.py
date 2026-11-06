"""CUDA execution MCP tools."""

from __future__ import annotations

from typing import Any

from remote_client.models import ExecutionConfig
from services.job_service import JobService
from tools import get_context, mcp


@mcp.tool()
async def cuda_run(
    workspace_id: str,
    execution_config: dict[str, Any],
    server_id: str | None = None,
) -> dict[str, Any]:
    ExecutionConfig.model_validate(execution_config)
    return await JobService(get_context()).run(workspace_id, execution_config, server_id=server_id)


@mcp.tool()
async def cuda_job_status(job_id: str) -> dict[str, Any]:
    return await JobService(get_context()).status(job_id)


@mcp.tool()
async def cuda_job_logs(job_id: str) -> dict[str, Any]:
    return await JobService(get_context()).logs(job_id)


@mcp.tool()
async def cuda_job_cancel(job_id: str) -> dict[str, Any]:
    return await JobService(get_context()).cancel(job_id)
