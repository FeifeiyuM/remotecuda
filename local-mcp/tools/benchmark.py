"""Benchmark MCP tools."""

from __future__ import annotations

from typing import Any

from remote_client.models import BenchmarkConfig, ProfilerConfig
from services.job_service import JobService
from tools import get_context, mcp


@mcp.tool()
async def cuda_benchmark(
    workspace_id: str,
    benchmark_config: dict[str, Any],
    server_id: str | None = None,
) -> dict[str, Any]:
    BenchmarkConfig.model_validate(benchmark_config)
    return await JobService(get_context()).benchmark(workspace_id, benchmark_config, server_id=server_id)


@mcp.tool()
async def benchmark_result(job_id: str) -> dict[str, Any]:
    context = get_context()
    job = context.jobs.require(job_id)
    remote = await context.client_for(job.server_id).job(job.remote_job_id)
    result = remote.get("result", {})
    artifact_path = result.get("artifact_path")
    if not artifact_path:
        raise RuntimeError("benchmark result is not ready")
    return await context.client_for(job.server_id).benchmark_result(job.remote_job_id)


@mcp.tool()
async def profiler_result(job_id: str) -> dict[str, Any]:
    context = get_context()
    job = context.jobs.require(job_id)
    return await context.client_for(job.server_id).profiler_result(job.remote_job_id)


@mcp.tool()
async def cuda_profile(
    workspace_id: str,
    profiler_config: dict[str, Any],
    server_id: str | None = None,
) -> dict[str, Any]:
    ProfilerConfig.model_validate(profiler_config)
    return await JobService(get_context()).profile(workspace_id, profiler_config, server_id=server_id)
