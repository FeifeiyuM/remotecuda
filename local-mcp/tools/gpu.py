"""GPU query MCP tools."""

from __future__ import annotations

from typing import Any

from tools import get_context, mcp


@mcp.tool()
async def cuda_list_devices(server_id: str) -> list[dict[str, Any]]:
    return await get_context().client_for(server_id).list_gpus()


@mcp.tool()
async def cuda_device_info(server_id: str, device_id: int) -> dict[str, Any]:
    return await get_context().client_for(server_id).gpu_info(device_id)


@mcp.tool()
async def cuda_get_status(server_id: str, device_id: int) -> dict[str, Any]:
    return await get_context().client_for(server_id).gpu_info(device_id)


@mcp.tool()
async def cuda_list_processes(server_id: str, device_id: int) -> list[dict[str, Any]]:
    return await get_context().client_for(server_id).gpu_processes(device_id)


@mcp.tool()
async def cuda_resource_check(server_id: str, device_id: int, min_memory_mb: int = 0) -> dict[str, Any]:
    gpu = await get_context().client_for(server_id).gpu_info(device_id)
    return {
        "server_id": server_id,
        "available": int(gpu.get("memory_free_mb", 0)) >= min_memory_mb,
        "gpu": gpu,
    }
