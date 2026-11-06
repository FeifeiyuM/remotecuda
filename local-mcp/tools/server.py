"""Server management MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from registry import ServerConfig
from tools import get_context, mcp


class ServerAddConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    endpoint: str
    auth_enabled: bool = True
    token_env: str | None = Field(default=None, min_length=1, max_length=128)
    enabled: bool = True
    priority: int = Field(default=10, ge=0, le=10_000)
    tags: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30, gt=0, le=600)

    @model_validator(mode="after")
    def validate_auth_pair(self) -> "ServerAddConfig":
        if self.auth_enabled and not self.token_env:
            raise ValueError("token_env is required when auth_enabled is true")
        return self


@mcp.tool()
async def server_list() -> list[dict[str, Any]]:
    return get_context().servers.list_servers()


@mcp.tool()
async def server_add(config: ServerAddConfig) -> dict[str, Any]:
    return get_context().servers.add_server(ServerConfig.model_validate(config.model_dump())).safe_dict()


@mcp.tool()
async def server_update(server_id: str, config: ServerAddConfig) -> dict[str, Any]:
    return (
        get_context()
        .servers.update_server(server_id, ServerConfig.model_validate(config.model_dump()))
        .safe_dict()
    )


@mcp.tool()
async def server_remove(server_id: str) -> dict[str, Any]:
    context = get_context()
    if context.workspaces.list(server_id=server_id):
        raise RuntimeError("server still has workspaces")
    if context.jobs.has_active_jobs(server_id):
        raise RuntimeError("server still has active jobs")
    context.servers.remove_server(server_id)
    context.clients.remove_client(server_id)
    return {"server_id": server_id, "removed": True}


@mcp.tool()
async def server_info(server_id: str) -> dict[str, Any]:
    server = get_context().servers.get_server(server_id, include_runtime=True)
    if not server:
        raise RuntimeError(f"server not found: {server_id}")
    return server


@mcp.tool()
async def server_health(server_id: str) -> dict[str, Any]:
    return await get_context().monitor.check_server(server_id)


@mcp.tool()
async def server_connect(server_id: str) -> dict[str, Any]:
    context = get_context()
    context.client_for(server_id)
    result = await context.monitor.check_server(server_id)
    return result


@mcp.tool()
async def server_gpu_list(server_id: str) -> list[dict[str, Any]]:
    context = get_context()
    return await context.client_for(server_id).list_gpus()


@mcp.tool()
async def cuda_cluster_status() -> dict[str, Any]:
    context = get_context()
    health = await context.monitor.check_all(concurrent=True)
    return {
        "servers": health,
        "active_jobs": len(context.jobs.list(active_only=True)),
        "workspaces": len(context.workspaces.list()),
    }
