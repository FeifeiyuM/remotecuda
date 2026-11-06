"""Health monitoring for all configured remote agents."""

from __future__ import annotations

import asyncio
from typing import Any

from registry import ServerRegistry
from registry.models import ServerRuntimeStatus
from remote_client import RemoteClientPool


class HealthMonitor:
    def __init__(self, server_registry: ServerRegistry, client_pool: RemoteClientPool) -> None:
        self.servers = server_registry
        self.clients = client_pool

    async def check_server(self, server_id: str) -> dict[str, Any]:
        config = self.servers.require_server(server_id)
        if not config.enabled:
            runtime = ServerRuntimeStatus(
                connection_status="DISCONNECTED",
                health_status="UNKNOWN",
                last_error="server disabled",
            )
            self.servers.update_runtime_status(server_id, runtime)
            return {"server_id": server_id, **runtime.model_dump(mode="json")}
        try:
            client = self.clients.get_client(server_id, config)
        except Exception:
            client = self.clients.register_client(config)
        try:
            health = await client.health()
            capabilities = await client.capabilities()
            gpus = await client.list_gpus()
            runtime = ServerRuntimeStatus(
                connection_status="CONNECTED",
                health_status=health.get("status", "HEALTHY"),
                capabilities=capabilities,
                gpus=gpus,
                active_jobs=int(capabilities.get("active_jobs", 0) or 0),
                last_heartbeat=health.get("timestamp"),
            )
            self.servers.update_runtime_status(server_id, runtime)
            return {"server_id": server_id, **runtime.model_dump(mode="json")}
        except Exception as exc:
            runtime = ServerRuntimeStatus(
                connection_status="DISCONNECTED", health_status="UNHEALTHY", last_error=str(exc)
            )
            self.servers.update_runtime_status(server_id, runtime)
            return {"server_id": server_id, **runtime.model_dump(mode="json")}

    async def check_all(self, *, concurrent: bool = False) -> list[dict[str, Any]]:
        server_ids = [item["server_id"] for item in self.servers.list_servers(enabled_only=True)]
        if concurrent:
            results = await asyncio.gather(*(self.check_server(server_id) for server_id in server_ids))
            return list(results)
        return [await self.check_server(server_id) for server_id in server_ids]
