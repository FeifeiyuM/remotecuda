"""Cross-server resource scheduler."""

from __future__ import annotations

from typing import Any

from registry import ServerRegistry, ServerRegistryError
from scheduler.selection_policy import (
    ServerCandidate,
    ScheduleRequirements,
    filter_candidates,
    sort_by_policy,
)


class SchedulerError(RuntimeError):
    pass


class ResourceScheduler:
    def __init__(self, server_registry: ServerRegistry) -> None:
        self._servers = server_registry

    def select_server(
        self,
        *,
        server_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        selection_policy: str = "least_loaded",
    ) -> str:
        if server_id:
            config = self._servers.get_server(server_id)
            if not config:
                raise ServerRegistryError(f"server not found: {server_id}")
            if not config["enabled"]:
                raise SchedulerError(f"server is disabled: {server_id}")
            return server_id

        candidates: list[ServerCandidate] = []
        for server in self._servers.list_servers(enabled_only=True):
            runtime = server.get("runtime", {})
            candidates.append(
                ServerCandidate(
                    server_id=server["server_id"],
                    enabled=server["enabled"],
                    priority=server["priority"],
                    tags=server["tags"],
                    capabilities=runtime.get("capabilities", {}),
                    gpus=runtime.get("gpus", []),
                    active_jobs=runtime.get("active_jobs", 0),
                    healthy=runtime.get("health_status") == "HEALTHY",
                    connected=runtime.get("connection_status") == "CONNECTED",
                )
            )
        filtered = filter_candidates(candidates, ScheduleRequirements.model_validate(requirements or {}))
        ordered = sort_by_policy(filtered, selection_policy)
        if not ordered:
            raise SchedulerError("no eligible Remote CUDA server")
        return ordered[0].server_id
