"""Composition root passed to MCP tool modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from monitor.health_monitor import HealthMonitor
from registry import JobRegistry, ServerRegistry, WorkspaceRegistry
from remote_client import RemoteClient, RemoteClientPool
from scheduler import ResourceScheduler

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "runtime"


@dataclass
class AppContext:
    data_dir: Path
    servers: ServerRegistry
    workspaces: WorkspaceRegistry
    jobs: JobRegistry
    clients: RemoteClientPool
    scheduler: ResourceScheduler
    monitor: HealthMonitor

    def client_for(self, server_id: str) -> RemoteClient:
        config = self.servers.require_server(server_id)
        if not config.enabled:
            raise RuntimeError(f"server is disabled: {server_id}")
        try:
            return self.clients.get_client(server_id, config)
        except Exception:
            return self.clients.register_client(config)

    async def close(self) -> None:
        await self.clients.close_all()
        self.servers.close()
        self.workspaces.close()
        self.jobs.close()


def build_context(data_dir: str | Path | None = None, servers_config: str | Path | None = None) -> AppContext:
    resolved_data = Path(data_dir or os.getenv("REMOTE_CUDA_HOME", DEFAULT_DATA_DIR)).expanduser().resolve()
    resolved_data.mkdir(parents=True, exist_ok=True)
    config_path = Path(
        servers_config
        or os.getenv("REMOTE_CUDA_SERVERS_CONFIG", Path.cwd() / "local.yaml")
    ).expanduser()
    servers = ServerRegistry(resolved_data / "servers.db", config_path)
    workspaces = WorkspaceRegistry(resolved_data / "workspaces.db")
    jobs = JobRegistry(resolved_data / "jobs.db")
    clients = RemoteClientPool()
    scheduler = ResourceScheduler(servers)
    monitor = HealthMonitor(servers, clients)
    return AppContext(
        data_dir=resolved_data,
        servers=servers,
        workspaces=workspaces,
        jobs=jobs,
        clients=clients,
        scheduler=scheduler,
        monitor=monitor,
    )
