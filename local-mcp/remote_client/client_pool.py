"""Connection pool which keeps one HTTP client per GPU server."""

from __future__ import annotations

import json
import os
from typing import Any

from registry import ServerConfig
from remote_client.client import RemoteClient


class RemoteClientPoolError(RuntimeError):
    pass


class RemoteClientPool:
    def __init__(self) -> None:
        self._clients: dict[str, RemoteClient] = {}
        self._retired: list[RemoteClient] = []
        self._fingerprints: dict[str, str] = {}

    @staticmethod
    def _fingerprint(config: ServerConfig) -> str:
        return json.dumps(
            {
                "endpoint": config.endpoint,
                "auth_enabled": config.auth_enabled,
                "token_env": config.token_env,
                "timeout_seconds": config.timeout_seconds,
            },
            sort_keys=True,
        )

    def register_client(
        self, server_config: ServerConfig | dict[str, Any], token: str | None = None
    ) -> RemoteClient:
        config = ServerConfig.model_validate(server_config)
        resolved_token = None
        if config.auth_enabled:
            resolved_token = token or os.getenv(config.token_env or "")
        if config.auth_enabled and not resolved_token:
            raise RemoteClientPoolError(f"environment variable {config.token_env} is not set")
        if not config.enabled:
            raise RemoteClientPoolError(f"server is disabled: {config.server_id}")

        old = self._clients.pop(config.server_id, None)
        if old:
            self._retired.append(old)
        self._fingerprints.pop(config.server_id, None)

        client = RemoteClient(
            server_id=config.server_id,
            endpoint=config.endpoint,
            token=resolved_token,
            timeout_seconds=config.timeout_seconds,
        )
        self._clients[config.server_id] = client
        self._fingerprints[config.server_id] = self._fingerprint(config)
        return client

    def get_client(self, server_id: str, expected_config: ServerConfig | None = None) -> RemoteClient:
        client = self._clients.get(server_id)
        if client and expected_config:
            expected_fingerprint = self._fingerprint(expected_config)
            if self._fingerprints.get(server_id) != expected_fingerprint:
                return self.register_client(expected_config)
        if not client:
            raise RemoteClientPoolError(f"client is not connected: {server_id}")
        return client

    def remove_client(self, server_id: str) -> None:
        client = self._clients.pop(server_id, None)
        if client:
            self._retired.append(client)
        self._fingerprints.pop(server_id, None)

    async def close_all(self) -> None:
        clients = list(self._clients.values()) + self._retired
        self._clients.clear()
        self._retired.clear()
        self._fingerprints.clear()
        for client in clients:
            await client.aclose()
