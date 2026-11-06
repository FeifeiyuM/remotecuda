"""Persistent registry for Remote CUDA Agent servers."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from registry.models import (
    ConnectionStatus,
    HealthStatus,
    ServerConfig,
    ServerRuntimeStatus,
)


class ServerRegistryError(RuntimeError):
    pass


class ServerRegistry:
    """SQLite-backed registry containing static config and volatile status."""

    def __init__(self, database_path: str | Path, config_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.touch(mode=0o600, exist_ok=True)
        self.database_path.chmod(0o600)
        self.config_path = Path(config_path) if config_path else None
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.database_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        self._migrate_legacy_server_config()
        if self.config_path and self.config_path.exists():
            self._import_config_file()

    def _init_schema(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS servers (
                    server_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    runtime_json TEXT NOT NULL
                );
                """
            )

    def _migrate_legacy_server_config(self) -> None:
        """Normalize retired settings persisted by older Local MCP versions."""

        rows = self._db.execute("SELECT server_id, config_json FROM servers").fetchall()
        with self._db:
            for row in rows:
                payload = json.loads(row["config_json"])
                changed = False

                endpoint = str(payload.get("endpoint", ""))
                parsed = urlsplit(endpoint)
                if parsed.scheme != "http":
                    endpoint = urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
                    payload["endpoint"] = endpoint
                    changed = True

                if changed:
                    self._db.execute(
                        "UPDATE servers SET config_json=? WHERE server_id=?",
                        (json.dumps(payload, separators=(",", ":")), row["server_id"]),
                    )

    def _import_config_file(self) -> None:
        assert self.config_path is not None
        text = self.config_path.read_text(encoding="utf-8")
        if self.config_path.suffix.lower() == ".json":
            import json

            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ServerRegistryError(
                    "PyYAML is required to load local.yaml; install local-mcp dependencies"
                ) from exc
            payload = yaml.safe_load(text)
        entries = payload.get("servers", []) if isinstance(payload, dict) else []
        for entry in entries:
            config = ServerConfig.model_validate(entry)
            existing = self.get_server(config.server_id, include_runtime=False)
            if existing is None:
                self.add_server(config)
            elif existing != config.safe_dict():
                self.update_server(config.server_id, config)

    def add_server(self, config: ServerConfig) -> ServerConfig:
        config = ServerConfig.model_validate(config)
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM servers WHERE server_id=?", (config.server_id,)).fetchone():
                raise ServerRegistryError(f"server already exists: {config.server_id}")
            runtime = ServerRuntimeStatus()
            self._db.execute(
                "INSERT INTO servers(server_id, config_json, runtime_json) VALUES(?,?,?)",
                (config.server_id, config.model_dump_json(), runtime.model_dump_json()),
            )
        return config

    def update_server(self, server_id: str, config: ServerConfig) -> ServerConfig:
        config = ServerConfig.model_validate(config)
        if config.server_id != server_id:
            raise ServerRegistryError("server_id in path and config must match")
        with self._lock, self._db:
            if not self._db.execute("SELECT 1 FROM servers WHERE server_id=?", (server_id,)).fetchone():
                raise ServerRegistryError(f"server not found: {server_id}")
            self._db.execute(
                "UPDATE servers SET config_json=? WHERE server_id=?",
                (config.model_dump_json(), server_id),
            )
        return config

    def remove_server(self, server_id: str) -> None:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM servers WHERE server_id=?", (server_id,))
            if cur.rowcount == 0:
                raise ServerRegistryError(f"server not found: {server_id}")

    def get_server(self, server_id: str, include_runtime: bool = False) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM servers WHERE server_id=?", (server_id,)).fetchone()
        if not row:
            return None
        config = ServerConfig.model_validate_json(row["config_json"]).safe_dict()
        if include_runtime:
            config["runtime"] = ServerRuntimeStatus.model_validate_json(row["runtime_json"]).model_dump(mode="json")
        return config

    def require_server(self, server_id: str) -> ServerConfig:
        row = self._db.execute("SELECT config_json FROM servers WHERE server_id=?", (server_id,)).fetchone()
        if not row:
            raise ServerRegistryError(f"server not found: {server_id}")
        return ServerConfig.model_validate_json(row["config_json"])

    def list_servers(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM servers ORDER BY server_id").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self.get_server(row["server_id"], include_runtime=True)
            if item and (not enabled_only or item["enabled"]):
                result.append(item)
        return result

    def update_runtime_status(self, server_id: str, status: ServerRuntimeStatus | dict[str, Any]) -> ServerRuntimeStatus:
        runtime = ServerRuntimeStatus.model_validate(status)
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE servers SET runtime_json=? WHERE server_id=?",
                (runtime.model_dump_json(), server_id),
            )
            if cur.rowcount == 0:
                raise ServerRegistryError(f"server not found: {server_id}")
        return runtime

    def get_server_status(self, server_id: str) -> ServerRuntimeStatus:
        row = self._db.execute("SELECT runtime_json FROM servers WHERE server_id=?", (server_id,)).fetchone()
        if not row:
            raise ServerRegistryError(f"server not found: {server_id}")
        return ServerRuntimeStatus.model_validate_json(row["runtime_json"])

    def set_connection(self, server_id: str, connected: bool, error: str | None = None) -> None:
        runtime = self.get_server_status(server_id)
        runtime.connection_status = ConnectionStatus.CONNECTED if connected else ConnectionStatus.DISCONNECTED
        runtime.last_error = error
        self.update_runtime_status(server_id, runtime)

    def mark_healthy(self, server_id: str, error: str | None = None) -> None:
        runtime = self.get_server_status(server_id)
        runtime.health_status = HealthStatus.UNHEALTHY if error else HealthStatus.HEALTHY
        if error:
            runtime.last_error = error
        self.update_runtime_status(server_id, runtime)

    @staticmethod
    def validate_config(data: dict[str, Any]) -> ServerConfig:
        try:
            return ServerConfig.model_validate(data)
        except ValidationError as exc:
            raise ServerRegistryError(f"invalid server config: {exc}") from exc

    def close(self) -> None:
        self._db.close()
