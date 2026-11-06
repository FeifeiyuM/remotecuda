"""Persistent local index of remote workspaces."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from registry.models import WorkspaceRecord, utc_now


class WorkspaceRegistryError(RuntimeError):
    pass


class WorkspaceRegistry:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.touch(mode=0o600, exist_ok=True)
        self.database_path.chmod(0o600)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.database_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                )
                """
            )

    def create(self, record: WorkspaceRecord | dict[str, Any]) -> WorkspaceRecord:
        try:
            item = WorkspaceRecord.model_validate(record)
        except ValidationError as exc:
            raise WorkspaceRegistryError(f"invalid workspace: {exc}") from exc
        item.updated_at = utc_now()
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM workspaces WHERE workspace_id=?", (item.workspace_id,)).fetchone():
                raise WorkspaceRegistryError(f"workspace already exists: {item.workspace_id}")
            self._db.execute("INSERT INTO workspaces VALUES(?,?)", (item.workspace_id, item.model_dump_json()))
        return item

    def get(self, workspace_id: str) -> WorkspaceRecord | None:
        row = self._db.execute("SELECT record_json FROM workspaces WHERE workspace_id=?", (workspace_id,)).fetchone()
        return WorkspaceRecord.model_validate_json(row["record_json"]) if row else None

    def require(self, workspace_id: str) -> WorkspaceRecord:
        item = self.get(workspace_id)
        if not item:
            raise WorkspaceRegistryError(f"workspace not found: {workspace_id}")
        return item

    def list(self, server_id: str | None = None) -> list[WorkspaceRecord]:
        rows = self._db.execute("SELECT record_json FROM workspaces ORDER BY workspace_id").fetchall()
        items = [WorkspaceRecord.model_validate_json(row["record_json"]) for row in rows]
        return [item for item in items if server_id is None or item.server_id == server_id]

    def update(self, workspace_id: str, **changes: Any) -> WorkspaceRecord:
        item = self.require(workspace_id)
        payload = item.model_dump()
        protected = {"workspace_id", "server_id", "created_at"}
        unknown = set(changes) - set(payload)
        if unknown:
            raise WorkspaceRegistryError(f"unknown workspace fields: {sorted(unknown)}")
        if protected.intersection(changes):
            raise WorkspaceRegistryError(f"cannot change immutable fields: {sorted(protected.intersection(changes))}")
        payload.update(changes)
        item = WorkspaceRecord.model_validate(payload)
        item.updated_at = utc_now()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE workspaces SET record_json=? WHERE workspace_id=?",
                (item.model_dump_json(), workspace_id),
            )
        return item

    def delete(self, workspace_id: str) -> None:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM workspaces WHERE workspace_id=?", (workspace_id,))
            if cur.rowcount == 0:
                raise WorkspaceRegistryError(f"workspace not found: {workspace_id}")

    def migrate(self, workspace_id: str, target_server_id: str, remote_path: str | None = None) -> WorkspaceRecord:
        """Explicitly move the local workspace binding to another server."""

        item = self.require(workspace_id)
        with self._lock, self._db:
            payload = item.model_dump()
            payload["server_id"] = target_server_id
            if remote_path is not None:
                payload["remote_path"] = remote_path
            payload["updated_at"] = utc_now()
            migrated = WorkspaceRecord.model_validate(payload)
            self._db.execute(
                "UPDATE workspaces SET record_json=? WHERE workspace_id=?",
                (migrated.model_dump_json(), workspace_id),
            )
        return migrated

    def count_for_server(self, server_id: str) -> int:
        row = self._db.execute("SELECT COUNT(*) AS n FROM workspaces").fetchone()
        return sum(1 for item in self.list() if item.server_id == server_id) if row["n"] else 0

    def close(self) -> None:
        self._db.close()
