"""Workspace lifecycle management on one GPU server."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from security import atomic_write
from storage.database import Database
from storage.models import WorkspaceMetadata


class WorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=128, pattern=r"^workspace-[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_. -]*$")


class WorkspaceManager:
    def __init__(self, workspace_root: Path, database: Database) -> None:
        self.root = workspace_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = database

    def workspace_root(self, workspace_id: str) -> Path:
        if not re.fullmatch(r"workspace-[A-Za-z0-9_.-]+", workspace_id):
            raise ValueError("invalid workspace id")
        candidate = self.root / workspace_id
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ValueError("workspace path escapes root")
        return resolved

    def create(self, request: WorkspaceCreateRequest) -> dict[str, Any]:
        if self.database.get_workspace(request.workspace_id):
            raise ValueError(f"workspace already exists: {request.workspace_id}")
        workspace_root = self.workspace_root(request.workspace_id)
        if workspace_root.exists():
            raise ValueError("workspace directory already exists")
        for directory in (
            "source",
            "build",
            "artifacts/binaries",
            "artifacts/benchmark",
            "artifacts/profiler",
            "logs",
        ):
            (workspace_root / directory).mkdir(parents=True, exist_ok=True, mode=0o700)
        workspace_root.chmod(0o700)
        metadata = WorkspaceMetadata(workspace_id=request.workspace_id, name=request.name)
        self.database.create_workspace(metadata)
        atomic_write(workspace_root / "workspace.json", metadata.model_dump_json().encode("utf-8"))
        self.database.audit("workspace.create", workspace_id=request.workspace_id, name=request.name)
        return self.info(request.workspace_id)

    def metadata(self, workspace_id: str) -> WorkspaceMetadata:
        metadata = self.database.get_workspace(workspace_id)
        if not metadata:
            raise FileNotFoundError(f"workspace not found: {workspace_id}")
        return metadata

    def info(self, workspace_id: str) -> dict[str, Any]:
        metadata = self.metadata(workspace_id)
        root = self.workspace_root(workspace_id)
        source_count = sum(1 for path in (root / "source").rglob("*") if path.is_file())
        return {
            **metadata.model_dump(mode="json"),
            "remote_path": str(root),
            "source_files": source_count,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.info(metadata.workspace_id) for metadata in self.database.list_workspaces()]

    def update_status(self, workspace_id: str, status: str) -> WorkspaceMetadata:
        metadata = self.metadata(workspace_id)
        metadata.status = status
        self.database.update_workspace(metadata)
        atomic_write(
            self.workspace_root(workspace_id) / "workspace.json", metadata.model_dump_json().encode("utf-8")
        )
        return metadata

    def update_sync_time(self, workspace_id: str) -> WorkspaceMetadata:
        from storage.models import utc_now

        metadata = self.metadata(workspace_id)
        metadata.last_sync_at = utc_now()
        self.database.update_workspace(metadata)
        atomic_write(
            self.workspace_root(workspace_id) / "workspace.json", metadata.model_dump_json().encode("utf-8")
        )
        return metadata

    def delete(self, workspace_id: str) -> None:
        metadata = self.metadata(workspace_id)
        root = self.workspace_root(workspace_id)
        if not root.is_relative_to(self.root):
            raise ValueError("refusing to delete outside workspace root")
        if self.database.active_jobs(workspace_id):
            raise ValueError("workspace has active jobs")
        self.database.delete_workspace(workspace_id)
        if root.exists():
            shutil.rmtree(root)
        self.database.audit("workspace.delete", workspace_id=workspace_id, name=metadata.name)
