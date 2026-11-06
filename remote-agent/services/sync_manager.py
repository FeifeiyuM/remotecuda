"""Manifest-based incremental source synchronization."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from security import PathSecurityError, atomic_write, resolve_inside, safe_relative_path
from .workspace_manager import WorkspaceManager


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=1024**3)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        safe_relative_path(value)
        return value


class SyncManifestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    files: list[ManifestFile] = Field(default_factory=list, max_length=50_000)


class FileUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    path: str = Field(min_length=1, max_length=1024)
    content_base64: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class SyncManager:
    def __init__(self, workspaces: WorkspaceManager, max_upload_bytes: int) -> None:
        self.workspaces = workspaces
        self.max_upload_bytes = max_upload_bytes

    def source_root(self, workspace_id: str) -> Path:
        self.workspaces.metadata(workspace_id)
        return self.workspaces.workspace_root(workspace_id) / "source"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def plan(self, request: SyncManifestRequest) -> dict[str, Any]:
        self.workspaces.metadata(request.workspace_id)
        source_root = self.source_root(request.workspace_id)
        source_root.mkdir(parents=True, exist_ok=True)
        desired = {entry.path: entry for entry in request.files}
        actual: dict[str, Path] = {
            path.relative_to(source_root).as_posix(): path
            for path in source_root.rglob("*")
            if path.is_file() and path.name != ".manifest.json"
        }
        upload: list[str] = []
        unchanged: list[str] = []
        for relative, entry in desired.items():
            current = actual.get(relative)
            if current is None or current.stat().st_size != entry.size or self._sha256(current) != entry.sha256:
                upload.append(relative)
            else:
                unchanged.append(relative)
        remove = sorted(set(actual) - set(desired))
        return {
            "workspace_id": request.workspace_id,
            "upload": sorted(upload),
            "unchanged": sorted(unchanged),
            "remove": remove,
        }

    def apply_manifest(self, request: SyncManifestRequest) -> dict[str, Any]:
        plan = self.plan(request)
        source_root = self.source_root(request.workspace_id)
        for relative in plan["remove"]:
            path = resolve_inside(source_root, relative, must_exist=True)
            path.unlink()
            parent = path.parent
            while parent != source_root and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        atomic_write(
            source_root / ".manifest.json",
            request.model_dump_json().encode("utf-8"),
        )
        self.workspaces.update_sync_time(request.workspace_id)
        return plan

    def upload(self, request: FileUploadRequest) -> dict[str, Any]:
        relative = safe_relative_path(request.path)
        source_root = self.source_root(request.workspace_id)
        try:
            content = base64.b64decode(request.content_base64, validate=True)
        except Exception as exc:
            raise ValueError("invalid base64 content") from exc
        if len(content) > self.max_upload_bytes:
            raise ValueError("uploaded file exceeds size limit")
        digest = hashlib.sha256(content).hexdigest()
        if digest != request.sha256:
            raise ValueError("uploaded file SHA256 mismatch")
        target = resolve_inside(source_root, relative)
        atomic_write(target, content)
        self.workspaces.metadata(request.workspace_id)
        return {"workspace_id": request.workspace_id, "path": relative.as_posix(), "size": len(content), "sha256": digest}
