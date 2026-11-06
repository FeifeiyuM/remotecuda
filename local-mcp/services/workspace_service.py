"""High-level workspace and synchronization operations."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

from registry import WorkspaceRegistry
from registry.models import WorkspaceRecord, utc_now
from remote_client import RemoteClient
from remote_client.models import ManifestFile, SyncManifest
from security_paths import is_safe_relative_path
from services.context import AppContext


class WorkspaceServiceError(RuntimeError):
    pass


DEFAULT_IGNORES = {
    ".git/",
    ".remote-cuda-artifacts/",
    "build/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    "*.o",
    "*.a",
    "*.so",
    "*.bin",
    "*.fatbin",
}


class WorkspaceService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    @staticmethod
    def _local_root(local_path: str | Path) -> Path:
        root = Path(local_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise WorkspaceServiceError(f"local path is not a directory: {root}")
        return root

    @staticmethod
    def _ignore_rules(root: Path) -> set[str]:
        rules = set(DEFAULT_IGNORES)
        ignore_file = root / ".cudaignore"
        if ignore_file.is_file():
            for raw in ignore_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    rules.add(line)
        return rules

    @staticmethod
    def _ignored(relative_path: str, rules: set[str]) -> bool:
        parts = Path(relative_path).parts
        for rule in rules:
            normalized = rule.rstrip("/")
            if fnmatch.fnmatch(relative_path, rule) or any(fnmatch.fnmatch(part, normalized) for part in parts):
                return True
        return False

    def scan_local(self, workspace: WorkspaceRecord) -> list[ManifestFile]:
        root = self._local_root(workspace.local_path)
        rules = self._ignore_rules(root)
        max_files = int(os.getenv("REMOTE_CUDA_MAX_SYNC_FILES", "20000"))
        max_size = int(os.getenv("REMOTE_CUDA_MAX_FILE_BYTES", str(512 * 1024 * 1024)))
        files: list[ManifestFile] = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if self._ignored(rel, rules):
                continue
            if path.is_symlink():
                real = path.resolve(strict=True)
                if not real.is_relative_to(root):
                    raise WorkspaceServiceError(f"symbolic link escapes workspace: {rel}")
                if not path.is_file():
                    raise WorkspaceServiceError(f"symbolic-link directories are not supported: {rel}")
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > max_size:
                raise WorkspaceServiceError(f"file exceeds sync size limit: {rel}")
            files.append(ManifestFile(path=rel, size=size, sha256=self._sha256(path)))
            if len(files) > max_files:
                raise WorkspaceServiceError("sync manifest exceeds file-count limit")
        return files

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def create(self, server_id: str, name: str, local_path: str, workspace_id: str | None = None) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name):
            raise WorkspaceServiceError("invalid workspace name")
        root = self._local_root(local_path)
        if self.context.workspaces.list(server_id=server_id):
            existing = [item for item in self.context.workspaces.list(server_id=server_id) if item.local_path == str(root)]
            if existing:
                raise WorkspaceServiceError(f"workspace already exists for this local path: {existing[0].workspace_id}")
        workspace_id = workspace_id or f"workspace-{uuid.uuid4().hex[:12]}"
        client = self.context.client_for(server_id)
        remote = await client.create_workspace(
            {"workspace_id": workspace_id, "name": name}
        )
        try:
            record = WorkspaceRecord(
                workspace_id=workspace_id,
                name=name,
                server_id=server_id,
                local_path=str(root),
                remote_path=remote.get("remote_path"),
            )
            return self.context.workspaces.create(record).model_dump(mode="json")
        except Exception:
            await client.delete_workspace(workspace_id)
            raise

    def list(self, server_id: str | None = None) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.context.workspaces.list(server_id)]

    async def info(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.context.workspaces.require(workspace_id)
        remote = await self.context.client_for(workspace.server_id).workspace_info(workspace_id)
        return {"local": workspace.model_dump(mode="json"), "remote": remote}

    async def _sync_files(self, workspace: WorkspaceRecord, client: RemoteClient) -> dict[str, Any]:
        manifest = SyncManifest(workspace_id=workspace.workspace_id, files=self.scan_local(workspace))
        plan = await client.sync_manifest(manifest)
        upload_paths = set(plan.get("upload", []))
        local_root = self._local_root(workspace.local_path)
        for entry in manifest.files:
            if entry.path in upload_paths:
                await client.upload_file(workspace.workspace_id, entry.path, local_root / entry.path)
        verification = await client.sync_manifest(manifest)
        if verification.get("upload"):
            raise WorkspaceServiceError(
                f"remote verification requested another upload for: {', '.join(verification['upload'][:10])}"
            )
        return {
            "workspace_id": workspace.workspace_id,
            "server_id": workspace.server_id,
            "files": len(manifest.files),
            "uploaded": len(upload_paths),
            "unchanged": len(verification.get("unchanged", [])),
            "removed": len(verification.get("remove", [])),
        }

    async def sync(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.context.workspaces.require(workspace_id)
        client = self.context.client_for(workspace.server_id)
        result = await self._sync_files(workspace, client)
        self.context.workspaces.update(workspace_id, status="READY", last_sync_at=utc_now())
        return result

    async def delete(self, workspace_id: str, force: bool = False) -> dict[str, Any]:
        workspace = self.context.workspaces.require(workspace_id)
        active_jobs = self.context.jobs.list(
            server_id=workspace.server_id, workspace_id=workspace_id, active_only=True
        )
        if not force and active_jobs:
            raise WorkspaceServiceError("workspace has active jobs; cancel them before deletion")
        await self.context.client_for(workspace.server_id).delete_workspace(workspace_id)
        self.context.workspaces.delete(workspace_id)
        return {"workspace_id": workspace_id, "deleted": True}

    async def migrate(self, workspace_id: str, target_server_id: str) -> dict[str, Any]:
        workspace = self.context.workspaces.require(workspace_id)
        if workspace.server_id == target_server_id:
            raise WorkspaceServiceError("workspace already belongs to target server")
        if self.context.jobs.list(
            server_id=workspace.server_id, workspace_id=workspace_id, active_only=True
        ):
            raise WorkspaceServiceError("workspace has active jobs; cancel them before migration")
        target_health = await self.context.monitor.check_server(target_server_id)
        if (
            target_health.get("connection_status") != "CONNECTED"
            or target_health.get("health_status") != "HEALTHY"
        ):
            raise WorkspaceServiceError(f"target server is unavailable: {target_server_id}")
        target_client = self.context.client_for(target_server_id)
        remote = await target_client.create_workspace(
            {"workspace_id": workspace.workspace_id, "name": workspace.name}
        )
        staged = workspace.model_copy(update={"server_id": target_server_id})
        try:
            sync_result = await self._sync_files(staged, target_client)
            migrated = self.context.workspaces.migrate(
                workspace_id, target_server_id, remote.get("remote_path")
            )
        except Exception:
            try:
                await target_client.delete_workspace(workspace_id)
            except Exception:
                pass
            raise
        source_delete_error = None
        try:
            await self.context.client_for(workspace.server_id).delete_workspace(workspace_id)
        except Exception as exc:
            source_delete_error = str(exc)
        return {
            "workspace_id": workspace_id,
            "from_server_id": workspace.server_id,
            "to_server_id": target_server_id,
            "remote_path": migrated.remote_path,
            "sync": sync_result,
            "source_delete_error": source_delete_error,
        }
