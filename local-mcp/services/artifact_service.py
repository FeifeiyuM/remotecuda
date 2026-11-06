"""Artifact listing and download orchestration."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any, TYPE_CHECKING

from security_paths import is_safe_relative_path

if TYPE_CHECKING:
    from services.context import AppContext


class ArtifactServiceError(RuntimeError):
    pass


class ArtifactService:
    def __init__(self, context: "AppContext") -> None:
        self.context = context

    async def list(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.context.workspaces.require(workspace_id)
        payload = await self.context.client_for(workspace.server_id).artifacts(workspace_id)
        return {"workspace_id": workspace_id, "server_id": workspace.server_id, **payload}

    async def download(self, workspace_id: str, artifact_path: str) -> dict[str, Any]:
        if not is_safe_relative_path(artifact_path) or ".." in PurePosixPath(artifact_path).parts:
            raise ArtifactServiceError("unsafe artifact path")
        requested_path = PurePosixPath(artifact_path)
        if len(requested_path.parts) > 1 and requested_path.parts[0] == "artifacts":
            requested_path = PurePosixPath(*requested_path.parts[1:])
            artifact_path = requested_path.as_posix()
        workspace = self.context.workspaces.require(workspace_id)
        project_root = Path(workspace.local_path).expanduser().resolve(strict=True)
        destination_root = (project_root / ".remote-cuda-artifacts").resolve(strict=False)
        destination = destination_root.joinpath(*PurePosixPath(artifact_path).parts)
        if not destination_root.is_relative_to(project_root):
            raise ArtifactServiceError("artifact destination escapes local workspace")
        resolved_destination = destination.resolve(strict=False)
        if not resolved_destination.is_relative_to(destination_root):
            raise ArtifactServiceError("artifact destination escapes local artifact directory")
        destination = resolved_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.urandom(6).hex()}")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in self.context.client_for(workspace.server_id).artifact_stream(
                    workspace_id, artifact_path
                ):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "workspace_id": workspace_id,
            "server_id": workspace.server_id,
            "artifact_path": artifact_path,
            "local_path": str(destination),
            "size": size,
            "sha256": digest.hexdigest(),
        }
