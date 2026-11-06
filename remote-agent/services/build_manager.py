"""Build API service."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .job_manager import JobManager
from .workspace_manager import WorkspaceManager


class BuildManager:
    def __init__(self, workspaces: WorkspaceManager, jobs: JobManager) -> None:
        self.workspaces = workspaces
        self.jobs = jobs

    async def submit(self, workspace_id: str, config: dict[str, Any], idempotency_key: str | None) -> dict[str, Any]:
        return await self.jobs.submit(
            workspace_id=workspace_id,
            job_type="BUILD",
            config=config,
            idempotency_key=idempotency_key,
        )

    def clean(self, workspace_id: str) -> dict[str, Any]:
        metadata = self.workspaces.metadata(workspace_id)
        if self.jobs.database.active_jobs(workspace_id):
            raise RuntimeError("workspace has active build jobs")
        build_root = self.workspaces.workspace_root(workspace_id) / "build"
        if build_root.exists():
            shutil.rmtree(build_root)
        build_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata.status = "READY"
        self.workspaces.update_status(workspace_id, metadata.status)
        return {"workspace_id": workspace_id, "cleaned": True, "build_path": str(build_root)}
