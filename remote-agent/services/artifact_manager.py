"""Artifact discovery, collection, and safe downloads."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from security import resolve_inside
from .workspace_manager import WorkspaceManager


class ArtifactManager:
    def __init__(self, workspaces: WorkspaceManager, max_artifact_bytes: int) -> None:
        self.workspaces = workspaces
        self.max_artifact_bytes = max_artifact_bytes

    def artifact_root(self, workspace_id: str) -> Path:
        self.workspaces.metadata(workspace_id)
        return self.workspaces.workspace_root(workspace_id) / "artifacts"

    def list(self, workspace_id: str) -> list[dict[str, Any]]:
        root = self.artifact_root(workspace_id)
        artifacts: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "downloadable": path.stat().st_size <= self.max_artifact_bytes,
                    "sha256": self._sha256(path),
                }
            )
        return artifacts

    def download_path(self, workspace_id: str, relative_path: str) -> Path:
        root = self.artifact_root(workspace_id)
        target = resolve_inside(root, self._canonical_path(relative_path), must_exist=True)
        if not target.is_file():
            raise ValueError("artifact is not a regular file")
        if target.stat().st_size > self.max_artifact_bytes:
            raise ValueError("artifact exceeds download size limit")
        return target

    @staticmethod
    def _canonical_path(relative_path: str) -> str:
        """Normalize paths emitted by older jobs that included ``artifacts/``."""

        path = PurePosixPath(relative_path)
        parts = path.parts
        if len(parts) > 1 and parts[0] == "artifacts":
            path = PurePosixPath(*parts[1:])
        return path.as_posix()

    def collect_binaries(self, workspace_id: str, build_root: Path) -> list[str]:
        artifact_root = self.artifact_root(workspace_id) / "binaries"
        artifact_root.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []
        build_root = build_root.resolve(strict=True)
        for path in build_root.rglob("*"):
            if not path.is_file() or path.is_symlink() or not path.stat().st_mode & 0o111:
                continue
            if not path.resolve(strict=True).is_relative_to(build_root):
                continue
            destination = artifact_root / path.name
            if destination.exists():
                destination.unlink()
            shutil.copy2(path, destination)
            collected.append(destination.relative_to(self.artifact_root(workspace_id)).as_posix())
        return sorted(set(collected))

    def collect_log(self, workspace_id: str, job_id: str, source: Path) -> str | None:
        """Copy a completed job log into the downloadable artifact tree."""

        if not source.is_file() or source.is_symlink():
            return None
        root = self.artifact_root(workspace_id)
        target = root / "logs" / f"{job_id}.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target.relative_to(root).as_posix()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def read_json_result(self, workspace_id: str, relative_path: str) -> dict[str, Any]:
        import json

        path = self.download_path(workspace_id, relative_path)
        return json.loads(path.read_text(encoding="utf-8"))
