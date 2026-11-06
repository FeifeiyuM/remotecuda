"""Small path-safety helpers shared by local tools."""

from __future__ import annotations

from pathlib import PurePosixPath


def is_safe_relative_path(path: str) -> bool:
    """Validate a portable, workspace-relative path without touching the filesystem."""

    if not path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return False
    return path == candidate.as_posix()
