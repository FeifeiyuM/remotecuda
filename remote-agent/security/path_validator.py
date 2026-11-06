"""Filesystem boundary validation utilities."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class PathSecurityError(ValueError):
    pass


def safe_relative_path(path: str) -> PurePosixPath:
    if not path or "\x00" in path:
        raise PathSecurityError("empty or null-byte path")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise PathSecurityError(f"unsafe relative path: {path}")
    return candidate


def resolve_inside(base: Path, relative_path: str | PurePosixPath, *, must_exist: bool = False) -> Path:
    root = base.expanduser().resolve(strict=True)
    relative = safe_relative_path(str(relative_path))
    target = root.joinpath(*relative.parts)
    # Existing parent symlinks must resolve inside the same root. This avoids
    # TOCTOU-prone comparisons against unrelated filesystem ancestors.
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() and not current.resolve(strict=True).is_relative_to(root):
            raise PathSecurityError(f"path escapes workspace: {relative}")
    if target.exists() and target.is_symlink() and not target.resolve(strict=True).is_relative_to(root):
        raise PathSecurityError(f"path escapes workspace: {relative}")
    if must_exist:
        resolved = target.resolve(strict=True)
    else:
        resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise PathSecurityError(f"path escapes workspace: {relative}")
    return resolved


def atomic_write(target: Path, content: bytes, *, mode: int = 0o600) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.urandom(6).hex()}")
    temporary.write_bytes(content)
    temporary.chmod(mode)
    temporary.replace(target)

