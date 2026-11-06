"""Small path-safety helpers shared by local tools."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
REQUEST_ENV_PREFIX = "RC_ENV_"
MAX_ENV_VARS = 32
MAX_ENV_NAME_CHARS = 128
MAX_ENV_VALUE_CHARS = 4_096
MAX_ENV_TOTAL_CHARS = 32_768
RESERVED_ENV_NAMES = frozenset(
    {
        "BASH_ENV",
        "CPATH",
        "C_INCLUDE_PATH",
        "CUDA_CACHE_PATH",
        "CUDA_DEVICE_ORDER",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDACXX",
        "CPLUS_INCLUDE_PATH",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GLIBC_TUNABLES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LIBRARY_PATH",
        "LD_AUDIT",
        "LD_DEBUG",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NVCC_CCBIN",
        "NVIDIA_VISIBLE_DEVICES",
        "OLDPWD",
        "PATH",
        "PERL5LIB",
        "PERL5OPT",
        "PERLLIB",
        "PWD",
        "PYTHONHOME",
        "PYTHONPATH",
        "RUBYOPT",
        "SHELL",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)
RESERVED_ENV_PREFIXES = ("REMOTE_AGENT_", "REMOTE_CUDA_", "LD_", "DYLD_")


def is_safe_relative_path(path: str) -> bool:
    """Validate a portable, workspace-relative path without touching the filesystem."""

    if not path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return False
    return path == candidate.as_posix()


def validate_environment_entries(entries: dict[str, str]) -> dict[str, str]:
    """Validate prefixed request entries before sending them remotely."""

    if len(entries) > MAX_ENV_VARS:
        raise ValueError(f"at most {MAX_ENV_VARS} environment variables are allowed")

    validated: dict[str, str] = {}
    total_chars = 0
    for name, value in entries.items():
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if len(name) > MAX_ENV_NAME_CHARS:
            raise ValueError("environment variable name is too long")
        if not name.startswith(REQUEST_ENV_PREFIX) or name == REQUEST_ENV_PREFIX:
            raise ValueError(
                f"request environment variable must use the {REQUEST_ENV_PREFIX} prefix"
            )
        child_name = name.removeprefix(REQUEST_ENV_PREFIX)
        if child_name in RESERVED_ENV_NAMES or child_name.startswith(RESERVED_ENV_PREFIXES):
            raise ValueError(f"reserved environment variable cannot be overridden: {child_name}")
        if not isinstance(value, str):
            raise ValueError(f"environment variable value must be text: {name}")
        if "\x00" in value:
            raise ValueError(f"environment variable value contains NUL: {name}")
        if len(value) > MAX_ENV_VALUE_CHARS:
            raise ValueError(f"environment variable value is too long: {name}")
        if child_name in validated:
            raise ValueError(f"duplicate environment variable after prefix removal: {child_name}")
        validated[child_name] = value
        total_chars += len(child_name) + len(value)
    if total_chars > MAX_ENV_TOTAL_CHARS:
        raise ValueError("environment variables exceed the total size limit")
    return validated
