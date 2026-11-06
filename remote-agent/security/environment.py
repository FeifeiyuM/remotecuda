"""Validation for prefixed, request-scoped environment variables."""

from __future__ import annotations

import re
from collections.abc import Mapping


ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
REQUEST_ENV_PREFIX = "RC_ENV_"
MAX_ENV_VARS = 32
MAX_ENV_NAME_CHARS = 128
MAX_ENV_VALUE_CHARS = 4_096
MAX_ENV_TOTAL_CHARS = 32_768

# These target names control process loading, interpreter behavior, service
# secrets, or GPU/device selection. They are never request-settable.
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


class EnvironmentPolicyError(ValueError):
    pass


def validate_child_environment_name(name: str) -> str:
    if not isinstance(name, str) or not ENV_NAME_PATTERN.fullmatch(name):
        raise EnvironmentPolicyError(f"invalid environment variable name: {name!r}")
    if name in RESERVED_ENV_NAMES or name.startswith(RESERVED_ENV_PREFIXES):
        raise EnvironmentPolicyError(f"reserved environment variable cannot be overridden: {name}")
    return name


def request_environment_name(name: str) -> str:
    """Validate an RC_ENV_-prefixed request name and return its child name."""

    if not isinstance(name, str) or not ENV_NAME_PATTERN.fullmatch(name):
        raise EnvironmentPolicyError(f"invalid environment variable name: {name!r}")
    if len(name) > MAX_ENV_NAME_CHARS:
        raise EnvironmentPolicyError("environment variable name is too long")
    if not name.startswith(REQUEST_ENV_PREFIX) or name == REQUEST_ENV_PREFIX:
        raise EnvironmentPolicyError(
            f"request environment variable must use the {REQUEST_ENV_PREFIX} prefix"
        )
    return validate_child_environment_name(name.removeprefix(REQUEST_ENV_PREFIX))


def validate_environment_entries(entries: Mapping[str, str]) -> dict[str, str]:
    """Return child-process names after validating prefixed request entries."""

    if len(entries) > MAX_ENV_VARS:
        raise EnvironmentPolicyError(f"at most {MAX_ENV_VARS} environment variables are allowed")

    validated: dict[str, str] = {}
    total_chars = 0
    for request_name, value in entries.items():
        child_name = request_environment_name(request_name)
        if not isinstance(value, str):
            raise EnvironmentPolicyError(f"environment variable value must be text: {request_name}")
        if "\x00" in value:
            raise EnvironmentPolicyError(f"environment variable value contains NUL: {request_name}")
        if len(value) > MAX_ENV_VALUE_CHARS:
            raise EnvironmentPolicyError(f"environment variable value is too long: {request_name}")
        if child_name in validated:
            raise EnvironmentPolicyError(
                f"duplicate environment variable after prefix removal: {child_name}"
            )
        validated[child_name] = value
        total_chars += len(child_name) + len(value)

    if total_chars > MAX_ENV_TOTAL_CHARS:
        raise EnvironmentPolicyError("environment variables exceed the total size limit")
    return validated


class EnvironmentPolicy:
    """Validate request names and return a child-process environment mapping."""

    def validate(self, entries: Mapping[str, str] | None) -> dict[str, str]:
        return validate_environment_entries(entries or {})
