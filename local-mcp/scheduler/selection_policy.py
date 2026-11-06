"""Pure selection policies for Remote CUDA servers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScheduleRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_name: str | None = None
    min_memory_mb: int = Field(default=0, ge=0)
    cuda_version: str | None = None
    tags: list[str] = Field(default_factory=list)


class ServerCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    server_id: str
    enabled: bool = True
    priority: int = 10
    tags: list[str] = Field(default_factory=list)
    capabilities: dict[str, object] = Field(default_factory=dict)
    gpus: list[dict[str, object]] = Field(default_factory=list)
    active_jobs: int = 0
    healthy: bool = True
    connected: bool = True


def filter_candidates(candidates: list[ServerCandidate], requirements: ScheduleRequirements) -> list[ServerCandidate]:
    required_tags = set(requirements.tags)
    result: list[ServerCandidate] = []
    for candidate in candidates:
        if not candidate.enabled or not candidate.healthy or not candidate.connected:
            continue
        if not required_tags.issubset(set(candidate.tags)):
            continue
        gpus = candidate.gpus or []
        if requirements.gpu_name:
            gpus = [gpu for gpu in gpus if requirements.gpu_name.lower() in str(gpu.get("name", "")).lower()]
        if requirements.min_memory_mb:
            gpus = [
                gpu
                for gpu in gpus
                if int(gpu.get("memory_free_mb", gpu.get("memory_total_mb", 0)) or 0)
                >= requirements.min_memory_mb
            ]
        if requirements.cuda_version:
            actual = str(candidate.capabilities.get("cuda_version", ""))
            if actual and requirements.cuda_version not in actual:
                continue
        if requirements.gpu_name or requirements.min_memory_mb:
            candidate = candidate.model_copy(update={"gpus": gpus})
        if gpus or not (requirements.gpu_name or requirements.min_memory_mb):
            result.append(candidate)
    return result


def sort_by_policy(candidates: list[ServerCandidate], policy: str) -> list[ServerCandidate]:
    if policy == "least_loaded":
        return sorted(candidates, key=lambda item: (item.active_jobs, item.priority, item.server_id))
    if policy == "most_free_memory":
        return sorted(
            candidates,
            key=lambda item: (
                -max((int(gpu.get("memory_free_mb", 0) or 0) for gpu in item.gpus), default=0),
                item.active_jobs,
                item.priority,
            ),
        )
    if policy == "gpu_model_match":
        return sorted(candidates, key=lambda item: (-len(item.gpus), item.active_jobs, item.priority))
    if policy == "priority":
        return sorted(candidates, key=lambda item: (item.priority, item.active_jobs, item.server_id))
    raise ValueError(f"unsupported selection policy: {policy}")
