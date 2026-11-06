"""Wire models shared by the local gateway."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RemoteErrorPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    error: dict[str, object] | None = None
    detail: object | None = None


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=1024**3)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        from security_paths import is_safe_relative_path

        if not is_safe_relative_path(value):
            raise ValueError("unsafe file path")
        return value


class SyncManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    files: list[ManifestFile] = Field(default_factory=list)


class SyncPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    workspace_id: str
    upload: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


class FileUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    path: str
    content_base64: str
    sha256: str = Field(min_length=64, max_length=64)


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_system: str = Field(default="cmake", pattern="^cmake$")
    build_type: str = Field(default="Release", pattern="^(Debug|Release|RelWithDebInfo|MinSizeRel)$")
    cuda_arch: str | None = Field(default=None, pattern=r"^[0-9]{2,3}$")
    parallel_jobs: int = Field(default=2, ge=1, le=256)
    target: str | None = Field(default=None, max_length=200, pattern=r"^[A-Za-z0-9_.+-]+$")
    clean: bool = False


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str = Field(min_length=1, max_length=512)
    args: list[str] = Field(default_factory=list, max_length=128)
    device_id: int = Field(default=0, ge=0, le=1023)
    timeout_seconds: int = Field(default=300, ge=1, le=86_400)
    env: dict[str, str] = Field(default_factory=dict, max_length=32)

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        if Path(value).is_absolute() or ".." in PurePosixPath(value).parts:
            raise ValueError("executable must be relative to the workspace build directory")
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        from security_paths import validate_environment_entries

        validate_environment_entries(value)
        return value


class BenchmarkConfig(ExecutionConfig):
    benchmark_name: str = Field(default="benchmark", min_length=1, max_length=200)
    iterations: int = Field(default=1, ge=1, le=10_000)
    warmup_iterations: int = Field(default=0, ge=0, le=10_000)


class NcuOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_set: str = Field(default="detailed", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    kernel_name: str | None = Field(default=None, min_length=1, max_length=512)
    launch_count: int | None = Field(default=None, ge=1, le=1_000_000)
    launch_skip: int | None = Field(default=None, ge=0, le=1_000_000)
    target_processes: str = Field(default="all", pattern=r"^(all|application|none)$")

    @field_validator("kernel_name")
    @classmethod
    def validate_kernel_name(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("kernel_name must be non-empty text")
        return value


class NsysOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: list[str] = Field(default_factory=lambda: ["cuda", "nvtx", "osrt"], max_length=16)
    sample_cpu: bool = True
    duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    delay_seconds: float | None = Field(default=None, ge=0, le=86_400)
    cuda_memory_usage: bool = False

    @field_validator("trace")
    @classmethod
    def validate_trace(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in value))
        if not normalized or any(
            not item or not item.isascii() or not item.replace("_", "").isalnum()
            for item in normalized
        ):
            raise ValueError("trace entries must be simple Nsight Systems feature names")
        return normalized


class ProfilerConfig(ExecutionConfig):
    profiler: str = Field(pattern=r"^(ncu|nsys)$")
    profile_name: str = Field(default="profile", min_length=1, max_length=200)
    ncu: NcuOptions | None = None
    nsys: NsysOptions | None = None

    @model_validator(mode="after")
    def validate_profiler_options(self) -> "ProfilerConfig":
        if self.profiler == "ncu" and self.nsys is not None:
            raise ValueError("nsys options require profiler=nsys")
        if self.profiler == "nsys" and self.ncu is not None:
            raise ValueError("ncu options require profiler=ncu")
        return self


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    size: int
    sha256: str | None = None
