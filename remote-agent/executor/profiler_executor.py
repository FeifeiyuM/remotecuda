"""Controlled Nsight Compute and Nsight Systems profiling."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from security import EnvironmentPolicy, atomic_write

from .cuda_executor import CudaExecutor
from .logging_utils import log_event
from .process_runner import ProcessFailure, ProcessResult, ProcessRunner


LOGGER = logging.getLogger("remote-cuda-agent.executor.profiler")


class ProfilerExecutor:
    def __init__(
        self,
        runner: ProcessRunner,
        ncu_path: str | Path | None,
        nsys_path: str | Path | None,
    ) -> None:
        self.runner = runner
        self.ncu_path = Path(ncu_path) if ncu_path is not None else None
        self.nsys_path = Path(nsys_path) if nsys_path is not None else None
        self.environment_policy = EnvironmentPolicy()

    @staticmethod
    def require_tool(path: str | Path | None, name: str) -> Path:
        if path is None:
            raise ProcessFailure(f"required profiler is not configured: {name}")
        resolved = Path(path)
        if not resolved.is_absolute() or not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ProcessFailure(f"required profiler is unavailable: {name}")
        return resolved

    @staticmethod
    def _safe_name(value: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        if not name:
            raise ProcessFailure("invalid profiler name")
        return name

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_ncu_command(
        self,
        *,
        tool: Path,
        executable: Path,
        args: list[str],
        report_path: Path,
        config: dict[str, Any],
    ) -> list[str]:
        options = config.get("ncu") or {}
        command = [
            str(tool),
            "--target-processes",
            str(options.get("target_processes", "all")),
            "--set",
            str(options.get("metric_set", "detailed")),
            "--force-overwrite",
            "--export",
            str(report_path),
        ]
        kernel_name = options.get("kernel_name")
        if kernel_name is not None:
            command.extend(["--kernel-name", str(kernel_name)])
        if options.get("launch_count") is not None:
            command.extend(["--launch-count", str(options["launch_count"])])
        if options.get("launch_skip") is not None:
            command.extend(["--launch-skip", str(options["launch_skip"])])
        return [*command, str(executable), *args]

    def _build_nsys_command(
        self,
        *,
        tool: Path,
        executable: Path,
        args: list[str],
        report_path: Path,
        config: dict[str, Any],
    ) -> list[str]:
        options = config.get("nsys") or {}
        traces = options.get("trace") or ["cuda", "nvtx", "osrt"]
        command = [
            str(tool),
            "profile",
            "--output",
            str(report_path),
            "--force-overwrite",
            "true",
            "--trace",
            ",".join(str(value) for value in traces),
            # Nsight Systems profiles the launched process tree by default.
            # --target-processes is an Nsight Compute option and is rejected by
            # Nsight Systems 2026.1 on Linux.
            "--sample",
            "cpu" if options.get("sample_cpu", True) else "none",
        ]
        if options.get("duration_seconds") is not None:
            command.extend(["--duration", str(options["duration_seconds"])])
        if options.get("delay_seconds") is not None:
            command.extend(["--delay", str(options["delay_seconds"])])
        if options.get("cuda_memory_usage"):
            command.append("--cuda-memory-usage")
        return [*command, "--", str(executable), *args]

    @staticmethod
    def _resolve_report(primary: Path, profiler: str) -> Path:
        candidates = [primary]
        if profiler == "nsys" and primary.suffix != ".nsys-rep":
            candidates.append(Path(f"{primary}.nsys-rep"))
        if profiler == "ncu" and primary.suffix != ".ncu-rep":
            candidates.append(Path(f"{primary}.ncu-rep"))
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        raise ProcessFailure(f"profiler did not create a report: {primary}")

    async def profile(
        self,
        *,
        job_id: str,
        workspace_root: Path,
        config: dict[str, Any],
        timeout_seconds: float,
        log_path: Path,
    ) -> dict[str, Any]:
        started = time.monotonic()
        profiler = str(config.get("profiler", "")).lower()
        profile_name = self._safe_name(str(config.get("profile_name", "profile")))
        custom_env = self.environment_policy.validate(config.get("env") or {})
        environment = {**custom_env, "CUDA_VISIBLE_DEVICES": str(config.get("device_id", 0))}
        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "profiler.start",
            job_id=job_id,
            profiler=profiler,
            profile_name=profile_name,
            workspace_root=workspace_root,
            executable=config.get("executable"),
            timeout_seconds=timeout_seconds,
            environment_keys=sorted(custom_env),
        )
        try:
            if profiler not in {"ncu", "nsys"}:
                raise ProcessFailure("profiler must be ncu or nsys")
            tool = self.require_tool(self.ncu_path if profiler == "ncu" else self.nsys_path, profiler)
            executable = CudaExecutor.executable_path(
                workspace_root,
                str(config.get("executable", "")),
            )
            args = [str(value) for value in config.get("args", [])]
            if any("\x00" in value for value in args):
                raise ProcessFailure("invalid executable argument")

            artifact_root = workspace_root / "artifacts" / "profiler"
            artifact_root.mkdir(parents=True, exist_ok=True)
            base = artifact_root / f"{profile_name}-{job_id}"
            extension = ".ncu-rep" if profiler == "ncu" else ".nsys-rep"
            report_path = Path(f"{base}{extension}")
            if profiler == "ncu":
                command = self._build_ncu_command(
                    tool=tool,
                    executable=executable,
                    args=args,
                    report_path=report_path,
                    config=config,
                )
            else:
                command = self._build_nsys_command(
                    tool=tool,
                    executable=executable,
                    args=args,
                    report_path=report_path,
                    config=config,
                )

            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "profiler.process_start",
                job_id=job_id,
                profiler=profiler,
                tool_path=tool,
                executable=executable,
                report_path=report_path,
            )
            result: ProcessResult = await self.runner.run(
                job_id=f"{job_id}:profile",
                command=command,
                cwd=workspace_root / "build",
                timeout_seconds=timeout_seconds,
                env=environment,
                log_path=log_path,
            )
            report = self._resolve_report(report_path, profiler)
            metadata_path = Path(f"{base}.json")
            download_root = workspace_root / "artifacts"
            report_artifact_path = report.relative_to(download_root).as_posix()
            result_artifact_path = metadata_path.relative_to(download_root).as_posix()
            metadata = {
                "job_id": job_id,
                "profiler": profiler,
                "profile_name": profile_name,
                "tool_path": str(tool),
                "executable": str(executable),
                "args": args,
                "device_id": int(config.get("device_id", 0)),
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "report": {
                    "path": report_artifact_path,
                    "size": report.stat().st_size,
                    "sha256": self._sha256(report),
                },
            }
            atomic_write(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))
            payload = {
                **metadata,
                "artifact_path": report_artifact_path,
                "profiler_result_path": result_artifact_path,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "profiler.complete",
                job_id=job_id,
                profiler=profiler,
                artifact_path=payload["artifact_path"],
                profiler_result_path=payload["profiler_result_path"],
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
                total_duration_seconds=time.monotonic() - started,
            )
            return payload
        except BaseException as exc:
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "profiler.failed",
                job_id=job_id,
                profiler=profiler,
                profile_name=profile_name,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=time.monotonic() - started,
            )
            raise
