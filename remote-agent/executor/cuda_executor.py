"""Controlled CUDA executable execution and benchmarking."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from security import atomic_write, resolve_inside

from .logging_utils import log_event
from .process_runner import ProcessFailure, ProcessResult, ProcessRunner, ProcessTimeout


LOGGER = logging.getLogger("remote-cuda-agent.executor.cuda")


class GpuReader(Protocol):
    async def gpu(self, device_id: int) -> dict[str, Any]:
        """Read one device snapshot."""


class CudaExecutor:
    def __init__(self, runner: ProcessRunner, gpu_manager: GpuReader | None = None) -> None:
        self.runner = runner
        self.gpu_manager = gpu_manager

    @staticmethod
    def executable_path(workspace_root: Path, relative_executable: str) -> Path:
        if not relative_executable or "\x00" in relative_executable:
            raise ProcessFailure("invalid executable path")
        candidate = Path(relative_executable)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ProcessFailure("executable must be relative to build directory")
        workspace_root = workspace_root.resolve(strict=True)
        build_root = (workspace_root / "build").resolve(strict=True)
        if not build_root.is_relative_to(workspace_root):
            raise ProcessFailure("build directory escapes workspace")
        target = resolve_inside(build_root, relative_executable, must_exist=True)
        if not target.is_file() or not os.access(target, os.X_OK):
            raise ProcessFailure(f"executable is missing or not executable: {relative_executable}")
        return target

    async def execute(
        self,
        *,
        job_id: str,
        workspace_root: Path,
        config: dict[str, Any],
        timeout_seconds: float,
        log_path: Path,
    ) -> dict[str, Any]:
        started = time.monotonic()
        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "cuda.run_start",
            job_id=job_id,
            workspace_root=workspace_root,
            executable=config.get("executable"),
            args=config.get("args", []),
            device_id=config.get("device_id", 0),
            timeout_seconds=timeout_seconds,
        )
        try:
            executable = self.executable_path(workspace_root, str(config.get("executable", "")))
            args = [str(value) for value in config.get("args", [])]
            if any("\x00" in value for value in args):
                raise ProcessFailure("invalid executable argument")
            device_id = int(config.get("device_id", 0))
            result = await self.runner.run(
                job_id=job_id,
                command=[str(executable), *args],
                cwd=workspace_root / "build",
                timeout_seconds=timeout_seconds,
                env={"CUDA_VISIBLE_DEVICES": str(device_id)},
                log_path=log_path,
            )
            payload = {
                "executable": str(executable),
                "args": args,
                "device_id": device_id,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "cuda.run_complete",
                job_id=job_id,
                executable=executable,
                device_id=device_id,
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
                stdout_chars=len(result.stdout),
                stderr_chars=len(result.stderr),
                total_duration_seconds=time.monotonic() - started,
            )
            return payload
        except BaseException as exc:
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "cuda.run_failed",
                job_id=job_id,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=time.monotonic() - started,
            )
            raise

    async def _run_once(
        self,
        *,
        job_id: str,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        device_id: int,
        log_path: Path,
    ) -> tuple[ProcessResult, list[dict[str, Any]]]:
        samples: list[dict[str, Any]] = []
        stop = asyncio.Event()

        async def sample_gpu() -> None:
            if self.gpu_manager is None:
                return
            while not stop.is_set():
                try:
                    samples.append(await self.gpu_manager.gpu(device_id))
                except Exception as exc:
                    # GPU telemetry is diagnostic and must never be fabricated.
                    log_event(
                        log_path,
                        LOGGER,
                        logging.WARNING,
                        "cuda.gpu_monitor_stopped",
                        job_id=job_id,
                        device_id=device_id,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        sample_count=len(samples),
                    )
                    return
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.2)
                except TimeoutError:
                    pass

        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "cuda.sample_monitor_start",
            job_id=job_id,
            device_id=device_id,
            telemetry_enabled=self.gpu_manager is not None,
        )
        monitor = asyncio.create_task(sample_gpu())
        try:
            result = await self.runner.run(
                job_id=job_id,
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                env={"CUDA_VISIBLE_DEVICES": str(device_id)},
                log_path=log_path,
            )
            return result, samples
        finally:
            stop.set()
            await asyncio.gather(monitor, return_exceptions=True)
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "cuda.sample_monitor_complete",
                job_id=job_id,
                device_id=device_id,
                sample_count=len(samples),
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        low, high = math.floor(index), math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - index) + ordered[high] * (index - low)

    async def benchmark(
        self,
        *,
        job_id: str,
        workspace_root: Path,
        config: dict[str, Any],
        timeout_seconds: float,
        log_path: Path,
    ) -> dict[str, Any]:
        started = time.monotonic()
        warmup = int(config.get("warmup_iterations", 0))
        iterations = int(config.get("iterations", 1))
        benchmark_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "-",
            str(config.get("benchmark_name", "benchmark")),
        )
        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "cuda.benchmark_start",
            job_id=job_id,
            benchmark_name=benchmark_name,
            workspace_root=workspace_root,
            executable=config.get("executable"),
            iterations=iterations,
            warmup_iterations=warmup,
            device_id=config.get("device_id", 0),
            timeout_seconds=timeout_seconds,
        )
        try:
            executable = self.executable_path(workspace_root, str(config.get("executable", "")))
            args = [str(value) for value in config.get("args", [])]
            if any("\x00" in value for value in args):
                raise ProcessFailure("invalid executable argument")
            device_id = int(config.get("device_id", 0))
            durations: list[float] = []
            gpu_utilization: list[int] = []
            gpu_memory_used: list[int] = []
            outputs: list[str] = []

            async def one_iteration(label: str, sequence: int) -> None:
                iteration_started = time.monotonic()
                log_event(
                    log_path,
                    LOGGER,
                    logging.INFO,
                    "cuda.benchmark_iteration_start",
                    job_id=job_id,
                    label=label,
                    sequence=sequence,
                )
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProcessTimeout("benchmark exceeded total timeout")
                result, samples = await self._run_once(
                    job_id=f"{job_id}:{label}",
                    command=[str(executable), *args],
                    cwd=workspace_root / "build",
                    timeout_seconds=remaining,
                    device_id=device_id,
                    log_path=log_path,
                )
                if label != "warmup":
                    durations.append(result.duration_seconds * 1000)
                gpu_utilization.extend(int(sample.get("utilization", 0)) for sample in samples)
                gpu_memory_used.extend(int(sample.get("memory_used_mb", 0)) for sample in samples)
                outputs.append(result.stdout)
                log_event(
                    log_path,
                    LOGGER,
                    logging.INFO,
                    "cuda.benchmark_iteration_complete",
                    job_id=job_id,
                    label=label,
                    sequence=sequence,
                    exit_code=result.exit_code,
                    duration_seconds=result.duration_seconds,
                    gpu_samples=len(samples),
                    total_iteration_seconds=time.monotonic() - iteration_started,
                )

            for sequence in range(warmup):
                await one_iteration("warmup", sequence + 1)
            for sequence in range(iterations):
                await one_iteration("iteration", sequence + 1)

            mean = sum(durations) / len(durations)
            metrics: dict[str, float | int] = {
                "mean_ms": mean,
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": self._percentile(durations, 0.50),
                "p90_ms": self._percentile(durations, 0.90),
                "p95_ms": self._percentile(durations, 0.95),
                "throughput_ops_per_second": 1000 / mean if mean else 0,
            }
            if gpu_utilization:
                metrics["gpu_utilization_max_percent"] = max(gpu_utilization)
            if gpu_memory_used:
                metrics["gpu_memory_used_max_mb"] = max(gpu_memory_used)
            payload = {
                "job_id": job_id,
                "benchmark_name": benchmark_name,
                "device_id": device_id,
                "iterations": iterations,
                "warmup_iterations": warmup,
                "metrics": metrics,
                "measured_iteration_ms": durations,
                "gpu_sample_count": len(gpu_utilization),
            }
            artifact = workspace_root / "artifacts" / "benchmark" / f"{benchmark_name}-{job_id}.json"
            atomic_write(artifact, json.dumps(payload, indent=2).encode("utf-8"))
            payload["artifact_path"] = artifact.relative_to(workspace_root / "artifacts").as_posix()
            payload["stdout"] = "\n".join(outputs)
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "cuda.benchmark_complete",
                job_id=job_id,
                benchmark_name=benchmark_name,
                artifact_path=payload["artifact_path"],
                iterations=iterations,
                warmup_iterations=warmup,
                gpu_sample_count=len(gpu_utilization),
                duration_seconds=time.monotonic() - started,
            )
            return payload
        except BaseException as exc:
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "cuda.benchmark_failed",
                job_id=job_id,
                benchmark_name=benchmark_name,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=time.monotonic() - started,
            )
            raise
