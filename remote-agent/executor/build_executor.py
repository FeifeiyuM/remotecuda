"""CMake-only build executor."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .logging_utils import log_event
from .process_runner import ProcessFailure, ProcessResult, ProcessRunner


LOGGER = logging.getLogger("remote-cuda-agent.executor.build")


class BuildExecutor:
    def __init__(
        self,
        runner: ProcessRunner,
        max_parallel: int,
        cmake_path: str | Path,
        nvcc_path: str | Path,
    ) -> None:
        self.runner = runner
        self.max_parallel = max_parallel
        self.cmake_path = Path(cmake_path)
        self.nvcc_path = Path(nvcc_path)

    @staticmethod
    def require_tool(path: str | Path, name: str) -> str:
        resolved = Path(path)
        if not resolved.is_absolute() or not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ProcessFailure(f"required tool is unavailable: {name}")
        return str(resolved)

    def build_dir(self, workspace_root: Path, build_type: str) -> Path:
        return workspace_root / "build" / build_type.lower()

    async def execute(
        self,
        *,
        job_id: str,
        workspace_root: Path,
        config: dict[str, Any],
        timeout_seconds: int,
        log_path: Path,
    ) -> dict[str, Any]:
        started = time.monotonic()
        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "build.start",
            job_id=job_id,
            workspace_root=workspace_root,
            build_system=config.get("build_system", "cmake"),
            timeout_seconds=timeout_seconds,
        )
        try:
            if config.get("build_system") != "cmake":
                raise ProcessFailure("only the CMake build system is supported")
            build_type = config.get("build_type", "Release")
            if build_type not in {"Debug", "Release", "RelWithDebInfo", "MinSizeRel"}:
                raise ProcessFailure("invalid CMAKE_BUILD_TYPE")
            parallel = min(int(config.get("parallel_jobs", 2)), self.max_parallel)
            if parallel < 1:
                raise ProcessFailure("parallel_jobs must be positive")
            cuda_arch = config.get("cuda_arch")
            if cuda_arch is not None and not re.fullmatch(r"[0-9]{2,3}", str(cuda_arch)):
                raise ProcessFailure("invalid CUDA architecture")

            source = workspace_root / "source"
            if not (source / "CMakeLists.txt").is_file():
                raise ProcessFailure("source/CMakeLists.txt not found")
            build_base = (workspace_root / "build").resolve(strict=False)
            workspace_root = workspace_root.resolve(strict=True)
            if not build_base.is_relative_to(workspace_root):
                raise ProcessFailure("build directory escapes workspace")
            build = self.build_dir(workspace_root, build_type)
            if build.exists() and not build.resolve(strict=True).is_relative_to(build_base):
                raise ProcessFailure("build directory escapes workspace")
            if config.get("clean") and build.exists():
                log_event(
                    log_path,
                    LOGGER,
                    logging.INFO,
                    "build.clean_start",
                    job_id=job_id,
                    build_dir=build,
                )
                shutil.rmtree(build)
                log_event(
                    log_path,
                    LOGGER,
                    logging.INFO,
                    "build.clean_complete",
                    job_id=job_id,
                    build_dir=build,
                )
            build.mkdir(parents=True, exist_ok=True)

            cmake = self.require_tool(self.cmake_path, "cmake")
            nvcc = self.require_tool(self.nvcc_path, "nvcc")
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "build.tools_resolved",
                job_id=job_id,
                cmake_path=cmake,
                nvcc_path=nvcc,
            )
            configure_command = [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}",
            ]
            if cuda_arch:
                configure_command.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "build.configure_start",
                job_id=job_id,
                source_dir=source,
                build_dir=build,
                build_type=build_type,
                cuda_arch=cuda_arch,
                nvcc_path=nvcc,
            )
            configure: ProcessResult = await self.runner.run(
                job_id=f"{job_id}:configure",
                command=configure_command,
                cwd=workspace_root,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                env={"CUDACXX": nvcc},
            )
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "build.configure_complete",
                job_id=job_id,
                exit_code=configure.exit_code,
                duration_seconds=configure.duration_seconds,
            )

            build_command = [cmake, "--build", str(build), "--parallel", str(parallel)]
            target = config.get("target")
            if target:
                if not re.fullmatch(r"[A-Za-z0-9_.+-]+", str(target)):
                    raise ProcessFailure("invalid CMake target")
                build_command.extend(["--target", target])
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "build.compile_start",
                job_id=job_id,
                build_dir=build,
                target=target,
                parallel_jobs=parallel,
            )
            built = await self.runner.run(
                job_id=f"{job_id}:build",
                command=build_command,
                cwd=workspace_root,
                timeout_seconds=max(1, timeout_seconds - int(configure.duration_seconds)),
                log_path=log_path,
            )
            result = {
                "build_type": build_type,
                "build_dir": str(build),
                "parallel_jobs": parallel,
                "configure_exit_code": configure.exit_code,
                "build_exit_code": built.exit_code,
                "duration_seconds": configure.duration_seconds + built.duration_seconds,
            }
            log_event(
                log_path,
                LOGGER,
                logging.INFO,
                "build.complete",
                job_id=job_id,
                duration_seconds=time.monotonic() - started,
                result=result,
            )
            return result
        except BaseException as exc:
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "build.failed",
                job_id=job_id,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=time.monotonic() - started,
            )
            raise
