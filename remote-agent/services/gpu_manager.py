"""Safe nvidia-smi and CUDA toolkit discovery."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class GPUManager:
    GPU_FIELDS = (
        "index",
        "name",
        "driver_version",
        "pci.bus_id",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    )

    def __init__(
        self,
        cmake_path: str | Path,
        nvcc_path: str | Path,
        ncu_path: str | Path | None = None,
        nsys_path: str | Path | None = None,
    ) -> None:
        self.cmake_path = Path(cmake_path)
        self.nvcc_path = Path(nvcc_path)
        self.ncu_path = Path(ncu_path) if ncu_path is not None else None
        self.nsys_path = Path(nsys_path) if nsys_path is not None else None

    async def _run(self, command: list[str], timeout: float = 10) -> tuple[int, str, str] | None:
        executable = os.path.isabs(command[0]) and command[0] or shutil.which(command[0])
        if not executable:
            return None

        try:
            completed = subprocess.run(
                [executable, *command[1:]],
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("GPU query timed out")
        return (
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )

    async def list_gpus(self) -> list[dict[str, Any]]:
        query = ",".join(self.GPU_FIELDS)
        result = await self._run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
        if result is None:
            return []
        code, stdout, stderr = result
        if code != 0:
            raise RuntimeError(f"nvidia-smi failed: {stderr.strip()}")
        gpus: list[dict[str, Any]] = []
        keys = [field.replace(".", "_") for field in self.GPU_FIELDS]
        for line in stdout.splitlines():
            values = [part.strip() for part in line.split(",")]
            if len(values) != len(keys):
                continue
            row = dict(zip(keys, values, strict=True))
            gpu: dict[str, Any] = {
                "device_id": int(row["index"]),
                "name": row["name"],
                "driver_version": row["driver_version"],
                "pci_bus_id": row["pci_bus_id"],
                "memory_total_mb": int(row["memory_total"]),
                "memory_used_mb": int(row["memory_used"]),
                "memory_free_mb": int(row["memory_free"]),
                "utilization": int(row["utilization_gpu"]),
                "temperature_c": int(row["temperature_gpu"]),
                "power_draw_w": row["power_draw"],
            }
            gpus.append(gpu)
        return gpus

    async def gpu(self, device_id: int) -> dict[str, Any]:
        gpus = await self.list_gpus()
        for gpu in gpus:
            if gpu["device_id"] == device_id:
                return gpu
        raise IndexError(f"GPU device not found: {device_id}")

    async def processes(self, device_id: int | None = None) -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        result = await self._run(command)
        if result is None or result[0] != 0:
            return []
        processes: list[dict[str, Any]] = []
        for line in result[1].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            processes.append(
                {
                    "gpu_bus_id": parts[0],
                    "pid": int(parts[1]) if parts[1].isdigit() else parts[1],
                    "process_name": parts[2],
                    "used_memory_mb": int(parts[3]) if parts[3].isdigit() else parts[3],
                }
            )
        if device_id is None:
            return processes
        gpus = await self.list_gpus()
        gpu = next((item for item in gpus if item["device_id"] == device_id), None)
        if gpu is None:
            raise IndexError(f"GPU device not found: {device_id}")
        return [process for process in processes if process["gpu_bus_id"] == gpu["pci_bus_id"]]

    async def toolkit_info(self) -> dict[str, Any]:
        result = await self._run([str(self.nvcc_path), "--version"])
        cuda_version = None
        if result and result[0] == 0:
            for line in result[1].splitlines():
                if "release" in line.lower():
                    cuda_version = line.strip()
                    break
        return {
            "cuda_version": cuda_version,
            "nvcc_available": result is not None and result[0] == 0,
            "cmake_available": shutil.which(str(self.cmake_path)) is not None,
            "cmake_path": str(self.cmake_path),
            "nvcc_path": str(self.nvcc_path),
            "ncu_path": str(self.ncu_path) if self.ncu_path else None,
            "nsys_path": str(self.nsys_path) if self.nsys_path else None,
            "nsight_compute_available": self.ncu_path is not None and shutil.which(str(self.ncu_path)) is not None,
            "nsight_systems_available": self.nsys_path is not None and shutil.which(str(self.nsys_path)) is not None,
        }

    async def capabilities(self) -> dict[str, Any]:
        toolkit = await self.toolkit_info()
        gpus = await self.list_gpus()
        return {
            **toolkit,
            "build_available": toolkit["nvcc_available"] and toolkit["cmake_available"],
            "profile_available": toolkit["nsight_compute_available"] or toolkit["nsight_systems_available"],
            "gpu_count": len(gpus),
            "gpus": gpus,
        }

    async def resource_check(self, device_id: int, min_memory_mb: int = 0) -> dict[str, Any]:
        gpu = await self.gpu(device_id)
        enough = gpu["memory_free_mb"] >= min_memory_mb and gpu["utilization"] < 100
        return {"device_id": device_id, "available": enough, "gpu": gpu}
