import asyncio
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import CudaExecutor, ProcessRunner


class FakeGpuReader:
    async def gpu(self, device_id: int) -> dict:
        return {
            "device_id": device_id,
            "utilization": 37,
            "memory_used_mb": 128,
        }


class BenchmarkTest(unittest.TestCase):
    def test_metrics_use_measured_process_and_gpu_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace-test"
            build = workspace / "build"
            artifacts = workspace / "artifacts" / "benchmark"
            for directory in (build, artifacts):
                directory.mkdir(parents=True)
            executable = build / "benchmark"
            shutil.copy2("/bin/echo", executable)
            executable.chmod(0o700)
            executor = CudaExecutor(ProcessRunner(10_000), FakeGpuReader())
            result = asyncio.run(
                executor.benchmark(
                    job_id="remote-test",
                    workspace_root=workspace,
                    config={
                        "executable": "benchmark",
                        "args": ["ok"],
                        "device_id": 0,
                        "iterations": 3,
                        "warmup_iterations": 1,
                        "benchmark_name": "echo benchmark",
                        "timeout_seconds": 30,
                    },
                    timeout_seconds=30,
                    log_path=workspace / "logs" / "benchmark.log",
                )
            )
            self.assertEqual(len(result["measured_iteration_ms"]), 3)
            self.assertGreater(result["metrics"]["mean_ms"], 0)
            self.assertEqual(result["metrics"]["gpu_utilization_max_percent"], 37)
            self.assertEqual(result["metrics"]["gpu_memory_used_max_mb"], 128)
            self.assertTrue((workspace / "artifacts" / result["artifact_path"]).is_file())
            self.assertFalse(result["artifact_path"].startswith("artifacts/"))
            log_content = (workspace / "logs" / "benchmark.log").read_text(encoding="utf-8")
            self.assertIn('"event":"cuda.benchmark_start"', log_content)
            self.assertIn('"event":"cuda.benchmark_iteration_start"', log_content)
            self.assertIn('"event":"cuda.sample_monitor_complete"', log_content)
            self.assertIn('"event":"cuda.benchmark_complete"', log_content)


if __name__ == "__main__":
    unittest.main()
