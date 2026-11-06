import asyncio
import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import BuildExecutor, ProcessResult, ProcessRunner


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ProcessResult:
        self.calls.append(kwargs)
        return ProcessResult(
            command=kwargs["command"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
        )


class BuildExecutorTest(unittest.TestCase):
    def test_build_writes_lifecycle_events_to_job_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            source = workspace / "source"
            source.mkdir(parents=True)
            (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\n")

            cmake = root / "cmake"
            nvcc = root / "nvcc"
            for executable in (cmake, nvcc):
                executable.write_text("#!/bin/sh\n", encoding="utf-8")
                executable.chmod(0o755)

            log_path = workspace / "logs" / "build.log"
            runner = FakeRunner()
            executor = BuildExecutor(runner, 2, cmake_path=cmake, nvcc_path=nvcc)
            result = asyncio.run(
                executor.execute(
                    job_id="build-log-test",
                    workspace_root=workspace,
                    config={
                        "build_system": "cmake",
                        "build_type": "Release",
                        "cuda_arch": "86",
                        "parallel_jobs": 2,
                        "clean": True,
                    },
                    timeout_seconds=30,
                    log_path=log_path,
                )
            )

            self.assertEqual(result["configure_exit_code"], 0)
            self.assertEqual(result["build_exit_code"], 0)
            self.assertEqual(len(runner.calls), 2)
            content = log_path.read_text(encoding="utf-8")
            self.assertIn('"event":"build.start"', content)
            self.assertIn('"event":"build.tools_resolved"', content)
            self.assertIn('"event":"build.configure_start"', content)
            self.assertIn('"event":"build.compile_start"', content)
            self.assertIn('"event":"build.complete"', content)


if __name__ == "__main__":
    unittest.main()
