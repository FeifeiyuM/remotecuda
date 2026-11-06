import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import CudaExecutor, ProcessRunner
from executor.process_runner import ProcessResult
from security import EnvironmentPolicy, EnvironmentPolicyError


class EnvironmentPolicyTest(unittest.TestCase):
    def test_accepts_prefixed_names_and_removes_prefix_for_child(self):
        policy = EnvironmentPolicy()
        validated = policy.validate(
            {
                "RC_ENV_CUDA_LAUNCH_BLOCKING": "1",
                "RC_ENV_OMP_NUM_THREADS": "4",
            }
        )
        self.assertEqual(validated, {"CUDA_LAUNCH_BLOCKING": "1", "OMP_NUM_THREADS": "4"})

        with self.assertRaisesRegex(EnvironmentPolicyError, "RC_ENV_ prefix"):
            policy.validate({"CUSTOM_VALUE": "1"})
        with self.assertRaisesRegex(EnvironmentPolicyError, "RC_ENV_ prefix"):
            policy.validate({"RC_ENV_": "1"})

    def test_reserved_target_names_cannot_be_overridden(self):
        policy = EnvironmentPolicy()
        with self.assertRaisesRegex(EnvironmentPolicyError, "reserved"):
            policy.validate({"RC_ENV_CUDA_VISIBLE_DEVICES": "0,1"})
        with self.assertRaisesRegex(EnvironmentPolicyError, "reserved"):
            policy.validate({"RC_ENV_LD_LIBRARY_PATH": "/tmp"})

    def test_duplicate_and_invalid_entries_are_rejected(self):
        policy = EnvironmentPolicy()
        with self.assertRaisesRegex(EnvironmentPolicyError, "invalid environment"):
            policy.validate({"lower-case": "value"})
        with self.assertRaisesRegex(EnvironmentPolicyError, "NUL"):
            policy.validate({"RC_ENV_CUSTOM_VALUE": "bad\x00value"})


class CudaEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / "workspace-test"
        self.build = self.workspace / "build"
        self.build.mkdir(parents=True)
        self.executable = self.build / "print-env"
        self.executable.write_text(
            """#!/bin/sh
if [ "$CUSTOM_TEST_VALUE" = "ephemeral-test-value" ]; then
  printf 'custom-value-present\\n'
fi
if [ -n "$REMOTE_CUDA_AGENT_TOKEN" ]; then
  printf 'agent-secret-present\\n'
fi
printf 'device=%s\\n' "$CUDA_VISIBLE_DEVICES"
""",
            encoding="utf-8",
        )
        self.executable.chmod(0o700)
        self.log_path = self.workspace / "logs" / "run.log"

    def tearDown(self):
        self.temp.cleanup()

    def test_custom_environment_is_ephemeral_and_not_persisted_to_logs(self):
        executor = CudaExecutor(ProcessRunner())

        async def run():
            return await executor.execute(
                job_id="env-test",
                workspace_root=self.workspace,
                config={
                    "executable": "print-env",
                    "device_id": 0,
                    "timeout_seconds": 10,
                    "env": {"RC_ENV_CUSTOM_TEST_VALUE": "ephemeral-test-value"},
                },
                timeout_seconds=10,
                log_path=self.log_path,
            )

        secret = "agent-secret-value"
        with mock.patch.dict(
            os.environ,
            {"REMOTE_CUDA_AGENT_TOKEN": secret, "CUSTOM_TEST_VALUE": "inherited-value"},
        ):
            result = asyncio.run(run())

        self.assertIn("custom-value-present", result["stdout"])
        self.assertNotIn("agent-secret-present", result["stdout"])
        self.assertNotIn("inherited-value", result["stdout"])
        log = self.log_path.read_text(encoding="utf-8")
        self.assertIn("CUSTOM_TEST_VALUE", log)
        self.assertIn("CUDA_VISIBLE_DEVICES", log)
        self.assertNotIn("ephemeral-test-value", log)

    def test_cuda_visible_devices_remains_server_controlled(self):
        executor = CudaExecutor(ProcessRunner())

        async def run():
            return await executor.execute(
                job_id="device-test",
                workspace_root=self.workspace,
                config={
                    "executable": "print-env",
                    "device_id": 2,
                    "timeout_seconds": 10,
                    "env": {"RC_ENV_CUSTOM_TEST_VALUE": "ephemeral-test-value"},
                },
                timeout_seconds=10,
                log_path=self.log_path,
            )

        result = asyncio.run(run())
        self.assertIn("device=2", result["stdout"])
        self.assertIn("custom-value-present", result["stdout"])


class FakeRecordingRunner:
    def __init__(self) -> None:
        self.environments: list[dict[str, str]] = []

    async def run(self, *, command, **kwargs) -> ProcessResult:
        self.environments.append(dict(kwargs["env"]))
        return ProcessResult(
            command=command,
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
        )


class BenchmarkEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / "workspace-test"
        self.build = self.workspace / "build"
        self.build.mkdir(parents=True)
        self.executable = self.build / "benchmark"
        shutil.copy2("/bin/true", self.executable)
        self.executable.chmod(0o700)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_benchmark_iteration_receives_same_environment(self):
        runner = FakeRecordingRunner()
        executor = CudaExecutor(runner)

        async def benchmark():
            return await executor.benchmark(
                job_id="benchmark-env",
                workspace_root=self.workspace,
                config={
                    "executable": "benchmark",
                    "iterations": 2,
                    "warmup_iterations": 1,
                    "device_id": 3,
                    "env": {"RC_ENV_CUSTOM_TEST_VALUE": "stable"},
                },
                timeout_seconds=10,
                log_path=self.workspace / "logs" / "benchmark.log",
            )

        result = asyncio.run(benchmark())
        self.assertEqual(len(result["measured_iteration_ms"]), 2)
        self.assertEqual(
            runner.environments,
            [
                {"CUSTOM_TEST_VALUE": "stable", "CUDA_VISIBLE_DEVICES": "3"}
                for _ in range(3)
            ],
        )


if __name__ == "__main__":
    unittest.main()
