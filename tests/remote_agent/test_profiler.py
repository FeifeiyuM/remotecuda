import asyncio
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import ProcessResult, ProcessRunner, ProfilerExecutor


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    async def run(self, *, command, **kwargs) -> ProcessResult:
        self.commands.append(command)
        self.environments.append(dict(kwargs["env"]))
        if "ncu" in Path(command[0]).name:
            report = Path(command[command.index("--export") + 1])
        else:
            report = Path(command[command.index("--output") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(b"report-data")
        return ProcessResult(
            command=command,
            exit_code=0,
            stdout="profile complete",
            stderr="",
            duration_seconds=0.01,
        )


class ProfilerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace-test"
        self.build = self.workspace / "build"
        self.build.mkdir(parents=True)
        self.executable = self.build / "cuda_app"
        shutil.copy2("/bin/echo", self.executable)
        self.executable.chmod(0o700)

        self.ncu = self.root / "ncu"
        self.nsys = self.root / "nsys"
        for path in (self.ncu, self.nsys):
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def test_ncu_profile_creates_report_and_metadata(self):
        runner = FakeRunner()
        executor = ProfilerExecutor(runner, self.ncu, self.nsys)
        result = asyncio.run(
            executor.profile(
                job_id="profile-test",
                workspace_root=self.workspace,
                config={
                    "profiler": "ncu",
                    "profile_name": "kernel profile",
                    "executable": "cuda_app",
                    "args": ["one"],
                    "device_id": 0,
                    "timeout_seconds": 30,
                    "ncu": {"metric_set": "full", "launch_count": 2},
                },
                timeout_seconds=30,
                log_path=self.workspace / "logs" / "profile.log",
            )
        )
        self.assertEqual(result["profiler"], "ncu")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue((self.workspace / "artifacts" / result["artifact_path"]).is_file())
        self.assertTrue((self.workspace / "artifacts" / result["profiler_result_path"]).is_file())
        self.assertFalse(result["artifact_path"].startswith("artifacts/"))
        self.assertFalse(result["profiler_result_path"].startswith("artifacts/"))
        self.assertIn("--set", runner.commands[0])
        self.assertIn("full", runner.commands[0])
        log = (self.workspace / "logs" / "profile.log").read_text(encoding="utf-8")
        self.assertIn('"event":"profiler.start"', log)
        self.assertIn('"event":"profiler.complete"', log)

    def test_nsys_profile_uses_selected_traces_without_ncu_process_option(self):
        runner = FakeRunner()
        executor = ProfilerExecutor(runner, self.ncu, self.nsys)
        result = asyncio.run(
            executor.profile(
                job_id="nsys-test",
                workspace_root=self.workspace,
                config={
                    "profiler": "nsys",
                    "profile_name": "system profile",
                    "executable": "cuda_app",
                    "args": [],
                    "device_id": 0,
                    "timeout_seconds": 30,
                    "nsys": {"trace": ["cuda", "nvtx"], "sample_cpu": False},
                },
                timeout_seconds=30,
                log_path=self.workspace / "logs" / "profile.log",
            )
        )
        self.assertTrue(result["artifact_path"].endswith(".nsys-rep"))
        command = runner.commands[0]
        self.assertIn("--trace", command)
        self.assertIn("cuda,nvtx", command)
        self.assertIn("--sample", command)
        self.assertIn("none", command)
        self.assertNotIn("--target-processes", command)

    def test_prefixed_environment_is_passed_to_profiler_process(self):
        runner = FakeRunner()
        executor = ProfilerExecutor(runner, self.ncu, self.nsys)
        asyncio.run(
            executor.profile(
                job_id="profile-env",
                workspace_root=self.workspace,
                config={
                    "profiler": "ncu",
                    "executable": "cuda_app",
                    "device_id": 4,
                    "timeout_seconds": 30,
                    "env": {"RC_ENV_CUSTOM_PROFILE_VALUE": "one-profile"},
                },
                timeout_seconds=30,
                log_path=self.workspace / "logs" / "profile.log",
            )
        )
        self.assertEqual(
            runner.environments,
            [{"CUSTOM_PROFILE_VALUE": "one-profile", "CUDA_VISIBLE_DEVICES": "4"}],
        )

    def test_missing_tool_is_rejected(self):
        executor = ProfilerExecutor(ProcessRunner(), None, self.nsys)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            asyncio.run(
                executor.profile(
                    job_id="missing-tool",
                    workspace_root=self.workspace,
                    config={"profiler": "ncu", "executable": "cuda_app"},
                    timeout_seconds=30,
                    log_path=self.workspace / "logs" / "profile.log",
                )
            )


if __name__ == "__main__":
    unittest.main()
