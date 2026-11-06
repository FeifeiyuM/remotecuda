import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import BuildExecutor, CudaExecutor, ProcessRunner, ProfilerExecutor
from services.artifact_manager import ArtifactManager
from services.job_manager import JobManager
from services.workspace_manager import WorkspaceCreateRequest, WorkspaceManager
from storage.database import Database


class JobPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = Database(self.root / "agent.db")
        self.workspaces = WorkspaceManager(self.root / "workspaces", self.database)
        self.workspaces.create(WorkspaceCreateRequest(workspace_id="workspace-test", name="test"))
        self.artifacts = ArtifactManager(self.workspaces, 1024 * 1024)
        self.runner = ProcessRunner()
        self.jobs = JobManager(
            database=self.database,
            workspaces=self.workspaces,
            artifacts=self.artifacts,
            runner=self.runner,
            build_executor=BuildExecutor(
                self.runner,
                2,
                cmake_path=Path("/bin/true"),
                nvcc_path=Path("/bin/true"),
            ),
            cuda_executor=CudaExecutor(self.runner),
            profiler_executor=ProfilerExecutor(self.runner, None, None),
            max_running_jobs=2,
            default_timeout_seconds=10,
        )

    def tearDown(self):
        self.database.close()
        self.temp.cleanup()

    def test_run_job_measures_real_process_and_persists_log(self):
        executable = self.workspaces.workspace_root("workspace-test") / "build" / "hello"
        executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2("/bin/echo", executable)
        executable.chmod(0o700)
        async def submit_and_wait():
            submitted = await self.jobs.submit(
                workspace_id="workspace-test",
                job_type="RUN",
                config={"executable": "hello", "args": ["measured-output"], "device_id": 0, "timeout_seconds": 30},
                idempotency_key="run-1",
            )
            await self.jobs._tasks[submitted["job_id"]]
            return submitted

        submitted = asyncio.run(submit_and_wait())
        job = self.jobs.get(submitted["job_id"])
        self.assertEqual(
            job.status,
            "SUCCESS",
            f"job error: {job.error}; log: {self.jobs.logs(job.job_id)['content']}",
        )
        self.assertIn("measured-output", job.result["stdout"])
        self.assertIn("$", self.jobs.logs(submitted["job_id"])["content"])
        duplicate = asyncio.run(
            self.jobs.submit(
                workspace_id="workspace-test",
                job_type="RUN",
                config={"executable": "hello", "timeout_seconds": 30},
                idempotency_key="run-1",
            )
        )
        self.assertEqual(duplicate["job_id"], submitted["job_id"])

    def test_restart_marks_active_job_interrupted(self):
        job = self.database.create_job(
            __import__("storage.models", fromlist=["JobMetadata"]).JobMetadata(
                job_id="remote-active",
                workspace_id="workspace-test",
                job_type="RUN",
                status="RUNNING",
            )
        )
        self.assertEqual(job.status, "RUNNING")
        self.database.close()
        reopened = Database(self.root / "agent.db")
        self.assertEqual(reopened.get_job("remote-active").status, "INTERRUPTED")
        reopened.close()


if __name__ == "__main__":
    unittest.main()
