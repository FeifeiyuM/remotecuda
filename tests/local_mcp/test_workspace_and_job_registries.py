import sys
import tempfile
import unittest
from pathlib import Path


LOCAL_MCP = str(Path(__file__).resolve().parents[2] / "local-mcp")
if LOCAL_MCP not in sys.path:
    sys.path.insert(0, LOCAL_MCP)

from registry import JobRegistry, WorkspaceRecord, WorkspaceRegistry


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspaces = WorkspaceRegistry(Path(self.temp.name) / "workspaces.db")
        self.jobs = JobRegistry(Path(self.temp.name) / "jobs.db")

    def tearDown(self):
        self.workspaces.close()
        self.jobs.close()
        self.temp.cleanup()

    def test_workspace_binding_is_immutable_and_migrate_is_explicit(self):
        record = WorkspaceRecord(
            workspace_id="workspace-test",
            name="test",
            server_id="gpu-one",
            local_path="/tmp/project",
        )
        self.workspaces.create(record)
        with self.assertRaises(Exception):
            self.workspaces.update("workspace-test", server_id="gpu-two")
        migrated = self.workspaces.migrate("workspace-test", "gpu-two", "/remote/workspace-test")
        self.assertEqual(migrated.server_id, "gpu-two")
        self.assertEqual(self.workspaces.require("workspace-test").remote_path, "/remote/workspace-test")

    def test_job_index_keeps_remote_and_server_mapping(self):
        job = self.jobs.create(
            job_id="local-job-1",
            remote_job_id="remote-job-1",
            server_id="gpu-one",
            workspace_id="workspace-test",
            job_type="BUILD",
            idempotency_key="build-1",
        )
        self.assertEqual(job.status, "PENDING")
        self.jobs.update_status("local-job-1", "RUNNING")
        self.assertTrue(self.jobs.has_active_jobs("gpu-one"))
        self.assertIsNotNone(self.jobs.get_by_idempotency_key("build-1"))


if __name__ == "__main__":
    unittest.main()
