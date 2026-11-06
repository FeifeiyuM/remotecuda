import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from services.sync_manager import FileUploadRequest, SyncManifestRequest, SyncManager
from services.workspace_manager import WorkspaceCreateRequest, WorkspaceManager
from storage.database import Database


class WorkspaceSyncTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database = Database(root / "agent.db")
        self.workspaces = WorkspaceManager(root / "workspaces", self.database)
        self.sync = SyncManager(self.workspaces, 1024 * 1024)
        self.workspaces.create(WorkspaceCreateRequest(workspace_id="workspace-test", name="test"))

    def tearDown(self):
        self.database.close()
        self.temp.cleanup()

    @staticmethod
    def upload_object(path: str, content: bytes) -> FileUploadRequest:
        return FileUploadRequest(
            workspace_id="workspace-test",
            path=path,
            content_base64=__import__("base64").b64encode(content).decode("ascii"),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def test_manifest_upload_hash_and_removal(self):
        content = b"__global__ void k() {}\n"
        upload = self.upload_object("src/kernel.cu", content)
        self.sync.upload(upload)
        manifest = SyncManifestRequest(
            workspace_id="workspace-test",
            files=[
                {
                    "path": "src/kernel.cu",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        )
        first = self.sync.apply_manifest(manifest)
        self.assertEqual(first["unchanged"], ["src/kernel.cu"])
        empty_plan = self.sync.apply_manifest(
            SyncManifestRequest(workspace_id="workspace-test", files=[])
        )
        self.assertEqual(empty_plan["remove"], ["src/kernel.cu"])
        self.assertFalse((self.workspaces.workspace_root("workspace-test") / "source/src/kernel.cu").exists())

    def test_hash_mismatch_is_rejected(self):
        upload = self.upload_object("bad.cu", b"data")
        upload.sha256 = "0" * 64
        with self.assertRaises(ValueError):
            self.sync.upload(upload)


if __name__ == "__main__":
    unittest.main()
