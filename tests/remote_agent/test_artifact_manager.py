import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ArtifactManagerPathTest(unittest.TestCase):
    def test_download_accepts_legacy_workspace_relative_artifact_paths(self):
        script = r'''
import sys
from pathlib import Path
from services.artifact_manager import ArtifactManager

class FakeWorkspaces:
    def __init__(self, workspace_root):
        self.workspace_root_value = workspace_root

    @staticmethod
    def metadata(_workspace_id):
        return None

    def workspace_root(self, _workspace_id):
        return self.workspace_root_value

root = Path(sys.argv[1])
workspace_root = root / "workspace-test"
artifact = workspace_root / "artifacts" / "profiler" / "report.ncu-rep"
artifact.parent.mkdir(parents=True)
artifact.write_bytes(b"report")
manager = ArtifactManager(FakeWorkspaces(workspace_root), max_artifact_bytes=1024)
canonical = manager.download_path("workspace-test", "profiler/report.ncu-rep")
legacy = manager.download_path("workspace-test", "artifacts/profiler/report.ncu-rep")
assert canonical == legacy
assert canonical.is_file()
'''
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "remote-agent")
            result = subprocess.run(
                [sys.executable, "-c", script, temp],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
