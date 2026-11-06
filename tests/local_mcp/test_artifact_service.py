import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ArtifactServiceTest(unittest.TestCase):
    def test_download_returns_hex_digest_and_normalizes_legacy_path(self):
        script = r'''
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from services.artifact_service import ArtifactService

class FakeStreamClient:
    @staticmethod
    async def artifact_stream(_workspace_id, _artifact_path):
        yield b"hello "
        yield b"world"

class FakeContext:
    def __init__(self, project_root):
        self.workspaces = SimpleNamespace(
            require=lambda _workspace_id: SimpleNamespace(
                local_path=str(project_root), server_id="gpu-test"
            )
        )

    def client_for(self, _server_id):
        return FakeStreamClient()

project = Path(sys.argv[1]) / "project"
project.mkdir()
service = ArtifactService(FakeContext(project))
result = asyncio.run(
    service.download("workspace-test", "artifacts/profiler/report.ncu-rep")
)
assert result["size"] == 11
assert result["sha256"] == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
assert result["local_path"] == str(
    project / ".remote-cuda-artifacts" / "profiler" / "report.ncu-rep"
)
'''
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "local-mcp")
            python = Path(__file__).resolve().parents[2] / "local-mcp" / ".venv" / "bin" / "python"
            if not python.is_file():
                self.skipTest("local MCP virtual environment is not installed")
            result = subprocess.run(
                [str(python), "-c", script, temp],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
