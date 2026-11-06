import unittest
from pathlib import Path


class ImportPathsTest(unittest.TestCase):
    def test_project_packages_are_declared(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / "local-mcp/server.py").is_file())
        self.assertTrue((root / "remote-agent/main.py").is_file())
        self.assertIn("fastmcp", (root / "local-mcp/pyproject.toml").read_text())
        self.assertIn("fastapi", (root / "remote-agent/requirements.txt").read_text())

    def test_services_start_directly_with_python(self):
        root = Path(__file__).resolve().parents[2]
        remote_start = (root / "scripts/start-remote.sh").read_text()
        local_start = (root / "scripts/start-local.sh").read_text()
        self.assertIn('exec "$PYTHON" main.py', remote_start)
        self.assertIn('"$PYTHON" server.py', local_start)
        self.assertFalse((root / "scripts/install-remote.sh").exists())
        self.assertFalse((root / "scripts/install-local.sh").exists())


if __name__ == "__main__":
    unittest.main()
