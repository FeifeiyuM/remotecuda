import sys
import tempfile
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from security import PathSecurityError, resolve_inside, safe_relative_path


class PathSecurityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_traversal_absolute_and_escape_symlink(self):
        with self.assertRaises(PathSecurityError):
            safe_relative_path("../outside.txt")
        with self.assertRaises(PathSecurityError):
            safe_relative_path("/etc/passwd")
        outside = self.root.parent / "outside-path-security"
        outside.write_text("outside", encoding="utf-8")
        link = self.root / "link"
        link.symlink_to(outside)
        try:
            with self.assertRaises(PathSecurityError):
                resolve_inside(self.root, "link")
        finally:
            link.unlink()
            outside.unlink()

    def test_accepts_child_path(self):
        target = self.root / "source" / "kernel.cu"
        target.parent.mkdir()
        target.write_text("kernel", encoding="utf-8")
        self.assertEqual(resolve_inside(self.root, "source/kernel.cu"), target.resolve())


if __name__ == "__main__":
    unittest.main()
