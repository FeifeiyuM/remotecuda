import sys
import tempfile
import unittest
from pathlib import Path


LOCAL_MCP = str(Path(__file__).resolve().parents[2] / "local-mcp")
if LOCAL_MCP not in sys.path:
    sys.path.insert(0, LOCAL_MCP)

from registry import ServerConfig, ServerRegistry, ServerRegistryError


class ServerRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ServerRegistry(Path(self.temp.name) / "servers.db")

    def tearDown(self):
        self.registry.close()
        self.temp.cleanup()

    def test_server_lifecycle_and_runtime_status(self):
        config = ServerConfig(
            server_id="gpu-test",
            name="Test GPU",
            endpoint="http://gpu.example/",
            token_env="GPU_TEST_TOKEN",
            tags=["cuda"],
        )
        self.registry.add_server(config)
        self.assertEqual(self.registry.list_servers()[0]["server_id"], "gpu-test")
        runtime = self.registry.get_server_status("gpu-test")
        self.assertEqual(runtime.connection_status, "DISCONNECTED")
        self.registry.set_connection("gpu-test", True)
        self.assertEqual(self.registry.get_server_status("gpu-test").connection_status, "CONNECTED")
        self.registry.remove_server("gpu-test")
        self.assertEqual(self.registry.list_servers(), [])

    def test_duplicate_and_invalid_ids_are_rejected(self):
        config = ServerConfig(
            server_id="gpu-test",
            name="Test GPU",
            endpoint="http://127.0.0.1:8443",
            token_env="GPU_TEST_TOKEN",
        )
        self.registry.add_server(config)
        with self.assertRaises(ServerRegistryError):
            self.registry.add_server(config)
        with self.assertRaises(Exception):
            self.registry.add_server(
                ServerConfig(
                    server_id="../bad",
                    name="Bad",
                    endpoint="http://127.0.0.1:8443",
                    token_env="GPU_TEST_TOKEN",
                )
            )

    def test_no_auth_server_does_not_require_token_environment(self):
        config = ServerConfig(
            server_id="gpu-no-auth",
            name="No Auth Server",
            endpoint="http://127.0.0.1:8443",
            auth_enabled=False,
        )
        self.assertIsNone(config.token_env)

if __name__ == "__main__":
    unittest.main()
