import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


LOCAL_MCP = str(Path(__file__).resolve().parents[2] / "local-mcp")
if LOCAL_MCP not in sys.path:
    sys.path.insert(0, LOCAL_MCP)

try:
    from remote_client.client import RemoteClient
    from remote_client.models import BenchmarkConfig, ExecutionConfig, ProfilerConfig
except ModuleNotFoundError as exc:
    if exc.name != "httpx":
        raise
    raise unittest.SkipTest("local MCP virtual environment is not installed")


class ExecutionEnvironmentModelTest(unittest.TestCase):
    def test_execution_models_inherit_request_scoped_env(self):
        for model in (ExecutionConfig, BenchmarkConfig, ProfilerConfig):
            payload = {
                "executable": "release/app",
                "env": {"RC_ENV_CUSTOM_TEST_VALUE": "one-run"},
            }
            if model is ProfilerConfig:
                payload["profiler"] = "ncu"
            config = model(**payload)
            self.assertEqual(config.env, {"RC_ENV_CUSTOM_TEST_VALUE": "one-run"})

    def test_reserved_environment_is_rejected_locally(self):
        with self.assertRaises(ValidationError):
            ExecutionConfig(
                executable="release/app",
                env={"RC_ENV_CUDA_VISIBLE_DEVICES": "0,1"},
            )
        with self.assertRaises(ValidationError):
            BenchmarkConfig(
                executable="release/app",
                env={"RC_ENV_LD_PRELOAD": "/tmp/lib.so"},
            )
        with self.assertRaises(ValidationError):
            BenchmarkConfig(executable="release/app", env={"CUSTOM_TEST_VALUE": "value"})

    def test_empty_environment_is_omitted_for_older_remote_agents(self):
        payload = RemoteClient._execution_payload(ExecutionConfig(executable="release/app"))
        self.assertNotIn("env", payload)

        payload = RemoteClient._execution_payload(
            ExecutionConfig(
                executable="release/app",
                env={"RC_ENV_CUSTOM_TEST_VALUE": "one-run"},
            )
        )
        self.assertEqual(payload["env"], {"RC_ENV_CUSTOM_TEST_VALUE": "one-run"})


if __name__ == "__main__":
    unittest.main()
