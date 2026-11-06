import os
import json
import types
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from pydantic import ValidationError

from agent_config import AgentSettings, load_settings
import agent_config as agent_config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tool_temp = tempfile.TemporaryDirectory()
        tool_root = Path(self.tool_temp.name)
        self.cmake_path = tool_root / "cmake"
        self.nvcc_path = tool_root / "nvcc"
        for path in (self.cmake_path, self.nvcc_path):
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)

    def tearDown(self):
        self.tool_temp.cleanup()

    def tool_kwargs(self):
        return {"cmake_path": self.cmake_path, "nvcc_path": self.nvcc_path}

    def test_defaults_are_loopback_http(self):
        environment = {
            "REMOTE_CUDA_CMAKE_PATH": str(self.cmake_path),
            "REMOTE_CUDA_NVCC_PATH": str(self.nvcc_path),
        }
        with mock.patch.dict(os.environ, environment):
            settings = load_settings("/does/not/exist.yaml")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.cmake_path, self.cmake_path)
        self.assertEqual(settings.nvcc_path, self.nvcc_path)

    def test_token_can_be_set_in_config_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "remote.json"
            config_path.write_text(
                json.dumps(
                    {
                        "token": "config-token",
                        "cmake_path": str(self.cmake_path),
                        "nvcc_path": str(self.nvcc_path),
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(config_path)
            self.assertEqual(settings.token, "config-token")

            with mock.patch.dict(os.environ, {"REMOTE_CUDA_AGENT_TOKEN": "environment-token"}):
                overridden = load_settings(config_path)
            self.assertEqual(overridden.token, "environment-token")

    def test_authentication_can_be_disabled(self):
        settings = AgentSettings(auth_enabled=False, **self.tool_kwargs())
        self.assertFalse(settings.auth_enabled)

    def test_authentication_requires_token_when_enabled(self):
        with self.assertRaises(ValidationError):
            AgentSettings(auth_enabled=True, token="", **self.tool_kwargs())

    def test_runtime_paths_default_to_install_directory(self):
        settings = AgentSettings(**self.tool_kwargs())
        runtime_dir = agent_config.INSTALL_DIR / "runtime"
        self.assertEqual(settings.workspace_root, runtime_dir / "workspaces")
        self.assertEqual(settings.database_path, runtime_dir / "agent.db")

    def test_venv_executables_derive_install_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            install_dir = Path(temporary_dir)
            bin_dir = install_dir / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            executable = bin_dir / "uvicorn"
            executable.touch()
            with mock.patch.object(agent_config.sys, "argv", [str(executable)]):
                self.assertEqual(agent_config._default_install_dir(), install_dir.resolve())

    def test_relative_config_paths_resolve_beside_config_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "agent.json"
            config_path.write_text(
                json.dumps(
                    {
                        "workspace_root": "runtime/workspaces",
                        "database_path": "runtime/agent.db",
                        "cmake_path": str(self.cmake_path),
                        "nvcc_path": str(self.nvcc_path),
                    }
                ),
                encoding="utf-8",
            )
            settings = load_settings(config_path)
            self.assertEqual(settings.workspace_root, Path(temporary_dir) / "runtime/workspaces")
            self.assertEqual(settings.database_path, Path(temporary_dir) / "runtime/agent.db")

    def test_default_config_is_remote_yaml_in_current_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                Path("remote.yaml").write_text("port: 12345\n", encoding="utf-8")
                yaml_module = types.ModuleType("yaml")
                yaml_module.safe_load = lambda _text: {"port": 12345, **self.tool_kwargs()}
                with mock.patch.dict(sys.modules, {"yaml": yaml_module}):
                    settings = load_settings()
                self.assertEqual(settings.port, 12345)
            finally:
                os.chdir(original_cwd)

    def test_tool_paths_are_required(self):
        with self.assertRaises(ValidationError):
            AgentSettings()

    def test_tool_paths_must_be_absolute_executables(self):
        kwargs = self.tool_kwargs()
        kwargs["cmake_path"] = self.cmake_path.relative_to(self.cmake_path.parent)
        with self.assertRaises(ValidationError):
            AgentSettings(**kwargs)

    def test_nsight_tool_paths_are_optional_but_validated(self):
        settings = AgentSettings(**self.tool_kwargs())
        self.assertIsNone(settings.ncu_path)
        self.assertIsNone(settings.nsys_path)

        kwargs = self.tool_kwargs()
        kwargs["ncu_path"] = self.cmake_path
        kwargs["nsys_path"] = self.nvcc_path
        settings = AgentSettings(**kwargs)
        self.assertEqual(settings.ncu_path, self.cmake_path)
        self.assertEqual(settings.nsys_path, self.nvcc_path)

        kwargs = self.tool_kwargs()
        kwargs["nsys_path"] = self.nvcc_path.parent / "missing-nsys"
        with self.assertRaises(ValidationError):
            AgentSettings(**kwargs)

if __name__ == "__main__":
    unittest.main()
