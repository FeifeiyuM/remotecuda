import sys
import unittest
from pathlib import Path


LOCAL_MCP = str(Path(__file__).resolve().parents[2] / "local-mcp")
if LOCAL_MCP not in sys.path:
    sys.path.insert(0, LOCAL_MCP)

from scheduler.selection_policy import (
    ServerCandidate,
    ScheduleRequirements,
    filter_candidates,
    sort_by_policy,
)


class SchedulerTest(unittest.TestCase):
    def test_filters_tags_memory_and_gpu_model(self):
        candidates = [
            ServerCandidate(
                server_id="small",
                priority=1,
                tags=["development"],
                gpus=[{"name": "RTX 3060", "memory_free_mb": 8000}],
            ),
            ServerCandidate(
                server_id="large",
                priority=2,
                tags=["benchmark"],
                gpus=[{"name": "RTX 4090", "memory_free_mb": 20000}],
            ),
        ]
        requirements = ScheduleRequirements(
            gpu_name="4090", min_memory_mb=10000, tags=["benchmark"]
        )
        result = filter_candidates(candidates, requirements)
        self.assertEqual([item.server_id for item in result], ["large"])

    def test_least_loaded_and_most_free_memory(self):
        candidates = [
            ServerCandidate(server_id="busy", active_jobs=5, gpus=[{"memory_free_mb": 24000}]),
            ServerCandidate(server_id="idle", active_jobs=0, gpus=[{"memory_free_mb": 8000}]),
        ]
        self.assertEqual(sort_by_policy(candidates, "least_loaded")[0].server_id, "idle")
        self.assertEqual(sort_by_policy(candidates, "most_free_memory")[0].server_id, "busy")


if __name__ == "__main__":
    unittest.main()
