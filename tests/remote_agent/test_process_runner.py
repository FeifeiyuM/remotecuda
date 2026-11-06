import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path


REMOTE_AGENT = str(Path(__file__).resolve().parents[2] / "remote-agent")
if REMOTE_AGENT not in sys.path:
    sys.path.insert(0, REMOTE_AGENT)

from executor import ProcessRunner


class ProcessRunnerTest(unittest.TestCase):
    def test_cancellation_terminates_process_group_quickly(self):
        async def cancel_long_process():
            runner = ProcessRunner()
            task = asyncio.create_task(
                runner.run(
                    job_id="cancel-test",
                    command=["/bin/sleep", "30"],
                    cwd=Path(tempfile.gettempdir()),
                    timeout_seconds=30,
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return

        started = time.monotonic()
        asyncio.run(cancel_long_process())
        self.assertLess(time.monotonic() - started, 5)

    def test_successful_process_writes_lifecycle_events_to_job_log(self):
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "jobs" / "process.log"

            async def run_process():
                return await ProcessRunner().run(
                    job_id="log-test",
                    command=["/bin/echo", "executor-log"],
                    cwd=Path(temp),
                    timeout_seconds=5,
                    log_path=log_path,
                )

            result = asyncio.run(run_process())
            self.assertEqual(result.exit_code, 0)
            content = log_path.read_text(encoding="utf-8")
            self.assertIn('"event":"process.start"', content)
            self.assertIn('"event":"process.spawned"', content)
            self.assertIn('"event":"process.complete"', content)
            self.assertIn('"job_id":"log-test"', content)
            self.assertIn('"exit_code":0', content)


if __name__ == "__main__":
    unittest.main()
