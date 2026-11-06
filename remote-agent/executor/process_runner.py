"""Controlled process execution with timeout and cancellation.

The operating-system wait runs in a worker thread. This keeps the FastAPI event
loop responsive and avoids tying child-process reaping to a particular asyncio
child watcher implementation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .logging_utils import log_event


LOGGER = logging.getLogger("remote-cuda-agent.executor.process")


class ProcessTimeout(RuntimeError):
    pass


class ProcessFailure(RuntimeError):
    pass


@dataclass
class ProcessResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


class ProcessRunner:
    def __init__(self, max_output_chars: int = 1_000_000) -> None:
        self.max_output_chars = max_output_chars
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._process_lock = threading.Lock()

    def _run_sync(
        self,
        *,
        job_id: str,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None,
        log_path: Path | None,
        cancel_event: threading.Event,
    ) -> ProcessResult:
        started = time.perf_counter()
        if cancel_event.is_set():
            log_event(
                log_path,
                LOGGER,
                logging.WARNING,
                "process.cancelled_before_start",
                job_id=job_id,
                command=command,
            )
            raise ProcessFailure("process cancelled before start")

        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "process.start",
            job_id=job_id,
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            environment_keys=sorted(env.keys()) if env else [],
        )
        timed_out = False
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._child_environment(env),
                start_new_session=True,
            )
        except BaseException as exc:
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "process.spawn_failed",
                job_id=job_id,
                command=command,
                cwd=cwd,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=time.perf_counter() - started,
            )
            raise

        with self._process_lock:
            self._processes[job_id] = process
        log_event(
            log_path,
            LOGGER,
            logging.INFO,
            "process.spawned",
            job_id=job_id,
            pid=process.pid,
        )

        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            log_event(
                log_path,
                LOGGER,
                logging.WARNING,
                "process.timeout",
                job_id=job_id,
                pid=process.pid,
                timeout_seconds=timeout_seconds,
            )
            self._signal_process(process, signal.SIGKILL)
            stdout_bytes, stderr_bytes = process.communicate()
            log_event(
                log_path,
                LOGGER,
                logging.WARNING,
                "process.killed_after_timeout",
                job_id=job_id,
                pid=process.pid,
            )
        finally:
            with self._process_lock:
                self._processes.pop(job_id, None)

        duration = time.perf_counter() - started
        stdout = self._truncate(stdout_bytes.decode("utf-8", "replace"))
        stderr = self._truncate(stderr_bytes.decode("utf-8", "replace"))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(command)}\n{stdout}\n--- stderr ---\n{stderr}\n")

        result = ProcessResult(command, process.returncode or 0, stdout, stderr, duration, timed_out)
        finish_event = "process.timeout_complete" if timed_out else "process.complete"
        finish_level = logging.WARNING if timed_out or process.returncode else logging.INFO
        log_event(
            log_path,
            LOGGER,
            finish_level,
            finish_event,
            job_id=job_id,
            pid=process.pid,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            stdout_chars=len(result.stdout),
            stderr_chars=len(result.stderr),
            timed_out=result.timed_out,
        )
        if timed_out:
            raise ProcessTimeout(f"process timed out after {timeout_seconds} seconds")
        if process.returncode != 0:
            raise ProcessFailure(
                f"process exited with code {process.returncode}: {stderr[-2000:] or stdout[-2000:]}"
            )
        return result

    async def run(
        self,
        *,
        job_id: str,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
        log_path: Path | None = None,
    ) -> ProcessResult:
        if not command or not os.path.isabs(command[0]):
            log_event(
                log_path,
                LOGGER,
                logging.ERROR,
                "process.rejected",
                job_id=job_id,
                command=command,
                cwd=cwd,
                reason="command must start with an absolute executable path",
            )
            raise ProcessFailure("process command must start with an absolute executable path")

        cancel_event = threading.Event()
        future: Future[ProcessResult] = Future()

        def thread_main() -> None:
            try:
                future.set_result(
                    self._run_sync(
                        job_id=job_id,
                        command=command,
                        cwd=cwd,
                        timeout_seconds=timeout_seconds,
                        env=env,
                        log_path=log_path,
                        cancel_event=cancel_event,
                    )
                )
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(
            target=thread_main,
            name=f"cuda-process-{job_id}",
            daemon=True,
        ).start()
        try:
            while not future.done():
                await asyncio.sleep(0.02)
            return future.result()
        except asyncio.CancelledError:
            cancel_event.set()
            log_event(
                log_path,
                LOGGER,
                logging.WARNING,
                "process.cancel_requested",
                job_id=job_id,
                command=command,
            )
            terminated = self.terminate(job_id)
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                pass
            killed = False
            if not future.done():
                killed = self.kill(job_id)
            while not future.done():
                await asyncio.sleep(0.02)
            log_event(
                log_path,
                LOGGER,
                logging.WARNING,
                "process.cancel_complete",
                job_id=job_id,
                terminated=terminated,
                killed=killed,
            )
            raise

    def terminate(self, job_id: str) -> bool:
        return self._signal_job(job_id, signal.SIGTERM)

    def kill(self, job_id: str) -> bool:
        return self._signal_job(job_id, signal.SIGKILL)

    def _signal_job(self, job_id: str, sig: signal.Signals) -> bool:
        with self._process_lock:
            process = self._processes.get(job_id)
        if process is None or process.poll() is not None:
            return False
        self._signal_process(process, sig)
        return True

    @staticmethod
    def _child_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
        """Build a minimal child environment without agent secrets."""

        allowed = {"PATH", "LANG", "LC_ALL", "TMPDIR", "TERM", "NVCC_CCBIN"}
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.setdefault("PATH", "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin")
        environment.setdefault("LANG", "C.UTF-8")
        environment.update(overrides or {})
        return environment

    @staticmethod
    def _signal_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        """Signal the child's process group so build-tool descendants also stop."""

        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            pass

    def _truncate(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        suffix = f"\n... output truncated at {self.max_output_chars} characters ...\n"
        return value[: self.max_output_chars] + suffix
