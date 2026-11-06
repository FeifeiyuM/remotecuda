"""SQLite persistence for workspaces, jobs, locks, and audit events."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import ACTIVE_STATUSES, JobMetadata, JobStatus, WorkspaceMetadata, utc_now


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.chmod(0o600)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        self._recover_after_restart()

    def _init_schema(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gpu_locks (
                    device_id INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    workspace_id TEXT,
                    job_id TEXT,
                    detail_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_workspace ON jobs(json_extract(metadata_json, '$.workspace_id'));
                """
            )

    def _recover_after_restart(self) -> None:
        with self._lock, self._db:
            rows = self._db.execute("SELECT job_id, metadata_json FROM jobs").fetchall()
            for row in rows:
                job = JobMetadata.model_validate_json(row["metadata_json"])
                if job.status in ACTIVE_STATUSES:
                    job.status = JobStatus.INTERRUPTED
                    job.finished_at = utc_now()
                    job.error = "agent restarted while job was active"
                    self._db.execute(
                        "UPDATE jobs SET metadata_json=? WHERE job_id=?",
                        (job.model_dump_json(), job.job_id),
                    )
            self._db.execute("DELETE FROM gpu_locks")

    def create_workspace(self, metadata: WorkspaceMetadata) -> None:
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM workspaces WHERE workspace_id=?", (metadata.workspace_id,)).fetchone():
                raise ValueError(f"workspace already exists: {metadata.workspace_id}")
            self._db.execute("INSERT INTO workspaces VALUES(?,?)", (metadata.workspace_id, metadata.model_dump_json()))

    def get_workspace(self, workspace_id: str) -> WorkspaceMetadata | None:
        row = self._db.execute("SELECT metadata_json FROM workspaces WHERE workspace_id=?", (workspace_id,)).fetchone()
        return WorkspaceMetadata.model_validate_json(row["metadata_json"]) if row else None

    def list_workspaces(self) -> list[WorkspaceMetadata]:
        rows = self._db.execute("SELECT metadata_json FROM workspaces ORDER BY workspace_id").fetchall()
        return [WorkspaceMetadata.model_validate_json(row["metadata_json"]) for row in rows]

    def update_workspace(self, metadata: WorkspaceMetadata) -> None:
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE workspaces SET metadata_json=? WHERE workspace_id=?",
                (metadata.model_dump_json(), metadata.workspace_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"workspace not found: {metadata.workspace_id}")

    def delete_workspace(self, workspace_id: str) -> None:
        if self.active_jobs(workspace_id):
            raise ValueError("workspace has active jobs")
        with self._lock, self._db:
            self._db.execute("DELETE FROM workspaces WHERE workspace_id=?", (workspace_id,))
            self._db.execute("DELETE FROM jobs WHERE workspace_id=?", (workspace_id,))

    def create_job(self, job: JobMetadata) -> JobMetadata:
        with self._lock, self._db:
            try:
                self._db.execute(
                    "INSERT INTO jobs(job_id,idempotency_key,metadata_json) VALUES(?,?,?)",
                    (job.job_id, job.idempotency_key, job.model_dump_json()),
                )
            except sqlite3.IntegrityError as exc:
                existing = self.get_job_by_idempotency_key(job.idempotency_key) if job.idempotency_key else None
                if existing:
                    raise ValueError("idempotent job already exists") from exc
                raise ValueError(f"job already exists: {job.job_id}") from exc
        return job

    def get_job(self, job_id: str) -> JobMetadata | None:
        row = self._db.execute("SELECT metadata_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return JobMetadata.model_validate_json(row["metadata_json"]) if row else None

    def get_job_by_idempotency_key(self, key: str | None) -> JobMetadata | None:
        if not key:
            return None
        row = self._db.execute("SELECT metadata_json FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
        return JobMetadata.model_validate_json(row["metadata_json"]) if row else None

    def list_jobs(self, workspace_id: str | None = None, active_only: bool = False) -> list[JobMetadata]:
        rows = self._db.execute("SELECT metadata_json FROM jobs").fetchall()
        jobs = [JobMetadata.model_validate_json(row["metadata_json"]) for row in rows]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        if workspace_id:
            jobs = [job for job in jobs if job.workspace_id == workspace_id]
        if active_only:
            jobs = [job for job in jobs if job.status in ACTIVE_STATUSES]
        return jobs

    def active_jobs(self, workspace_id: str | None = None) -> list[JobMetadata]:
        return self.list_jobs(workspace_id=workspace_id, active_only=True)

    def update_job(self, job: JobMetadata) -> JobMetadata:
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE jobs SET metadata_json=? WHERE job_id=?", (job.model_dump_json(), job.job_id)
            )
            if cur.rowcount == 0:
                raise ValueError(f"job not found: {job.job_id}")
        return job

    def acquire_gpu(self, device_id: int, job_id: str) -> bool:
        with self._lock, self._db:
            try:
                self._db.execute("INSERT INTO gpu_locks(device_id,job_id) VALUES(?,?)", (device_id, job_id))
            except sqlite3.IntegrityError:
                return False
        return True

    def release_gpu(self, device_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM gpu_locks WHERE device_id=?", (device_id,))

    def audit(self, event: str, *, workspace_id: str | None = None, job_id: str | None = None, **detail: Any) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO audit_log(timestamp,event,workspace_id,job_id,detail_json) VALUES(?,?,?,?,?)",
                (utc_now(), event, workspace_id, job_id, json.dumps(detail, sort_keys=True)),
            )

    def close(self) -> None:
        self._db.close()
