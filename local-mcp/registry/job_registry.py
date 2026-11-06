"""Persistent cross-server job index."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from registry.models import JobRecord, JobStatus, JobType, utc_now


class JobRegistryError(RuntimeError):
    pass


class JobRegistry:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.touch(mode=0o600, exist_ok=True)
        self.database_path.chmod(0o600)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.database_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_server ON jobs(json_extract(record_json, '$.server_id'));
                """
            )

    def create(
        self,
        *,
        job_id: str,
        remote_job_id: str,
        server_id: str,
        workspace_id: str,
        job_type: JobType | str,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        try:
            item = JobRecord(
                job_id=job_id,
                remote_job_id=remote_job_id,
                server_id=server_id,
                workspace_id=workspace_id,
                job_type=JobType(job_type),
                idempotency_key=idempotency_key,
            )
        except (ValidationError, ValueError) as exc:
            raise JobRegistryError(f"invalid job: {exc}") from exc
        with self._lock, self._db:
            try:
                self._db.execute(
                    "INSERT INTO jobs(job_id,idempotency_key,record_json) VALUES(?,?,?)",
                    (item.job_id, item.idempotency_key, item.model_dump_json()),
                )
            except sqlite3.IntegrityError as exc:
                existing = self.get_by_idempotency_key(item.idempotency_key) if item.idempotency_key else None
                if existing:
                    raise JobRegistryError("duplicate idempotency key") from exc
                raise JobRegistryError(f"job already exists: {item.job_id}") from exc
        return item

    def get(self, job_id: str) -> JobRecord | None:
        row = self._db.execute("SELECT record_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return JobRecord.model_validate_json(row["record_json"]) if row else None

    def require(self, job_id: str) -> JobRecord:
        item = self.get(job_id)
        if not item:
            raise JobRegistryError(f"job not found: {job_id}")
        return item

    def get_by_idempotency_key(self, key: str | None) -> JobRecord | None:
        if not key:
            return None
        row = self._db.execute("SELECT record_json FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
        return JobRecord.model_validate_json(row["record_json"]) if row else None

    def list(
        self,
        server_id: str | None = None,
        workspace_id: str | None = None,
        active_only: bool = False,
    ) -> list[JobRecord]:
        rows = self._db.execute("SELECT record_json FROM jobs").fetchall()
        result = [JobRecord.model_validate_json(row["record_json"]) for row in rows]
        result.sort(key=lambda item: item.created_at, reverse=True)
        result = [item for item in result if server_id in (None, item.server_id)]
        result = [item for item in result if workspace_id in (None, item.workspace_id)]
        if active_only:
            result = [item for item in result if item.status in {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING}]
        return result

    def update_status(self, job_id: str, status: JobStatus | str) -> JobRecord:
        item = self.require(job_id)
        item.status = JobStatus(status)
        item.updated_at = utc_now()
        with self._lock, self._db:
            self._db.execute("UPDATE jobs SET record_json=? WHERE job_id=?", (item.model_dump_json(), job_id))
        return item

    def has_active_jobs(self, server_id: str) -> bool:
        return bool(self.list(server_id=server_id, active_only=True))

    def close(self) -> None:
        self._db.close()
