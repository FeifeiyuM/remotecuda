"""Structured logging helpers for executor lifecycle events."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import Any


def log_event(
    log_path: Path | None,
    logger: Logger,
    level: int,
    event: str,
    *,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Write one structured event to the service logger and optional job log."""

    payload = {"event": event, **fields}
    if message is not None:
        payload["message"] = message
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    logger.log(level, serialized)
    if log_path is None:
        return

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    level_name = logging.getLevelName(level)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {level_name} {serialized}\n")
