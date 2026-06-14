"""Structured JSON logging for agent observability."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

_RESERVED_RECORD_KEYS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line — suitable for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        run_id = _run_id.get()
        if run_id:
            entry["run_id"] = run_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                entry[key] = value

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def setup_logging(*, level: int = logging.INFO, stream: Any = None) -> None:
    """Configure root logger with JSON output to stderr."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def set_run_id(run_id: str) -> None:
    _run_id.set(run_id)


def clear_run_id() -> None:
    _run_id.set(None)
