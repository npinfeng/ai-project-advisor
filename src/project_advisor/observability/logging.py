"""Context-local structured logging for concurrent Agent Runs."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_log_context: ContextVar[dict[str, str]] = ContextVar(
    "project_advisor_log_context",
    default={},
)


@contextmanager
def bind_log_context(**values: Any) -> Iterator[None]:
    """Temporarily add correlation fields without leaking across async tasks."""
    current = dict(_log_context.get())
    current.update({key: str(value) for key, value in values.items() if value})
    token = _log_context.set(current)
    try:
        yield
    finally:
        _log_context.reset(token)


def current_log_context() -> dict[str, str]:
    return dict(_log_context.get())


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit one machine-readable event with the active correlation context."""
    payload = {"event": event, **current_log_context(), **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))
