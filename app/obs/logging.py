"""Structured logging.

Context (run_id / step_id / attempt / worker_id) is bound into contextvars once, at
the top of a step's execution, so every log line emitted anywhere below it carries the
correlation keys without threading them through call signatures.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(*, level: str | None = None, json_output: bool | None = None) -> None:
    global _configured
    if _configured:
        return

    from app.config import settings

    lvl = (level or settings.log_level).upper()
    as_json = settings.log_json if json_output is None else json_output

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=lvl)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if as_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, lvl, logging.INFO)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


bind = structlog.contextvars.bind_contextvars
unbind = structlog.contextvars.unbind_contextvars
clear = structlog.contextvars.clear_contextvars


__all__ = ["bind", "clear", "configure_logging", "get_logger", "unbind"]
