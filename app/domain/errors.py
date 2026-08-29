"""Error taxonomy.

The engine never does `except Exception: retry`. Every failure is classified into
exactly one of RETRYABLE / TERMINAL / AMBIGUOUS, and the classification decides what
the state machine does next. Unclassified exceptions are TERMINAL on purpose: an
unknown failure that silently retries is how a bug becomes an outage.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any


class ErrorClass(StrEnum):
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    #: The side effect may or may not have happened. Resolution is per-tool policy,
    #: never a blind retry. See DESIGN.md section 5.2.
    AMBIGUOUS = "ambiguous"


class ErrorCode(StrEnum):
    # infrastructure / transient
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    DB_SERIALIZATION_FAILURE = "db_serialization_failure"
    # request / logic
    INVALID_INPUT = "invalid_input"
    UNKNOWN_TOOL = "unknown_tool"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    UNAUTHORIZED = "unauthorized"
    ILLEGAL_TRANSITION = "illegal_transition"
    # engine lifecycle
    LEASE_LOST = "lease_lost"
    TOO_MANY_RECOVERIES = "too_many_recoveries"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    STEP_FAILED = "step_failed"
    CANCELLED = "cancelled"
    # guardrails
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    # effects
    AMBIGUOUS_EFFECT = "ambiguous_effect"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    # catch-all
    INTERNAL = "internal"
    INJECTED_FAULT = "injected_fault"


class EngineError(Exception):
    """Base for everything the engine raises deliberately."""

    error_class: ErrorClass = ErrorClass.TERMINAL
    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}
        self.retry_after_seconds = retry_after_seconds

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": str(self.code),
            "class": str(self.error_class),
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


class RetryableError(EngineError):
    error_class = ErrorClass.RETRYABLE
    code = ErrorCode.INTERNAL


class TerminalError(EngineError):
    error_class = ErrorClass.TERMINAL
    code = ErrorCode.INTERNAL


class AmbiguousEffectError(EngineError):
    """Raised when we cannot tell whether an external side effect landed."""

    error_class = ErrorClass.AMBIGUOUS
    code = ErrorCode.AMBIGUOUS_EFFECT


class IllegalTransitionError(TerminalError):
    """An attempt to move an entity along an edge the state machine does not have.

    This is always a programming error, never a runtime condition, so it is terminal
    and loud rather than retried.
    """

    code = ErrorCode.ILLEGAL_TRANSITION


class LeaseLostError(EngineError):
    """This worker no longer owns the step it is executing: it has been fenced.

    Not retryable *by this worker* — someone else already owns the work. The worker
    discards its result and does not ack.
    """

    error_class = ErrorClass.TERMINAL
    code = ErrorCode.LEASE_LOST


class CancelledError(EngineError):
    error_class = ErrorClass.TERMINAL
    code = ErrorCode.CANCELLED


#: Exception types from the standard library / drivers that are transient by nature.
_RETRYABLE_BUILTINS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    asyncio.TimeoutError,
    OSError,  # includes socket errors; superset of ConnectionError, kept last
)


def classify(exc: BaseException) -> ErrorClass:
    """Map any exception onto a retry class.

    Order matters: an `EngineError` carries its own verdict and wins over the
    structural checks below it.
    """
    if isinstance(exc, EngineError):
        return exc.error_class
    # psycopg raises SerializationFailure/DeadlockDetected for concurrency losses;
    # matched by name to avoid importing the driver into the domain layer.
    name = type(exc).__name__
    if name in {"SerializationFailure", "DeadlockDetected", "OperationalError"}:
        return ErrorClass.RETRYABLE
    if isinstance(exc, _RETRYABLE_BUILTINS):
        return ErrorClass.RETRYABLE
    return ErrorClass.TERMINAL


def code_of(exc: BaseException) -> ErrorCode:
    if isinstance(exc, EngineError):
        return exc.code
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ErrorCode.TIMEOUT
    if isinstance(exc, ConnectionError):
        return ErrorCode.CONNECTION_ERROR
    return ErrorCode.INTERNAL


def error_to_json(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, EngineError):
        return exc.to_json()
    return {
        "code": str(code_of(exc)),
        "class": str(classify(exc)),
        "message": f"{type(exc).__name__}: {exc}",
    }


def retry_after_of(exc: BaseException) -> float | None:
    return exc.retry_after_seconds if isinstance(exc, EngineError) else None


__all__ = [
    "AmbiguousEffectError",
    "CancelledError",
    "EngineError",
    "ErrorClass",
    "ErrorCode",
    "IllegalTransitionError",
    "LeaseLostError",
    "RetryableError",
    "TerminalError",
    "classify",
    "code_of",
    "error_to_json",
    "retry_after_of",
]
