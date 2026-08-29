"""Deterministic fault injection.

Chaos tests that rely on `sleep` and hope are flaky and prove nothing. These faults
fire at an exact point in a step's execution, so "the worker died after the side
effect but before the result was committed" is a reproducible scenario rather than a
race the test occasionally wins.

Gated behind `AGENTORC_ENABLE_FAULT_INJECTION`; a fault requested without it is a
hard error, never a silent no-op, so a fault spec that leaks into a real deployment
is loud instead of invisible.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from app.config import settings
from app.domain.errors import ErrorCode, RetryableError, TerminalError
from app.obs.logging import get_logger

log = get_logger("faults")

#: Leases whose heartbeat has been deliberately suppressed, so they expire under the
#: worker while its process stays alive. Simulates a stalled (not crashed) worker,
#: which is the case that exercises fencing.
#:
#: Keyed by "<step_id>:<epoch>", not by step id: with several workers in one process
#: (as in the chaos suite) a step-id key would also suppress the heartbeat of the
#: worker that takes the step over, and recovery would never converge.
_heartbeat_suppressed: set[str] = set()


def make_lease_key(step_id: object, epoch: int) -> str:
    return f"{step_id}:{epoch}"


def heartbeat_suppressed(key: str) -> bool:
    return key in _heartbeat_suppressed


def suppress_heartbeat(key: str) -> None:
    _heartbeat_suppressed.add(key)


def clear_suppressions() -> None:
    _heartbeat_suppressed.clear()


class FaultSpec:
    """Parsed `fault` block from a step's input.

    Kinds:
      crash            -- kill the worker process outright (SIGKILL-equivalent)
      error            -- raise a retryable or terminal error
      hang             -- sleep, optionally without heartbeating (lease expiry)
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.kind: str = raw.get("kind", "")
        self.phase: str = raw.get("phase", "after_effect")
        self.times: int = int(raw.get("times", 1))
        self.retryable: bool = bool(raw.get("retryable", True))
        self.seconds: float = float(raw.get("seconds", 0))
        self.suppress_heartbeat: bool = bool(raw.get("suppress_heartbeat", False))
        self.exit_code: int = int(raw.get("exit_code", 137))


def parse(raw: Any) -> FaultSpec | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise TerminalError("fault spec must be an object", code=ErrorCode.INVALID_INPUT)
    if not settings.enable_fault_injection:
        raise TerminalError(
            "step requested fault injection but AGENTORC_ENABLE_FAULT_INJECTION is off",
            code=ErrorCode.INVALID_INPUT,
        )
    return FaultSpec(raw)


async def apply(fault: FaultSpec | None, *, phase: str, attempt: int, lease_key: str) -> None:
    """Fire the fault if this is its phase. Some of these do not return."""
    if fault is None or fault.phase != phase:
        return

    if fault.kind == "error":
        if attempt <= fault.times:
            log.warning("injecting_error", phase=phase, attempt=attempt, lease=lease_key)
            exc_type = RetryableError if fault.retryable else TerminalError
            raise exc_type(
                f"injected {'retryable' if fault.retryable else 'terminal'} fault "
                f"at {phase} (attempt {attempt})",
                code=ErrorCode.INJECTED_FAULT,
            )
        return

    if fault.kind == "hang":
        # Bounded by `times` like the error fault: without it, the worker that takes
        # the step over after recovery would hang too, and the run would never finish.
        if attempt > fault.times:
            return
        if fault.suppress_heartbeat:
            suppress_heartbeat(lease_key)
        log.warning(
            "injecting_hang",
            phase=phase,
            seconds=fault.seconds,
            heartbeat_suppressed=fault.suppress_heartbeat,
            lease=lease_key,
        )
        await asyncio.sleep(fault.seconds)
        return

    if fault.kind == "crash":
        if attempt > fault.times:
            return
        log.critical("injecting_crash", phase=phase, lease=lease_key, attempt=attempt)
        # os._exit, not sys.exit: no atexit hooks, no finally blocks, no graceful
        # lease release. That is the point — it must be indistinguishable from
        # `kill -9`, so recovery has to come from lease expiry rather than cleanup.
        os._exit(fault.exit_code)

    raise TerminalError(f"unknown fault kind {fault.kind!r}", code=ErrorCode.INVALID_INPUT)


__all__ = [
    "FaultSpec",
    "apply",
    "clear_suppressions",
    "fault_for_tool",
    "heartbeat_suppressed",
    "make_lease_key",
    "parse",
    "suppress_heartbeat",
]


def fault_for_tool(run_input: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    """Fault spec a run declares for a given tool, if any.

    A tool_call step's input is generated by the model, so there is nowhere for a
    test to attach a fault to it. Instead a run may declare
    `{"tool_faults": {"send_email": {...}}}` and the agent loop attaches the matching
    spec when it creates the step.

    Returns None unless fault injection is enabled, so this channel is inert in any
    deployment that has not explicitly opted in.
    """
    if not settings.enable_fault_injection:
        return None
    faults = run_input.get("tool_faults")
    if not isinstance(faults, dict):
        return None
    spec = faults.get(tool_name)
    return spec if isinstance(spec, dict) else None
