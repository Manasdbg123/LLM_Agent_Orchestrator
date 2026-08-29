"""Handler contract.

A handler is given a claimed step and returns what should happen next. It never
writes a status column and never decides retry policy — that is the executor's job.
Keeping handlers free of state-machine concerns is what lets Phase 3 add LLM and tool
handlers without touching the reliability layer.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.core.leases import Lease
from app.domain.models import AgentRun
from app.domain.states import StepKind


@dataclass(frozen=True, slots=True)
class NextStep:
    """A step to create once the current one is committed as succeeded."""

    kind: StepKind
    input: dict[str, Any] = field(default_factory=dict)
    parent_step_id: Any = None
    #: Seconds to delay before the step becomes runnable.
    delay_seconds: float = 0.0
    #: Retry budget for the new step, from the tool's declaration.
    max_attempts: int | None = None
    #: Why this step needs a human before it may run. None means it may run freely.
    #: Set from `Tool.approval_reason(arguments)`, so the gate can depend on the
    #: arguments rather than only on which tool was chosen.
    approval_reason: str | None = None
    #: Tool identity, needed to record the approval request.
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepContext:
    """Everything a handler is allowed to know.

    Deliberately carries no open transaction: handlers that need the database open
    their own short transaction. A handler that ran inside the executor's commit
    transaction would hold it open for the entire duration of an LLM call.
    """

    run: AgentRun
    lease: Lease
    worker_id: str

    @property
    def run_id(self) -> Any:
        return self.run.id

    @property
    def step_id(self) -> Any:
        return self.lease.step_id

    @property
    def input(self) -> dict[str, Any]:
        return self.lease.input


@dataclass(frozen=True, slots=True)
class HandlerResult:
    """What the handler produced.

    `next_step` and `run_output` are mutually exclusive in practice: a handler either
    continues the run or ends it.
    """

    output: dict[str, Any] = field(default_factory=dict)
    next_step: NextStep | None = None
    #: Set by a terminal handler (finalize) to close the run successfully.
    run_output: dict[str, Any] | None = None


Handler = Callable[[StepContext], Coroutine[Any, Any, HandlerResult]]

_REGISTRY: dict[StepKind, Handler] = {}


def register(kind: StepKind) -> Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        if kind in _REGISTRY:
            raise RuntimeError(f"handler for {kind} already registered")
        _REGISTRY[kind] = fn
        return fn

    return decorator


def get_handler(kind: StepKind) -> Handler:
    try:
        return _REGISTRY[kind]
    except KeyError:
        from app.domain.errors import ErrorCode, TerminalError

        # Unknown kind is a deployment error (a worker older than the API that
        # created the step). Terminal, not retryable: retrying will not upgrade it.
        raise TerminalError(
            f"no handler registered for step kind {kind!r}",
            code=ErrorCode.INVALID_INPUT,
        ) from None


__all__ = [
    "Handler",
    "HandlerResult",
    "NextStep",
    "StepContext",
    "get_handler",
    "register",
]
