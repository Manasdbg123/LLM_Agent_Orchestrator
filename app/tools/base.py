"""Tool contract.

Every tool declares how it behaves under replay. That declaration is not
documentation — it is what the effect ledger consults when a step is re-executed
after a crash and finds a previous attempt that started but never committed
(`app.core.idempotency`). A tool that cannot say what happens on replay cannot be
made safe by the engine, so the field is required rather than defaulted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from app.llm.base import ToolSchema


class EffectPolicy(StrEnum):
    """What the engine may do with an effect whose outcome is unknown."""

    #: Naturally idempotent: pure computation, a GET, or an upsert on a natural key.
    #: Re-running it produces the same state, so an ambiguous attempt is simply redone.
    SAFE_TO_REPLAY = "safe_to_replay"

    #: The downstream service deduplicates on a key we pass it (Stripe-style
    #: `Idempotency-Key`). We re-send with the same key and the provider returns the
    #: original result instead of performing the effect twice.
    REQUIRES_PROVIDER_KEY = "requires_provider_key"

    #: Fire-and-forget with no dedupe available. On an ambiguous outcome the engine
    #: refuses to guess: the step fails with `ambiguous_effect` and a human decides.
    UNSAFE_TO_REPLAY = "unsafe_to_replay"


class ToolArgs(BaseModel):
    """Base for tool argument models.

    `extra="forbid"` makes Pydantic emit `additionalProperties: false`, which is what
    lets us send the tool with `strict: true` and have the API guarantee that
    `tool_use.input` validates. Validation still runs on our side — `strict` covers
    the schema, not the semantics (a well-formed expression can still be nonsense).
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool may know about the call it is serving."""

    run_id: uuid.UUID
    step_id: uuid.UUID
    #: Stable across retries of this step. Side-effecting tools pass it downstream.
    idempotency_key: str
    #: Metadata only. Deliberately NOT part of the idempotency key — see DESIGN.md
    #: section 5.2 for why including it would defeat the whole mechanism.
    attempt: int
    worker_id: str


@dataclass(slots=True)
class ToolResult:
    """What the model sees, plus what we record."""

    #: Rendered into the `tool_result` block the model reads next turn.
    content: str
    #: Sets `is_error` on that block. An errored tool result is a normal, recoverable
    #: event: the model reads the message and adapts. It is not a step failure.
    is_error: bool = False
    #: Structured payload persisted on the `tool_calls` row for the dashboard/audit.
    data: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[ToolArgs]]
    effect_policy: ClassVar[EffectPolicy]
    #: Wired into the approval gate in Phase 4. Declared here so tools can state
    #: their risk now rather than being retrofitted later.
    requires_approval: ClassVar[bool]
    timeout_seconds: ClassVar[float]

    async def execute(self, ctx: ToolContext, args: Any) -> ToolResult: ...


class BaseTool:
    """Convenience base supplying schema generation and defaults."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_model: ClassVar[type[ToolArgs]]
    effect_policy: ClassVar[EffectPolicy] = EffectPolicy.SAFE_TO_REPLAY
    requires_approval: ClassVar[bool] = False
    timeout_seconds: ClassVar[float] = 30.0
    #: Retry budget for a step running this tool, overriding the engine default.
    #: A tool that reaches a flaky network deserves more attempts than one that
    #: cannot fail transiently.
    max_attempts: ClassVar[int] = 3

    @classmethod
    def approval_reason(cls, arguments: dict[str, Any]) -> str | None:
        """Why this specific call needs a human, or None if it does not.

        Taking the arguments rather than being a flat boolean is what lets approval
        be *data-dependent*: `database_write` gates on which namespace is being
        written to, not on the fact that it is a write. A blanket per-tool flag would
        force a choice between gating every write or none of them.
        """
        return "tool is marked as requiring approval" if cls.requires_approval else None

    @classmethod
    def schema(cls) -> ToolSchema:
        schema = cls.args_model.model_json_schema()
        # The API requires both for strict mode; Pydantic emits `required` only when
        # a model has required fields, so an all-optional tool needs the key added.
        schema.setdefault("additionalProperties", False)
        schema.setdefault("required", [])
        schema.pop("title", None)
        return ToolSchema(
            name=cls.name, description=cls.description, input_schema=schema, strict=True
        )

    async def execute(self, ctx: ToolContext, args: Any) -> ToolResult:
        raise NotImplementedError


__all__ = [
    "BaseTool",
    "EffectPolicy",
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolResult",
]
