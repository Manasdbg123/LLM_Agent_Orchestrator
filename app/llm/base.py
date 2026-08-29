"""The LLM provider port.

Everything above this layer speaks in these types, so swapping Anthropic for another
provider is one adapter. Two things about the shape are deliberate:

1. `LLMResponse.content` carries the provider's **raw content blocks**, not a parsed
   summary. Thinking blocks in particular must be replayed to the model byte-identical
   on the next turn, so a lossy internal representation would silently degrade the
   model's reasoning across a multi-step run. We persist the raw blocks and replay
   them verbatim.

2. Providers raise our own `RetryableError` / `TerminalError`, never SDK-specific
   exceptions. The engine's retry classifier (`app.core.retry`) then works unchanged
   regardless of who is behind the port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolUse:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    model: str
    #: "end_turn" | "tool_use" | "max_tokens" | "refusal" | ...
    stop_reason: str
    #: Raw provider content blocks, replayed verbatim on the next turn.
    content: list[dict[str, Any]]
    text: str
    tool_uses: list[ToolUse]
    usage: Usage
    latency_ms: int
    #: Populated only when stop_reason == "refusal".
    stop_details: dict[str, Any] | None = None
    request_id: str | None = None
    #: Redacted request/response, persisted to `llm_calls` as debugging evidence.
    raw_request: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_uses)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the provider needs to see it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    #: `strict: true` guarantees `tool_use.input` validates against the schema, which
    #: removes a whole class of malformed-argument handling. Requires the schema to
    #: set `additionalProperties: false` and list `required`.
    strict: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": self.strict,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """Ports are narrow on purpose: one call, no session state.

    Conversation state lives in Postgres and is rebuilt per turn, so a provider
    implementation never needs memory of its own — which is what lets a step be
    retried on a different worker without changing its meaning.
    """

    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str = "high",
    ) -> LLMResponse: ...


__all__ = ["LLMProvider", "LLMResponse", "ToolSchema", "ToolUse", "Usage"]
