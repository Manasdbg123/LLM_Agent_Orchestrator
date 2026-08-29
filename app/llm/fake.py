"""A scripted provider for tests and the eval harness.

Not a mock in the usual sense — it implements the full port and produces real content
blocks, real tool_use ids, and real token/cost accounting. What it removes is
non-determinism and cost, so that a test asserting "the agent recovered from a
malformed tool call" fails when the *engine* breaks rather than when the model has an
off day.

The critical property is that it is a **pure function of the messages it is given**.
The turn it emits is chosen by counting assistant turns in the transcript, never by
internal call-count state. That mirrors the real provider closely enough to matter:
after a crash, a replacement worker rebuilds the transcript and gets the same answer,
so recovery tests exercise the engine rather than an artifact of the fake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.domain.errors import ErrorCode, RetryableError, TerminalError
from app.llm.base import LLMResponse, ToolSchema, ToolUse, Usage

#: Where a script is carried. The engine puts the run's input into the first user
#: message, so the provider can find its script without extra plumbing.
SCRIPT_MARKER = "<<script>>"


@dataclass(frozen=True, slots=True)
class Turn:
    """One scripted assistant turn."""

    #: Emit tool_use blocks for each (tool_name, input) pair.
    tools: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    #: Emit a final text answer and stop.
    text: str | None = None
    #: Emit a thinking block, to exercise verbatim replay.
    thinking: str | None = None
    #: Raise instead of responding, to exercise the retry classifier.
    raise_retryable: bool = False
    raise_terminal: bool = False
    #: Return stop_reason="refusal".
    refuse: bool = False
    input_tokens: int = 1000
    output_tokens: int = 200

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return cls(
            tools=[(t["name"], t.get("input") or {}) for t in data.get("tools", [])],
            text=data.get("text"),
            thinking=data.get("thinking"),
            raise_retryable=bool(data.get("raise_retryable")),
            raise_terminal=bool(data.get("raise_terminal")),
            refuse=bool(data.get("refuse")),
            input_tokens=int(data.get("input_tokens", 1000)),
            output_tokens=int(data.get("output_tokens", 200)),
        )


def encode_script(turns: list[dict[str, Any]]) -> str:
    """Embed a script in a task string."""
    return f"{SCRIPT_MARKER}{json.dumps(turns)}"


def _extract_script(messages: list[dict[str, Any]]) -> list[Turn] | None:
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if SCRIPT_MARKER in text:
            raw = text.split(SCRIPT_MARKER, 1)[1]
            return [Turn.from_dict(t) for t in json.loads(raw)]
    return None


def _assistant_turns(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        default_script: list[Turn] | None = None,
        *,
        model: str = "fake-model",
    ) -> None:
        self.default_script = default_script or [Turn(text="done")]
        self.model = model
        #: Observability for tests: every request seen, in order.
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str = "high",
    ) -> LLMResponse:
        self.calls.append(
            {
                "model": model,
                "message_count": len(messages),
                "tools": [t.name for t in tools],
                "effort": effort,
            }
        )

        script = _extract_script(messages) or self.default_script
        index = _assistant_turns(messages)
        # Running off the end of a script means the engine kept looping when the
        # script expected it to stop. Answer rather than crash, so the test failure
        # points at the loop, not at the fake.
        turn = script[index] if index < len(script) else Turn(text="script exhausted")

        if turn.raise_retryable:
            raise RetryableError(
                "fake provider: injected retryable failure", code=ErrorCode.UPSTREAM_UNAVAILABLE
            )
        if turn.raise_terminal:
            raise TerminalError(
                "fake provider: injected terminal failure", code=ErrorCode.INVALID_INPUT
            )

        content: list[dict[str, Any]] = []
        if turn.thinking:
            content.append({"type": "thinking", "thinking": turn.thinking, "signature": "fake-sig"})

        tool_uses: list[ToolUse] = []
        if turn.tools:
            for position, (name, payload) in enumerate(turn.tools):
                # Deterministic ids: a recovered step rebuilding this turn produces
                # the same tool_use ids, so tool_result pairing stays valid.
                use_id = f"toolu_fake_{index}_{position}"
                content.append({"type": "tool_use", "id": use_id, "name": name, "input": payload})
                tool_uses.append(ToolUse(id=use_id, name=name, input=payload))
            stop_reason = "tool_use"
            text = ""
        elif turn.refuse:
            stop_reason = "refusal"
            text = ""
        else:
            text = turn.text or "done"
            content.append({"type": "text", "text": text})
            stop_reason = "end_turn"

        usage = Usage(input_tokens=turn.input_tokens, output_tokens=turn.output_tokens)
        return LLMResponse(
            model=self.model,
            stop_reason=stop_reason,
            stop_details={"type": "refusal", "category": "fake"} if turn.refuse else None,
            content=content,
            text=text,
            tool_uses=tool_uses,
            usage=usage,
            latency_ms=1,
            request_id=f"req_fake_{index}",
            raw_request={"model": self.model, "message_count": len(messages)},
            raw_response={"stop_reason": stop_reason, "content": content},
        )


__all__ = ["SCRIPT_MARKER", "FakeProvider", "Turn", "encode_script"]
