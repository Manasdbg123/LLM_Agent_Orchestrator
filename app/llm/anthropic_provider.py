"""Anthropic adapter.

Three decisions here are load-bearing for the reliability layer:

**SDK retries are switched off (`max_retries=0`).** The SDK retries 429/5xx by
default, which would be a second, invisible retry loop nested inside a step that
already has one. That hides rate limiting from the state machine, makes step
durations unpredictable, and can outlive the lease — a worker still retrying inside
the SDK while the reaper has already handed its step to someone else. The engine owns
retry: it classifies the error, persists a backoff, and releases the lease.

**The manual loop, not the SDK's tool runner.** The tool runner would drive the whole
agent loop in memory, which is precisely the thing this project exists to make
durable. One call here is one turn is one committed step.

**Raw content blocks are returned and replayed verbatim.** Thinking blocks must go
back to the model unchanged on the same model, so the adapter never reduces the
response to text.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

from app.domain.errors import ErrorCode, RetryableError, TerminalError
from app.llm.base import LLMResponse, ToolSchema, ToolUse, Usage
from app.obs.logging import get_logger

log = get_logger("llm.anthropic")

#: Models whose thinking is adaptive and which accept the server-side fallback param.
_FALLBACK_CAPABLE = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")

#: Server-side refusal fallbacks. On a policy decline the API re-runs the request on
#: a fallback model inside the same call, which for an autonomous agent is the
#: difference between a recoverable hiccup and a dead run.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        timeout_seconds: float = 120.0,
        enable_fallbacks: bool = True,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            # See module docstring: the engine owns retry, not the SDK.
            max_retries=0,
            timeout=timeout_seconds,
        )
        self.enable_fallbacks = enable_fallbacks

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
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            # Adaptive thinking: `budget_tokens` is rejected on current models, and
            # depth is steered with `effort` instead.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if tools:
            request["tools"] = [t.to_wire() for t in tools]

        betas: list[str] = []
        if self.enable_fallbacks and model in _FALLBACK_CAPABLE:
            betas.append(_FALLBACK_BETA)
            request["fallbacks"] = "default"

        started = time.perf_counter()
        try:
            if betas:
                raw = await self._client.beta.messages.create(betas=betas, **request)
            else:
                raw = await self._client.messages.create(**request)
        except BaseException as exc:
            raise _translate(exc) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        payload = raw.model_dump(mode="json")
        content: list[dict[str, Any]] = payload.get("content") or []

        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        tool_uses = [
            # Inputs are parsed JSON from the SDK, never string-matched: current
            # models vary their JSON escaping and raw matching breaks on it.
            ToolUse(id=b["id"], name=b["name"], input=b.get("input") or {})
            for b in content
            if b.get("type") == "tool_use"
        ]

        usage_raw = payload.get("usage") or {}
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens") or 0,
            output_tokens=usage_raw.get("output_tokens") or 0,
            cache_read_tokens=usage_raw.get("cache_read_input_tokens") or 0,
            cache_write_tokens=usage_raw.get("cache_creation_input_tokens") or 0,
        )

        stop_reason = payload.get("stop_reason") or "end_turn"
        # `stop_details` is populated only for refusals and is None otherwise, so it
        # must be guarded rather than read unconditionally.
        stop_details = payload.get("stop_details") if stop_reason == "refusal" else None

        return LLMResponse(
            model=payload.get("model") or model,
            stop_reason=stop_reason,
            stop_details=stop_details,
            content=content,
            text=text,
            tool_uses=tool_uses,
            usage=usage,
            latency_ms=latency_ms,
            request_id=getattr(raw, "_request_id", None),
            raw_request=_redact(request),
            raw_response=payload,
        )


def _redact(request: dict[str, Any]) -> dict[str, Any]:
    """Persist the request's shape, not its full content.

    Whole conversations would bloat `llm_calls` and duplicate what the step timeline
    already holds. What is worth keeping is everything needed to explain a call's
    cost and behaviour.
    """
    return {
        "model": request.get("model"),
        "max_tokens": request.get("max_tokens"),
        "thinking": request.get("thinking"),
        "output_config": request.get("output_config"),
        "tools": [t["name"] for t in request.get("tools", [])],
        "message_count": len(request.get("messages", [])),
        "system_chars": len(request.get("system") or ""),
        "fallbacks": request.get("fallbacks"),
    }


def _translate(exc: BaseException) -> BaseException:
    """Map SDK exceptions onto the engine's retry taxonomy.

    Ordered most-specific first. A single broad `except APIStatusError` would erase
    the difference between "the service is busy, come back" and "this request is
    malformed and always will be" — and retrying the latter is how a bad prompt turns
    into a rate-limit incident.
    """
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except (TypeError, ValueError):
                retry_after = None
        return RetryableError(
            f"rate limited by Anthropic: {exc}",
            code=ErrorCode.RATE_LIMITED,
            retry_after_seconds=retry_after,
        )

    if isinstance(exc, anthropic.APITimeoutError):
        return RetryableError(f"Anthropic request timed out: {exc}", code=ErrorCode.TIMEOUT)

    if isinstance(exc, anthropic.APIConnectionError):
        return RetryableError(f"could not reach Anthropic: {exc}", code=ErrorCode.CONNECTION_ERROR)

    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        # Retrying will not conjure credentials.
        return TerminalError(f"Anthropic auth failed: {exc}", code=ErrorCode.UNAUTHORIZED)

    if isinstance(exc, anthropic.BadRequestError | anthropic.NotFoundError):
        return TerminalError(f"invalid Anthropic request: {exc}", code=ErrorCode.INVALID_INPUT)

    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return RetryableError(
                f"Anthropic server error {exc.status_code}: {exc}",
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
            )
        return TerminalError(
            f"Anthropic returned {exc.status_code}: {exc}", code=ErrorCode.INVALID_INPUT
        )

    return exc


__all__ = ["AnthropicProvider"]
