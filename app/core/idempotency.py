"""The effect ledger.

This is the mechanism that turns Phase 2's visible duplicate into an at-most-once
effect. `tool_calls` is both the record of a tool invocation and the dedupe ledger —
one table, because the guarantee being made is exactly "this tool call happened
once", and splitting the record from the ledger would let the two disagree.

## The key

    idempotency_key = sha256(run_id | step_id | tool_name | canonical_json(args))

**`attempt` is deliberately absent.** The brief specified `(run_id, step_id,
attempt)`, and including `attempt` would defeat the entire feature: the dangerous
case is an *ambiguous* failure — the email was sent, then the connection dropped
before success was recorded — and on retry a key containing `attempt` is a
*different* key, so the ledger has no memory of the first attempt and the customer
gets two emails. The key must be stable across attempts for the dedupe to mean
anything. `attempt` is carried as metadata on the row instead.

Arguments are in the key so that a step whose arguments somehow differ is treated as
a different effect rather than silently deduplicated against an unrelated call.
Canonical JSON (sorted keys, no insignificant whitespace) makes the hash independent
of dict ordering.

## The protocol

    CLAIM    INSERT ... ON CONFLICT DO NOTHING
             inserted             -> we own it, execute
             conflict + committed -> return the stored result, do NOT re-execute
             conflict + in_flight -> AMBIGUOUS: resolve by the tool's effect policy
    EXECUTE  perform the effect (the only step that touches the outside world)
    COMMIT   mark committed, store the result

## What this does and does not guarantee

It does not provide exactly-once effects. Exactly-once across a process boundary and
an external service is not achievable without that service's cooperation. What it
guarantees is **at-most-once per effect key**, plus a machine-readable record of
every case where the outcome was genuinely unknowable.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.errors import AmbiguousEffectError, ErrorCode
from app.domain.models import ToolCall
from app.obs.logging import get_logger
from app.tools.base import EffectPolicy

log = get_logger("idempotency")


class EffectStatus(StrEnum):
    IN_FLIGHT = "in_flight"
    COMMITTED = "committed"
    #: An in-flight effect the engine decided not to resume (UNSAFE_TO_REPLAY).
    ABANDONED = "abandoned"


def canonical_json(value: Any) -> str:
    """Stable JSON: sorted keys, tight separators.

    Without this, `{"a":1,"b":2}` and `{"b":2,"a":1}` hash differently and the same
    effect gets two keys.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def effect_key(
    *, run_id: uuid.UUID, step_id: uuid.UUID, tool_name: str, arguments: dict[str, Any]
) -> str:
    """The dedupe key. Stable across attempts — see the module docstring."""
    payload = "|".join([str(run_id), str(step_id), tool_name, canonical_json(arguments)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ClaimOutcome(StrEnum):
    #: We inserted the row; nobody has performed this effect. Proceed.
    OWNED = "owned"
    #: Already committed; the stored result is returned and the tool is NOT re-run.
    ALREADY_COMMITTED = "already_committed"
    #: A previous attempt started and never finished. Outcome unknown.
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class Claim:
    outcome: ClaimOutcome
    tool_call_id: uuid.UUID
    idempotency_key: str
    #: Present when ALREADY_COMMITTED.
    result: dict[str, Any] | None = None
    is_error: bool = False

    @property
    def should_execute(self) -> bool:
        return self.outcome is ClaimOutcome.OWNED


async def claim_effect(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    step_id: uuid.UUID,
    tool_name: str,
    arguments: dict[str, Any],
    effect_policy: EffectPolicy,
    attempt: int,
    provider_tool_use_id: str | None = None,
) -> Claim:
    """Phase 1 of the protocol. Must be committed before the effect is performed."""
    key = effect_key(run_id=run_id, step_id=step_id, tool_name=tool_name, arguments=arguments)

    inserted = (
        await session.execute(
            pg_insert(ToolCall)
            .values(
                step_id=step_id,
                run_id=run_id,
                tool_name=tool_name,
                provider_tool_use_id=provider_tool_use_id,
                arguments=arguments,
                idempotency_key=key,
                effect_status=EffectStatus.IN_FLIGHT,
                effect_policy=str(effect_policy),
                attempt_observed=attempt,
            )
            .on_conflict_do_nothing(index_elements=[ToolCall.idempotency_key])
            .returning(ToolCall.id)
        )
    ).scalar_one_or_none()

    if inserted is not None:
        return Claim(ClaimOutcome.OWNED, tool_call_id=inserted, idempotency_key=key)

    existing = (
        await session.execute(
            sa.select(
                ToolCall.id, ToolCall.effect_status, ToolCall.result, ToolCall.is_error
            ).where(ToolCall.idempotency_key == key)
        )
    ).one()

    if existing.effect_status == EffectStatus.COMMITTED:
        log.info("effect_deduplicated", tool=tool_name, key=key[:16])
        return Claim(
            ClaimOutcome.ALREADY_COMMITTED,
            tool_call_id=existing.id,
            idempotency_key=key,
            result=existing.result,
            is_error=existing.is_error,
        )

    log.warning(
        "effect_ambiguous",
        tool=tool_name,
        key=key[:16],
        policy=str(effect_policy),
        prior_status=existing.effect_status,
    )
    return Claim(ClaimOutcome.AMBIGUOUS, tool_call_id=existing.id, idempotency_key=key)


def resolve_ambiguous(policy: EffectPolicy, *, tool_name: str) -> bool:
    """Decide whether an ambiguous effect may be re-executed.

    This is a policy question, not a technical one — we genuinely cannot know whether
    the effect landed — so each tool declares its answer up front and the engine
    obeys it rather than guessing.
    """
    if policy is EffectPolicy.SAFE_TO_REPLAY:
        # Re-running converges to the same state (upsert, pure computation, GET).
        return True
    if policy is EffectPolicy.REQUIRES_PROVIDER_KEY:
        # Safe *because* we re-send the same key and the provider deduplicates. If it
        # already performed the effect, it returns the original result instead.
        return True
    raise AmbiguousEffectError(
        f"tool {tool_name!r} may have already performed its effect and cannot be "
        "safely replayed; a human must confirm the outcome",
        code=ErrorCode.AMBIGUOUS_EFFECT,
        details={"tool": tool_name, "policy": str(policy)},
    )


async def commit_effect(
    session: AsyncSession,
    *,
    tool_call_id: uuid.UUID,
    result: dict[str, Any],
    is_error: bool = False,
) -> bool:
    """Phase 3. Guarded on `in_flight` so a late writer cannot overwrite a commit."""
    updated = (
        await session.execute(
            sa.update(ToolCall)
            .where(
                ToolCall.id == tool_call_id,
                ToolCall.effect_status == EffectStatus.IN_FLIGHT,
            )
            .values(
                effect_status=EffectStatus.COMMITTED,
                result=result,
                is_error=is_error,
                committed_at=sa.func.now(),
            )
            .returning(ToolCall.id)
        )
    ).scalar_one_or_none()
    return updated is not None


async def abandon_effect(
    session: AsyncSession, *, tool_call_id: uuid.UUID, error: dict[str, Any]
) -> None:
    """Mark an effect we refuse to resume, so it is visible for human review."""
    await session.execute(
        sa.update(ToolCall)
        .where(ToolCall.id == tool_call_id)
        .values(
            effect_status=EffectStatus.ABANDONED,
            error=error,
            is_error=True,
            committed_at=sa.func.now(),
        )
    )


__all__ = [
    "Claim",
    "ClaimOutcome",
    "EffectStatus",
    "abandon_effect",
    "canonical_json",
    "claim_effect",
    "commit_effect",
    "effect_key",
    "resolve_ambiguous",
]
