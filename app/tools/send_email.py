"""Send an email through a mock provider that implements real dedupe semantics.

The provider is mocked; its *contract* is not. It behaves the way Stripe, SendGrid
and Postmark behave: you pass an `Idempotency-Key`, it stores the key with the first
response, and any later request carrying that key returns the original result without
performing the effect again.

That fidelity is the entire point. This project claims "we cooperate correctly with
an idempotent downstream", and the only way to demonstrate that is against a
downstream that actually deduplicates. A mock that just appends to a list would let
a broken implementation pass.

Sending is the one effect here that cannot be made naturally idempotent — there is
no natural key to upsert on — so it is classified `REQUIRES_PROVIDER_KEY`, and the
recovery path depends on the provider honouring the key rather than on the engine
being clever.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar

import sqlalchemy as sa
from pydantic import Field
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import session_scope
from app.domain.models import EmailOutbox
from app.obs.logging import get_logger
from app.tools.base import BaseTool, EffectPolicy, ToolArgs, ToolContext, ToolResult

log = get_logger("tool.send_email")


@dataclass(frozen=True, slots=True)
class SendResult:
    message_id: str
    #: False when the provider recognised the key and returned the original send.
    #: This is the flag the crash test asserts on.
    delivered_now: bool


class MockEmailProvider:
    """Stands in for an SMTP/API provider, with a persistent dedupe store.

    The store is a table rather than an in-memory dict on purpose: after a worker is
    killed, the *replacement process* must be able to see that the first attempt
    already sent. An in-memory store would die with the worker and the second attempt
    would send a second email — passing a test that proves nothing.
    """

    async def send(
        self,
        *,
        idempotency_key: str,
        to: str,
        subject: str,
        body: str,
        run_id: object,
        step_id: object,
    ) -> SendResult:
        message_id = "msg_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]

        async with session_scope() as session:
            # INSERT ... ON CONFLICT DO NOTHING is the dedupe. If a row already
            # exists for this key, no send happens and the stored message id is
            # returned, exactly as a real provider would.
            inserted = (
                await session.execute(
                    pg_insert(EmailOutbox)
                    .values(
                        idempotency_key=idempotency_key,
                        message_id=message_id,
                        recipient=to,
                        subject=subject,
                        body=body,
                        run_id=run_id,
                        step_id=step_id,
                    )
                    .on_conflict_do_nothing(index_elements=[EmailOutbox.idempotency_key])
                    .returning(EmailOutbox.message_id)
                )
            ).scalar_one_or_none()

            if inserted is not None:
                log.info("email_sent", message_id=inserted, recipient=to)
                return SendResult(message_id=inserted, delivered_now=True)

            existing = (
                await session.execute(
                    sa.select(EmailOutbox.message_id).where(
                        EmailOutbox.idempotency_key == idempotency_key
                    )
                )
            ).scalar_one()
            # Also record that a duplicate was suppressed, so the demo can show the
            # provider doing the work rather than merely assert that it did.
            await session.execute(
                sa.update(EmailOutbox)
                .where(EmailOutbox.idempotency_key == idempotency_key)
                .values(duplicate_attempts=EmailOutbox.duplicate_attempts + 1)
            )
            log.warning("email_deduplicated", message_id=existing, recipient=to)
            return SendResult(message_id=existing, delivered_now=False)


class SendEmailArgs(ToolArgs):
    to: str = Field(..., min_length=3, max_length=200, description="Recipient address")
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=5000)


class SendEmailTool(BaseTool):
    name: ClassVar[str] = "send_email"
    description: ClassVar[str] = (
        "Send an email. This has a real external effect and cannot be undone, so it "
        "requires human approval before it runs."
    )
    args_model: ClassVar[type[ToolArgs]] = SendEmailArgs
    effect_policy: ClassVar[EffectPolicy] = EffectPolicy.REQUIRES_PROVIDER_KEY
    #: Gate enforced in Phase 4; declared now so the tool states its own risk.
    requires_approval: ClassVar[bool] = True
    timeout_seconds: ClassVar[float] = 20.0
    #: Deliberately fewer attempts than the default. Every retry of a send is a
    #: round trip through the dedupe path; if that is not working, retrying harder
    #: makes the blast radius bigger rather than smaller.
    max_attempts: ClassVar[int] = 2

    def __init__(self, provider: MockEmailProvider | None = None) -> None:
        self.provider = provider or MockEmailProvider()

    async def execute(self, ctx: ToolContext, args: SendEmailArgs) -> ToolResult:
        # The step's idempotency key is passed straight through as the provider's
        # dedupe key. It is stable across attempts, which is what makes a retry after
        # an ambiguous failure safe.
        result = await self.provider.send(
            idempotency_key=ctx.idempotency_key,
            to=args.to,
            subject=args.subject,
            body=args.body,
            run_id=ctx.run_id,
            step_id=ctx.step_id,
        )
        note = "" if result.delivered_now else " (duplicate suppressed by provider)"
        return ToolResult(
            content=f"Email sent to {args.to}, id {result.message_id}{note}.",
            data={
                "message_id": result.message_id,
                "recipient": args.to,
                "delivered_now": result.delivered_now,
            },
        )


__all__ = ["MockEmailProvider", "SendEmailArgs", "SendEmailTool", "SendResult"]
