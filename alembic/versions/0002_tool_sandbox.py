"""tool sandbox tables

Targets for the Phase 3 side-effecting tools:

* `sandbox_records` - what `database_write` upserts into. The composite primary key
  is what makes that tool honestly SAFE_TO_REPLAY.
* `email_outbox` - the mock email provider's dedupe store. The unique constraint on
  `idempotency_key` IS the dedupe, and it lives in Postgres so a worker that replaces
  a killed one can see that the first attempt already sent.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sandbox_records",
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("writes", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "run_id", sa.Uuid(), sa.ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "written_by_step_id",
            sa.Uuid(),
            sa.ForeignKey("steps.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("namespace", "key", name="pk_sandbox_records"),
    )
    op.create_index("ix_sandbox_records_run", "sandbox_records", ["run_id"])

    op.create_table(
        "email_outbox",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("recipient", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "duplicate_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "run_id", sa.Uuid(), sa.ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "step_id", sa.Uuid(), sa.ForeignKey("steps.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "sent_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_email_outbox_idempotency_key"),
    )
    op.create_index("ix_email_outbox_run", "email_outbox", ["run_id"])


def downgrade() -> None:
    op.drop_table("email_outbox")
    op.drop_table("sandbox_records")
