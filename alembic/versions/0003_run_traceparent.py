"""run trace context

Adds `agent_runs.traceparent`. A run outlives the process that created it, so its
trace context has to be stored rather than held in memory — that is what makes a
crash and its recovery appear in a single trace.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("traceparent", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "traceparent")
