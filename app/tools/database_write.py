"""Write to a sandbox key-value store.

Classified `SAFE_TO_REPLAY`, and the implementation is what earns that label: the
write is an **upsert on a natural key** (`namespace`, `key`), so performing it twice
leaves exactly the state performing it once would. The classification is a claim
about the SQL, not a hope about the caller.

Contrast with `send_email`, which cannot be made naturally idempotent and therefore
has to cooperate with a deduplicating provider instead.
"""

from __future__ import annotations

from typing import Any, ClassVar

import sqlalchemy as sa
from pydantic import Field
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import session_scope
from app.domain.models import SandboxRecord
from app.tools.base import BaseTool, EffectPolicy, ToolArgs, ToolContext, ToolResult

#: Namespaces treated as production-like. Writing to one requires human approval,
#: which demonstrates that approval can be a function of the *arguments* and not
#: merely of the tool. Wired into the gate in Phase 4.
SENSITIVE_NAMESPACES = frozenset({"customers", "billing", "production"})


class DatabaseWriteArgs(ToolArgs):
    namespace: str = Field(
        ..., min_length=1, max_length=64, description="Logical table or collection name"
    )
    key: str = Field(..., min_length=1, max_length=200, description="Unique record key")
    value: str = Field(..., max_length=4000, description="Value to store")


class DatabaseWriteTool(BaseTool):
    name: ClassVar[str] = "database_write"
    description: ClassVar[str] = (
        "Store a value under a key in a namespace. Overwrites any existing value for "
        "that key. Use it to record results that later steps or the user will need."
    )
    args_model: ClassVar[type[ToolArgs]] = DatabaseWriteArgs
    effect_policy: ClassVar[EffectPolicy] = EffectPolicy.SAFE_TO_REPLAY
    requires_approval: ClassVar[bool] = False
    timeout_seconds: ClassVar[float] = 10.0

    @classmethod
    def approval_reason(cls, arguments: dict[str, Any]) -> str | None:
        """Gate on the target, not on the operation.

        Writing scratch notes is routine; writing to `customers` or `billing` is not.
        Demonstrates that the approval gate is a function of the arguments.
        """
        namespace = str(arguments.get("namespace", "")).lower()
        if namespace in SENSITIVE_NAMESPACES:
            return f"write targets the sensitive namespace {namespace!r}"
        return None

    async def execute(self, ctx: ToolContext, args: DatabaseWriteArgs) -> ToolResult:
        async with session_scope() as session:
            stmt = (
                pg_insert(SandboxRecord)
                .values(
                    namespace=args.namespace,
                    key=args.key,
                    value=args.value,
                    run_id=ctx.run_id,
                    written_by_step_id=ctx.step_id,
                )
                .on_conflict_do_update(
                    index_elements=[SandboxRecord.namespace, SandboxRecord.key],
                    set_={
                        "value": args.value,
                        "run_id": ctx.run_id,
                        "written_by_step_id": ctx.step_id,
                        "updated_at": sa.func.now(),
                        # Counts executions rather than distinct records. Nothing in
                        # the system depends on it; it exists so a test can prove the
                        # replay actually happened and still converged.
                        "writes": SandboxRecord.writes + 1,
                    },
                )
                .returning(SandboxRecord.writes)
            )
            writes = int((await session.execute(stmt)).scalar_one())

        return ToolResult(
            content=f"Stored {args.key!r} in {args.namespace!r}.",
            data={
                "namespace": args.namespace,
                "key": args.key,
                "bytes": len(args.value),
                "writes": writes,
            },
        )


__all__ = ["SENSITIVE_NAMESPACES", "DatabaseWriteArgs", "DatabaseWriteTool"]
