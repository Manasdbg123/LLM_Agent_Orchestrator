"""Concurrent creation of the same agent definition.

Regression test for a bug the load test found: `get_or_create_definition` read, saw
nothing, and inserted. At a submission concurrency of 20 that raced -- several
requests naming the same new agent all missed, all inserted, and every one but the
winner took a unique violation that surfaced as an HTTP 500 on 13% of submissions.

The fix is an `ON CONFLICT DO NOTHING` insert followed by a re-read. This test drives
the race deliberately rather than hoping to hit it: N concurrent tasks, one brand-new
name, all of them must come back with the same definition and none may raise.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa

from app.core.runs import get_or_create_definition
from app.db import session_scope
from app.domain.models import AgentDefinition

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

CONCURRENCY = 12


async def _create(name: str) -> uuid.UUID:
    """One caller, in its own transaction -- as two API requests would be."""
    async with session_scope() as session:
        definition = await get_or_create_definition(
            session, name=name, model="fake-model", tools=["calculator"]
        )
        return definition.id


async def test_concurrent_creation_of_one_name_yields_one_definition() -> None:
    name = f"race-{uuid.uuid4().hex[:8]}"

    ids = await asyncio.gather(*(_create(name) for _ in range(CONCURRENCY)))

    # Every caller got the same row. Not merely "nobody crashed": two definitions
    # under one name would silently give concurrent runs different tool lists.
    assert len(set(ids)) == 1, f"{len(set(ids))} distinct definitions created for one name"

    async with session_scope() as session:
        count = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(AgentDefinition)
                .where(AgentDefinition.name == name)
            )
        ).scalar_one()
    assert count == 1


async def test_concurrent_creation_of_distinct_names_is_unaffected() -> None:
    """The conflict path must not swallow legitimate parallel creations."""
    names = [f"race-{uuid.uuid4().hex[:8]}" for _ in range(CONCURRENCY)]

    ids = await asyncio.gather(*(_create(n) for n in names))

    assert len(set(ids)) == CONCURRENCY

    async with session_scope() as session:
        count = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(AgentDefinition)
                .where(AgentDefinition.name.in_(names))
            )
        ).scalar_one()
    assert count == CONCURRENCY


async def test_existing_definition_is_returned_not_duplicated() -> None:
    """The ordinary path: a second call for a known name re-reads, never inserts."""
    name = f"race-{uuid.uuid4().hex[:8]}"

    first = await _create(name)
    second = await _create(name)

    assert first == second
