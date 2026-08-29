"""Test harness.

Integration and chaos tests run against a **real** Postgres and a real Redis. Mocking
the database here would mock away the entire subject of the tests: `SELECT ... FOR
UPDATE`, partial unique indexes, `now()` evaluated server-side, and atomic conditional
UPDATEs are the mechanism, not an implementation detail behind it.

The schema under test is created by `alembic upgrade head`, never by
`metadata.create_all`, so drift between the migration and the ORM models fails a test
rather than surviving to production.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest

# Must be set before app.config is imported anywhere.
_TEST_DB = os.environ.get(
    "AGENTORC_TEST_DATABASE_URL", "postgresql+psycopg://orch:orch@localhost:5432/orch_test"
)
os.environ.setdefault("AGENTORC_DATABASE_URL", _TEST_DB)
os.environ.setdefault("AGENTORC_ENABLE_FAULT_INJECTION", "true")
os.environ.setdefault("AGENTORC_LOG_JSON", "false")
os.environ.setdefault("AGENTORC_LOG_LEVEL", "WARNING")
# Compressed reliability timings so recovery is observable inside a test's lifetime.
os.environ.setdefault("AGENTORC_LEASE_TTL_SECONDS", "4")
os.environ.setdefault("AGENTORC_REAPER_INTERVAL_SECONDS", "0.5")
os.environ.setdefault("AGENTORC_ENQUEUE_GRACE_SECONDS", "1")
os.environ.setdefault("AGENTORC_STALLED_MESSAGE_IDLE_MS", "8000")
os.environ.setdefault("AGENTORC_WORKER_POLL_BLOCK_MS", "500")

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import dispose_engine, session_scope  # noqa: E402
from app.domain.models import AgentDefinition  # noqa: E402


def pytest_asyncio_loop_factories():
    """Force a selector-based loop for every async test.

    psycopg's async driver cannot run on Windows' default ProactorEventLoop, so tests
    must build their loop the same way the application entry points do.
    """
    from app.runtime import new_event_loop

    return {"selector": new_event_loop}


TABLES = [
    "email_outbox",
    "sandbox_records",
    "dummy_effects",
    "state_transitions",
    "approval_requests",
    "llm_calls",
    "tool_calls",
    "steps",
    "agent_runs",
    "agent_definitions",
]


def _ensure_database() -> str | None:
    """Create the test database if needed. Returns a skip reason, or None."""
    import sqlalchemy

    url = sqlalchemy.engine.make_url(settings.database_url)
    admin = url.set(database="postgres")
    try:
        engine = sqlalchemy.create_engine(admin, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            exists = conn.execute(
                sa.text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ).scalar()
            if not exists:
                conn.execute(sa.text(f'CREATE DATABASE "{url.database}"'))
        engine.dispose()
    except Exception as exc:
        return f"Postgres unavailable at {admin.render_as_string(hide_password=True)}: {exc}"
    return None


@pytest.fixture(scope="session")
def database() -> Iterator[None]:
    reason = _ensure_database()
    if reason:
        pytest.skip(reason, allow_module_level=True)

    from alembic.config import Config

    from alembic import command

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config(os.path.join(root, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(root, "alembic"))
    if os.environ.get("AGENTORC_TEST_RESET"):
        command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield


@pytest.fixture
async def clean_db(database: None) -> AsyncIterator[None]:
    """Truncate between tests.

    TRUNCATE rather than a rolled-back outer transaction: the code under test opens
    its own connections and commits deliberately (the crash window in the dummy
    handler depends on it), so a shared open transaction would deadlock against it.
    """
    async with session_scope() as session:
        await session.execute(sa.text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
    yield
    await dispose_engine()


@pytest.fixture
async def session(clean_db: None) -> AsyncIterator[AsyncSession]:
    async with session_scope() as s:
        yield s


@pytest.fixture
async def definition(clean_db: None) -> AgentDefinition:
    from app.core.runs import get_or_create_definition

    async with session_scope() as s:
        d = await get_or_create_definition(s, name=f"test-{uuid.uuid4().hex[:6]}")
        await s.flush()
        s.expunge(d)
        return d


@pytest.fixture
async def redis_queue(clean_db: None) -> AsyncIterator[object]:
    """A Redis-backed queue with a test-scoped stream, or skip if Redis is absent."""
    from app.queue.redis_streams import RedisStreamQueue

    q = RedisStreamQueue(stream=f"test:steps:{uuid.uuid4().hex[:8]}", group="workers")
    try:
        await q.setup()
    except Exception as exc:
        await q.close()
        pytest.skip(f"Redis unavailable at {settings.redis_url}: {exc}")
    try:
        yield q
    finally:
        try:
            await q.redis.delete(q.stream)
        finally:
            await q.close()


@pytest.fixture
def pg_queue(clean_db: None) -> object:
    """The Redis-free backend. Every engine test that runs on this proves that
    correctness does not depend on the queue."""
    from app.queue.postgres import PostgresQueue

    return PostgresQueue(poll_interval_seconds=0.05)
