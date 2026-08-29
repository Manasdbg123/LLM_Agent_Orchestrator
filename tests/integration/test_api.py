"""The HTTP surface, exercised against the real ASGI app and a real database.

The API is the part an operator actually touches, and the approval endpoints are the
only way a human can unblock a run — so they get tested rather than assumed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import sqlalchemy as sa

from app.api.main import app
from app.db import session_scope
from app.domain.models import ApprovalRequest
from app.llm.fake import encode_script
from tests.helpers import Cluster, wait_for

pytestmark = [pytest.mark.integration]


@pytest.fixture
async def client(clean_db: None, pg_queue) -> AsyncIterator[httpx.AsyncClient]:
    # The queue is injected rather than built by the lifespan, so the test drives the
    # same backend the workers are consuming from.
    app.state.queue = pg_queue
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _payload(script: list[dict], **overrides: object) -> dict:
    body = {
        "agent": f"api-{uuid.uuid4().hex[:6]}",
        "task": f"do the thing {encode_script(script)}",
        "model": "fake-model",
        "tools": ["calculator", "send_email"],
    }
    body.update(overrides)
    return body


async def test_health_and_readiness(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json()["status"] == "ok"
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


async def test_create_run_is_idempotent_per_key(client: httpx.AsyncClient) -> None:
    body = _payload([{"text": "hello"}])
    headers = {"Idempotency-Key": f"key-{uuid.uuid4().hex[:8]}"}

    first = await client.post("/v1/runs", json=body, headers=headers)
    second = await client.post("/v1/runs", json=body, headers=headers)

    assert first.status_code == 201
    # A retried create returns the original run rather than starting a second one.
    assert second.json()["id"] == first.json()["id"]


async def test_run_timeline_cost_and_audit_log(client: httpx.AsyncClient, pg_queue) -> None:
    created = await client.post(
        "/v1/runs",
        json=_payload(
            [
                {"tools": [{"name": "calculator", "input": {"expression": "6*7"}}]},
                {"text": "42"},
            ],
            tools=["calculator"],
        ),
    )
    run_id = created.json()["id"]

    async with Cluster(pg_queue, workers=1):
        assert await wait_for(lambda: _run_done(client, run_id), timeout=60)

    run = (await client.get(f"/v1/runs/{run_id}")).json()
    assert run["status"] == "succeeded"
    assert run["output"]["answer"] == "42"
    assert run["input_tokens"] > 0

    steps = (await client.get(f"/v1/runs/{run_id}/steps")).json()
    assert [s["kind"] for s in steps] == ["agent_turn", "tool_call", "agent_turn", "finalize"]
    assert all(s["duration_ms"] is not None for s in steps)

    cost = (await client.get(f"/v1/runs/{run_id}/cost")).json()
    assert cost["llm_call_count"] == 2
    assert cost["cost_usd"] > 0
    assert cost["budget_remaining_usd"] == pytest.approx(cost["max_cost_usd"] - cost["cost_usd"])
    assert all(c["price_version"] for c in cost["calls"])

    transitions = (await client.get(f"/v1/runs/{run_id}/transitions")).json()
    assert [t["to_status"] for t in transitions if t["entity"] == "run"] == [
        "pending",
        "running",
        "succeeded",
    ]


async def _run_done(client: httpx.AsyncClient, run_id: str) -> bool:
    body = (await client.get(f"/v1/runs/{run_id}")).json()
    return body["status"] in {"succeeded", "failed", "cancelled"}


async def test_approval_queue_and_decision_round_trip(client: httpx.AsyncClient, pg_queue) -> None:
    created = await client.post(
        "/v1/runs",
        json=_payload(
            [
                {
                    "tools": [
                        {
                            "name": "send_email",
                            "input": {"to": "a@example.test", "subject": "S", "body": "B"},
                        }
                    ]
                },
                {"text": "sent"},
            ],
            tools=["send_email"],
        ),
    )
    run_id = created.json()["id"]

    async with Cluster(pg_queue, workers=1):
        assert await wait_for(lambda: _has_pending(run_id), timeout=45)

        queue_body = (await client.get("/v1/approvals")).json()
        assert len(queue_body) == 1
        approval = queue_body[0]
        assert approval["tool_name"] == "send_email"
        assert approval["decision"] == "pending"
        assert approval["run_id"] == run_id

        # The run is parked, so the operator can see why without joining steps.
        assert (await client.get(f"/v1/runs/{run_id}")).json()["status"] == "awaiting_approval"

        decided = await client.post(
            f"/v1/approvals/{approval['id']}/decision",
            json={"decision": "approve", "decided_by": "operator", "reason": "ok"},
        )
        assert decided.status_code == 200
        assert decided.json()["decision"] == "approved"
        assert decided.json()["decided_by"] == "operator"

        assert await wait_for(lambda: _run_done(client, run_id), timeout=60)

    assert (await client.get(f"/v1/runs/{run_id}")).json()["status"] == "succeeded"


async def test_a_decided_approval_cannot_be_flipped(client: httpx.AsyncClient, pg_queue) -> None:
    created = await client.post(
        "/v1/runs",
        json=_payload(
            [
                {
                    "tools": [
                        {
                            "name": "send_email",
                            "input": {"to": "b@example.test", "subject": "S", "body": "B"},
                        }
                    ]
                },
                {"text": "sent"},
            ],
            tools=["send_email"],
        ),
    )
    run_id = created.json()["id"]

    async with Cluster(pg_queue, workers=1):
        assert await wait_for(lambda: _has_pending(run_id), timeout=45)
        approval_id = (await client.get("/v1/approvals")).json()[0]["id"]

        first = await client.post(
            f"/v1/approvals/{approval_id}/decision",
            json={"decision": "reject", "decided_by": "operator", "reason": "no"},
        )
        assert first.status_code == 200

        second = await client.post(
            f"/v1/approvals/{approval_id}/decision",
            json={"decision": "approve", "decided_by": "someone-else"},
        )
        assert second.status_code == 409
        assert "not reversible" in second.json()["detail"]


async def test_unknown_ids_are_404_not_500(client: httpx.AsyncClient) -> None:
    missing = uuid.uuid4()
    assert (await client.get(f"/v1/runs/{missing}")).status_code == 404
    assert (await client.get(f"/v1/approvals/{missing}")).status_code == 404
    decision = await client.post(
        f"/v1/approvals/{missing}/decision",
        json={"decision": "approve", "decided_by": "operator"},
    )
    assert decision.status_code == 404


async def test_cancelling_a_terminal_run_is_a_conflict(client: httpx.AsyncClient, pg_queue) -> None:
    created = await client.post("/v1/runs", json=_payload([{"text": "done"}], tools=[]))
    run_id = created.json()["id"]

    async with Cluster(pg_queue, workers=1):
        assert await wait_for(lambda: _run_done(client, run_id), timeout=60)

    conflict = await client.post(f"/v1/runs/{run_id}/cancel")
    assert conflict.status_code == 409


async def _has_pending(run_id: str) -> bool:
    async with session_scope() as session:
        return (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(ApprovalRequest)
                .where(
                    ApprovalRequest.run_id == sa.cast(run_id, sa.Uuid),
                    ApprovalRequest.decision == "pending",
                )
            )
        ).scalar_one() > 0
