"""The dashboard's write paths: start a run, cancel it, filter the list.

Each goes through the same core service call as the API, so these tests assert on
database state, not just on the HTML that came back.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.api.main import app
from app.domain.states import RunStatus
from app.llm.fake import encode_script
from tests.helpers import Cluster, get_run, make_agent_run, wait_for_run_terminal

pytestmark = [pytest.mark.integration]


@pytest.fixture
async def client(clean_db: None, pg_queue):
    app.state.queue = pg_queue
    from app.api.deps import queue_holder

    queue_holder.set(pg_queue)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _run_id_from(response: httpx.Response) -> uuid.UUID:
    assert response.status_code == 303
    return uuid.UUID(response.headers["location"].rsplit("/", 1)[-1])


async def test_a_run_started_from_the_dashboard_executes(client, pg_queue) -> None:
    task = encode_script([{"text": "done from the dashboard"}])
    response = await client.post("/ui/runs", data={"task": task, "max_steps": "5"})
    run_id = _run_id_from(response)

    run = await get_run(run_id)
    assert run.max_steps == 5
    async with Cluster(pg_queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    detail = await client.get(f"/ui/runs/{run_id}")
    assert "done from the dashboard" in detail.text
    # A finished run offers no cancel button and stops refreshing itself.
    assert "Cancel run" not in detail.text
    assert 'id="live-toggle"' not in detail.text


async def test_an_empty_task_is_rejected(client) -> None:
    response = await client.post("/ui/runs", data={"task": "   "})
    assert response.status_code == 422


async def test_cancel_from_the_dashboard(client, pg_queue) -> None:
    run_id, _ = await make_agent_run([{"text": "never reached"}], queue=pg_queue)
    page = await client.get(f"/ui/runs/{run_id}")
    assert "Cancel run" in page.text

    response = await client.post(f"/ui/runs/{run_id}/cancel")
    assert response.status_code == 303
    run = await get_run(run_id)
    assert run.cancel_requested or RunStatus(run.status) is RunStatus.CANCELLED


async def test_runs_can_be_filtered_by_status_and_searched(client, pg_queue) -> None:
    await make_agent_run([{"text": "x"}], queue=pg_queue, task="alpha")
    await make_agent_run([{"text": "x"}], queue=pg_queue, task="bravo")

    listing = await client.get("/ui?q=alpha")
    assert "alpha" in listing.text and "bravo" not in listing.text

    succeeded = await client.get("/ui?status=succeeded")
    assert "No runs match this filter" in succeeded.text

    # An unknown status is ignored rather than erroring.
    assert (await client.get("/ui?status=bogus")).status_code == 200

