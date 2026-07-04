"""Phase 5 dashboard endpoints — FREE (overlay seeded via add_data_points, no LLM).

The audit overlay is process-global, so everything is scoped to a unique dataset
and the assertions filter to it.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from aeg import cognee_client, config
from aeg.gateway import app
from aeg.ontology import Claim, Contradiction, ImmuneEvent, deterministic_id

DATASET = "aeg_test_dashboard"


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as c:
        yield c


def _claim(text, *, status="active", confidence=0.6, data_id="d1"):
    return Claim(id=deterministic_id(DATASET, "claim", text), text=text, subject=text.split()[0],
                 predicate="is", object="x", confidence=confidence, status=status,
                 data_id=data_id, dataset=DATASET)


async def _seed_overlay():
    truth = _claim("Atlas uses Postgres", confidence=0.6)
    poison = _claim("Atlas uses MongoDB", status="quarantined", confidence=0.1)
    conflict = Contradiction(
        id=deterministic_id("contradiction", str(truth.id), str(poison.id)),
        claim_a=truth, claim_b=poison, claim_a_id=str(truth.id), claim_b_id=str(poison.id),
        dataset=DATASET, confidence=0.95, verdict="a_wins", rationale="mutually exclusive DBs")
    event = ImmuneEvent(action="quarantine", target="Atlas uses MongoDB",
                        details="contradiction loser", dataset=DATASET, occurred_at="2026-07-03T00:00:01")
    await cognee_client.add_data_points([truth, poison, conflict, event])
    return truth, poison


async def test_root_serves_landing(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Aeg" in resp.text and "Memory" in resp.text  # the landing hero


async def test_how_page_serves(client):
    resp = await client.get("/how")
    assert resp.status_code == 200
    assert "manifesto" in resp.text.lower()


async def test_proof_page_serves(client):
    resp = await client.get("/proof")
    assert resp.status_code == 200
    assert "receipts" in resp.text.lower() and "18" in resp.text


async def test_dashboard_route_serves_monitor(client):
    resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert "Memory graph" in resp.text  # the live monitor


async def test_favicon_and_assets(client):
    assert (await client.get("/favicon.svg")).status_code == 200
    css = await client.get("/assets/brand.css")
    assert css.status_code == 200 and "--vital" in css.text
    # path traversal is refused
    assert (await client.get("/assets/../gateway.py")).status_code in (404, 400)


async def test_dashboard_state_shape_and_counts(client):
    truth, poison = await _seed_overlay()
    st = (await client.get("/dashboard/state", params={"dataset": DATASET})).json()

    assert set(st) >= {"health", "graph", "quarantine", "contradictions", "events"}
    assert st["health"]["open_contradictions"] == 1
    assert st["health"]["label"] == "compromised"

    node_ids = {n["id"] for n in st["graph"]["nodes"]}
    assert {str(truth.id), str(poison.id)} <= node_ids
    assert len(st["graph"]["edges"]) == 1
    edge = st["graph"]["edges"][0]
    assert edge["source"] in node_ids and edge["target"] in node_ids
    assert edge["open"] is True

    assert any(q["text"] == "Atlas uses MongoDB" for q in st["quarantine"])
    assert st["contradictions"][0]["open"] is True


async def test_events_scoped_and_sorted(client):
    await _seed_overlay()
    resp = (await client.get("/events", params={"dataset": DATASET})).json()
    assert resp["count"] >= 1
    assert all(e.get("dataset") == DATASET for e in resp["items"])
    times = [e["occurred_at"] for e in resp["items"]]
    assert times == sorted(times, reverse=True)


async def test_demo_reset_is_admin_only(client, monkeypatch):
    # NEVER run the real global prune in-session — it wipes sibling tests' store
    fake = AsyncMock(return_value={"status": "reset"})
    monkeypatch.setattr(cognee_client, "reset_all", fake)

    # unauthenticated (no AEG_API_KEY configured): the destructive route is disabled
    monkeypatch.setattr(config, "AEG_API_KEY", "")
    disabled = await client.post("/demo/reset")
    assert disabled.status_code == 403
    fake.assert_not_awaited()

    # with a key configured, a wrong/missing key is rejected...
    monkeypatch.setattr(config, "AEG_API_KEY", "s3cret")
    assert (await client.post("/demo/reset")).status_code == 401
    assert (await client.post("/demo/reset", headers={"X-Aeg-Key": "nope"})).status_code == 401
    fake.assert_not_awaited()

    # ...and the correct key runs the reset and clears in-process state
    app.state.seen_ids["x"].add("y")
    ok = await client.post("/demo/reset", headers={"X-Aeg-Key": "s3cret"})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    fake.assert_awaited_once()
    assert "x" not in app.state.seen_ids
