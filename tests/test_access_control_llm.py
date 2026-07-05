"""Real per-user access control (Feature 4) — Postgres-gated, LLM tier.

Tenant isolation is enforced by cognee backend access control on the Postgres
profile; this proves through the Aeg gateway that user B cannot recall user A's
memory, while user A can. Embedded SQLite cannot back access control, so this is
skipped unless run with the Postgres profile + a running pg:

    docker compose up -d
    AEG_PROFILE=postgres AEG_ACCESS_CONTROL=true uv run pytest tests/test_access_control_llm.py

The raw cognee isolation is separately proven by scripts/verify_access_control.py.
"""

import os
import uuid

import httpx
import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("AEG_PROFILE") != "postgres"
        or os.environ.get("AEG_ACCESS_CONTROL", "").lower() not in ("1", "true", "yes", "on"),
        reason="needs AEG_PROFILE=postgres + AEG_ACCESS_CONTROL=true + docker compose up",
    ),
]

QUERY = "What database does Project Atlas use?"


@pytest_asyncio.fixture
async def client():
    from aeg import config
    from aeg.gateway import create_app
    assert config.AEG_ACCESS_CONTROL, "access control flag must be on for this test"
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://aeg") as c:
        yield c


async def test_tenant_isolation(client):
    # unique users per run — self-contained, no collision with persisted pg state
    tag = uuid.uuid4().hex[:8]
    owner, intruder = f"alice{tag}", f"bob{tag}"

    r = await client.post("/remember", json={
        "content": "Project Atlas uses Postgres as its primary database.",
        "source": {"kind": "user", "identifier": owner},
        "dataset": "aeg_main", "user_id": owner,
    }, timeout=200)
    assert r.status_code == 200

    async def recall_text(user_id: str) -> str:
        resp = await client.post("/recall", json={
            "query": QUERY, "dataset": "aeg_main", "user_id": user_id, "top_k": 10,
        }, timeout=200)
        return " ".join(e["text"] for e in resp.json().get("entries", [])).lower()

    # the intruder cannot see the owner's memory; the owner can
    assert "postgres" not in await recall_text(intruder), "user B must not read user A's memory"
    assert "postgres" in await recall_text(owner), "user A must read her own memory"
