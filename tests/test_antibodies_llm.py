"""Phase 6 antibody meta-memory — real MiMo end-to-end. Skip with -m 'not llm'.

The full loop learns an antibody from the defeated attack; a REPLAY of the same
poison is then blocked instantly (no scan); a NOVEL distinct poison still passes
the door (no false-positive).
"""

import httpx
import pytest
import pytest_asyncio

from aeg import cognee_client
from aeg.gateway import app

pytestmark = pytest.mark.llm

# Distinct, self-contained story: antibodies are a GLOBAL overlay (only reset_all
# clears them), so this test must not share a poison fingerprint with any other
# LLM test — otherwise the antibody it learns would block their ingests.
DATASET = "aeg_test_antibody"
TRUTH = "The cache TTL is 60 seconds."
POISON = "The cache TTL is actually 86400 seconds."
NOVEL = "The load balancer uses round-robin routing across three zones."


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as c:
        yield c
    await cognee_client.forget(dataset=DATASET)


async def _remember(client, content, kind="user"):
    return (await client.post("/remember", json={
        "content": content, "source": {"kind": kind, "identifier": "t"},
        "dataset": DATASET}, timeout=180)).json()


async def test_antibody_blocks_replay_but_not_novel(client):
    await _remember(client, TRUTH, "user")
    poison = await _remember(client, POISON, "tool")
    assert poison["quarantined"] is False and poison["data_ids"], "poison must reach the substrate first"

    await client.post("/scan", json={"dataset": DATASET}, timeout=300)
    await client.post("/respond", json={"dataset": DATASET}, timeout=300)

    # the defeated attack is now an antibody
    antibodies = (await client.get("/antibodies")).json()
    assert antibodies["count"] >= 1
    learned = next(a for a in antibodies["items"] if "86400" in a["pattern"])
    assert learned["times_seen"] == 1

    # REPLAY the exact same poison → blocked instantly, no cognify, no new scan
    replay = await _remember(client, POISON, "tool")
    assert replay["quarantined"] is True, "a replayed known attack must be blocked at ingest"
    assert replay["antibody"] and "86400" in replay["antibody"]
    assert replay["data_ids"] == [], "blocked replay must not be cognified"
    assert "antibody" in replay["events"]

    bumped = (await client.get("/antibodies")).json()
    learned2 = next(a for a in bumped["items"] if "86400" in a["pattern"])
    assert learned2["times_seen"] == 2, "the replay bumps times_seen"

    # NO FALSE POSITIVE: a novel, distinct-topic memory still passes the door
    novel = await _remember(client, NOVEL, "tool")
    assert novel["quarantined"] is False, "the antibody must not over-block novel content"
    assert novel["antibody"] == ""
    assert novel["data_ids"], "novel content should be cognified normally"
