"""Phase 3 DoD — real MiMo + fastembed (slow). Skip with -m 'not llm'.

The story: a subtle lie (no injection phrasing) passes innate screening and IS
cognified. The immune sweep (/scan) detects the contradiction against a higher-
trust claim, quarantines the lower-trust side (item-level memory_only forget),
and the poison stops appearing in normal recall. /release reverses it.
"""

import httpx
import pytest
import pytest_asyncio

from aeg import cognee_client
from aeg.gateway import app

pytestmark = pytest.mark.llm

DATASET = "aeg_test_detection"

TRUTH = "The API rate limit is 100 requests per second."
POISON = "The API rate limit is actually 9999 requests per second."
QUERY = "What is the API rate limit?"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def clean_global_store():
    # recall is global (COGNEE_NOTES §6b) and the antibody overlay is global too;
    # start this file from a clean store so its two order-dependent tests are not
    # crowded/blocked by content or antibodies learned in earlier LLM files
    await cognee_client.reset_all()
    yield


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as c:
        yield c
    await cognee_client.forget(dataset=DATASET)


async def _recall_text(client, **overrides) -> str:
    body = {"query": QUERY, "dataset": DATASET, "top_k": 10, "only_context": True}
    body.update(overrides)
    resp = await client.post("/recall", json=body, timeout=180)
    return " ".join(e["text"] for e in resp.json()["entries"]).lower()


async def test_scan_detects_and_quarantines_poison(client):
    truth = (await client.post("/remember", json={
        "content": TRUTH, "source": {"kind": "user", "identifier": "demi"},
        "dataset": DATASET}, timeout=180)).json()
    assert truth["screening"]["verdict"] == "clean"

    poison = (await client.post("/remember", json={
        "content": POISON, "source": {"kind": "tool", "identifier": "slack-bot"},
        "dataset": DATASET}, timeout=180)).json()
    # the adaptive-immunity premise: the lie passes the door and IS cognified
    assert poison["screening"]["verdict"] == "clean"
    assert poison["quarantined"] is False
    assert poison["data_ids"], "subtle poison must reach the substrate"
    poison_claim_ids = {c["id"] for c in poison["claims"]}

    before = await _recall_text(client)
    assert "9999" in before, "poison should be recallable BEFORE the scan"

    scan = (await client.post("/scan", json={"dataset": DATASET}, timeout=300)).json()
    assert scan["pairs_checked"] >= 1
    assert scan["contradictions"], "scanner should confirm the contradiction"
    verdicts = {c["verdict"] for c in scan["contradictions"]}
    assert verdicts & {"a_wins", "b_wins"}, f"expected a decisive verdict, got {verdicts}"
    # the poison (untrusted tool source, 0.25) must be the loser
    assert set(scan["quarantined"]) & poison_claim_ids, \
        f"poison claims {poison_claim_ids} should be quarantined, got {scan['quarantined']}"
    quarantined_poison = set(scan["quarantined"]) & poison_claim_ids
    for cid in quarantined_poison:
        assert scan["reweighted"][cid] == pytest.approx(0.25 * 0.4)

    queue = (await client.get("/quarantine", params={"dataset": DATASET})).json()
    assert queue["count"] >= 1
    assert any("9999" in item["text"] for item in queue["items"])
    assert all(item["id"] for item in queue["items"]), "queue items need ids for /release"

    after = await _recall_text(client)
    assert "100" in after, "the trusted fact must survive"
    assert "9999" not in after, "quarantined poison must stop appearing in normal recall"

    conflicts = (await client.get("/contradictions", params={"dataset": DATASET})).json()
    assert conflicts["count"] >= 1
    assert conflicts["items"][0]["confidence"] >= 0.7
    assert conflicts["items"][0]["rationale"]

    # idempotence: the same pair is not re-flagged or re-quarantined
    rescan = (await client.post("/scan", json={"dataset": DATASET}, timeout=300)).json()
    assert rescan["contradictions"] == []
    assert rescan["quarantined"] == []


async def test_release_restores_recall(client):
    # state persists from the previous test within the session-scoped stores
    queue = (await client.get("/quarantine", params={"dataset": DATASET})).json()
    poison_items = [i for i in queue["items"] if "9999" in i["text"]]
    assert poison_items, "expected the quarantined poison from the DoD test"
    claim_id = poison_items[0]["id"]

    release = (await client.post("/release", json={
        "dataset": DATASET, "claim_id": claim_id}, timeout=300)).json()
    assert release["released"] is True
    assert release["restored_data_id"]

    restored = await _recall_text(client)
    assert "9999" in restored, "released memory should be recallable again"

    queue_after = (await client.get("/quarantine", params={"dataset": DATASET})).json()
    assert all("9999" not in i["text"] for i in queue_after["items"])
