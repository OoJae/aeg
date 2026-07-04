"""Phase 6 antibody meta-memory — free units (fingerprint + match + upsert + gateway CHECK)."""

import httpx
import pytest
import pytest_asyncio

from aeg import antibodies, config
from aeg.antibodies import MIN_CORE_TOKENS, fingerprint, match_antibody, record_antibody
from aeg.gateway import app

POISON = "The API rate limit is actually 9999 requests per second."
TRUTH = "The API rate limit is 100 requests per second."


# --- fingerprint ------------------------------------------------------------- #

def test_fingerprint_is_stable_across_reorder_case_whitespace():
    a = fingerprint("The API rate limit is 9999 requests per second.")
    b = fingerprint("  requests  PER second: 9999 API rate LIMIT  ")
    assert a == b, "reordering, case and whitespace must not change the signature"


def test_fingerprint_number_is_the_anticollision_key():
    # truth and lie share every token except the number — must differ
    assert fingerprint(TRUTH) != fingerprint(POISON)
    assert "9999" in fingerprint(POISON) and "100" in fingerprint(TRUTH)


def test_fingerprint_from_claim_uses_subject_object_text():
    claim = {"text": "Atlas uses MongoDB", "subject": "Atlas", "object": "MongoDB", "predicate": "uses"}
    fp = fingerprint(claim)
    assert "atlas" in fp and "mongodb" in fp


# --- match / record (monkeypatched client) ----------------------------------- #

class Store:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.added = []

    async def list_typed_nodes(self, node_type, **filters):
        return [r for r in self.rows if r.get("type", "Antibody") == node_type
                and all(r.get(k) == v for k, v in filters.items())]

    async def add_data_points(self, points):
        self.added.extend(points)
        return points


@pytest.fixture
def store(monkeypatch):
    def _install(rows=()):
        s = Store(rows)
        monkeypatch.setattr(antibodies.cognee_client, "list_typed_nodes", s.list_typed_nodes)
        monkeypatch.setattr(antibodies.cognee_client, "add_data_points", s.add_data_points)
        return s
    return _install


def _ab(pattern, times_seen=1):
    import uuid
    return {"id": str(uuid.uuid4()), "type": "Antibody", "pattern": pattern,
            "attack_type": "false_fact", "times_seen": times_seen, "last_seen": "", "dataset": "aeg_antibodies"}


async def test_match_positive_subset(store):
    fp = fingerprint(POISON)
    store([_ab(fp)])
    hit = await match_antibody("hey, the api rate limit is actually 9999 requests per second, ok?")
    assert hit is not None and hit.pattern == fp


async def test_match_negative_disjoint(store):
    store([_ab(fingerprint(POISON))])
    assert await match_antibody("Maya Chen owns the billing service.") is None


async def test_match_guard_below_min_core(store):
    store([_ab("a|b")])  # only 2 tokens, below MIN_CORE_TOKENS
    assert MIN_CORE_TOKENS >= 3
    assert await match_antibody("a b c d e f g") is None


async def test_record_antibody_upserts_and_increments(store):
    fp = fingerprint(POISON)
    s = store([_ab(fp, times_seen=1)])
    result = await record_antibody("aeg_demo", pattern=fp, attack_type="false_fact")
    assert result.times_seen == 2
    assert s.added and s.added[0].times_seen == 2


async def test_record_antibody_first_sighting(store):
    s = store([])
    result = await record_antibody("aeg_demo", pattern=fingerprint(POISON), attack_type="false_fact")
    assert result.times_seen == 1
    assert s.added[0].dataset == config.DATASET_ANTIBODIES


# --- gateway CHECK integration (monkeypatched match + client) ---------------- #

@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as c:
        yield c


async def test_gateway_blocks_on_antibody_match(client, monkeypatch):
    from aeg.ontology import Antibody, deterministic_id
    fp = fingerprint(POISON)
    stub = Antibody(id=deterministic_id("antibody", fp), pattern=fp,
                    attack_type="false_fact", times_seen=1)

    async def fake_match(content):
        return stub

    recorded = []

    async def fake_record(dataset, *, pattern, attack_type):
        recorded.append(pattern)
        return stub

    async def no_store(points):
        return points

    monkeypatch.setattr("aeg.gateway.antibodies.match_antibody", fake_match)
    monkeypatch.setattr("aeg.gateway.antibodies.record_antibody", fake_record)
    monkeypatch.setattr("aeg.gateway.cognee_client.add_data_points", no_store)

    resp = (await client.post("/remember", json={
        "content": POISON, "source": {"kind": "tool", "identifier": "x"},
        "dataset": "aeg_test_ab"}, timeout=60)).json()
    assert resp["quarantined"] is True
    assert resp["antibody"] == fp
    assert resp["data_ids"] == [], "antibody-blocked content must not be cognified"
    assert "antibody" in resp["events"]
    assert recorded == [fp], "a matched replay bumps the antibody's times_seen"


async def test_gateway_flag_off_bypasses_check(client, monkeypatch):
    called = False

    async def fake_match(content):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(config, "AEG_ANTIBODIES_ENABLED", False)
    monkeypatch.setattr("aeg.gateway.antibodies.match_antibody", fake_match)

    async def no_store(points):
        return points

    async def fake_remember(*a, **k):
        return {"status": "completed", "items": [{"id": "i1"}]}

    monkeypatch.setattr("aeg.gateway.cognee_client.add_data_points", no_store)
    monkeypatch.setattr("aeg.gateway.cognee_client.remember", fake_remember)
    monkeypatch.setattr("aeg.gateway.screening.extract_claims",
                        lambda content: _immediate([]))

    await client.post("/remember", json={
        "content": "A brand new fact about widgets.",
        "source": {"kind": "user", "identifier": "x"}, "dataset": "aeg_test_ab2"}, timeout=60)
    assert called is False, "flag off must skip the antibody lookup entirely"


async def _immediate(value):
    return value
