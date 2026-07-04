"""Phase 4 response/reinforce units — no LLM, cognee I/O monkeypatched."""

import uuid

import pytest

from aeg import response
from aeg.response import BadMemoryVerdict, should_forget

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "aeg.tests.response")


def cid(name: str) -> str:
    """Readable name -> stable real UUID string (claim_from_payload parses ids)."""
    return str(uuid.uuid5(_NS, name))


def payload(name: str, *, status: str = "quarantined", data_id: str = "d1",
            confidence: float = 0.25, text: str = "poison") -> dict:
    return {"id": cid(name), "status": status, "data_id": data_id, "dataset": "ds",
            "confidence": confidence, "text": text, "subject": "s",
            "predicate": "p", "object": "o"}


def contradiction(a: str, b: str) -> dict:
    return {"id": "x", "claim_a_id": cid(a), "claim_b_id": cid(b), "verdict": "a_wins",
            "rationale": "conflict", "dataset": "ds"}


async def test_verify_quarantined_fail_safe(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(response.cognee_client, "llm_structured", boom)
    verdict = await response.verify_quarantined(payload("q"), [payload("t")], contradiction("t", "q"))
    assert verdict.confirmed_bad is False, "a flaked verifier must never delete memory"


def test_should_forget_threshold_gating():
    assert should_forget(BadMemoryVerdict(confirmed_bad=True, confidence=0.70))
    assert not should_forget(BadMemoryVerdict(confirmed_bad=True, confidence=0.69))
    assert not should_forget(BadMemoryVerdict(confirmed_bad=False, confidence=0.99))


class ClientLog:
    """Monkeypatch surface capturing cognee_client calls in order."""

    def __init__(self, claims, contradictions):
        self.claims = claims
        self.contradictions = contradictions
        self.calls: list[tuple] = []

    async def list_typed_nodes(self, node_type, **filters):
        rows = self.claims if node_type == "Claim" else self.contradictions
        return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]

    async def forget(self, **kwargs):
        self.calls.append(("forget", kwargs))
        return {"status": "success"}

    async def add_data_points(self, points):
        self.calls.append(("add_data_points", points))
        return points

    async def remember(self, *args, **kwargs):
        self.calls.append(("remember", kwargs))
        return {"status": "completed", "items": []}

    async def improve(self, **kwargs):
        self.calls.append(("improve", kwargs))
        return {}

    async def list_data_ids(self, dataset):
        return []


@pytest.fixture
def wire(monkeypatch):
    def _wire(claims, contradictions, release_result=None):
        log = ClientLog(claims, contradictions)
        for name in ("list_typed_nodes", "forget", "add_data_points",
                     "remember", "improve"):
            monkeypatch.setattr(response.cognee_client, name, getattr(log, name))

        async def fake_release(dataset, claim_id):
            log.calls.append(("release", claim_id))
            return release_result or {"released": True, "released_claims": [claim_id]}

        monkeypatch.setattr(response.detection, "release_claim", fake_release)
        return log

    return _wire


async def confirming_verifier(claim, against, conflict):
    return BadMemoryVerdict(confirmed_bad=True, confidence=0.95, rationale="bad")


async def denying_verifier(claim, against, conflict):
    return BadMemoryVerdict(confirmed_bad=False, confidence=0.9, rationale="fine")


async def test_respond_confirmed_routes_to_full_forget(wire):
    log = wire(
        claims=[payload("q"), payload("t", status="active", data_id="d2",
                                      confidence=0.7, text="truth")],
        contradictions=[contradiction("t", "q")],
    )
    report = await response.respond("ds", verifier=confirming_verifier)
    assert report.forgotten == [cid("q")]
    assert report.forgotten_data_ids == ["d1"]
    forget_calls = [c for c in log.calls if c[0] == "forget"]
    assert forget_calls == [("forget", {"dataset": "ds", "data_id": "d1"})], \
        "confirmed-bad must be a FULL forget (no memory_only)"
    # respond() may also write an antibody (its own add_data_points call); target
    # the batch that carries the forgotten claim + forget event
    batches = [c[1] for c in log.calls if c[0] == "add_data_points"]
    batch = next(b for b in batches
                 if any(getattr(p, "action", None) == "forget" for p in b))
    statuses = {getattr(p, "status", None) for p in batch}
    actions = {getattr(p, "action", None) for p in batch}
    assert "forgotten" in statuses and "forget" in actions
    assert "antibody" in report.events, "a defeated attack should record an antibody"


async def test_respond_unconfirmed_routes_to_release(wire):
    log = wire(
        claims=[payload("q"), payload("t", status="active", data_id="d2")],
        contradictions=[contradiction("t", "q")],
    )
    report = await response.respond("ds", verifier=denying_verifier)
    assert report.forgotten == []
    assert report.released == [cid("q")]
    assert ("release", cid("q")) in log.calls
    assert not [c for c in log.calls if c[0] == "forget"]


async def test_respond_skips_without_evidence_or_data_id(wire):
    calls = []

    async def counting_verifier(claim, against, conflict):
        calls.append(claim["id"])
        return BadMemoryVerdict(confirmed_bad=True, confidence=1.0)

    wire(
        claims=[payload("no-item", data_id=""),  # ingest-time quarantine
                payload("no-evidence", data_id="d9")],  # no contradiction record
        contradictions=[],
    )
    report = await response.respond("ds", verifier=counting_verifier)
    assert calls == [], "verifier must never run without data_id AND evidence"
    assert set(report.skipped) == {cid("no-item"), cid("no-evidence")}
    assert report.forgotten == []


async def test_reinforce_unique_session_and_sweep_ordering(wire):
    still_quarantined = payload("q2", data_id="d3")
    log = wire(
        claims=[payload("t", status="active", data_id="d2", confidence=0.7,
                        text="the truth"), still_quarantined],
        contradictions=[],
    )
    first = await response.reinforce("ds", cid("t"))
    second = await response.reinforce("ds", cid("t"))
    assert first.session_id != second.session_id, "session ids must be unique per call"
    assert "Verified: the truth" in first.note
    assert first.new_confidence > first.old_confidence

    ordered = [c[0] for c in log.calls]
    improve_at = ordered.index("improve")
    sweep = [i for i, c in enumerate(log.calls)
             if c[0] == "forget" and c[1].get("memory_only")]
    assert sweep and min(sweep) > improve_at, \
        "resurrection sweep must re-forget quarantined items AFTER improve()"
    assert first.requarantined == [cid("q2")]


async def test_reinforce_unknown_claim_returns_none(wire):
    wire(claims=[], contradictions=[])
    assert await response.reinforce("ds", "ghost") is None
