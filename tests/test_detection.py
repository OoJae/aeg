"""Phase 3 detection + trust units — pure functions, no LLM, no cognee I/O."""

import pytest

from aeg import detection, trust
from aeg.detection import PairVerdict, candidate_pairs, claim_tokens, contradiction_id, decide, tokens


def claim(cid: str, subject: str = "", object_: str = "", text: str = "",
          confidence: float = 0.5, data_id: str = "") -> dict:
    return {"id": cid, "subject": subject, "object": object_, "text": text,
            "confidence": confidence, "status": "active", "data_id": data_id}


# --- stage 1: tokens + pairing ---------------------------------------------- #

def test_tokens_strips_stopwords_and_case():
    assert tokens("The rate limit of the API") == {"rate", "limit", "api"}
    assert tokens("") == frozenset()


def test_claim_tokens_falls_back_to_text():
    empty_spo = claim("x", text="The deploy password rotates on Friday.")
    assert "friday" in claim_tokens(empty_spo)


def test_candidate_pairs_lexical_variance():
    a = claim("a", subject="API rate limit", object_="100 requests per second")
    b = claim("b", subject="the rate limit of the API", object_="9999 requests per second")
    pairs = candidate_pairs([a, b])
    assert len(pairs) == 1
    assert {p["id"] for p in pairs[0]} == {"a", "b"}


def test_candidate_pairs_no_self_and_no_disjoint():
    a = claim("a", subject="API rate limit")
    b = claim("b", subject="deploy password schedule")
    assert candidate_pairs([a]) == []
    assert candidate_pairs([a, b]) == []


def test_candidate_pairs_skips_existing_contradiction():
    a = claim("a", subject="API limit")
    b = claim("b", subject="API limit")
    existing = frozenset({str(contradiction_id("a", "b"))})
    assert candidate_pairs([a, b], existing_contradiction_ids=existing) == []


def test_candidate_pairs_cap_and_deterministic_ordering():
    claims = [claim(f"c{i}", subject="shared topic here") for i in range(6)]
    pairs = candidate_pairs(claims, max_pairs=3)
    assert len(pairs) == 3
    for first, second in pairs:
        assert str(first["id"]) < str(second["id"])  # lexicographic within pair


def test_contradiction_id_order_independent():
    assert contradiction_id("a", "b") == contradiction_id("b", "a")
    assert contradiction_id("a", "b") != contradiction_id("a", "c")


# --- trust resolution -------------------------------------------------------- #

def test_trust_resolve_clear_winner():
    truth = claim("a", confidence=0.6)
    poison = claim("b", confidence=0.25)
    res = trust.resolve(truth, poison)
    assert res.verdict == "a_wins"
    assert res.winner_id == "a" and res.loser_id == "b"
    assert res.new_confidence["b"] == pytest.approx(0.25 * trust.LOSER_FACTOR)
    assert res.new_confidence["a"] == pytest.approx(0.6 + trust.WINNER_BONUS)


def test_trust_resolve_margin_boundary():
    # gap exactly at the margin -> decisive (>= semantics)
    res = trust.resolve(claim("a", confidence=0.60), claim("b", confidence=0.45))
    assert res.verdict == "a_wins"
    # just inside the margin -> tie
    res = trust.resolve(claim("a", confidence=0.60), claim("b", confidence=0.46))
    assert res.verdict == "both_suspect"


def test_trust_resolve_both_suspect_never_quarantines():
    res = trust.resolve(claim("a", confidence=0.5), claim("b", confidence=0.5))
    assert res.verdict == "both_suspect"
    assert res.winner_id is None and res.loser_id is None
    assert res.new_confidence["a"] == pytest.approx(0.5 * trust.TIE_FACTOR)


def test_trust_confidence_clamped_to_one():
    res = trust.resolve(claim("a", confidence=0.95), claim("b", confidence=0.2))
    assert res.new_confidence["a"] == 1.0


# --- stage 2 gating ----------------------------------------------------------- #

def test_decide_threshold_gating():
    a, b = claim("a", confidence=0.6), claim("b", confidence=0.25)
    assert decide(PairVerdict(relation="contradicts", confidence=0.69), a, b) is None
    assert decide(PairVerdict(relation="contradicts", confidence=0.70), a, b) is not None
    assert decide(PairVerdict(relation="consistent", confidence=0.99), a, b) is None
    assert decide(PairVerdict(relation="unrelated", confidence=1.0), a, b) is None


async def test_verify_pair_fail_safe(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(detection.cognee_client, "llm_structured", boom)
    verdict = await detection.verify_pair(claim("a"), claim("b"))
    assert verdict.relation == "unrelated"
    assert verdict.confidence == 0.0
