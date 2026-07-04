"""Memory health score units — pure compute_score, no LLM, no I/O."""

from aeg.health import compute_score


def claim(cid, *, status="active", confidence=0.6, data_id="d1"):
    return {"id": cid, "status": status, "confidence": confidence, "data_id": data_id}


def contra(a, b, verdict="a_wins"):
    return {"id": f"{a}-{b}", "claim_a_id": a, "claim_b_id": b, "verdict": verdict}


def test_baseline_all_active_high_trust():
    result = compute_score([claim(str(i)) for i in range(6)], [])
    assert result["score"] == 100
    assert result["label"] == "healthy"


def test_no_claims_no_divide_by_zero():
    result = compute_score([], [])
    assert result["score"] == 100
    assert result["total"] == 0


def test_live_low_trust_claim_dips():
    base = [claim(str(i)) for i in range(6)]
    poisoned = base + [claim("p", confidence=0.25)]
    result = compute_score(poisoned, [])
    assert result["label"] == "degraded"
    assert result["low_trust"] == 1
    assert result["score"] < 100


def test_open_contradiction_compromises():
    claims = [claim("t"), claim("p", status="quarantined", confidence=0.1)]
    result = compute_score(claims, [contra("t", "p")])
    assert result["open_contradictions"] == 1
    assert result["label"] == "compromised"


def test_contradiction_with_forgotten_side_is_resolved():
    claims = [claim("t"), claim("p", status="forgotten", confidence=0.1)]
    result = compute_score(claims, [contra("t", "p")])
    assert result["open_contradictions"] == 0, "a forgotten side closes the contradiction"


def test_innate_quarantine_does_not_count_but_substrate_does():
    innate = [claim(str(i)) for i in range(6)] + [
        claim("doc", status="quarantined", confidence=0.25, data_id="")]
    substrate = [claim(str(i)) for i in range(6)] + [
        claim("tool", status="quarantined", confidence=0.25, data_id="d9")]
    # innate block (data_id="") holds health; a substrate threat drags it
    assert compute_score(innate, [])["score"] > compute_score(substrate, [])["score"]


def test_full_heal_recovers():
    base = [claim(str(i)) for i in range(6)]
    healed = base + [claim("p", status="forgotten", confidence=0.1)]
    result = compute_score(healed, [contra("0", "p")])
    assert result["score"] == 100
    assert result["label"] == "healthy"


def test_both_suspect_counts_open():
    claims = [claim("a", status="quarantined"), claim("b", status="quarantined")]
    result = compute_score(claims, [contra("a", "b", verdict="both_suspect")])
    assert result["open_contradictions"] == 1
