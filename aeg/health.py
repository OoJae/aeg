"""Memory health score — a legible UX signal, not a research metric (build guide §6).

Pure like trust.py: takes Claim + Contradiction payloads (list_typed_nodes dicts),
returns a 0-100 score and a label. The async fetch lives in the gateway.

The formula penalizes, in order of weight:
  - open contradictions   — two conflicting memories, neither yet resolved
  - substrate threats      — quarantined memory that actually reached the graph
  - low-trust drag         — untrusted claims sitting in active memory

Innate blocks (data_id=="") are DELIBERATELY excluded from the substrate-threat
count: they never entered the recall substrate, so memory health is genuinely
unaffected — and respond() can't clear them, so counting them would make the
heal irrecoverable. They still surface in the threat feed and quarantine queue.
"""

from __future__ import annotations

HEALTHY_THRESHOLD = 85
DEGRADED_THRESHOLD = 50


def _clamp(value: float) -> int:
    return max(0, min(100, round(value)))


def compute_score(claims: list[dict], contradictions: list[dict]) -> dict:
    active = [c for c in claims if c.get("status") == "active"]
    quarantined = [c for c in claims if c.get("status") == "quarantined"]
    forgotten = [c for c in claims if c.get("status") == "forgotten"]
    low_trust = [c for c in active if float(c.get("confidence", 0.5)) < 0.5]
    high_trust = [c for c in active if float(c.get("confidence", 0.5)) >= 0.5]

    # substrate-backed threats only (innate blocks never entered memory)
    live_quar = [c for c in quarantined if c.get("data_id")]

    status_by_id = {str(c.get("id")): c.get("status") for c in claims}
    open_contra = sum(
        1 for k in contradictions
        if status_by_id.get(str(k.get("claim_a_id"))) != "forgotten"
        and status_by_id.get(str(k.get("claim_b_id"))) != "forgotten"
    )

    low_ratio = len(low_trust) / max(1, len(active))
    score = _clamp(
        100
        - 15 * (1 if low_trust else 0)
        - 15 * low_ratio
        - 40 * open_contra
        - 12 * min(len(live_quar), 3)
    )

    if score >= HEALTHY_THRESHOLD:
        label = "healthy"
    elif score >= DEGRADED_THRESHOLD:
        label = "degraded"
    else:
        label = "compromised"

    return {
        "score": score,
        "label": label,
        "active": len(active),
        "high_trust": len(high_trust),
        "low_trust": len(low_trust),
        "quarantined": len(quarantined),
        "forgotten": len(forgotten),
        "open_contradictions": open_contra,
        "total": len(claims),
    }
