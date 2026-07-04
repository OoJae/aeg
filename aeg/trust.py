"""Trust engine — pure confidence arithmetic for contradiction resolution.

No cognee imports, no I/O: everything here is unit-testable for free. The
inputs are Claim payloads from cognee_client.list_typed_nodes (need "id" and
"confidence"); the output says who wins, who gets quarantined, and what the new
confidences are. Ties never quarantine — reversible-before-irreversible is the
working agreement, and a coin-flip deletion is neither.
"""

from __future__ import annotations

from dataclasses import dataclass

from aeg.ontology import Verdict

TRUST_MARGIN = 0.15  # min confidence gap for a decisive verdict
LOSER_FACTOR = 0.4  # loser confidence multiplier
WINNER_BONUS = 0.10  # winner confidence bonus (clamped to 1.0)
TIE_FACTOR = 0.7  # both_suspect: both sides downweighted
VERIFIER_THRESHOLD = 0.7  # min LLM-verifier confidence to confirm a contradiction


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class TrustResolution:
    verdict: Verdict  # "a_wins" | "b_wins" | "both_suspect"
    winner_id: str | None  # None on both_suspect
    loser_id: str | None
    new_confidence: dict[str, float]  # claim_id -> updated value
    rationale: str


def resolve(claim_a: dict, claim_b: dict, *, margin: float = TRUST_MARGIN) -> TrustResolution:
    """Resolve a CONFIRMED contradiction between two claim payloads by trust.

    abs(conf_a - conf_b) >= margin → the higher-trust side wins; the loser is
    downweighted (×LOSER_FACTOR) and marked for quarantine; the winner gets a
    small bonus. Below the margin → both_suspect: both downweighted
    (×TIE_FACTOR), NEITHER quarantined.
    """
    id_a, id_b = str(claim_a["id"]), str(claim_b["id"])
    conf_a = float(claim_a.get("confidence", 0.5))
    conf_b = float(claim_b.get("confidence", 0.5))

    # epsilon keeps ">= margin" true at the exact boundary despite float error
    # (0.60 - 0.45 == 0.14999999999999997)
    if abs(conf_a - conf_b) >= margin - 1e-9:
        a_wins = conf_a > conf_b
        winner_id, winner_conf = (id_a, conf_a) if a_wins else (id_b, conf_b)
        loser_id, loser_conf = (id_b, conf_b) if a_wins else (id_a, conf_a)
        return TrustResolution(
            verdict="a_wins" if a_wins else "b_wins",
            winner_id=winner_id,
            loser_id=loser_id,
            new_confidence={
                winner_id: _clamp(winner_conf + WINNER_BONUS),
                loser_id: _clamp(loser_conf * LOSER_FACTOR),
            },
            rationale=(f"trust {conf_a:.2f} vs {conf_b:.2f} (margin {margin:.2f}): "
                       f"lower-trust side loses and is quarantined"),
        )

    return TrustResolution(
        verdict="both_suspect",
        winner_id=None,
        loser_id=None,
        new_confidence={id_a: _clamp(conf_a * TIE_FACTOR), id_b: _clamp(conf_b * TIE_FACTOR)},
        rationale=(f"trust {conf_a:.2f} vs {conf_b:.2f} within margin {margin:.2f}: "
                   f"both downweighted, neither quarantined (ties never destroy)"),
    )
