"""Aeg's typed memory ontology — cognee DataPoint subclasses.

Mechanism verified in COGNEE_NOTES §5: subclass DataPoint, declare
metadata={"index_fields": [...]}; no registration step. Relationship fields
(annotations referencing other DataPoint types) become graph edges on insert
via add_data_points(). Dedup() showed no observable effect on 1.2.2, so stable
identity uses explicit deterministic ids — see deterministic_id().

Field lists follow build guide §6 (TrustSignal is the canonical name, not §4's
TrustScore). Base DataPoint already provides id/created_at/updated_at/type.

These types are an AUDIT/PROVENANCE OVERLAY, surfaced to the dashboard via
export_graph() and traversed by Phase-3 detection — NOT recall content. So they
declare empty index_fields: recall's vector search must not seed on them.
COGNEE_NOTES finding: recall ignores its `datasets` arg in the embedded
access-control-off config (search is global), so node_set facets are the only
recall-exclusion mechanism — and an indexed overlay node carrying poison text
would be an untagged vector seed that defeats the quarantine filter.
"""

from __future__ import annotations

import uuid
from typing import Literal

from aeg.cognee_client import DataPoint  # cognee access stays behind the client

AEG_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "aeg.memory-immune-system")


def deterministic_id(*parts: str) -> uuid.UUID:
    """Stable UUID5 for a DataPoint so identical facts map to the same node
    across re-ingests (COGNEE_NOTES §5: no automatic dedup on 1.2.2).

    Gateway id conventions:
      Claim         = deterministic_id(dataset, "claim", text)
      Source        = deterministic_id("source", kind, identifier)
      TrustSignal   = deterministic_id("trust", str(claim.id), str(source.id))
      Contradiction = deterministic_id("contradiction", min(ids), max(ids))
                      (order-independent; ids are str(claim.id))
      ImmuneEvent   = random id (events are occurrences, never deduped)
    """
    return uuid.uuid5(AEG_NAMESPACE, "|".join(parts))


ClaimStatus = Literal["active", "quarantined", "forgotten"]
SourceKind = Literal["user", "document", "tool", "agent", "unknown"]
TrustTier = Literal["trusted", "normal", "untrusted"]
Verdict = Literal["unresolved", "a_wins", "b_wins", "both_suspect"]
ImmuneAction = Literal["screen", "quarantine", "forget", "reinforce", "antibody"]
AttackType = Literal["injection", "false_fact", "stale", "duplicate"]


class Claim(DataPoint):
    """An atomic statement the agent may rely on.

    data_id/dataset link this typed claim back to the raw cognee memory item it
    was extracted from — the handle Phase 4 needs to forget() the underlying
    memory when a claim is confirmed bad (COGNEE_NOTES §2, §4).
    """

    text: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5  # 0..1
    status: ClaimStatus = "active"
    data_id: str = ""  # cognee data item this claim was extracted from
    dataset: str = ""  # dataset that data item lives in
    metadata: dict = {"index_fields": []}


def claim_from_payload(payload: dict) -> Claim:
    """Rebuild a Claim DataPoint from a list_typed_nodes payload.

    Status/confidence updates persist via same-id upsert, and the graph MERGE
    replaces properties WHOLESALE — so an upsert must carry every field, not
    just the changed ones (COGNEE_NOTES §5 Phase-3 addendum).
    """
    return Claim(
        id=uuid.UUID(payload["id"]),
        text=payload.get("text", ""),
        subject=payload.get("subject", ""),
        predicate=payload.get("predicate", ""),
        object=payload.get("object", ""),
        confidence=payload.get("confidence", 0.5),
        status=payload.get("status", "active"),
        data_id=payload.get("data_id", ""),
        dataset=payload.get("dataset", ""),
    )


class Source(DataPoint):
    """Where a memory came from — the provenance anchor for trust decisions."""

    kind: SourceKind
    identifier: str
    trust_tier: TrustTier = "normal"
    first_seen: str = ""  # ISO timestamp
    metadata: dict = {"index_fields": []}


class TrustSignal(DataPoint):
    """Scored provenance link between a Claim and its Source."""

    claim: Claim
    source: Source
    weight: float = 0.5  # 0..1
    rationale: str = ""
    metadata: dict = {"index_fields": []}


class Contradiction(DataPoint):
    """A detected conflict between two Claims — adaptive immunity's output.

    claim_a/claim_b render as graph EDGES (relationship fields are never payload
    properties — COGNEE_NOTES §5 Phase-3 addendum), so claim_a_id/claim_b_id
    scalar mirrors carry the references readable from list_typed_nodes payloads.
    claim_a is always the lexicographically smaller str(id) — verdicts are
    deterministic.
    """

    claim_a: Claim
    claim_b: Claim
    claim_a_id: str = ""
    claim_b_id: str = ""
    dataset: str = ""
    confidence: float = 0.0  # verifier confidence in the contradiction
    detected_at: str = ""  # ISO timestamp
    verdict: Verdict = "unresolved"
    rationale: str = ""
    metadata: dict = {"index_fields": []}


class ImmuneEvent(DataPoint):
    """Audit record of an action Aeg took — feeds the response log/dashboard.

    dataset scopes the threat feed (the overlay is process-global and accumulates;
    COGNEE_NOTES §6b). Uses the random uuid4 default id — events are occurrences,
    never deduped.
    """

    action: ImmuneAction
    target: str  # id/text of the memory acted on
    details: str = ""
    dataset: str = ""
    occurred_at: str = ""  # ISO timestamp
    metadata: dict = {"index_fields": []}


class Antibody(DataPoint):
    """Remembered attack pattern — innate memory of a defeated attack (Phase 6).

    `pattern` is a canonical sorted token signature of the attack; a replayed
    attack is blocked instantly at ingest when its tokens contain this pattern's
    core (no LLM). Lives in the global overlay tagged dataset=aeg_antibodies.
    """

    pattern: str
    attack_type: AttackType = "false_fact"
    times_seen: int = 1
    last_seen: str = ""  # ISO timestamp
    dataset: str = ""
    metadata: dict = {"index_fields": []}


def antibody_from_payload(payload: dict) -> Antibody:
    """Rebuild an Antibody from a list_typed_nodes payload for same-id upsert
    (MERGE replaces properties wholesale — carry every field, COGNEE_NOTES §5)."""
    return Antibody(
        id=uuid.UUID(payload["id"]),
        pattern=payload.get("pattern", ""),
        attack_type=payload.get("attack_type", "false_fact"),
        times_seen=payload.get("times_seen", 1),
        last_seen=payload.get("last_seen", ""),
        dataset=payload.get("dataset", ""),
    )


__all__ = [
    "Claim", "Source", "TrustSignal", "Contradiction", "ImmuneEvent", "Antibody",
    "deterministic_id", "claim_from_payload", "antibody_from_payload",
    "ClaimStatus", "SourceKind", "TrustTier", "Verdict", "ImmuneAction", "AttackType",
]
