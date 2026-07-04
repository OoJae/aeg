"""Phase 1 DoD: ontology types import without error (no LLM calls)."""

from aeg.cognee_client import DataPoint
from aeg.ontology import (
    Antibody,
    Claim,
    Contradiction,
    ImmuneEvent,
    Source,
    TrustSignal,
    deterministic_id,
)

ALL_TYPES = [Claim, Source, TrustSignal, Contradiction, ImmuneEvent, Antibody]


def test_all_types_subclass_datapoint():
    for cls in ALL_TYPES:
        assert issubclass(cls, DataPoint), cls.__name__


def test_claim_fields_and_defaults():
    claim = Claim(
        text="Project Atlas uses Postgres.",
        subject="Project Atlas",
        predicate="uses",
        object="Postgres",
    )
    assert claim.confidence == 0.5
    assert claim.status == "active"
    # overlay types are non-indexed so they never seed recall (see ontology docstring)
    assert claim.metadata["index_fields"] == []
    assert claim.id is not None  # inherited from DataPoint


def test_relationship_fields_accept_datapoints():
    claim = Claim(text="t", subject="s", predicate="p", object="o")
    source = Source(kind="tool", identifier="slack-bot")
    signal = TrustSignal(claim=claim, source=source, weight=0.2)
    assert signal.claim is claim
    assert signal.source.trust_tier == "normal"
    conflict = Contradiction(claim_a=claim, claim_b=claim)
    assert conflict.verdict == "unresolved"


def test_deterministic_id_is_stable():
    assert deterministic_id("a", "b") == deterministic_id("a", "b")
    assert deterministic_id("a", "b") != deterministic_id("a", "c")
