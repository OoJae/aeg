"""Phase 1 DoD: a fact written via cognee_client.remember() comes back from
recall() — a behavioral assertion, not an "it ran" check.

Marked llm: makes real LLM + local embedding calls. Skip with -m "not llm".
"""

import pytest

from aeg import cognee_client

DATASET = "aeg_test_roundtrip"
pytestmark = pytest.mark.llm


async def test_remember_then_recall_roundtrip():
    result = await cognee_client.remember(
        "Maya owns the billing service.", dataset=DATASET
    )
    assert result["status"] == "completed"
    assert result["items"], "expected data ids captured at ingest (COGNEE_NOTES §2)"

    try:
        entries = await cognee_client.recall(
            "Who owns the billing service?",
            datasets=[DATASET],
            only_context=True,
            top_k=5,
        )
        joined = " ".join(entry["text"] for entry in entries).lower()
        assert "maya" in joined, f"fact not recovered; got: {joined[:300]}"
        assert entries[0]["source"] is not None
    finally:
        await cognee_client.forget(dataset=DATASET)

    assert await cognee_client.recall(
        "Who owns the billing service?", datasets=[DATASET], only_context=True
    ) == [], "forgotten dataset should recall as empty via the wrapper"
