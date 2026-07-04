"""Phase 4 spine contracts — real MiMo + fastembed. Skip with -m 'not llm'.

test_full_forget_after_memory_only_quarantine is the Phase-4 gate (probe-as-
regression-test): respond() permanently forgets items that detection already
memory_only-forgot, so that exact sequence must succeed.

test_forget_changes_recall / test_improve_changes_recall are the build-guide
DoD tests: forget() and improve() must actually CHANGE recall() output — not
merely run without error. Both use cognee_client.recall_diff, built for this.
"""

import uuid

import pytest

from aeg import cognee_client, config

pytestmark = pytest.mark.llm

DATASET = "aeg_test_response"


@pytest.fixture(autouse=True)
async def _teardown():
    yield
    await cognee_client.forget(dataset=DATASET)


async def _ingest(text: str) -> str:
    result = await cognee_client.remember(text, dataset=DATASET)
    return str(result["items"][-1]["id"])


async def test_full_forget_after_memory_only_quarantine():
    # the exact respond() sequence: quarantine (memory_only) then permanent forget
    data_id = await _ingest("The Vega service depends on RabbitMQ.")
    assert data_id in await cognee_client.list_data_ids(DATASET)

    quarantine = await cognee_client.forget(dataset=DATASET, data_id=data_id,
                                            memory_only=True)
    assert quarantine.get("status") == "success"
    assert data_id in await cognee_client.list_data_ids(DATASET), \
        "memory_only must keep the raw Data record"

    permanent = await cognee_client.forget(dataset=DATASET, data_id=data_id)
    assert permanent.get("status") == "success"
    assert data_id not in await cognee_client.list_data_ids(DATASET), \
        "full forget after quarantine must remove the raw Data record"


async def test_forget_changes_recall():
    poison = "The backup schedule runs every 42 minutes."
    data_id = await _ingest(poison)

    diff = await cognee_client.recall_diff(
        "How often does the backup schedule run?",
        lambda: cognee_client.forget(dataset=DATASET, data_id=data_id),
        datasets=[DATASET],
        query_type=cognee_client.LANES["chunks"],
        top_k=5,
        only_context=True,
    )
    assert diff.changed, "forget() must change recall output"
    assert diff.removed, "forget() should remove entries from recall"
    after_text = " ".join(e["text"] for e in diff.after).lower()
    assert "42" not in after_text, "the forgotten fact must be gone from recall"


async def test_improve_changes_recall():
    await _ingest("The analytics pipeline is owned by the data team.")
    session = config.session_id(f"test-{uuid.uuid4().hex[:8]}")
    note = "Verified: the analytics pipeline runs on Spark clusters nightly."
    await cognee_client.remember(note, dataset=DATASET, session_id=session)

    diff = await cognee_client.recall_diff(
        "What does the analytics pipeline run on?",
        lambda: cognee_client.improve(dataset=DATASET, session_ids=[session]),
        datasets=[DATASET],
        query_type=cognee_client.LANES["chunks"],
        top_k=5,
        only_context=True,
    )
    assert diff.changed, "improve(session_ids) must change recall output"
    assert diff.added, "bridged session memory should add recall entries"
    after_text = " ".join(e["text"] for e in diff.after).lower()
    assert "spark" in after_text, "the bridged note must be recallable"
