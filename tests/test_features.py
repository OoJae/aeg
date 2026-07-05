"""Free units for the four shipped 'What's next' features — no LLM/API calls.

Semantic-antibody and access-control END-TO-END behaviour (real embeddings /
Postgres) is exercised by the LLM tier + scripts/verify_access_control.py; here we
test the pure logic and wiring that must stay green without a key.
"""

import asyncio
import uuid

import pytest

from aeg import antibodies, config, detection, gateway


# --- config flags exist with sane defaults --------------------------------- #

def test_new_flags_present():
    assert config.AEG_SEMANTIC_ANTIBODIES is True
    assert 0.0 < config.AEG_ANTIBODY_SIM_THRESHOLD <= 1.0
    assert config.AEG_AUTO_SWEEP_ENABLED is False          # off by default
    assert config.AEG_SWEEP_INTERVAL_SECONDS > 0
    assert config.AEG_ACCESS_CONTROL is False              # off by default


# --- semantic antibody: cosine gate ---------------------------------------- #

def test_cosine():
    assert antibodies._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert antibodies._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert antibodies._cosine([], [1.0]) == 0.0


async def test_semantic_antibody_branch(monkeypatch):
    """Lexical miss + embedding cosine ≥ threshold ⇒ semantic block; below ⇒ none."""
    monkeypatch.setattr(config, "AEG_SEMANTIC_ANTIBODIES", True)
    monkeypatch.setattr(config, "AEG_ANTIBODY_SIM_THRESHOLD", 0.82)
    ab = {"id": str(uuid.uuid4()), "type": "Antibody",
          "pattern": "zzz|yyy|xxx|www", "times_seen": 1, "embedding": [1.0, 0.0, 0.0]}

    async def fake_list(node_type, **f):
        return [ab] if node_type == "Antibody" else []
    monkeypatch.setattr(antibodies.cognee_client, "list_typed_nodes", fake_list)

    # content shares NO tokens with the pattern -> lexical miss, semantic decides
    content = "some completely different words here"
    monkeypatch.setattr(antibodies.cognee_client, "embed",
                        lambda t: _coro([[0.99, 0.14, 0.0]]))   # cosine ~0.99
    assert await antibodies.match_antibody(content) is not None

    monkeypatch.setattr(antibodies.cognee_client, "embed",
                        lambda t: _coro([[0.0, 1.0, 0.0]]))      # cosine 0.0
    assert await antibodies.match_antibody(content) is None


async def test_semantic_off_skips_embedding(monkeypatch):
    monkeypatch.setattr(config, "AEG_SEMANTIC_ANTIBODIES", False)
    ab = {"id": str(uuid.uuid4()), "type": "Antibody", "pattern": "a|b|c|d",
          "times_seen": 1, "embedding": [1.0, 0.0]}
    monkeypatch.setattr(antibodies.cognee_client, "list_typed_nodes",
                        lambda nt, **f: _coro([ab] if nt == "Antibody" else []))

    async def boom(_):
        raise AssertionError("embed must not run when semantic antibodies are off")
    monkeypatch.setattr(antibodies.cognee_client, "embed", boom)
    assert await antibodies.match_antibody("nothing in common at all") is None


# --- scheduled background sweep: wiring ------------------------------------- #

async def test_auto_sweep_calls_scan(monkeypatch):
    monkeypatch.setattr(config, "AEG_SWEEP_INTERVAL_SECONDS", 0)   # sleep(0) → tight loop
    monkeypatch.setattr(config, "AEG_DAILY_LLM_BUDGET", 100000)
    calls: list[str] = []

    async def fake_scan(ds, **kw):
        calls.append(ds)
    monkeypatch.setattr(detection, "scan_dataset", fake_scan)

    class _State:
        llm_window_start = 0.0
        llm_calls = 0
        from collections import defaultdict
        locks = defaultdict(asyncio.Lock)

    class _App:
        state = _State()

    task = asyncio.create_task(gateway._auto_sweep_loop(_App()))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    assert calls, "auto-sweep should have run scan_dataset"
    assert all(ds == config.DATASET_MAIN for ds in calls)


# --- MCP server: tools registered ------------------------------------------ #

async def test_mcp_tools_registered():
    from aeg import mcp_server
    tools = await mcp_server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "aeg_remember", "aeg_recall", "aeg_scan", "aeg_respond",
        "aeg_reinforce", "aeg_quarantine", "aeg_contradictions", "aeg_health",
    }


def _coro(value):
    async def _c():
        return value
    return _c()
