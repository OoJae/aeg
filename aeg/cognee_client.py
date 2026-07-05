"""The ONLY module that imports cognee.

Every wrapper below is built against the empirically verified contract in
docs/COGNEE_NOTES.md (section references in each docstring). If cognee's API
drifts, this file is the one-file fix.

Wrapper names map 1:1 onto cognee's memory quartet — remember / recall /
improve / forget — so the demo and README can point at the real ops directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import UUID

from aeg import config

config.apply_cognee_env()  # must precede the cognee import (COGNEE_NOTES §11)

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402  (re-export: lane selector for aeg)
from cognee.infrastructure.databases.graph import get_graph_engine  # noqa: E402
from cognee.infrastructure.databases.vector import get_vector_engine  # noqa: E402
from cognee.infrastructure.llm import LLMGateway  # noqa: E402
from cognee.low_level import DataPoint  # noqa: E402  (re-export: base class for aeg.ontology)
from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError  # noqa: E402
from cognee.tasks.storage import add_data_points as _cognee_add_data_points  # noqa: E402

# Recall lanes, named so callers never import SearchType (COGNEE_NOTES §10). The
# graph lanes support node_name quarantine filtering; "chunks" is the pure-vector
# lane (no filtering, no synthesized answer) used for the graph-only proof.
LANES: dict[str, "SearchType"] = {
    "graph": SearchType.GRAPH_COMPLETION,
    "graph_cot": SearchType.GRAPH_COMPLETION_COT,
    "chunks": SearchType.CHUNKS,
}


def _to_dict(obj: Any) -> Any:
    for attr in ("to_dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return obj


_user_cache: dict[str, Any] = {}
_schema_ready = False


async def _ensure_schema() -> None:
    """Create cognee's relational tables (users/principals/permissions/…) once.
    On a fresh Postgres these don't exist until a table-creating op runs, so a
    first-call create_user would hit `relation "principals" does not exist`.
    Idempotent (CREATE IF NOT EXISTS)."""
    global _schema_ready
    if _schema_ready:
        return
    from cognee.infrastructure.databases.relational import create_db_and_tables  # lazy
    try:
        await create_db_and_tables()
    except Exception:
        pass
    _schema_ready = True


async def get_cognee_user(user_id: str):
    """Get-or-create the cognee user backing an Aeg user_id (access-control mode,
    Feature 4). cognee's add/cognify/search scope data to this user, so user A's
    memory is invisible to user B — proven by scripts/verify_access_control.py.
    Cached per process."""
    if user_id in _user_cache:
        return _user_cache[user_id]
    await _ensure_schema()
    from cognee.modules.users.methods import create_user, get_user_by_email  # lazy
    email = f"aeg-user-{user_id}@example.com"
    user = None
    try:
        user = await get_user_by_email(email)
    except Exception:
        user = None
    if user is None:
        user = await create_user(email=email, password=f"aeg-{user_id}", is_verified=True)
    _user_cache[user_id] = user
    return user


def _normalize_search(results: Any) -> list[dict]:
    """Normalize cognee.search output (access-control recall) into recall's
    {text, source, raw} shape. Search returns dicts like {search_result: [...]}."""
    out: list[dict] = []
    for item in (results or []):
        sr = item.get("search_result") if isinstance(item, dict) else None
        if isinstance(sr, list):
            out.extend({"text": str(s), "source": "graph", "raw": item} for s in sr)
        elif sr is not None:
            out.append({"text": str(sr), "source": "graph", "raw": item})
        else:
            out.append({"text": str(item), "source": "graph", "raw": item})
    return out


async def remember(
    data: str | list[str],
    *,
    dataset: str = config.DATASET_MAIN,
    session_id: str | None = None,
    node_set: list[str] | None = None,
    graph_model: type | None = None,
    self_improvement: bool = False,
    user_id: str | None = None,
) -> dict:
    """Write memory. COGNEE_NOTES §1, §2, §7.

    Defaults self_improvement OFF (cognee's default is ON and spawns a
    background improve — nondeterministic for short-lived processes).
    Returns the RememberResult dict; `items` holds data ids for forget(),
    but is cumulative for the dataset — diff against prior items to attribute
    a specific write (§2).

    Access-control mode (config.AEG_ACCESS_CONTROL + user_id): routes through the
    user-scoped add/cognify path (high-level remember takes no user) so the write
    is owned by, and isolated to, this tenant.
    """
    if config.AEG_ACCESS_CONTROL and user_id:
        user = await get_cognee_user(user_id)
        add_kwargs = {"node_set": node_set} if node_set else {}
        await cognee.add(data, dataset_name=dataset, user=user, **add_kwargs)
        await cognee.cognify(datasets=[dataset], user=user)
        return {"status": "completed", "items": []}  # per-tenant dataset; diff moot

    kwargs: dict[str, Any] = {}
    if node_set is not None:
        kwargs["node_set"] = node_set
    if graph_model is not None:
        kwargs["graph_model"] = graph_model
    result = await cognee.remember(
        data,
        dataset_name=dataset,
        session_id=session_id,
        self_improvement=self_improvement,
        **kwargs,
    )
    return _to_dict(result)


def _normalize_entry(entry: Any) -> dict:
    return {
        "text": getattr(entry, "text", None) or str(_to_dict(entry)),
        "source": getattr(entry, "source", None),
        "raw": entry,
    }


async def recall(
    query: str,
    *,
    datasets: list[str] | None = None,
    top_k: int = 10,
    session_id: str | None = None,
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    query_type: Any = None,
    user_id: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Read memory. COGNEE_NOTES §1, §2, §6.

    Entries are normalized to {"text", "source", "raw"}; `source` is cognee's
    provenance tag (graph/session/trace/...). node_name filters by node_set
    tags — the quarantine-exclusion mechanism (§6), graph lanes only.
    Recalling a fully forgotten dataset raises DatasetNotFoundError in cognee;
    here that is an empty result (§2).

    Access-control mode (config.AEG_ACCESS_CONTROL + user_id): searches as this
    tenant, so a user provably cannot recall another user's memory.
    """
    if config.AEG_ACCESS_CONTROL and user_id:
        user = await get_cognee_user(user_id)
        try:
            results = await cognee.search(
                query_text=query, query_type=query_type, user=user,
                datasets=datasets or [config.DATASET_MAIN], top_k=top_k,
            )
        except DatasetNotFoundError:
            return []
        return _normalize_search(results)

    try:
        results = await cognee.recall(
            query,
            query_type=query_type,
            datasets=datasets or [config.DATASET_MAIN],
            top_k=top_k,
            session_id=session_id,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            only_context=only_context,
            **kwargs,
        )
    except DatasetNotFoundError:
        return []
    return [_normalize_entry(entry) for entry in results]


async def improve(
    *,
    dataset: str = config.DATASET_MAIN,
    session_ids: list[str] | None = None,
    feedback_alpha: float | None = None,
    run_in_background: bool = False,
    **kwargs: Any,
) -> Any:
    """Reinforce/enrich memory (wraps memify). COGNEE_NOTES §3, §7.

    With session_ids, bridges screened session memory into the permanent graph
    (user_sessions_from_cache node set). feedback_alpha travels via ImproveKwargs,
    so it is forwarded only when explicitly set.
    """
    if feedback_alpha is not None:
        kwargs["feedback_alpha"] = feedback_alpha
    return await cognee.improve(
        dataset=dataset,
        session_ids=session_ids,
        run_in_background=run_in_background,
        **kwargs,
    )


async def forget(
    *,
    dataset: str,
    data_id: UUID | str | None = None,
    memory_only: bool = False,
) -> dict:
    """Delete memory — the immune response. COGNEE_NOTES §4.

    data_id given: surgical item-level removal (shared entities survive, zero
    LLM calls) — the agreed primary strategy. data_id None: drops the whole
    dataset (the untrusted-sub-dataset fallback). memory_only=True wipes
    graph+vectors but keeps raw data so cognify() can restore — reversible
    hard quarantine.

    Idempotent: forgetting a dataset that does not exist is a no-op (cognee
    raises when it can't resolve the dataset), so demo/test resets are safe.
    """
    if isinstance(data_id, str):
        data_id = UUID(data_id)
    try:
        return await cognee.forget(data_id=data_id, dataset=dataset, memory_only=memory_only)
    except (DatasetNotFoundError, AttributeError):
        return {"dataset": dataset, "data_id": str(data_id) if data_id else None,
                "status": "not_found"}


async def forget_everything() -> dict:
    """Global wipe of ALL datasets (COGNEE_NOTES §4, PX.1). Demo/test reset only —
    deliberately kept out of forget() so no code path reaches it by accident."""
    return await cognee.forget(everything=True)


async def recognify(dataset: str) -> Any:
    """Re-run cognify over a dataset — the quarantine-RELEASE primitive.

    Restores recall for items whose memory was wiped by forget(memory_only=True):
    cognify runs with use_pipeline_cache=False and incremental_loading=True, so
    only items with reset cognify status are re-processed, keeping their data_ids
    (COGNEE_NOTES §4 Phase-3 addendum). Dataset-granular: it restores ALL reset
    items — the caller must re-forget items that should stay quarantined.
    """
    try:
        result = await cognee.cognify(datasets=[dataset])
    except DatasetNotFoundError:
        return {"dataset": dataset, "status": "not_found"}
    return _to_dict(result)


async def export_graph() -> tuple[list, list]:
    """(nodes, edges) for the whole graph store — global, not dataset-scoped.
    COGNEE_NOTES §8."""
    engine = await get_graph_engine()
    return await engine.get_graph_data()


async def graph_metrics() -> dict:
    """Raw material for the memory health score. COGNEE_NOTES §8."""
    engine = await get_graph_engine()
    return await engine.get_graph_metrics()


async def overlay_snapshot(dataset: str) -> dict:
    """One graph read → the dashboard's dataset-scoped audit overlay.

    Partitions a single export_graph() into claims / contradictions / events for
    `dataset` (COGNEE_NOTES §8) — cheaper than three list_typed_nodes calls
    (each of which does its own global export) on a ~1.5s dashboard poll.
    """
    nodes, _ = await export_graph()
    out: dict[str, list[dict]] = {"claims": [], "contradictions": [], "events": []}
    bucket = {"Claim": "claims", "Contradiction": "contradictions", "ImmuneEvent": "events"}
    for node in nodes:
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            node_id, payload = node[0], node[1]
        else:
            node_id, payload = None, node
        if not isinstance(payload, dict):
            continue
        key = bucket.get(payload.get("type"))
        if key is None or payload.get("dataset") != dataset:
            continue
        out[key].append({"id": str(node_id), **payload})
    return out


async def dashboard_snapshot(dataset: str, nodes: list | None = None) -> dict:
    """Partition a graph node list → the dashboard's dataset-scoped overlay PLUS
    the global antibody list, in ONE pass (coalesces the old overlay_snapshot +
    list_typed_nodes("Antibody") double export). Pass `nodes` to reuse a cached
    global export so distinct `dataset` query params don't each force a fresh
    whole-graph read (adversarial-study MEDIUM: attacker-cycled dataset param).
    Antibodies are global immune memory, so they are collected unscoped.
    """
    if nodes is None:
        nodes, _ = await export_graph()
    out: dict[str, list[dict]] = {
        "claims": [], "contradictions": [], "events": [], "antibodies": []
    }
    bucket = {"Claim": "claims", "Contradiction": "contradictions", "ImmuneEvent": "events"}
    for node in nodes:
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            node_id, payload = node[0], node[1]
        else:
            node_id, payload = None, node
        if not isinstance(payload, dict):
            continue
        node_type = payload.get("type")
        if node_type == "Antibody":
            out["antibodies"].append({"id": str(node_id), **payload})
            continue
        key = bucket.get(node_type)
        if key is None or payload.get("dataset") != dataset:
            continue
        out[key].append({"id": str(node_id), **payload})
    return out


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed text via cognee's configured embedding engine (fastembed local,
    all-MiniLM-L6-v2, 384-dim, keyless by default). Used for semantic antibody
    matching — cosine over these vectors catches paraphrased replays that share
    no distinctive tokens with the recorded attack."""
    engine = get_vector_engine().embedding_engine
    return await engine.embed_text(texts)


async def reset_all() -> dict:
    """Full local reset — the ONLY primitive that clears the accumulating typed
    overlay (forget()/forget_everything() do not — COGNEE_NOTES §6b). Global and
    destructive; demo/test reset only.
    """
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(graph=True, vector=True, metadata=True, cache=True)
    return {"status": "reset"}


async def list_data_ids(dataset: str) -> list[str]:
    """Ids of the raw Data records currently in a dataset ([] if absent).

    Used to prove permanent forget() removed an item, and to refresh the
    gateway's per-dataset attribution set after improve() bridging creates new
    session-derived data items (COGNEE_NOTES §2, §3 Phase-4 addendum).
    """
    datasets = await cognee.datasets.list_datasets()
    ds_id = next((d.id for d in datasets if getattr(d, "name", None) == dataset), None)
    if ds_id is None:
        return []
    items = await cognee.datasets.list_data(dataset_id=ds_id)
    return [str(item.id) for item in items]


async def list_typed_nodes(node_type: str, **field_filters: Any) -> list[dict]:
    """Payloads of graph nodes whose `type` == node_type, optionally filtered by
    exact field values (e.g. status="quarantined", dataset="aeg_main").

    Reads the audit/provenance overlay (Claim/Source/ImmuneEvent/...) that
    add_data_points wrote. COGNEE_NOTES: these nodes are NOT dataset-owned, so
    forget()/forget_everything() do NOT remove them — the overlay accumulates and
    must be filtered by field (e.g. dataset) when scoping is needed.
    """
    nodes, _ = await export_graph()
    out = []
    for node in nodes:
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            node_id, payload = node[0], node[1]
        else:
            node_id, payload = None, node
        if not isinstance(payload, dict) or payload.get("type") != node_type:
            continue
        if all(payload.get(k) == v for k, v in field_filters.items()):
            # stored properties exclude the node id (popped at write time) — inject it
            out.append({"id": str(node_id), **payload})
    return out


async def export_snapshot(dataset: str = config.DATASET_MAIN) -> Any:
    """Dataset-scoped GraphSnapshot(dataset, nodes, edges) — the before/after
    view for the dashboard. COGNEE_NOTES §8."""
    return await cognee.export(dataset=dataset)


async def add_data_points(points: list[DataPoint]) -> list:
    """Insert typed DataPoint instances directly into the graph. COGNEE_NOTES §5.

    ⚠️ These land in the GLOBAL graph without belongs_to_set facets and without
    dataset ownership — so they are NOT hidden by node_name quarantine filtering
    and NOT removed by forget(data_id). This is Aeg's provenance/audit overlay
    (Claim/Source/TrustSignal/ImmuneEvent); the recall substrate that filtering
    and forget operate on is the remember()-ingested content. Deterministic ids
    (ontology.deterministic_id) are the dedup mechanism — cognee 1.2.2 does not
    dedup by content.
    """
    return await _cognee_add_data_points(points)


async def llm_structured(text_input: str, system_prompt: str, response_model: type) -> Any:
    """One structured-output call through cognee's configured LLM. COGNEE_NOTES §11.

    Uses the same MiMo(custom)+instructor json_mode path cognee uses internally,
    so no separate LLM client/config is introduced. ~5–15s per call.
    """
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,
        response_model=response_model,
    )


@dataclass
class RecallDiff:
    """Before/after recall comparison — Phase 4's test primitive: proof that
    forget()/improve() actually changed recall output."""

    before: list[dict]
    after: list[dict]
    added: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


async def recall_diff(
    query: str,
    action: Callable[[], Awaitable[Any]],
    **recall_kwargs: Any,
) -> RecallDiff:
    """recall -> await action() -> recall again, diffing normalized text."""
    before = await recall(query, **recall_kwargs)
    await action()
    after = await recall(query, **recall_kwargs)
    before_texts = {entry["text"] for entry in before}
    after_texts = {entry["text"] for entry in after}
    return RecallDiff(
        before=before,
        after=after,
        added=after_texts - before_texts,
        removed=before_texts - after_texts,
    )
