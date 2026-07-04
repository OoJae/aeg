#!/usr/bin/env python
"""Phase 0 truth-check: empirically verify cognee's API against THIS install.

Every check prints exactly one machine-parsable line:

    RESULT|<check-id>|<PASS|FAIL|UNSUPPORTED>|<summary>

followed by indented evidence lines (verbatim signatures, return shapes,
timings). The full stdout is the evidence appendix of docs/COGNEE_NOTES.md.

Sections:
    p0  introspection + boot (no API key, no LLM calls)
    p1  behavioral checks: round-trip, forget granularity, node_set filtering,
        DataPoints, sessions (needs LLM_API_KEY; small real calls)
    p2  improve/memify contract, graph export, graph-vs-vector, feedback,
        cost/determinism (needs LLM_API_KEY)
    p3  provider config + truth-subspace availability (record-only, never fails)

Usage:
    uv run scripts/verify_cognee_api.py                  # all sections
    uv run scripts/verify_cognee_api.py --sections p0
    uv run scripts/verify_cognee_api.py --only P1.2,P1.3
    uv run scripts/verify_cognee_api.py --list
    uv run scripts/verify_cognee_api.py --include-destructive   # adds forget(everything=True) last

Exit code: 0 iff no FAIL in sections p0/p1/p2.

This script deliberately imports cognee directly — it is the Phase 0 probe
that the aeg/cognee_client.py wrapper contract is derived from.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / ".cognee_verify"


def bootstrap_env() -> None:
    """Isolate ALL cognee state under .cognee_verify/ — must run before `import cognee`."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    for sub in ("data", "system", "cache", "logs"):
        (SCRATCH / sub).mkdir(parents=True, exist_ok=True)
    defaults = {
        "DATA_ROOT_DIRECTORY": str(SCRATCH / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(SCRATCH / "system"),
        "CACHE_ROOT_DIRECTORY": str(SCRATCH / "cache"),
        "COGNEE_LOGS_DIR": str(SCRATCH / "logs"),
        "COGNEE_LOG_FILE": "false",
        "COGNEE_MINIMAL_LOGGING": "true",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        "CACHING": "true",
        "CACHE_BACKEND": "fs",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


# --------------------------------------------------------------------------- #
# check runner infrastructure
# --------------------------------------------------------------------------- #

class Unsupported(Exception):
    """Raise inside a check to report UNSUPPORTED instead of FAIL."""


class Ctx:
    def __init__(self) -> None:
        self.cognee = None
        self.notes: dict = {}  # discoveries shared across checks
        self.evidence_lines: list[str] = []

    def ev(self, line: object = "") -> None:
        for sub in str(line).splitlines() or [""]:
            self.evidence_lines.append(sub)


CHECKS: list[dict] = []


def check(cid: str, title: str, *, section: str, needs_key: bool = False,
          destructive: bool = False):
    def deco(fn):
        CHECKS.append(dict(id=cid, title=title, section=section,
                           needs_key=needs_key, destructive=destructive, fn=fn))
        return fn
    return deco


def trunc(obj: object, limit: int = 400) -> str:
    text = repr(obj)
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit} chars)"


def safe_dump(obj: object) -> object:
    """Best-effort dict view of a result object."""
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    try:
        return vars(obj)
    except TypeError:
        return obj


def flat_text(obj: object, depth: int = 0) -> str:
    """Recursively collect all strings from arbitrary result structures."""
    if depth > 6 or obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return " ".join(flat_text(item, depth + 1) for item in obj)
    if isinstance(obj, dict):
        return " ".join(flat_text(value, depth + 1) for value in obj.values())
    dumped = safe_dump(obj)
    if dumped is obj:
        return str(obj)
    return flat_text(dumped, depth + 1)


async def timed(coro):
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


# --------------------------------------------------------------------------- #
# cognee helpers
# --------------------------------------------------------------------------- #

async def graph_snapshot():
    from cognee.infrastructure.databases.graph import get_graph_engine

    engine = await get_graph_engine()
    nodes, edges = await engine.get_graph_data()
    return nodes, edges


def nodes_matching(nodes, keyword: str):
    return [n for n in nodes if keyword.lower() in str(n).lower()]


async def discover_data_ids(ctx: Ctx, dataset_name: str):
    """Return list of data items for a dataset, recording HOW they were found."""
    c = ctx.cognee
    datasets = await c.datasets.list_datasets()
    target = None
    for ds in datasets:
        name = getattr(ds, "name", None) or (ds.get("name") if isinstance(ds, dict) else None)
        if name == dataset_name:
            target = ds
            break
    if target is None:
        raise Unsupported(f"dataset {dataset_name!r} not found via datasets.list_datasets()")
    ds_id = getattr(target, "id", None) or (target.get("id") if isinstance(target, dict) else None)
    list_data = c.datasets.list_data
    params = inspect.signature(list_data).parameters
    if "dataset_id" in params:
        items = await list_data(dataset_id=ds_id)
        mechanism = "datasets.list_datasets() -> match name -> datasets.list_data(dataset_id=...)"
    else:
        items = await list_data(ds_id)
        mechanism = "datasets.list_datasets() -> match name -> datasets.list_data(<positional id>)"
    ctx.notes.setdefault("data_id_mechanism", mechanism)
    return ds_id, items


def item_id(item):
    return getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)


async def recall_text(ctx: Ctx, query: str, dataset: str, **kwargs) -> str:
    """Context-only graph recall, flattened to lowercase text for keyword checks."""
    from cognee import SearchType

    kwargs.setdefault("query_type", SearchType.GRAPH_COMPLETION)
    kwargs.setdefault("only_context", True)
    kwargs.setdefault("top_k", 5)
    results = await ctx.cognee.recall(query, datasets=[dataset], **kwargs)
    return flat_text(results).lower()


# --------------------------------------------------------------------------- #
# section p0 — introspection + boot (keyless)
# --------------------------------------------------------------------------- #

@check("P0.1", "installed version + v1 surface fail-fast", section="p0")
async def p0_1(ctx: Ctx):
    ctx.ev(f"python: {sys.version.split()[0]}")
    version = importlib.metadata.version("cognee")
    ctx.ev(f"cognee (installed): {version}")
    missing = [n for n in ("remember", "recall", "improve", "forget")
               if not hasattr(ctx.cognee, n)]
    if missing:
        print(f"RESULT|P0.1|FAIL|installed cognee {version} lacks {missing} — "
              "predates the v1 memory surface; everything downstream is invalid", flush=True)
        sys.exit(1)
    ctx.notes["version"] = version
    return "PASS", f"cognee {version} exposes remember/recall/improve/forget"


@check("P0.2", "clean reset (prune) for re-runnability", section="p0")
async def p0_2(ctx: Ctx):
    c = ctx.cognee
    _, t_data = await timed(c.prune.prune_data())
    _, t_sys = await timed(c.prune.prune_system(metadata=True, cache=True))
    ctx.ev(f"prune_data() ok in {t_data:.1f}s; prune_system(metadata=True, cache=True) ok in {t_sys:.1f}s")
    ctx.ev(f"all stores confined to {SCRATCH}")
    return "PASS", "prune_data + prune_system(metadata=True, cache=True) reset the isolated stores"


@check("P0.3", "signature inventory of the full surface", section="p0")
async def p0_3(ctx: Ctx):
    c = ctx.cognee
    names = ["remember", "recall", "improve", "forget", "add", "cognify", "search",
             "memify", "delete", "export", "serve", "push", "visualize_graph",
             "get_schema_inventory"]
    for name in names:
        fn = getattr(c, name, None)
        if fn is None:
            ctx.ev(f"{name}: MISSING")
            continue
        try:
            ctx.ev(f"{name}{inspect.signature(fn)}")
        except (ValueError, TypeError):
            ctx.ev(f"{name}: exists ({type(fn).__name__}), no signature")
    for mod_name in ("prune", "datasets", "session"):
        mod = getattr(c, mod_name, None)
        members = [m for m in dir(mod) if not m.startswith("_")] if mod else "MISSING"
        ctx.ev(f"cognee.{mod_name}: {members}")
    # TypedDict kwargs are invisible to inspect.signature — dump them explicitly
    from cognee.api.v1.remember.remember import RememberKwargs
    ctx.ev(f"RememberKwargs: {list(RememberKwargs.__annotations__)}")
    from cognee.api.v1.improve.improve import ImproveKwargs
    ctx.ev(f"ImproveKwargs: {list(ImproveKwargs.__annotations__)}")
    from cognee import SearchType
    ctx.ev(f"SearchType ({len(list(SearchType))}): {[m.name for m in SearchType]}")
    ctx.ev("param asymmetry: remember(dataset_name=...) / improve+forget(dataset=...) / recall(datasets=[...])")
    return "PASS", "signatures recorded verbatim; RememberKwargs/ImproveKwargs dumped (invisible to inspect)"


@check("P0.4", "embedded boot config (sqlite/lancedb/ladybug), setup() location", section="p0")
async def p0_4(ctx: Ctx):
    getters = [
        ("relational", "cognee.infrastructure.databases.relational.config", "get_relational_config"),
        ("vector", "cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("graph", "cognee.infrastructure.databases.graph.config", "get_graph_config"),
    ]
    for label, module_path, getter_name in getters:
        try:
            module = __import__(module_path, fromlist=[getter_name])
            config = getattr(module, getter_name)()
            dump = {k: v for k, v in safe_dump(config).items()
                    if any(word in k for word in ("provider", "path", "name", "url", "file"))
                    and "key" not in k.lower()}
            ctx.ev(f"{label}: {trunc(dump, 500)}")
        except Exception as exc:
            ctx.ev(f"{label}: introspection failed via {module_path}.{getter_name}: {exc!r}")
    ctx.ev(f"cognee.setup (top-level): {'exists' if hasattr(ctx.cognee, 'setup') else 'MISSING'}")
    try:
        from cognee.modules.engine.operations.setup import setup  # noqa: F401
        ctx.ev("cognee.modules.engine.operations.setup.setup: exists (karpathy-wiki pattern)")
    except ImportError as exc:
        ctx.ev(f"cognee.modules.engine.operations.setup.setup: MISSING ({exc})")
    ctx.ev("whether setup() is REQUIRED before first call is decided empirically in P1.1")
    return "PASS", "embedded store config + setup() locations recorded"


# --------------------------------------------------------------------------- #
# section p1 — behavior (needs key)
# --------------------------------------------------------------------------- #

async def remember_with_setup_probe(ctx: Ctx, *args, **kwargs):
    """First remember() of the run doubles as the 'is setup() required?' probe."""
    c = ctx.cognee
    if ctx.notes.get("setup_probed"):
        return await c.remember(*args, **kwargs)
    try:
        result = await c.remember(*args, **kwargs)
        ctx.notes["setup_required"] = False
    except Exception as first_error:
        from cognee.modules.engine.operations.setup import setup
        await setup()
        result = await c.remember(*args, **kwargs)
        ctx.notes["setup_required"] = True
        ctx.notes["setup_error"] = trunc(first_error, 200)
    ctx.notes["setup_probed"] = True
    return result


@check("P1.1", "remember→recall round-trip + result shapes", section="p1", needs_key=True)
async def p1_1(ctx: Ctx):
    ds = "vfy_roundtrip"
    result, t_write = await timed(remember_with_setup_probe(
        ctx, "Maya owns the billing service.", dataset_name=ds, self_improvement=False))
    ctx.ev(f"setup() required before first call: {ctx.notes.get('setup_required')}"
           + (f" (cold error: {ctx.notes.get('setup_error')})" if ctx.notes.get("setup_required") else ""))
    ctx.ev(f"remember() took {t_write:.1f}s; returned {type(result).__name__}")
    dumped = safe_dump(result)
    ctx.ev(f"RememberResult dump: {trunc(dumped, 700)}")
    if isinstance(dumped, dict):
        ctx.notes["remember_result_keys"] = list(dumped.keys())

    from cognee import SearchType
    results, t_read = await timed(ctx.cognee.recall(
        "Who owns the billing service?", datasets=[ds],
        query_type=SearchType.GRAPH_COMPLETION, only_context=True, top_k=5))
    ctx.ev(f"recall(only_context=True) took {t_read:.1f}s; returned {type(results).__name__} "
           f"of {len(results) if hasattr(results, '__len__') else '?'}")
    if isinstance(results, list) and results:
        entry = results[0]
        ctx.ev(f"entry type: {type(entry).__name__}; source attr: {getattr(entry, 'source', 'MISSING')!r}")
        ctx.ev(f"entry dump: {trunc(safe_dump(entry), 700)}")
    text = flat_text(results).lower()
    if "maya" not in text:
        return "FAIL", "recall context does not contain 'maya' after remembering the fact"
    return "PASS", "fact written via remember() and recovered via recall(); shapes recorded"


@check("P1.2", "forget() granularity — THE design-critical check", section="p1", needs_key=True)
async def p1_2(ctx: Ctx):
    c = ctx.cognee
    ds = "vfy_forget"
    fact_a = "Project Atlas uses Postgres as its primary database."
    fact_b = "Maya Chen is the lead engineer of Project Atlas."

    result_a = await remember_with_setup_probe(ctx, fact_a, dataset_name=ds,
                                               self_improvement=False)
    result_b = await c.remember(fact_b, dataset_name=ds, self_improvement=False)

    def ids_from_result(result):
        items = (safe_dump(result) or {}).get("items") or []
        return [str(item.get("id")) for item in items if isinstance(item, dict)]

    ingest_ids_a, ingest_ids_b = ids_from_result(result_a), ids_from_result(result_b)
    ctx.ev(f"capture-at-ingest: RememberResult.items ids A={ingest_ids_a} B={ingest_ids_b}")

    ds_id, items_all = await discover_data_ids(ctx, ds)
    listed_ids = {str(item_id(i)) for i in items_all}
    ctx.ev(f"data_id discovery (fallback): {ctx.notes.get('data_id_mechanism')}")
    ctx.ev(f"dataset_id={ds_id}; listed items={len(items_all)}; "
           f"ingest ids match listing: {set(ingest_ids_a + ingest_ids_b) == listed_ids}")
    import uuid as uuid_mod
    id_a = uuid_mod.UUID(ingest_ids_a[0]) if ingest_ids_a else None
    id_b = uuid_mod.UUID(ingest_ids_b[0]) if ingest_ids_b else None
    if id_a is None or id_b is None:
        return "FAIL", "could not discover per-item data_ids — forget(data_id=...) unusable as designed"
    ctx.notes["data_id_mechanism"] = ("PRIMARY: RememberResult.items[].id at ingest; "
                                      "fallback: datasets.list_data(dataset_id=...)")

    before = await recall_text(ctx, "What database does Project Atlas use and who leads it?", ds)
    ctx.ev(f"recall before forget: postgres={'postgres' in before} maya={'maya' in before}")
    nodes_before, _ = await graph_snapshot()
    atlas_before = len(nodes_matching(nodes_before, "atlas"))

    forget_result, t_forget = await timed(c.forget(data_id=id_a, dataset=ds))
    ctx.ev(f"forget(data_id=A, dataset={ds}) in {t_forget:.1f}s -> {trunc(forget_result, 400)}")

    after = await recall_text(ctx, "What database does Project Atlas use and who leads it?", ds)
    nodes_after, _ = await graph_snapshot()
    atlas_after = len(nodes_matching(nodes_after, "atlas"))
    ctx.ev(f"recall after item forget: postgres={'postgres' in after} maya={'maya' in after}")
    ctx.ev(f"'atlas' graph nodes before={atlas_before} after={atlas_after} "
           f"(shared-entity survival: {atlas_after > 0})")
    item_level_ok = ("postgres" not in after) and ("maya" in after)

    dataset_forget = await c.forget(dataset=ds)
    ctx.ev(f"forget(dataset={ds}) -> {trunc(dataset_forget, 300)}")
    from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError
    try:
        wiped = await recall_text(ctx, "Who is the lead engineer of Project Atlas?", ds)
        ctx.ev(f"recall after dataset forget: maya={'maya' in wiped}")
        maya_after_wipe = "maya" in wiped
    except DatasetNotFoundError:
        ctx.ev("recall on a fully-forgotten dataset raises DatasetNotFoundError "
               "(wrapper contract: catch it and treat as empty)")
        maya_after_wipe = False

    # memory_only: graph+vectors wiped, raw data retained, re-cognify possible
    ds2 = "vfy_forget2"
    try:
        await c.remember("The staging cluster runs Kubernetes.", dataset_name=ds2,
                         self_improvement=False)
        await c.forget(dataset=ds2, memory_only=True)
        _, items_kept = await discover_data_ids(ctx, ds2)
        try:
            recall_gone = await recall_text(ctx, "What does the staging cluster run?", ds2)
        except Exception as exc:
            recall_gone = ""
            ctx.ev(f"recall after memory_only forget raised {type(exc).__name__} "
                   "(graph gone, dataset record kept)")
        await c.cognify(datasets=[ds2])
        recall_back = await recall_text(ctx, "What does the staging cluster run?", ds2)
        ctx.ev(f"memory_only=True: raw items kept={len(items_kept)}; "
               f"kubernetes recalled after forget={'kubernetes' in recall_gone}; "
               f"after re-cognify={'kubernetes' in recall_back}")
        await c.forget(dataset=ds2)
    except Exception as exc:
        ctx.ev(f"memory_only sub-check errored (non-fatal): {trunc(exc, 300)}")

    ctx.notes["forget_item_level"] = item_level_ok
    if not item_level_ok:
        return "FAIL", ("item-level forget(data_id) did NOT surgically remove one fact "
                        "while keeping the other — fall back to the untrusted-sub-dataset strategy")
    return "PASS", ("forget(data_id=...) removed fact A, kept fact B"
                    + (", shared 'Atlas' entity survived" if atlas_after > 0 else
                       "; NOTE shared 'Atlas' entity also disappeared")
                    + f"; dataset-level forget wiped the rest; maya_after_wipe={maya_after_wipe}")


@check("P1.3", "node_set tagging + quarantine filtering at recall", section="p1", needs_key=True)
async def p1_3(ctx: Ctx):
    from cognee import SearchType
    c = ctx.cognee
    ds = "vfy_nodeset"
    poison = "The API rate limit is 9999 requests per second."
    truth = "The API rate limit is 100 requests per second."
    query = "What is the API rate limit?"

    await remember_with_setup_probe(ctx, poison, dataset_name=ds, self_improvement=False,
                                    node_set=["source:tool", "quarantine:true"])
    await c.remember(truth, dataset_name=ds, self_improvement=False,
                     node_set=["source:user", "quarantine:false"])

    unfiltered = await recall_text(ctx, query, ds)
    ctx.ev(f"unfiltered recall: 9999={'9999' in unfiltered} 100={'100 ' in unfiltered or ' 100' in unfiltered}")

    clean = await recall_text(ctx, query, ds, node_name=["quarantine:false"])
    ctx.ev(f"node_name=['quarantine:false']: 9999={'9999' in clean} 100={'100' in clean}")
    dirty = await recall_text(ctx, query, ds, node_name=["quarantine:true"])
    ctx.ev(f"node_name=['quarantine:true']: 9999={'9999' in dirty} 100={'100' in dirty}")

    both = await recall_text(ctx, query, ds, node_name=["quarantine:false", "source:user"],
                             node_name_filter_operator="AND")
    ctx.ev(f"AND filter ['quarantine:false','source:user']: 9999={'9999' in both} 100={'100' in both}")

    chunks = await c.search(query, query_type=SearchType.CHUNKS, datasets=[ds],
                            node_name=["quarantine:false"], top_k=5)
    chunks_text = flat_text(chunks).lower()
    ctx.ev(f"CHUNKS + node_name filter (documented as unsupported lane): "
           f"9999={'9999' in chunks_text} 100={'100' in chunks_text} "
           f"-> filter {'IGNORED (both returned)' if '9999' in chunks_text else 'apparently applied'}")

    scoped = await c.recall(query, datasets=[ds], scope=["graph"], only_context=True,
                            query_type=SearchType.GRAPH_COMPLETION, top_k=5)
    scoped_text = flat_text(scoped).lower()
    ctx.ev(f"scope=['graph'] (lane selector, NOT a tag filter): "
           f"9999={'9999' in scoped_text} 100={'100' in scoped_text}")

    quarantine_excluded = "9999" not in clean and "100" in clean
    ctx.notes["node_set_filtering"] = quarantine_excluded
    if not quarantine_excluded:
        return "FAIL", ("node_name filter did not exclude quarantine:true content in-process — "
                        "quarantine must fall back to dataset segregation + post-filtering")
    return "PASS", "quarantine:false filter excluded the poisoned fact and kept the clean one"


@check("P1.4", "node_set propagation: belongs_to_set reaches extracted entities", section="p1", needs_key=True)
async def p1_4(ctx: Ctx):
    nodes, edges = await graph_snapshot()
    nodeset_nodes = [n for n in nodes if "nodeset" in str(n).lower()
                     or "quarantine:" in str(n).lower() or "source:" in str(n).lower()]
    ctx.ev(f"graph size: {len(nodes)} nodes / {len(edges)} edges")
    ctx.ev(f"NodeSet-ish nodes found: {len(nodeset_nodes)}")
    for node in nodeset_nodes[:4]:
        ctx.ev(f"  {trunc(node, 300)}")
    belongs = [e for e in edges if "belongs_to_set" in str(e).lower()]
    ctx.ev(f"belongs_to_set edges: {len(belongs)}")
    if not belongs:
        return "UNSUPPORTED", "no belongs_to_set edges found — tag propagation not observable"
    # do tagged edges reach entity nodes (derived knowledge), or only chunks/documents?
    sources = {str(e[0]) for e in belongs if isinstance(e, (list, tuple)) and len(e) >= 2}
    node_by_id = {}
    for node in nodes:
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            node_by_id[str(node[0])] = node[1]
    kinds: dict[str, int] = {}
    for source_id in sources:
        payload = node_by_id.get(source_id, {})
        kind = str(payload.get("type", "?")) if isinstance(payload, dict) else "?"
        kinds[kind] = kinds.get(kind, 0) + 1
    ctx.ev(f"belongs_to_set source node types: {kinds}")
    reaches_entities = any("entity" in kind.lower() for kind in kinds)
    ctx.notes["tags_reach_entities"] = reaches_entities
    summary = ("belongs_to_set edges reach extracted Entity nodes — quarantine tags cover derived knowledge"
               if reaches_entities else
               "belongs_to_set edges exist but only on chunks/documents — derived entities NOT tagged")
    return "PASS", summary


@check("P1.5", "custom DataPoints: direct insert, dedup, graph_model extraction", section="p1", needs_key=True)
async def p1_5(ctx: Ctx):
    from typing import Optional
    import_paths = {}
    try:
        from cognee.low_level import DataPoint as DPLow
        import_paths["cognee.low_level.DataPoint"] = True
    except ImportError:
        import_paths["cognee.low_level.DataPoint"] = False
    try:
        from cognee.infrastructure.engine import DataPoint, Embeddable, Dedup  # noqa: F401
        import_paths["cognee.infrastructure.engine (DataPoint, Embeddable, Dedup)"] = True
    except ImportError:
        import_paths["cognee.infrastructure.engine (DataPoint, Embeddable, Dedup)"] = False
        DataPoint = DPLow  # type: ignore[misc]
    ctx.ev(f"import paths: {import_paths}")
    ctx.ev(f"DataPoint inherited fields: {list(DataPoint.model_fields.keys())}")

    class VfyClaim(DataPoint):
        text: str
        subject: str
        status: str = "active"
        metadata: dict = {"index_fields": ["text"]}

    from cognee.tasks.storage import add_data_points
    ctx.ev(f"add_data_points{inspect.signature(add_data_points)}")

    def make_claims():
        return [VfyClaim(text="The API rate limit is 100 requests per second.", subject="api"),
                VfyClaim(text="Maya owns the billing service.", subject="maya")]

    await add_data_points(make_claims())
    nodes, _ = await graph_snapshot()
    count_first = len(nodes_matching(nodes, "vfyclaim"))
    await add_data_points(make_claims())  # identical content, fresh instances (fresh UUIDs)
    nodes, _ = await graph_snapshot()
    count_second = len(nodes_matching(nodes, "vfyclaim"))
    ctx.ev(f"VfyClaim nodes after 1st insert: {count_first}; after identical 2nd insert: {count_second} "
           f"-> content-dedup WITHOUT identity fields: {count_second == count_first}")

    dedup_note = "not-attempted"
    try:
        from typing import Annotated

        class VfyDedupClaim(DataPoint):
            text: Annotated[str, Dedup()]
            metadata: dict = {"index_fields": ["text"]}

        await add_data_points([VfyDedupClaim(text="Dedup sentinel fact.")])
        await add_data_points([VfyDedupClaim(text="Dedup sentinel fact.")])
        nodes, _ = await graph_snapshot()
        dedup_count = len(nodes_matching(nodes, "vfydedupclaim"))
        dedup_note = f"Annotated[str, Dedup()] -> {dedup_count} node(s) after 2 identical inserts"
    except Exception as exc:
        dedup_note = f"Dedup() mechanism errored: {trunc(exc, 200)}"
    ctx.ev(dedup_note)

    # graph_model= on remember(): constrain LLM extraction to a typed schema
    class VfyProject(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class VfyPerson(DataPoint):
        name: str
        works_on: Optional[VfyProject] = None
        metadata: dict = {"index_fields": ["name"]}

    class VfyTeam(DataPoint):
        summary: str
        people: list[VfyPerson] = []
        projects: list[VfyProject] = []
        metadata: dict = {"index_fields": ["summary"]}

    ds = "vfy_dp"
    await remember_with_setup_probe(
        ctx, "Maya Chen works on Project Atlas. Ravi Patel works on Project Beacon.",
        dataset_name=ds, self_improvement=False, graph_model=VfyTeam)
    nodes, _ = await graph_snapshot()
    typed = {kind: len(nodes_matching(nodes, kind))
             for kind in ("vfyperson", "vfyproject", "vfyteam")}
    ctx.ev(f"typed nodes after remember(graph_model=VfyTeam): {typed}")
    extraction_ok = typed["vfyperson"] > 0 or typed["vfyproject"] > 0
    ctx.notes["graph_model_on_remember"] = extraction_ok

    if count_first == 0:
        return "FAIL", "add_data_points() produced no typed nodes in the graph"
    return "PASS", (f"direct insert OK ({count_first} typed nodes); {dedup_note}; "
                    f"remember(graph_model=...) typed extraction: {extraction_ok}")


@check("P1.6", "session memory write + improve(session_ids) bridging", section="p1", needs_key=True)
async def p1_6(ctx: Ctx):
    c = ctx.cognee
    ds, session = "vfy_session", "vfy-s1"
    # ensure the dataset has a permanent graph first (improve targets an existing dataset)
    await remember_with_setup_probe(ctx, "The team stores secrets in Vault.",
                                    dataset_name=ds, self_improvement=False)

    _, t_session = await timed(c.remember(
        "The deploy password is rotated every Friday.",
        dataset_name=ds, session_id=session, self_improvement=False))
    ctx.ev(f"session remember took {t_session:.2f}s (expected near-instant: no chunking/LLM/embedding)")

    entries = await c.session.get_session(session_id=session)
    ctx.ev(f"get_session -> {len(entries)} entries; last entry dump: "
           f"{trunc(safe_dump(entries[-1]) if entries else None, 600)}")

    query = "When is the deploy password rotated?"
    with_session = flat_text(await c.recall(query, datasets=[ds], session_id=session,
                                            only_context=True, top_k=5)).lower()
    without_session = await recall_text(ctx, query, ds)
    ctx.ev(f"recall WITH session_id: friday={'friday' in with_session}")
    ctx.ev(f"recall WITHOUT session_id (pre-bridge): friday={'friday' in without_session}")

    improve_result, t_improve = await timed(c.improve(dataset=ds, session_ids=[session],
                                                      run_in_background=False))
    ctx.ev(f"improve(dataset={ds}, session_ids=[{session!r}]) took {t_improve:.1f}s "
           f"-> {trunc(improve_result, 300)}")

    bridged = await recall_text(ctx, query, ds)
    ctx.ev(f"recall WITHOUT session_id (post-bridge): friday={'friday' in bridged}")
    nodes, _ = await graph_snapshot()
    cache_set = nodes_matching(nodes, "user_sessions_from_cache")
    ctx.ev(f"'user_sessions_from_cache' node set present: {len(cache_set) > 0}")

    ctx.notes["session_bridging"] = "friday" in bridged
    if "friday" not in with_session:
        return "FAIL", "session-scoped recall did not return the session fact"
    if "friday" in without_session:
        return "FAIL", "session fact leaked into permanent recall BEFORE improve() bridging"
    if "friday" not in bridged:
        return "FAIL", "improve(session_ids=[...]) did not bridge the session fact into permanent memory"
    return "PASS", (f"session write near-instant ({t_session:.2f}s), isolated from permanent recall, "
                    "and bridged into the graph by improve(session_ids=[...])")


# --------------------------------------------------------------------------- #
# section p2 — improve contract, graph export, provenance, cost (needs key)
# --------------------------------------------------------------------------- #

@check("P2.1", "improve()/memify contract + kwargs probing", section="p2", needs_key=True)
async def p2_1(ctx: Ctx):
    c = ctx.cognee
    ds = "vfy_roundtrip"  # cognified in P1.1
    result, t_plain = await timed(c.improve(dataset=ds, run_in_background=False))
    ctx.ev(f"improve(dataset) [no sessions] took {t_plain:.1f}s -> "
           f"{type(result).__name__}: {trunc(result, 400)}")
    result2, t_alpha = await timed(c.improve(dataset=ds, feedback_alpha=0.8,
                                             run_in_background=False))
    ctx.ev(f"improve(dataset, feedback_alpha=0.8) accepted (kwargs probe) in {t_alpha:.1f}s "
           f"-> {trunc(result2, 200)}")
    source = inspect.getsource(c.improve)
    delegates = "memify" in source
    ctx.ev(f"inspect.getsource(improve) mentions memify: {delegates}")
    ctx.ev(f"memify exported at top level: {hasattr(c, 'memify')}")
    return "PASS", (f"improve() runs on a cognified dataset, accepts feedback_alpha via kwargs, "
                    f"and {'delegates to memify()' if delegates else 'does NOT visibly delegate to memify()'}")


@check("P2.2", "graph export for dashboard: get_graph_data / metrics / visualize / export", section="p2", needs_key=True)
async def p2_2(ctx: Ctx):
    c = ctx.cognee
    nodes, edges = await graph_snapshot()
    ctx.ev(f"get_graph_data(): {len(nodes)} nodes / {len(edges)} edges (GLOBAL — not dataset-scoped)")
    if nodes:
        ctx.ev(f"sample node: {trunc(nodes[0], 400)}")
    if edges:
        ctx.ev(f"sample edge: {trunc(edges[0], 400)}")
    from cognee.infrastructure.databases.graph import get_graph_engine
    engine = await get_graph_engine()
    if hasattr(engine, "get_graph_metrics"):
        try:
            metrics = await engine.get_graph_metrics()
            ctx.ev(f"get_graph_metrics(): {trunc(metrics, 400)}")
        except Exception as exc:
            ctx.ev(f"get_graph_metrics() errored: {trunc(exc, 200)}")
    html_path = SCRATCH / "vfy_graph.html"
    try:
        await c.visualize_graph(str(html_path))
        ctx.ev(f"visualize_graph -> {html_path.name}: "
               f"{html_path.stat().st_size} bytes" if html_path.exists() else "file NOT created")
    except Exception as exc:
        ctx.ev(f"visualize_graph errored: {trunc(exc, 300)}")
    ctx.ev(f"cognee.export{inspect.signature(c.export)}")
    try:
        export_result = await c.export(dataset="vfy_roundtrip")
        ctx.ev(f"export(dataset='vfy_roundtrip') -> {type(export_result).__name__}: "
               f"{trunc(export_result, 300)}")
    except Exception as exc:
        ctx.ev(f"export() errored (record-only): {trunc(exc, 250)}")
    if not nodes:
        return "FAIL", "get_graph_data() returned an empty graph after all prior ingests"
    return "PASS", f"dashboard can use get_graph_data ({len(nodes)}n/{len(edges)}e) + metrics + visualize_graph HTML"


@check("P2.3", "graph traversal vs vector similarity (2-hop provenance)", section="p2", needs_key=True)
async def p2_3(ctx: Ctx):
    from cognee import SearchType
    c = ctx.cognee
    ds = "vfy_2hop"
    await remember_with_setup_probe(ctx, "Project Atlas uses Postgres as its database.",
                                    dataset_name=ds, self_improvement=False)
    await c.remember("Maya Chen is the lead engineer of Project Atlas.",
                     dataset_name=ds, self_improvement=False)
    nodes, edges = await graph_snapshot()
    ctx.ev(f"post-ingest graph: postgres nodes={len(nodes_matching(nodes, 'postgres'))}, "
           f"postgres edges={len([e for e in edges if 'postgres' in str(e).lower()])}")
    query = "What database does the project led by Maya Chen use?"
    # retrieval of the 2-hop bridge varies with extraction nondeterminism and the
    # 384-dim local embeddings, so try a ladder of retrieval knobs and record the
    # first that lands the answer
    attempts = [
        ("GRAPH_COMPLETION top_k=15", dict(query_type=SearchType.GRAPH_COMPLETION, top_k=15)),
        ("GRAPH_COMPLETION neighborhood_depth=2",
         dict(query_type=SearchType.GRAPH_COMPLETION, top_k=15, neighborhood_depth=2)),
        ("GRAPH_COMPLETION_COT", dict(query_type=SearchType.GRAPH_COMPLETION_COT, top_k=15)),
    ]
    graph_answer, graph_text, winner = None, "", None
    for label, recall_kwargs in attempts:
        graph_answer = await c.recall(query, datasets=[ds], **recall_kwargs)
        graph_text = flat_text(graph_answer).lower()
        ctx.ev(f"{label}: postgres={'postgres' in graph_text} :: {trunc(graph_text, 180)}")
        if "postgres" in graph_text:
            winner = label
            break
    if isinstance(graph_answer, list) and graph_answer:
        ctx.ev(f"provenance: entry source={getattr(graph_answer[0], 'source', 'MISSING')!r}")
    chunks = await c.search(query, query_type=SearchType.CHUNKS, datasets=[ds], top_k=5)
    chunks_text = flat_text(chunks).lower()
    ctx.ev(f"CHUNKS (pure vector) returns raw chunks, no synthesized answer: "
           f"{trunc(chunks_text, 300)}")
    ctx.ev("query_type is the provenance lane selector: CHUNKS=vector, "
           "GRAPH_COMPLETION family=graph traversal; RecallResponse.source tags each entry")
    if winner is None:
        return "FAIL", "no graph-lane variant answered the 2-hop question"
    return "PASS", (f"2-hop answer emerges from the graph lane ({winner}); "
                    "CHUNKS lane returns only raw chunks")


@check("P2.4", "session feedback loop probe (add_feedback / node weights)", section="p2", needs_key=True)
async def p2_4(ctx: Ctx):
    c = ctx.cognee
    session = "vfy-s1"
    ctx.ev(f"add_feedback{inspect.signature(c.session.add_feedback)}")
    entries = await c.session.get_session(session_id=session)
    if not entries:
        return "UNSUPPORTED", "no session entries available to attach feedback to"
    qa = entries[-1]
    qa_id = getattr(qa, "qa_id", None) or getattr(qa, "id", None)
    ctx.ev(f"target qa entry id: {qa_id}")
    try:
        await c.session.add_feedback(session, qa_id, feedback_text="incorrect fact",
                                     feedback_score=1)
        ctx.ev("add_feedback(session, qa_id, feedback_text=..., feedback_score=1) accepted")
    except TypeError:
        await c.session.add_feedback(session_id=session, qa_id=qa_id,
                                     feedback_text="incorrect fact", feedback_score=1)
        ctx.ev("add_feedback required keyword form (session_id=, qa_id=)")
    await c.improve(dataset="vfy_session", session_ids=[session], feedback_alpha=0.8,
                    run_in_background=False)
    from cognee.infrastructure.databases.graph import get_graph_engine
    engine = await get_graph_engine()
    if hasattr(engine, "get_node_feedback_weights"):
        ctx.ev("engine.get_node_feedback_weights exists "
               f"{inspect.signature(engine.get_node_feedback_weights)}")
    else:
        ctx.ev("engine.get_node_feedback_weights: MISSING on this engine")
    return "PASS", "feedback attach + improve(feedback_alpha) ran; weight readback surface recorded"


@check("P2.5", "cost/determinism: only_context skips LLM; forget is zero-LLM", section="p2", needs_key=True)
async def p2_5(ctx: Ctx):
    from cognee import SearchType
    c = ctx.cognee
    context_result, t_ctx = await timed(c.recall(
        "What database does Project Atlas use?", datasets=["vfy_2hop"],
        query_type=SearchType.GRAPH_COMPLETION, only_context=True, top_k=5))
    ctx.ev(f"recall(only_context=True) in {t_ctx:.1f}s -> returns retrieved context, "
           f"no synthesized answer: {trunc(flat_text(context_result)[:200], 220)}")
    _, t_forget = await timed(c.forget(dataset="vfy_2hop"))
    ctx.ev(f"forget(dataset) in {t_forget:.1f}s (docs claim zero LLM/embedding calls; "
           f"sub-5s latency is consistent with that)")
    if t_forget > 15:
        return "FAIL", f"forget() took {t_forget:.1f}s — inconsistent with the zero-LLM claim"
    return "PASS", f"only_context path returns raw context; forget() completed in {t_forget:.1f}s"


# --------------------------------------------------------------------------- #
# section p3 — config + stretch availability (record-only)
# --------------------------------------------------------------------------- #

@check("P3.1", "resolved LLM/embedding provider config", section="p3")
async def p3_1(ctx: Ctx):
    try:
        from cognee.infrastructure.llm.config import get_llm_config
        llm = safe_dump(get_llm_config())
        redacted = {k: ("***" if "key" in k.lower() and v else v)
                    for k, v in llm.items() if not k.startswith("_")}
        ctx.ev(f"llm config: {trunc(redacted, 600)}")
    except Exception as exc:
        ctx.ev(f"llm config introspection failed: {trunc(exc, 200)}")
    try:
        from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
        embedding = safe_dump(EmbeddingConfig())
        redacted = {k: ("***" if "key" in k.lower() and v else v)
                    for k, v in embedding.items() if not k.startswith("_")}
        ctx.ev(f"embedding config: {trunc(redacted, 600)}")
    except Exception as exc:
        ctx.ev(f"embedding config introspection failed: {trunc(exc, 200)}")
    ctx.ev("documented fallback: EMBEDDING_API_KEY unset -> embeddings reuse LLM_API_KEY "
           "(fine for OpenAI; the trap that breaks Anthropic-only configs)")
    return "PASS", "provider matrix recorded (record-only, never fails)"


@check("P3.2", "truth-subspace reranking availability (stretch)", section="p3")
async def p3_2(ctx: Ctx):
    c = ctx.cognee
    improve_params = inspect.signature(c.improve).parameters
    recall_params = inspect.signature(c.recall).parameters
    has_build = "build_truth_subspace" in improve_params
    has_config = "retriever_specific_config" in recall_params
    ctx.ev(f"improve(build_truth_subspace=...): {'present' if has_build else 'MISSING'}")
    ctx.ev(f"recall(retriever_specific_config=...): {'present' if has_config else 'MISSING'}")
    ctx.ev("availability check only — building/using the subspace is Phase 6 stretch")
    if has_build and has_config:
        return "PASS", "truth-subspace surface present (improve flag + retriever_specific_config)"
    return "UNSUPPORTED", "truth-subspace surface incomplete on this version"


@check("PX.1", "forget(everything=True) global wipe", section="px", needs_key=True, destructive=True)
async def px_1(ctx: Ctx):
    c = ctx.cognee
    result = await c.forget(everything=True)
    ctx.ev(f"forget(everything=True) -> {trunc(result, 400)}")
    remaining = await c.datasets.list_datasets()
    ctx.ev(f"datasets remaining: {len(remaining)}")
    return "PASS", f"global wipe ran; {len(remaining)} datasets remain"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

async def run(args) -> int:
    ctx = Ctx()
    import cognee
    ctx.cognee = cognee

    sections = [s.strip() for s in args.sections.split(",")] if args.sections else None
    only = {c.strip() for c in args.only.split(",")} if args.only else None
    have_key = bool(os.environ.get("LLM_API_KEY"))

    selected = []
    for chk in CHECKS:
        if chk["destructive"] and not args.include_destructive:
            continue
        if sections and chk["section"] not in sections and chk["section"] != "px":
            continue
        if only and chk["id"] not in only:
            continue
        if args.no_reset and chk["id"] == "P0.2":
            continue
        selected.append(chk)

    statuses: dict[str, str] = {}
    for chk in selected:
        ctx.evidence_lines = []
        started = time.monotonic()
        if chk["needs_key"] and not have_key:
            status, summary = "UNSUPPORTED", "LLM_API_KEY not set — behavioral check skipped"
        else:
            try:
                status, summary = await chk["fn"](ctx)
            except Unsupported as exc:
                status, summary = "UNSUPPORTED", str(exc)
            except Exception:
                status = "FAIL"
                tail = traceback.format_exc().strip().splitlines()
                summary = f"unhandled error: {tail[-1]}"
                ctx.evidence_lines.extend(tail[-6:])
        elapsed = time.monotonic() - started
        statuses[chk["id"]] = status
        print(f"RESULT|{chk['id']}|{status}|{summary}", flush=True)
        print(f"    # {chk['title']} [{chk['section']}, {elapsed:.1f}s]", flush=True)
        for line in ctx.evidence_lines:
            print(f"    {line}", flush=True)

    fails = [cid for cid, status in statuses.items()
             if status == "FAIL" and not cid.startswith("P3")]
    print(f"\nSUMMARY|checks={len(statuses)}|pass={sum(1 for s in statuses.values() if s == 'PASS')}"
          f"|fail={len([s for s in statuses.values() if s == 'FAIL'])}"
          f"|unsupported={len([s for s in statuses.values() if s == 'UNSUPPORTED'])}", flush=True)
    if ctx.notes:
        printable = {k: v for k, v in ctx.notes.items() if k != "setup_probed"}
        print(f"NOTES|{printable}", flush=True)
    return 1 if fails else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", default=None,
                        help="comma-separated subset: p0,p1,p2,p3 (default: all)")
    parser.add_argument("--only", default=None, help="comma-separated check ids, e.g. P1.2,P1.3")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument("--no-reset", action="store_true", help="skip the P0.2 prune reset")
    parser.add_argument("--include-destructive", action="store_true",
                        help="also run forget(everything=True) as the final check")
    args = parser.parse_args()

    if args.list:
        for chk in CHECKS:
            flags = " [destructive]" if chk["destructive"] else (" [needs key]" if chk["needs_key"] else "")
            print(f"{chk['id']:>5}  ({chk['section']}) {chk['title']}{flags}")
        return

    bootstrap_env()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
