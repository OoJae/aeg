#!/usr/bin/env python
"""Phase 3 truth-check probes (Phase-0 discipline: verify before feature code).

P3.A — item-level forget(data_id, dataset, memory_only=True): substrate removal
       with raw Data record kept, and cognify(datasets=[ds]) restores ONLY the
       reset item with the SAME data_id. (Source-confirmed; this is the runtime
       confirmation.) This is the reversible-quarantine primitive.
P3.B — add_data_points with an existing node id is an UPSERT (MERGE ON MATCH),
       relationship fields become edges, and list_typed_nodes payloads carry the
       injected node id. This is the status/confidence-flip mechanism.

    uv run scripts/verify_phase3_probes.py

Output: RESULT|<id>|<PASS|FAIL>|<summary> + indented evidence (COGNEE_NOTES style).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("AEG_SCRATCH_DIR", str(REPO_ROOT / ".cognee_probe3"))

from aeg import cognee_client  # noqa: E402  (applies env before cognee import)
from aeg.ontology import Claim, Contradiction, deterministic_id  # noqa: E402

DS = "vfy3_items"


def ev(line: str) -> None:
    print(f"    {line}", flush=True)


async def probe_a() -> bool:
    from cognee import datasets as cognee_datasets  # probe-only import

    await cognee_client.forget(dataset=DS)
    r_a = await cognee_client.remember("The Vfy3 service uses Redis for caching.", dataset=DS)
    ids_a = [str(i["id"]) for i in r_a.get("items", [])]
    r_b = await cognee_client.remember("Nora Okafor leads the Vfy3 service.", dataset=DS)
    ids_b = [str(i["id"]) for i in r_b.get("items", [])]
    data_id_a = ids_a[0]
    ev(f"data_id_a={data_id_a}; items after 2nd ingest={len(ids_b)} (cumulative)")

    q = "What does the Vfy3 service use for caching?"
    before = " ".join(e["text"] for e in await cognee_client.recall(
        q, datasets=[DS], query_type=cognee_client.LANES["chunks"], top_k=5)).lower()
    ev(f"pre-forget CHUNKS recall: redis={'redis' in before}")
    if "redis" not in before:
        ev("baseline failed — redis fact not recallable before forget")
        return False

    result = await cognee_client.forget(dataset=DS, data_id=data_id_a, memory_only=True)
    ev(f"forget(data_id, dataset, memory_only=True) -> {result}")
    if result.get("status") not in ("success", "completed"):
        return False

    after = " ".join(e["text"] for e in await cognee_client.recall(
        q, datasets=[DS], query_type=cognee_client.LANES["chunks"], top_k=5)).lower()
    nora = " ".join(e["text"] for e in await cognee_client.recall(
        "Who leads the Vfy3 service?", datasets=[DS],
        query_type=cognee_client.LANES["chunks"], top_k=5)).lower()
    ev(f"post-forget: redis={'redis' in after} (want False); nora={'nora' in nora} (want True)")
    if "redis" in after or "nora" not in nora:
        return False

    all_datasets = await cognee_datasets.list_datasets()
    ds_id = next((d.id for d in all_datasets if getattr(d, "name", None) == DS), None)
    kept = await cognee_datasets.list_data(dataset_id=ds_id)
    kept_ids = [str(getattr(i, "id", None)) for i in kept]
    ev(f"raw Data records kept after memory_only forget: {kept_ids}")
    if data_id_a not in kept_ids:
        return False

    started = time.monotonic()
    await cognee_client.recognify(DS)
    elapsed = time.monotonic() - started
    restored = " ".join(e["text"] for e in await cognee_client.recall(
        q, datasets=[DS], query_type=cognee_client.LANES["chunks"], top_k=5)).lower()
    kept2 = [str(getattr(i, "id", None))
             for i in await cognee_datasets.list_data(dataset_id=ds_id)]
    ev(f"recognify took {elapsed:.1f}s (1 reset item); redis restored={'redis' in restored}; "
       f"data ids unchanged={sorted(kept2) == sorted(kept_ids)}")
    await cognee_client.forget(dataset=DS)
    return "redis" in restored and sorted(kept2) == sorted(kept_ids)


async def probe_b() -> bool:
    key = deterministic_id("vfy3", "claim", "upsert sentinel")

    def make(status: str, confidence: float) -> Claim:
        return Claim(id=key, text="upsert sentinel", subject="sentinel", predicate="is",
                     object="probed", confidence=confidence, status=status, dataset="vfy3")

    def by_key(payloads: list[dict]) -> list[dict]:
        # scope to THIS probe's sentinel id — overlay nodes persist across runs
        # (never removed by forget), so dataset-wide counts are not re-runnable
        return [p for p in payloads if p.get("id") == str(key)]

    await cognee_client.add_data_points([make("active", 0.6)])
    first = by_key(await cognee_client.list_typed_nodes("Claim", dataset="vfy3"))
    ev(f"after 1st insert: {len(first)} node(s) with sentinel id; "
       f"status={first[0].get('status')}; confidence={first[0].get('confidence')}; "
       f"id_injected={'id' in first[0]}")

    await cognee_client.add_data_points([make("quarantined", 0.24)])
    second = by_key(await cognee_client.list_typed_nodes("Claim", dataset="vfy3"))
    ev(f"after same-id re-insert: {len(second)} node(s); status={second[0].get('status')}; "
       f"confidence={second[0].get('confidence')}")
    upsert_ok = (len(second) == 1 and second[0].get("status") == "quarantined"
                 and abs(second[0].get("confidence", 0) - 0.24) < 1e-6
                 and second[0].get("id"))

    claim_a = make("active", 0.6)
    claim_b = Claim(id=deterministic_id("vfy3", "claim", "other"), text="other",
                    subject="sentinel", predicate="is not", object="probed",
                    confidence=0.25, status="active", dataset="vfy3")
    ordered = sorted([claim_a, claim_b], key=lambda c: str(c.id))
    conflict = Contradiction(
        id=deterministic_id("contradiction", str(ordered[0].id), str(ordered[1].id)),
        claim_a=ordered[0], claim_b=ordered[1],
        claim_a_id=str(ordered[0].id), claim_b_id=str(ordered[1].id),
        dataset="vfy3", confidence=0.9, verdict="a_wins", rationale="probe",
    )
    await cognee_client.add_data_points([conflict, *ordered])
    payloads = await cognee_client.list_typed_nodes("Contradiction", dataset="vfy3")
    _, edges = await cognee_client.export_graph()
    edge_names = {str(e[2]) for e in edges if isinstance(e, (list, tuple)) and len(e) >= 3}
    has_claim_edges = any("claim" in n for n in edge_names)
    ev(f"contradiction payloads: {len(payloads)}; scalar mirrors present="
       f"{bool(payloads and payloads[0].get('claim_a_id'))}; "
       f"verdict={payloads[0].get('verdict') if payloads else None}")
    ev(f"relationship edges present (claim_a/claim_b as edges): {has_claim_edges} "
       f"(sample edge names: {sorted(n for n in edge_names if 'claim' in n)[:4]})")
    return bool(upsert_ok and payloads and payloads[0].get("claim_a_id") and has_claim_edges)


async def main() -> int:
    ok = True
    for cid, title, fn in (("P3.A", "item-level memory_only forget + incremental restore", probe_a),
                           ("P3.B", "overlay same-id upsert + edge/payload semantics", probe_b)):
        try:
            passed = await fn()
        except Exception as exc:  # fail-soft with evidence
            import traceback
            ev("\n".join(traceback.format_exc().splitlines()[-4:]))
            passed = False
            ev(f"unhandled: {exc!r}")
        print(f"RESULT|{cid}|{'PASS' if passed else 'FAIL'}|{title}", flush=True)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
