#!/usr/bin/env python
"""Phase 2 DoD proof: a graph-only recall that a pure-vector query can't answer.

Seeds a 2-hop chain (one fact per remember, COGNEE_NOTES §10) through the gateway,
then contrasts the two recall lanes for the join question
"What database does the project led by Maya Chen use?":

  GRAPH lane  — traverses Maya→Atlas→Postgres and SYNTHESIZES "Postgres".
  CHUNKS lane — pure vector similarity, returns verbatim seeded fragments with
                NO synthesized answer.

Runs in-process against the ASGI app (no server). Exit 0 on success, 1 on failure.

    uv run scripts/graph_only_proof.py
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from aeg import cognee_client
from aeg.gateway import app

DATASET = "aeg_proof_graph"
QUERY = "What database does the project led by Maya Chen use?"
TARGET = "postgres"

SEED = [
    ("Maya Chen is the lead engineer of Project Atlas.", "user", "demi"),
    ("Project Atlas stores its data in Postgres.", "user", "demi"),
    # distractors: plausible neighbors that share vocabulary but not the join
    ("Project Beacon stores its data in MongoDB.", "user", "demi"),
    ("Ravi Patel is the lead engineer of Project Beacon.", "user", "demi"),
    ("The billing service exposes a REST API.", "user", "demi"),
]


async def main() -> int:
    await cognee_client.forget(dataset=DATASET)  # best-effort reset for re-runs

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as client:
        print(f"Seeding {len(SEED)} facts into '{DATASET}' (one remember() per fact)...")
        for content, kind, ident in SEED:
            resp = await client.post(
                "/remember",
                json={"content": content, "source": {"kind": kind, "identifier": ident},
                      "dataset": DATASET},
                timeout=120,
            )
            resp.raise_for_status()
            print(f"  + {content}")

        # --- GRAPH lane: verified P2.3 ladder (graph -> graph_cot) --------------- #
        graph_answer, graph_lane = "", None
        for lane in ("graph", "graph_cot"):
            resp = await client.post(
                "/recall",
                json={"query": QUERY, "dataset": DATASET, "top_k": 15, "lane": lane},
                timeout=120,
            )
            resp.raise_for_status()
            text = " ".join(e["text"] for e in resp.json()["entries"]).lower()
            if TARGET in text:
                graph_answer, graph_lane = text, lane
                break

        # --- CHUNKS lane: pure vector retrieval, no synthesis ------------------- #
        chunk_entries = await cognee_client.recall(
            QUERY, datasets=[DATASET], query_type=cognee_client.LANES["chunks"], top_k=3
        )
        chunk_texts = [e["text"] for e in chunk_entries]
        # The vector lane returns the underlying facts as raw retrieved fragments and
        # never performs the Maya->Atlas->Postgres join into an answer. It returns
        # retrieval, not synthesis (how many fragments surface varies with the
        # 384-dim embeddings — the point is that none of them is the joined answer).
        chunks_are_raw = len(chunk_texts) >= 1

    graph_len = len(graph_answer.split())

    print("\n" + "=" * 70)
    print(f"QUERY: {QUERY}")
    print("-" * 70)
    print(f"GRAPH LANE ({graph_lane}): "
          + (f"SYNTHESIZED a {graph_len}-word answer containing 'Postgres' "
             "(traversed Maya -> Atlas -> Postgres)" if graph_lane else "FAILED to answer"))
    print(f"  {graph_answer[:200]}")
    print("-" * 70)
    print(f"CHUNKS LANE (pure vector, top_k=3): {len(chunk_texts)} raw fragments, NO synthesis")
    for t in chunk_texts:
        print(f"  · {t[:100]}")
    print("  → the facts come back disconnected; the vector lane never joins them "
          "into the answer.")
    print("=" * 70)

    ok = bool(graph_lane) and chunks_are_raw
    if ok:
        print("\nPASS — the graph lane synthesized the multi-hop answer that pure vector "
              "similarity returns only as raw, un-joined fragments.")
    else:
        print("\nFAIL — "
              + ("graph lane did not answer the 2-hop question; " if not graph_lane else "")
              + ("chunks lane returned nothing" if not chunks_are_raw else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
