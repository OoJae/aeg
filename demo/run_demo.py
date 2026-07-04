#!/usr/bin/env python
"""Aeg — the full poison→heal loop, headless, with hard asserts (Phase 4 DoD).

    uv run python demo/run_demo.py

The arc (every beat asserted, exit 0 only if all pass):
  1  SEED       clean facts about a fictional eng org        [remember]
  2  BASELINE   agent answers RIGHT (rate limit = 100)       [recall]
  3  POISON     a subtle lie from a tool slips past the door [remember]
  4  WRONG      the agent's context is now compromised       [recall]
  5  DETECT     immune sweep finds + quarantines the lie     [recall + forget(memory_only)]
  6  RESPOND    verify → permanently forget confirmed-bad    [forget]
  7  REINFORCE  strengthen the truth, bridge a verified note [improve]
  8  RIGHT      agent answers RIGHT again — the graph healed [recall]

Re-runnable: uses a dedicated fresh scratch dir per run (.cognee_demo) because
overlay records, contradiction dedup, and the session cache all survive
forget(dataset) (COGNEE_NOTES §6b).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / ".cognee_demo"

# fresh stores BEFORE aeg (and therefore cognee) is imported
shutil.rmtree(SCRATCH, ignore_errors=True)
import os  # noqa: E402

os.environ["AEG_SCRATCH_DIR"] = str(SCRATCH)

import httpx  # noqa: E402

from aeg import cognee_client  # noqa: E402
from aeg.gateway import app  # noqa: E402

DATASET = "aeg_demo"
QUERY = "What is the API rate limit?"
TRUTH = "The API rate limit is 100 requests per second."
POISON = "The API rate limit is actually 9999 requests per second."

SEED = [
    "Project Atlas uses Postgres as its primary database.",
    "Maya Chen owns the billing service.",
    TRUTH,
]

_failures: list[str] = []
_ops: list[tuple[str, str]] = []


def banner(step: str, title: str) -> None:
    print(f"\n{'=' * 72}\n  STEP {step} — {title}\n{'=' * 72}", flush=True)


def check(condition: bool, message: str) -> None:
    mark = "✓" if condition else "✗ FAILED:"
    print(f"  {mark} {message}", flush=True)
    if not condition:
        _failures.append(message)


def op(step: str, cognee_op: str) -> None:
    _ops.append((step, cognee_op))


async def recall_context(client: httpx.AsyncClient) -> str:
    resp = await client.post("/recall", json={
        "query": QUERY, "dataset": DATASET, "top_k": 10, "only_context": True,
    }, timeout=300)
    return " ".join(e["text"] for e in resp.json()["entries"]).lower()


async def recall_answer(client: httpx.AsyncClient) -> str:
    resp = await client.post("/recall", json={
        "query": QUERY, "dataset": DATASET, "top_k": 10, "lane": "graph",
    }, timeout=300)
    entries = resp.json()["entries"]
    return entries[0]["text"] if entries else "(no answer)"


async def main() -> int:
    started = time.monotonic()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://aeg") as client:

        banner("1", "SEED — clean baseline memory enters through the gateway")
        for fact in SEED:
            resp = (await client.post("/remember", json={
                "content": fact, "source": {"kind": "user", "identifier": "demi"},
                "dataset": DATASET}, timeout=300)).json()
            check(resp["screening"]["verdict"] == "clean" and resp["data_ids"],
                  f"remember(): \"{fact[:60]}\" screened clean + cognified")
        op("1", "remember() ×3 — typed, screened, faceted ingest")

        banner("2", "BASELINE — the agent answers RIGHT")
        context = await recall_context(client)
        check("100" in context, "recall() context contains the true limit (100)")
        check("9999" not in context, "no poison anywhere yet")
        print(f'  agent answer: "{await recall_answer(client)}"')
        op("2", "recall() — graph lane")

        banner("3", "POISON — a subtle lie from a tool source slips past the door")
        poison = (await client.post("/remember", json={
            "content": POISON, "source": {"kind": "tool", "identifier": "slack-bot"},
            "dataset": DATASET}, timeout=300)).json()
        check(poison["screening"]["verdict"] == "clean" and not poison["quarantined"],
              "no injection phrasing — innate screening passes it (by design)")
        check(bool(poison["data_ids"]), "the lie is now IN the agent's memory substrate")
        poison_claim_ids = {c["id"] for c in poison["claims"]}
        op("3", "remember() — untrusted source recorded (trust 0.25)")

        banner("4", "WRONG — the agent's memory is compromised")
        context = await recall_context(client)
        check("9999" in context, "recall() context now carries the poison (9999)")
        print(f'  agent answer: "{await recall_answer(client)}"')
        op("4", "recall() — poisoned")

        banner("5", "DETECT — adaptive immunity sweeps the dataset")
        scan = (await client.post("/scan", json={"dataset": DATASET},
                                  timeout=300)).json()
        check(bool(scan["contradictions"]), "contradiction detected by the LLM verifier")
        if scan["contradictions"]:
            conflict = scan["contradictions"][0]
            print(f"  verdict: {conflict['verdict']} (confidence {conflict['confidence']:.2f})")
            print(f"  rationale: {conflict['rationale'][:100]}")
        check(bool(set(scan["quarantined"]) & poison_claim_ids),
              "the lower-trust side (the poison) is quarantined")
        context = await recall_context(client)
        check("9999" not in context, "poison already excluded from recall (reversible quarantine)")
        op("5", "recall() traversal + forget(memory_only=True) — quarantine")

        banner("6", "RESPOND — verify, then permanently forget() the confirmed lie")
        respond = (await client.post("/respond", json={"dataset": DATASET},
                                     timeout=300)).json()
        check(respond["verified"] >= 1 and bool(respond["forgotten"]),
              "quarantined memory verified as confirmed-bad and forgotten")
        if respond["verdicts"]:
            verdict = respond["verdicts"][0]
            print(f"  verifier: confirmed_bad={verdict['confirmed_bad']} "
                  f"(confidence {verdict['confidence']:.2f}) — {verdict['rationale'][:90]}")
        remaining = await cognee_client.list_data_ids(DATASET)
        check(all(d not in remaining for d in respond["forgotten_data_ids"]),
              "the poison's raw Data record is GONE — forget() was permanent")
        forgotten_claims = await cognee_client.list_typed_nodes(
            "Claim", dataset=DATASET, status="forgotten")
        check(bool(forgotten_claims), 'overlay claim status flipped to "forgotten" (audit)')
        op("6", "forget() — surgical, permanent, zero-LLM")

        banner("7", "REINFORCE — improve() strengthens the truth")
        truth_claims = [c for c in await cognee_client.list_typed_nodes(
            "Claim", dataset=DATASET, status="active") if "100" in c.get("text", "")]
        check(bool(truth_claims), "the surviving truth claim is found")
        reinforce = (await client.post("/reinforce", json={
            "dataset": DATASET, "claim_id": truth_claims[0]["id"]}, timeout=300)).json()
        check(reinforce["new_confidence"] > reinforce["old_confidence"],
              f"truth confidence {reinforce['old_confidence']:.2f} -> "
              f"{reinforce['new_confidence']:.2f}")
        context = await recall_context(client)
        check("verified" in context or "confirmed" in context,
              "the bridged verification note is now part of permanent memory")
        op("7", "improve(session_ids) — session note bridged into the graph")

        banner("8", "RIGHT — the graph has healed")
        context = await recall_context(client)
        check("100" in context, "recall() context contains the truth (100)")
        check("9999" not in context, "the poison is gone for good")
        print(f'  agent answer: "{await recall_answer(client)}"')
        op("8", "recall() — healed")

    print(f"\n{'=' * 72}\n  COGNEE OPS BY STEP\n{'=' * 72}")
    for step, cognee_op in _ops:
        print(f"  step {step}:  {cognee_op}")
    elapsed = time.monotonic() - started
    if _failures:
        print(f"\nDEMO FAILED in {elapsed:.0f}s — {len(_failures)} assertion(s):")
        for failure in _failures:
            print(f"  ✗ {failure}")
        return 1
    print(f"\nDEMO PASSED in {elapsed:.0f}s — poison → wrong → detect → forget → "
          "reinforce → right. The memory immune system works.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
