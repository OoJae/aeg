#!/usr/bin/env python
"""Phase 6 Item 2 probe — is truth-subspace reranking usable on MiMo+fastembed?

Surface confirmed present in Phase 0 (§9). This checks it RUNS end-to-end:
  improve(build_truth_subspace=True) and recall(retriever_specific_config=
  {"use_truth_weight": True}). Prints RESULT| lines; if either errors, the
  AEG_TRUTH_SUBSPACE flag stays off and we document UNSUPPORTED.

    uv run scripts/verify_phase6_truth.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("AEG_SCRATCH_DIR", str(REPO_ROOT / ".cognee_probe_truth"))

from aeg import cognee_client  # noqa: E402

DS = "vfy_truth"


async def main() -> int:
    await cognee_client.forget(dataset=DS)
    for fact in ("Vega is written in Rust.", "Vega ships weekly.", "Vega is owned by the core team."):
        await cognee_client.remember(fact, dataset=DS)

    ok = True

    try:
        result = await cognee_client.improve(dataset=DS, build_truth_subspace=True)
        print(f"RESULT|P6.TS-build|PASS|improve(build_truth_subspace=True) ran -> {type(result).__name__}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"RESULT|P6.TS-build|FAIL|{type(exc).__name__}: {str(exc)[:160]}")

    try:
        entries = await cognee_client.recall(
            "What language is Vega written in?", datasets=[DS],
            query_type=cognee_client.LANES["graph"], top_k=10,
            retriever_specific_config={"use_truth_weight": True},
        )
        text = " ".join(e["text"] for e in entries).lower()
        print(f"RESULT|P6.TS-recall|PASS|recall(retriever_specific_config=use_truth_weight) ran; "
              f"rust_in_answer={'rust' in text}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"RESULT|P6.TS-recall|FAIL|{type(exc).__name__}: {str(exc)[:160]}")

    await cognee_client.forget(dataset=DS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
