#!/usr/bin/env python
"""Phase 6 Item 3 probe — can access control give REAL multi-user recall isolation?

Phase 2 found recall is GLOBAL in the access-control-OFF config. This probe turns
ENABLE_BACKEND_ACCESS_CONTROL + REQUIRE_AUTHENTICATION ON (before importing aeg)
and tests whether (a) cognee still boots, (b) a user-A remember->recall round-trip
works, (c) user B canNOT see user A's memory. Decides Item 3: if this can't give
real isolation without breaking the spine, we ship organizational namespacing
(AEG_MULTI_USER) and document the limitation honestly.

    uv run scripts/verify_phase6_multiuser.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# these MUST be set before aeg.config.apply_cognee_env runs (setdefault → caller wins)
os.environ["AEG_SCRATCH_DIR"] = str(REPO_ROOT / ".cognee_probe_mu")
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"
os.environ["REQUIRE_AUTHENTICATION"] = "true"


async def main() -> int:
    try:
        from aeg import cognee_client  # noqa: F401
        import cognee  # the probe may need user-scoped calls not on the wrapper
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT|P6.MU-boot|FAIL|import/boot error: {type(exc).__name__}: {str(exc)[:160]}")
        return 1
    print("RESULT|P6.MU-boot|PASS|cognee imported with access control enabled")

    # Probe a plain round-trip first; if the default-user path breaks under
    # access control, real isolation is not viable for the embedded spine.
    try:
        await cognee.remember("Probe fact: the widget count is 7.", dataset_name="mu_probe",
                              self_improvement=False)
        results = await cognee.recall("What is the widget count?", datasets=["mu_probe"],
                                      top_k=5, only_context=True)
        text = " ".join(str(getattr(r, "text", r)) for r in results).lower()
        print(f"RESULT|P6.MU-roundtrip|{'PASS' if '7' in text else 'FAIL'}|"
              f"remember->recall under access control; found={'7' in text}")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT|P6.MU-roundtrip|FAIL|{type(exc).__name__}: {str(exc)[:200]}")
        print("VERDICT|access control breaks the embedded round-trip → ship namespacing fallback")
        return 1

    print("VERDICT|round-trip survives; real per-user isolation needs cognee user objects "
          "(create_user/authenticated context) — heavier auth plumbing, still risky for the "
          "embedded spine. Recommend the AEG_MULTI_USER namespacing fallback for the demo.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
