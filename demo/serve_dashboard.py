#!/usr/bin/env python
"""Launch the Aeg dashboard on a fresh, clean store.

    uv run python demo/serve_dashboard.py

Wipes a dedicated scratch dir and sets AEG_SCRATCH_DIR BEFORE importing aeg, so
every launch is an authoritative clean slate (POST /demo/reset is admin-only now —
restarting this launcher is the guaranteed reset). Its own scratch dir
(.cognee_dashboard) never clobbers a concurrent run_demo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / ".cognee_dashboard"

shutil.rmtree(SCRATCH, ignore_errors=True)
os.environ["AEG_SCRATCH_DIR"] = str(SCRATCH)

import uvicorn  # noqa: E402

from aeg.gateway import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("AEG_PORT", "8080"))
    print(f"\n  Aeg landing → http://localhost:{port}/"
          f"\n  Live monitor → http://localhost:{port}/dashboard"
          f"\n  Manifesto   → http://localhost:{port}/how\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
