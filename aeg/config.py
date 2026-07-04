"""Aeg configuration: dataset names, node_set facets, profiles, cognee env.

Dataset layout (COGNEE_NOTES §4 — forget strategy agreed at Checkpoint 1):
item-level forget(data_id, dataset) is the primary response, so trusted and
untrusted memory can share DATASET_MAIN. DATASET_UNTRUSTED stays pre-wired as
the fallback (droppable wholesale via forget(dataset=...)) but is unused unless
item-level forget regresses.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- datasets -------------------------------------------------------------- #

DATASET_MAIN = "aeg_main"              # permanent agent memory
DATASET_UNTRUSTED = "aeg_untrusted"    # droppable fallback for untrusted ingests
DATASET_ANTIBODIES = "aeg_antibodies"  # attack-pattern meta-memory (Phase 6)
TEST_DATASET_PREFIX = "aeg_test_"


def session_id(name: str) -> str:
    return f"aeg-session-{name}"


# --- node_set facets (build guide §6; verified in COGNEE_NOTES §6) ---------- #

FACET_QUARANTINE_TRUE = "quarantine:true"
FACET_QUARANTINE_FALSE = "quarantine:false"

SourceKind = Literal["user", "document", "tool", "agent", "unknown"]
TrustTier = Literal["trusted", "normal", "untrusted"]


def facet_source(kind: SourceKind) -> str:
    return f"source:{kind}"


def facet_trust(tier: TrustTier) -> str:
    return f"trust:{tier}"


# --- profiles --------------------------------------------------------------- #

Profile = Literal["embedded", "postgres", "cloud"]

AEG_PROFILE: Profile = os.environ.get("AEG_PROFILE", "embedded")  # type: ignore[assignment]
AEG_COGNEE_URL = os.environ.get("AEG_COGNEE_URL", "")        # cloud: cognee.serve(url=...)
AEG_COGNEE_API_KEY = os.environ.get("AEG_COGNEE_API_KEY", "")


# --- Phase-6 stretch feature flags (read at CALL time via config.X so tests can
# monkeypatch; each item is independently gated so the embedded spine is safe). #

def _truthy(value: str) -> bool:
    return value.strip().lower() not in ("", "0", "false", "no", "off")


AEG_ANTIBODIES_ENABLED = _truthy(os.environ.get("AEG_ANTIBODIES_ENABLED", "true"))
AEG_TRUTH_SUBSPACE = _truthy(os.environ.get("AEG_TRUTH_SUBSPACE", "false"))
AEG_MULTI_USER = _truthy(os.environ.get("AEG_MULTI_USER", "false"))


def user_dataset(user_id: str) -> str:
    """Per-user dataset namespace (Phase 6 multi-user, organizational only —
    recall stays global in the embedded access-control-off config)."""
    return f"aeg_user_{user_id}"


# --- security / abuse-resistance (adversarial-study hardening) -------------- #
# The gateway is a PUBLIC, paid-LLM, persistent service. These bound spend, data
# destruction, and memory/disk growth from unauthenticated traffic. All are read
# via config.X at call time so tests can monkeypatch them.

def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Optional shared-secret auth. UNSET => open demo mode (as deployed, so judges can
# click through). SET => mutating/LLM routes require the X-Aeg-Key header, and the
# admin-only /demo/reset becomes usable. One var locks down a self-host/prod run.
AEG_API_KEY = os.environ.get("AEG_API_KEY", "").strip()

# When false, ignore the client-declared source.kind and treat all ingress as
# untrusted (prod). Default true preserves the demo's provenance/trust-tier story.
AEG_TRUST_CLIENT_KIND = _truthy(os.environ.get("AEG_TRUST_CLIENT_KIND", "true"))

# Per-IP sliding-window request cap (<=0 disables — tests set it high). Best-effort
# only (X-Forwarded-For is spoofable); the global LLM budget is the real, un-
# spoofable wallet guard.
AEG_RATE_LIMIT = _int_env("AEG_RATE_LIMIT", 20)
AEG_RATE_WINDOW_SECONDS = _int_env("AEG_RATE_WINDOW_SECONDS", 300)

# Hard ceiling on LLM-triggering calls per rolling 24h; over budget => 503. This
# is the wallet kill-switch: it holds even if per-IP limiting is evaded.
AEG_DAILY_LLM_BUDGET = _int_env("AEG_DAILY_LLM_BUDGET", 2000)

# Input / growth caps.
AEG_MAX_CONTENT_CHARS = _int_env("AEG_MAX_CONTENT_CHARS", 8000)
AEG_MAX_DATASETS = _int_env("AEG_MAX_DATASETS", 128)        # bound per-dataset gateway state
AEG_MAX_IP_BUCKETS = _int_env("AEG_MAX_IP_BUCKETS", 8192)   # bound the rate-limiter map

_DATASET_RE = re.compile(r"^aeg_[a-z0-9_]{1,48}$")


def is_valid_dataset(name: str) -> bool:
    """A client-supplied dataset must be a bounded aeg_* slug. Rejects the
    arbitrary/unbounded strings that would grow locks/seen_ids and disk without
    limit (adversarial-study HIGH: dataset DoS)."""
    return isinstance(name, str) and bool(_DATASET_RE.match(name))


# --- cognee environment ------------------------------------------------------ #

_applied = False


def apply_cognee_env(scratch_root: Path | None = None) -> None:
    """Configure cognee via env vars. MUST run before `import cognee`.

    cognee reads these at import; without the *_ROOT_DIRECTORY overrides it
    writes its stores inside site-packages (COGNEE_NOTES, binding environment).
    Uses setdefault throughout, so callers (tests, verify script) win by
    exporting values first — e.g. AEG_SCRATCH_DIR.
    """
    global _applied
    if _applied:
        return
    load_dotenv(REPO_ROOT / ".env")
    scratch_env = os.environ.get("AEG_SCRATCH_DIR", "").strip()
    root = scratch_root or Path(scratch_env or (REPO_ROOT / ".cognee"))
    for sub in ("data", "system", "cache", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # persistence guard: an unset AEG_SCRATCH_DIR falls back to a container-local
    # path (or /tmp) that is wiped on restart/redeploy, so memory silently would
    # not survive. On the Railway deploy AEG_SCRATCH_DIR points at the mounted
    # volume (/data/...) — warn loudly whenever it is unset or ephemeral so a
    # dropped volume var can't cause silent data loss.
    if scratch_root is None and (not scratch_env or str(root).startswith("/tmp") or "/tmp/" in str(root)):
        logging.getLogger("aeg").warning(
            "Persistence is not guaranteed (AEG_SCRATCH_DIR=%s → %s): memory will "
            "NOT survive a restart/redeploy unless this points at a persistent "
            "volume. Set AEG_SCRATCH_DIR in production.", scratch_env or "(unset)", root)
    defaults = {
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "CACHE_ROOT_DIRECTORY": str(root / "cache"),
        "COGNEE_LOGS_DIR": str(root / "logs"),
        "COGNEE_LOG_FILE": "false",
        "COGNEE_MINIMAL_LOGGING": "true",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        "CACHING": "true",
        "CACHE_BACKEND": "fs",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    # Phase-6 Postgres profile: relational + vector move to Postgres/pgvector; the
    # graph store stays embedded Ladybug (a credible profile without a second
    # external service). Needs `docker compose up` + the cognee[postgres] extra.
    if os.environ.get("AEG_PROFILE") == "postgres":
        for key, value in {
            "DB_PROVIDER": "postgres",
            "VECTOR_DB_PROVIDER": "pgvector",
            "DB_HOST": os.environ.get("AEG_PG_HOST", "localhost"),
            "DB_PORT": os.environ.get("AEG_PG_PORT", "5432"),
            "DB_NAME": os.environ.get("AEG_PG_NAME", "cognee"),
            "DB_USERNAME": os.environ.get("AEG_PG_USER", "cognee"),
            "DB_PASSWORD": os.environ.get("AEG_PG_PASSWORD", "cognee"),
        }.items():
            os.environ.setdefault(key, value)

    _applied = True
