"""Phase 6 items 2-4 — config-surface units (free, no LLM, no cognee I/O)."""

import subprocess
import sys

from aeg import config
from aeg.gateway import effective_dataset


# --- Item 2: truth-subspace flag -------------------------------------------- #

def test_truth_subspace_default_off():
    assert config.AEG_TRUTH_SUBSPACE is False


# --- Item 3: multi-user namespacing ----------------------------------------- #

def test_user_dataset_namespacing():
    assert config.user_dataset("alice") == "aeg_user_alice"
    assert config.user_dataset("bob") != config.user_dataset("alice")


def test_effective_dataset_flag_off_ignores_user_id(monkeypatch):
    monkeypatch.setattr(config, "AEG_MULTI_USER", False)
    assert effective_dataset("aeg_main", "alice") == "aeg_main"


def test_effective_dataset_flag_on_routes_to_user_dataset(monkeypatch):
    monkeypatch.setattr(config, "AEG_MULTI_USER", True)
    assert effective_dataset("aeg_main", "alice") == "aeg_user_alice"
    assert effective_dataset("aeg_main", None) == "aeg_main", "no user_id → base dataset"


# --- Item 4: postgres profile config resolves (subprocess, no Docker) ------- #

def test_postgres_profile_resolves_env():
    code = (
        "import os; from aeg import config; config.apply_cognee_env();"
        "assert os.environ['DB_PROVIDER']=='postgres';"
        "assert os.environ['VECTOR_DB_PROVIDER']=='pgvector';"
        "assert os.environ['DB_HOST']=='localhost' and os.environ['DB_PORT']=='5432';"
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"AEG_PROFILE": "postgres", "AEG_SCRATCH_DIR": "/tmp/aeg_pg_test",
             "PATH": __import__("os").environ["PATH"]},
        capture_output=True, text=True, timeout=120,
    )
    assert "OK" in result.stdout, f"postgres profile did not resolve: {result.stderr[-400:]}"
