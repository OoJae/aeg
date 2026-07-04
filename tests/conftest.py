"""Test bootstrap: isolate cognee state under tests/.cognee_test BEFORE any
aeg import (aeg.cognee_client applies env with setdefault, so this wins)."""

import os
import shutil
from pathlib import Path

import pytest

TEST_SCRATCH = Path(__file__).resolve().parent / ".cognee_test"
os.environ["AEG_SCRATCH_DIR"] = str(TEST_SCRATCH)

# The gateway `app` is process-global (imported once), so the abuse guards would
# otherwise accumulate across the whole suite and start 429/503-ing. Disable the
# per-IP rate limit and the LLM budget for tests (0 => disabled); individual tests
# monkeypatch config.AEG_API_KEY when they need the auth path.
os.environ["AEG_RATE_LIMIT"] = "0"
os.environ["AEG_DAILY_LLM_BUDGET"] = "0"


@pytest.fixture(scope="session", autouse=True)
def clean_test_scratch():
    shutil.rmtree(TEST_SCRATCH, ignore_errors=True)
    yield
