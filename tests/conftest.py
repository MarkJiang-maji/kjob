"""Keep tests away from the real home directory: no automatic skill installation into ~/.claude or ~/.codex, and no
job history written to ~/.kjob (tests that check the history point HISTORY_FILE somewhere else themselves)."""
import os
import tempfile
from pathlib import Path

import pytest

os.environ["KJOB_NO_SKILLS"] = "1"


@pytest.fixture(autouse=True)
def _private_history(monkeypatch):
    from kjob import core
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(core, "HISTORY_FILE", Path(d) / "history.jsonl")
        yield
