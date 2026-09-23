"""Shared bits for the eval scripts: results directory and a scratch work directory."""

from __future__ import annotations

import tempfile
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def work_dir(name: str) -> Path:
    """A fresh scratch directory for one experiment."""
    return Path(tempfile.mkdtemp(prefix=f"tracememo-eval-{name}-"))
