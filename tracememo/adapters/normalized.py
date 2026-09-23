"""Adapter for data that is already in the normalized Parquet table format.

Used for synthetic data (see ``tracememo.synth``) and for any pre-normalized export.
Each ``<name>.parquet`` file in the directory becomes the table ``<name>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile


@register_adapter("normalized")
def ingest_normalized(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read ``<path>/<table>.parquet`` for each requested table (or all if ``tables`` is None)."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"normalized adapter: {path} is not a directory")
    names = tables if tables is not None else sorted(p.stem for p in path.glob("*.parquet"))
    out: list[IngestedTable] = []
    for name in names:
        file = path / f"{name}.parquet"
        if not file.exists():
            raise FileNotFoundError(f"normalized adapter: missing table file {file}")
        df = pd.read_parquet(file)
        out.append(
            IngestedTable(
                id=name,
                df=df,
                raw_inputs=[InputFile(path=str(file), sha256=sha256_file(file))],
            )
        )
    return out
