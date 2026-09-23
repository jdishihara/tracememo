"""Generic evaluation-results adapter: per-item metric scores from CSV or JSON.

Output: long-format ``eval_scores`` with columns ``item_id, config, metric_name, score``.

Options::

    options:
      format: long                    # long (one row per item/config/metric) | wide
      columns: {item_id: item_id, config: config, metric_name: metric_name, score: score}
      metrics: [correctness, ...]     # wide format only: which columns are metrics
                                      # (default: every column not used as an id)
      table: eval_scores

JSON input is a list of row objects (or ``{"data": [...]}``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile

DEFAULT_COLUMNS = {
    "item_id": "item_id",
    "config": "config",
    "metric_name": "metric_name",
    "score": "score",
}
OUT_COLUMNS = ["item_id", "config", "metric_name", "score"]


def read_rows(path: Path) -> pd.DataFrame:
    """Read a CSV or JSON file of rows."""
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]
        return pd.DataFrame(obj)
    return pd.read_csv(path)


def to_long(df: pd.DataFrame, options: dict[str, Any]) -> pd.DataFrame:
    """Normalize a long or wide frame to ``item_id, config, metric_name, score``."""
    cols = {**DEFAULT_COLUMNS, **options.get("columns", {})}
    fmt = options.get("format", "long")
    for key in ("item_id", "config"):
        if cols[key] not in df.columns:
            raise KeyError(
                f"eval_table adapter: column {cols[key]!r} for {key} not in {list(df.columns)}"
            )
    if fmt == "wide":
        ids = [cols["item_id"], cols["config"]]
        metrics = options.get("metrics") or [c for c in df.columns if c not in ids]
        long = df.melt(id_vars=ids, value_vars=metrics, var_name="metric_name", value_name="score")
        long = long.rename(columns={cols["item_id"]: "item_id", cols["config"]: "config"})
    elif fmt == "long":
        for key in ("metric_name", "score"):
            if cols[key] not in df.columns:
                raise KeyError(
                    f"eval_table adapter: column {cols[key]!r} for {key} not in {list(df.columns)}"
                )
        long = df.rename(columns={v: k for k, v in cols.items()})
    else:
        raise ValueError(f"eval_table adapter: format must be 'long' or 'wide', got {fmt!r}")
    out = long[OUT_COLUMNS].copy()
    out["item_id"] = out["item_id"].astype(str)
    out["config"] = out["config"].astype(str)
    out["metric_name"] = out["metric_name"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").astype(float)
    return out.dropna(subset=["score"]).reset_index(drop=True)


@register_adapter("eval_table")
def ingest_eval_table(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read evaluation results into a long-format ``eval_scores`` table."""
    path = Path(path)
    name = options.get("table", "eval_scores")
    if tables and name not in tables:
        raise KeyError(f"eval_table adapter produces {name!r}, not {tables}")
    df = to_long(read_rows(path), options)
    raw = [InputFile(path=str(path), sha256=sha256_file(path))]
    return [IngestedTable(id=name, df=df, raw_inputs=raw)]
