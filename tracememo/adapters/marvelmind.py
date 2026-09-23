"""Marvelmind ultrasonic positioning CSV adapter.

Marvelmind exports vary by dashboard version, so column names and units are options::

    inputs:
      - adapter: marvelmind
        path: data/hedge_log.csv
        options:
          table: beacon_raw          # output table name (use "beacon" to feed drone analyses)
          delimiter: ","
          columns: {time: Time, x: X, y: Y, z: Z, quality: Quality, address: Address}
          address: 12                # optional: keep only this hedgehog
          length_unit: m             # m | cm | mm
          time_unit: s               # s | ms | us
          time_origin: absolute      # absolute | first  (first: t=0 at the first row)
          time_offset_s: 0.0         # added after conversion, e.g. to align to arming

Output columns: ``t_s, x_m, y_m, z_m`` plus ``quality`` when configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile

DEFAULT_COLUMNS = {
    "time": "Time",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "quality": "Quality",
    "address": "Address",
}
LENGTH_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
TIME_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


@register_adapter("marvelmind")
def ingest_marvelmind(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read a Marvelmind CSV export into one normalized beacon table."""
    path = Path(path)
    table_name = options.get("table", "beacon_raw")
    if tables and table_name not in tables:
        raise KeyError(f"marvelmind adapter produces {table_name!r}, not {tables}")
    cols = {**DEFAULT_COLUMNS, **options.get("columns", {})}
    df = pd.read_csv(path, sep=options.get("delimiter", ","), skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    for key in ("time", "x", "y", "z"):
        if cols[key] not in df.columns:
            raise KeyError(
                f"marvelmind adapter: column {cols[key]!r} for {key} not in {list(df.columns)}"
            )
    if "address" in options and cols["address"] in df.columns:
        df = df[df[cols["address"]] == options["address"]]

    lscale = LENGTH_SCALE[options.get("length_unit", "m")]
    tscale = TIME_SCALE[options.get("time_unit", "s")]
    t = pd.to_numeric(df[cols["time"]], errors="coerce").astype(float) * tscale
    if options.get("time_origin", "absolute") == "first":
        t = t - t.iloc[0]
    t = t + float(options.get("time_offset_s", 0.0))
    out = pd.DataFrame({"t_s": t})
    for key in ("x", "y", "z"):
        out[f"{key}_m"] = pd.to_numeric(df[cols[key]], errors="coerce").astype(float) * lscale
    if cols["quality"] in df.columns:
        out["quality"] = pd.to_numeric(df[cols["quality"]], errors="coerce")
    out = out.dropna(subset=["t_s", "x_m", "y_m", "z_m"]).sort_values("t_s").reset_index(drop=True)
    if "quality" in out.columns:
        out["quality"] = out["quality"].round().astype("int64")
    raw = [InputFile(path=str(path), sha256=sha256_file(path))]
    return [IngestedTable(id=table_name, df=out, raw_inputs=raw)]
