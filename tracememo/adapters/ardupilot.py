"""ArduPilot DataFlash (``.bin``) adapter built on ``pymavlink.DFReader``.

Message names and fields differ between firmware versions, so the message-to-table mapping is
configuration. The built-in ``arducopter4`` profile matches ArduCopter 4.x logs; override any
part of it in ``project.yaml``::

    inputs:
      - adapter: ardupilot
        path: data/flight.bin
        tables: [pose, nav_target, beacon]
        options:
          profile: arducopter4
          mapping:                     # deep-merged over the profile
            beacon:
              sources:
                - message: BCN
                  fields: {x_m: PosY, y_m: PosX, z_m: {field: PosZ, scale: -1}, quality: Health}
          arming: {message: EV, field: Id, value: 10}
          drop_before_arming: true
          merge_tolerance_s: 0.02

A table is built from one or more ``sources``; the first is the base and later ones are
joined by nearest timestamp. Each field is ``NAME`` or ``{field: NAME, scale: k, offset: c}``;
``convert: deg2rad`` and ``wrap_pi: true`` (wrap an angle into (-pi, pi]) are also accepted.
``instance: {field: C, value: 0}`` filters a source to
one instance (for example one EKF core). ArduPilot positions are NED; the profile maps them to
the normalized ENU frame (``x_m`` east, ``y_m`` north, ``z_m`` up).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile

TIME_FIELD = "TimeUS"

PROFILES: dict[str, dict[str, Any]] = {
    "arducopter4": {
        "arming": {"message": "EV", "field": "Id", "value": 10},
        "drop_before_arming": True,
        "merge_tolerance_s": 0.02,
        "tables": {
            "pose": {
                "sources": [
                    {
                        "message": "XKF1",
                        "instance": {"field": "C", "value": 0},
                        "fields": {
                            "x_m": "PE",
                            "y_m": "PN",
                            "z_m": {"field": "PD", "scale": -1.0},
                            "roll_rad": {"field": "Roll", "convert": "deg2rad"},
                            "pitch_rad": {"field": "Pitch", "convert": "deg2rad"},
                            "yaw_rad": {"field": "Yaw", "convert": "deg2rad", "wrap_pi": True},
                        },
                    }
                ]
            },
            "nav_target": {
                "sources": [
                    {"message": "PSCN", "fields": {"y_m": "TPN"}},
                    {"message": "PSCE", "fields": {"x_m": "TPE"}},
                    {"message": "PSCD", "fields": {"z_m": {"field": "TPD", "scale": -1.0}}},
                ],
                "columns": ["t_s", "x_m", "y_m", "z_m"],
            },
            "beacon": {
                "sources": [
                    {
                        "message": "BCN",
                        "fields": {
                            "x_m": "PosY",
                            "y_m": "PosX",
                            "z_m": {"field": "PosZ", "scale": -1.0},
                            "quality": "Health",
                        },
                    }
                ]
            },
        },
    }
}

CONVERTERS = {"deg2rad": math.pi / 180.0, "rad2deg": 180.0 / math.pi, "cm2m": 0.01, "mm2m": 0.001}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_options(options: dict[str, Any]) -> dict[str, Any]:
    """Merge adapter options over the selected profile."""
    profile = options.get("profile", "arducopter4")
    if profile not in PROFILES:
        raise ValueError(
            f"ardupilot adapter: unknown profile {profile!r}; known: {sorted(PROFILES)}"
        )
    cfg = copy.deepcopy(PROFILES[profile])
    if "mapping" in options:
        cfg["tables"] = _deep_merge(cfg["tables"], options["mapping"])
    for key in ("arming", "drop_before_arming", "merge_tolerance_s"):
        if key in options:
            cfg[key] = options[key]
    return cfg


def read_messages(path: Path, types: list[str]) -> dict[str, pd.DataFrame]:
    """Read all messages of the given types from a DataFlash log into one frame per type."""
    from pymavlink import DFReader

    log = DFReader.DFReader_binary(str(path))
    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in types}
    while True:
        m = log.recv_match(type=types)
        if m is None:
            break
        rows[m.get_type()].append(m.to_dict())
    log.close()
    return {t: pd.DataFrame(r) for t, r in rows.items()}


def arming_time_us(frames: dict[str, pd.DataFrame], arming: dict[str, Any] | None) -> int | None:
    """Timestamp (us) of the first arming event, or None if not found."""
    if not arming:
        return None
    df = frames.get(arming["message"])
    if df is None or df.empty:
        return None
    hits = df[df[arming["field"]] == arming["value"]]
    return int(hits[TIME_FIELD].iloc[0]) if not hits.empty else None


def _field_series(df: pd.DataFrame, spec: str | dict[str, Any], message: str) -> pd.Series:
    if isinstance(spec, str):
        spec = {"field": spec}
    name = spec["field"]
    if name not in df.columns:
        raise KeyError(
            f"ardupilot adapter: message {message} has no field {name!r}; "
            f"fields: {list(df.columns)}"
        )
    s = pd.to_numeric(df[name], errors="coerce").astype(float)
    if "convert" in spec:
        s = s * CONVERTERS[spec["convert"]]
    s = s * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))
    if spec.get("wrap_pi"):
        s = (s + math.pi) % (2 * math.pi) - math.pi
    return s


def build_source(df: pd.DataFrame, source: dict[str, Any], t0_us: int) -> pd.DataFrame:
    """Turn one message frame into normalized columns with ``t_s`` from ``t0_us``."""
    message = source["message"]
    inst = source.get("instance")
    if inst:
        df = df[df[inst["field"]] == inst["value"]]
    out = pd.DataFrame({"t_s": (df[TIME_FIELD].astype("int64") - t0_us) / 1e6})
    for col, spec in source["fields"].items():
        out[col] = _field_series(df, spec, message).to_numpy()
    return out.sort_values("t_s").reset_index(drop=True)


def build_table(
    frames: dict[str, pd.DataFrame], spec: dict[str, Any], t0_us: int, tolerance_s: float
) -> pd.DataFrame:
    """Build a normalized table from its sources, joining extra sources by nearest time."""
    parts = []
    for source in spec["sources"]:
        df = frames.get(source["message"])
        if df is None or df.empty:
            raise ValueError(f"ardupilot adapter: no {source['message']} messages in log")
        parts.append(build_source(df, source, t0_us))
    table = parts[0]
    for extra in parts[1:]:
        table = pd.merge_asof(table, extra, on="t_s", direction="nearest", tolerance=tolerance_s)
    if "columns" in spec:
        table = table[spec["columns"]]
    return table


@register_adapter("ardupilot")
def ingest_ardupilot(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read an ArduPilot ``.bin`` log into normalized tables using the configured mapping."""
    path = Path(path)
    cfg = resolve_options(options)
    names = tables or list(cfg["tables"])
    unknown = [n for n in names if n not in cfg["tables"]]
    if unknown:
        raise KeyError(f"ardupilot adapter: no mapping for table(s) {unknown}")
    types = sorted({s["message"] for n in names for s in cfg["tables"][n]["sources"]})
    if cfg.get("arming"):
        types = sorted(set(types) | {cfg["arming"]["message"]})
    frames = read_messages(path, types)

    t_arm = arming_time_us(frames, cfg.get("arming"))
    if t_arm is None:
        first = min(int(df[TIME_FIELD].min()) for df in frames.values() if not df.empty)
        t_arm = first
    raw = [InputFile(path=str(path), sha256=sha256_file(path))]
    out = []
    for name in names:
        table = build_table(frames, cfg["tables"][name], t_arm, float(cfg["merge_tolerance_s"]))
        if cfg.get("drop_before_arming", True):
            table = table[table["t_s"] >= 0.0].reset_index(drop=True)
        table = table.dropna().reset_index(drop=True)
        for c in table.columns:
            if table[c].dtype == np.float64 and c != "t_s" and c == "quality":
                table[c] = table[c].astype("int64")
        out.append(IngestedTable(id=name, df=table, raw_inputs=raw))
    return out
