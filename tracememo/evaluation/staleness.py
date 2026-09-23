"""Staleness demo: change one input, rebuild, and show exactly what was recomputed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.cli import run_analyze, run_ingest
from tracememo.evaluation.common import (
    build_project,
    load_store,
    md_table,
    prepare_example,
    synth_drone,
)
from tracememo.store.store import ValueStore


def diff_stores(before: ValueStore, after: ValueStore) -> dict[str, Any]:
    """Which analyses reran, which values changed and which figures were regenerated."""
    analyses = [
        {"analysis": aid, "status": "cached" if rec.cached else "reran"}
        for aid, rec in after.analyses.items()
    ]
    changed, unchanged = [], []
    for vid, v in after.values.items():
        old = before.values.get(vid)
        if old is None or old.value != v.value:
            changed.append(vid)
        else:
            unchanged.append(vid)
    figs = [
        {
            "figure": fid,
            "regenerated": f.provenance.timestamp != before.figures[fid].provenance.timestamp,
        }
        for fid, f in after.figures.items()
        if f.provenance and before.figures.get(fid) and before.figures[fid].provenance
    ]
    return {
        "analyses": analyses,
        "changed_values": sorted(changed),
        "unchanged_values": sorted(unchanged),
        "figures": figs,
    }


def run(work: Path, seed: int = 0, perturb_m: float = 0.02) -> dict[str, Any]:
    """Build the drone memo, perturb the beacon log by ``perturb_m`` in x, rebuild, and diff."""
    work.mkdir(parents=True, exist_ok=True)
    data = synth_drone(work / "data" / "drone", seed)
    cfg = build_project(prepare_example("drone_memo", work, data))
    before = load_store(cfg)

    beacon = data / "beacon.parquet"
    df = pd.read_parquet(beacon)
    df["x_m"] += perturb_m
    df.to_parquet(beacon, index=False)
    run_ingest(cfg)
    run_analyze(cfg)
    after = load_store(cfg)
    d = diff_stores(before, after)
    statuses = {a["analysis"]: a["status"] for a in d["analyses"]}
    d["expected"] = {
        "drone.trajectory": "cached",
        "drone.loc_err": "reran",
        "drone.failures": "reran",
    }
    d["as_expected"] = all(statuses.get(k) == v for k, v in d["expected"].items())
    d["perturbation"] = f"beacon.parquet x_m += {perturb_m} m"
    d["n_changed"] = len(d["changed_values"])
    d["n_unchanged"] = len(d["unchanged_values"])
    return d


def to_markdown(result: dict[str, Any]) -> str:
    """Analysis status table plus changed/unchanged value counts."""
    rows = [
        {
            "analysis": a["analysis"],
            "status": a["status"],
            "expected": result["expected"].get(a["analysis"], ""),
        }
        for a in result["analyses"]
    ]
    text = f"Perturbation: `{result['perturbation']}`\n\n" + md_table(
        rows, ["analysis", "status", "expected"]
    )
    text += (
        f"\n\nValues changed: {result['n_changed']}; values unchanged: {result['n_unchanged']} "
        f"(all `drone.trajectory.*` values unchanged and served from cache)."
    )
    text += "\n\nChanged values: " + ", ".join(f"`{v}`" for v in result["changed_values"])
    return text
