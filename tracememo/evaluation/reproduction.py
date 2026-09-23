"""Reproduction test: every computed value matches the synthetic ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tracememo.evaluation.common import (
    build_project,
    load_store,
    md_table,
    prepare_example,
    synth_drone,
    synth_rag,
)
from tracememo.store.store import ValueStore

LOC_ERR_KEYS = [
    "n_samples",
    "mean_cm",
    "median_cm",
    "p95_cm",
    "max_cm",
    "rmse_cm",
    "pct_under_10cm",
    "mean_3d_cm",
    "median_3d_cm",
    "p95_3d_cm",
    "max_3d_cm",
    "rmse_3d_cm",
    "pct_under_10cm_3d",
]
TRAJ_KEYS = ["duration_s", "planned_length_m", "actual_length_m", "max_altitude_m"]


def _row(
    pipeline: str, vid: str, got: float, truth: float, rel: float, abs_: float
) -> dict[str, Any]:
    err = abs(got - truth)
    tol = max(abs_, rel * abs(truth))
    return {
        "pipeline": pipeline,
        "value_id": vid,
        "computed": got,
        "truth": truth,
        "abs_err": err,
        "rel_err": err / abs(truth) if truth else err,
        "tolerance": tol,
        "pass": err <= tol,
    }


def compare_drone(
    store: ValueStore, truth: dict[str, Any], pipeline: str, rel: float, abs_: float
) -> list[dict[str, Any]]:
    """Rows for the drone analyses against ``truth.json``."""
    rows = []
    ref = store.get_value("drone.loc_err.reference").value
    for k in LOC_ERR_KEYS:
        rows.append(
            _row(
                pipeline,
                f"drone.loc_err.{k}",
                float(store.get_value(f"drone.loc_err.{k}").value),
                float(truth["loc_err"][ref][k]),
                rel,
                abs_,
            )
        )
    for k in TRAJ_KEYS:
        rows.append(
            _row(
                pipeline,
                f"drone.trajectory.{k}",
                float(store.get_value(f"drone.trajectory.{k}").value),
                float(truth["trajectory"][k]),
                rel,
                abs_,
            )
        )
    counts = {t: 0 for t in ("dropout", "jump", "drift")}
    for e in truth["events"]:
        counts[e["type"]] += 1
    for t, n in counts.items():
        rows.append(
            _row(
                pipeline,
                f"drone.failures.n_{t}s",
                float(store.get_value(f"drone.failures.n_{t}s").value),
                float(n),
                0.0,
                0.0,
            )
        )
    return rows


def compare_rag(store: ValueStore, truth: dict[str, Any], pipeline: str) -> list[dict[str, Any]]:
    """Rows for the LLM analyses against ``truth.json``."""
    rows = []
    lat = truth["latency"]
    for v, t in lat.items():
        if not isinstance(t, dict):
            continue
        rows.append(
            _row(
                pipeline,
                f"llm.latency.{v}.e2e_median_ms",
                float(store.get_value(f"llm.latency.{v}.e2e_median_ms").value),
                t["e2e_median_ms"],
                1e-6,
                1e-3,
            )
        )
        rows.append(
            _row(
                pipeline,
                f"llm.latency.{v}.e2e_p95_ms",
                float(store.get_value(f"llm.latency.{v}.e2e_p95_ms").value),
                t["e2e_p95_ms"],
                1e-6,
                1e-3,
            )
        )
        for s, st in t["stages"].items():
            for stat in ("median_ms", "p95_ms"):
                rows.append(
                    _row(
                        pipeline,
                        f"llm.latency.{v}.{s}.{stat}",
                        float(store.get_value(f"llm.latency.{v}.{s}.{stat}").value),
                        st[stat],
                        1e-6,
                        1e-3,
                    )
                )
    rows.append(
        _row(
            pipeline,
            "llm.latency.reduction_pct",
            float(store.get_value("llm.latency.reduction_pct").value),
            lat["reduction_pct"],
            1e-6,
            1e-6,
        )
    )
    for c, metrics in truth["eval"].items():
        for m, t in metrics.items():
            base = f"llm.eval.{c}.{m}"
            mean = float(store.get_value(f"{base}.mean").value)
            rows.append(_row(pipeline, f"{base}.mean", mean, t["sample_mean"], 1e-9, 1e-9))
            lo, hi = (
                float(store.get_value(f"{base}.ci_low").value),
                float(store.get_value(f"{base}.ci_high").value),
            )
            rows.append(
                {
                    "pipeline": pipeline,
                    "value_id": f"{base}.ci",
                    "computed": f"[{lo:.3f}, {hi:.3f}]",
                    "truth": t["sample_mean"],
                    "abs_err": 0.0,
                    "rel_err": 0.0,
                    "tolerance": 0.0,
                    "pass": lo <= t["sample_mean"] <= hi,
                }
            )
    return rows


def run(work: Path, seed: int = 0) -> dict[str, Any]:
    """Build all example pipelines on fresh synthetic data and compare against truth."""
    work.mkdir(parents=True, exist_ok=True)
    drone_dir = synth_drone(work / "data" / "drone", seed)
    rag_dir = synth_rag(work / "data" / "rag", seed)
    drone_truth = json.loads((drone_dir / "truth.json").read_text())
    rag_truth = json.loads((rag_dir / "truth.json").read_text())

    rows: list[dict[str, Any]] = []
    cfg = build_project(prepare_example("drone_memo", work, drone_dir))
    rows += compare_drone(load_store(cfg), drone_truth, "drone (normalized tables)", 1e-9, 1e-9)
    cfg = build_project(prepare_example("drone_memo", work, drone_dir, "project_raw.yaml"))
    rows += compare_drone(
        load_store(cfg), drone_truth, "drone (ArduPilot .bin + Marvelmind CSV)", 0.02, 0.2
    )
    cfg = build_project(prepare_example("rag_memo", work, rag_dir))
    rows += compare_rag(load_store(cfg), rag_truth, "rag (Langfuse export + eval CSV)")

    by_pipeline: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = by_pipeline.setdefault(
            r["pipeline"], {"pipeline": r["pipeline"], "values": 0, "passed": 0, "max_rel_err": 0.0}
        )
        s["values"] += 1
        s["passed"] += int(r["pass"])
        s["max_rel_err"] = max(s["max_rel_err"], float(r["rel_err"]))
    summary = list(by_pipeline.values())
    return {
        "seed": seed,
        "rows": rows,
        "summary": summary,
        "all_pass": all(r["pass"] for r in rows),
    }


def to_markdown(result: dict[str, Any]) -> str:
    """Summary table plus the list of failures, if any."""
    text = md_table(result["summary"], ["pipeline", "values", "passed", "max_rel_err"])
    fails = [r for r in result["rows"] if not r["pass"]]
    if fails:
        text += "\n\nFailures:\n\n" + md_table(
            fails, ["pipeline", "value_id", "computed", "truth", "tolerance"]
        )
    return text
