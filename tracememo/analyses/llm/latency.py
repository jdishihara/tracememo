"""``llm.latency``: per-stage and end-to-end latency per system version."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tracememo.analyses.registry import analysis
from tracememo.figures import SERIES, new_figure
from tracememo.store.models import AnalysisResult, Figure, Table, Value, slugify

ID = "llm.latency"


def stage_spans(spans: pd.DataFrame, stage_names: list[str] | None) -> pd.DataFrame:
    """Spans that represent pipeline stages: the named ones, or all leaf spans."""
    if stage_names:
        return spans[spans["name"].isin(stage_names)]
    parents = set(spans["parent_id"].dropna())
    return spans[~spans["span_id"].isin(parents)]


def stage_order(spans: pd.DataFrame) -> list[str]:
    """Stage names ordered by their mean start offset within a trace."""
    t0 = spans.groupby("trace_id")["start"].transform("min")
    offset = (spans["start"] - t0).dt.total_seconds()
    return list(offset.groupby(spans["name"]).mean().sort_values().index)


@analysis(
    id=ID,
    inputs=["spans", "traces"],
    params={"baseline": None, "candidate": None, "stage_names": None},
)
def latency(spans: pd.DataFrame, traces: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Median and p95 latency per stage and end to end, per version, plus the reduction
    between ``baseline`` and ``candidate`` (default: first and last version, sorted)."""
    versions = sorted(v for v in traces["version"].dropna().unique())
    if not versions:
        raise ValueError(f"{ID}: traces have no version labels")
    baseline = params.get("baseline") or versions[0]
    candidate = params.get("candidate") or versions[-1]
    for v in (baseline, candidate):
        if v not in versions:
            raise ValueError(f"{ID}: version {v!r} not in traces; known: {versions}")

    st = stage_spans(spans, params.get("stage_names"))
    st = st.merge(traces[["trace_id", "version"]], on="trace_id", how="inner")
    stages = stage_order(st)
    values: list[Value] = [
        Value(id=f"{ID}.baseline_version", value=baseline, description="Baseline version"),
        Value(id=f"{ID}.candidate_version", value=candidate, description="Candidate version"),
    ]
    rows = []
    medians: dict[str, dict[str, float]] = {}
    for v in versions:
        vs = slugify(v)
        tv = traces[traces["version"] == v]["total_latency_ms"].dropna()
        e2e_med, e2e_p95 = float(np.median(tv)), float(np.percentile(tv, 95))
        values += [
            Value(
                id=f"{ID}.{vs}.n_traces",
                value=int(len(tv)),
                fmt="d",
                description=f"Number of traces for version {v}",
            ),
            Value(
                id=f"{ID}.{vs}.e2e_median_ms",
                value=e2e_med,
                unit="ms",
                fmt=".0f",
                description=f"Median end-to-end latency, version {v}",
            ),
            Value(
                id=f"{ID}.{vs}.e2e_p95_ms",
                value=e2e_p95,
                unit="ms",
                fmt=".0f",
                description=f"95th percentile end-to-end latency, version {v}",
            ),
        ]
        medians[v] = {}
        for s in stages:
            d = st[(st["version"] == v) & (st["name"] == s)]["duration_ms"]
            if d.empty:
                continue
            med, p95 = float(np.median(d)), float(np.percentile(d, 95))
            medians[v][s] = med
            values += [
                Value(
                    id=f"{ID}.{vs}.{slugify(s)}.median_ms",
                    value=med,
                    unit="ms",
                    fmt=".0f",
                    description=f"Median latency of stage {s}, version {v}",
                ),
                Value(
                    id=f"{ID}.{vs}.{slugify(s)}.p95_ms",
                    value=p95,
                    unit="ms",
                    fmt=".0f",
                    description=f"95th percentile latency of stage {s}, version {v}",
                ),
            ]
            rows.append({"stage": s, "version": v, "median_ms": med, "p95_ms": p95})
    base_med = next(v.value for v in values if v.id == f"{ID}.{slugify(baseline)}.e2e_median_ms")
    cand_med = next(v.value for v in values if v.id == f"{ID}.{slugify(candidate)}.e2e_median_ms")
    values.append(
        Value(
            id=f"{ID}.reduction_pct",
            value=100.0 * (1.0 - float(cand_med) / float(base_med)),
            unit="%",
            fmt=".1f",
            description=f"Reduction in median end-to-end latency from {baseline} to {candidate}",
        )
    )

    long = pd.DataFrame(rows, columns=["stage", "version", "median_ms", "p95_ms"])
    wide = long.pivot(index="stage", columns="version", values=["median_ms", "p95_ms"])
    wide = wide.reindex(stages)
    wide.columns = [f"{v}_{stat}" for stat, v in wide.columns]
    wide = wide[[f"{v}_{stat}" for v in versions for stat in ("median_ms", "p95_ms")]]
    table = Table(
        id=f"{ID}.by_stage",
        dataframe=wide.reset_index(),
        float_fmt=".0f",
        caption="Median and 95th percentile latency per pipeline stage and version (ms).",
    )

    fig = _stacked_figure(versions, stages, medians)
    return AnalysisResult(values=values, figures=[fig], tables=[table])


def _stacked_figure(
    versions: list[str], stages: list[str], medians: dict[str, dict[str, float]]
) -> Figure:
    fig, ax = new_figure(width=5.0, height=3.4)
    x = np.arange(len(versions))
    bottom = np.zeros(len(versions))
    for i, s in enumerate(stages):
        h = np.array([medians[v].get(s, 0.0) for v in versions])
        ax.bar(
            x,
            h,
            bottom=bottom,
            width=0.5,
            color=SERIES[i % len(SERIES)],
            label=s,
            edgecolor="white",
            linewidth=1.5,
        )
        bottom += h
    ax.set_xticks(x, versions)
    ax.set_ylabel("Median stage latency (ms)")
    ax.set_ylim(0, bottom.max() * 1.3)
    ax.legend(loc="upper center", ncols=2)
    return Figure(
        id=f"{ID}.stacked",
        figure=fig,
        caption="Median latency per pipeline stage, stacked, for each system version.",
    )
