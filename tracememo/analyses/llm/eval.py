"""``llm.eval``: mean and bootstrap confidence interval of each metric per configuration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tracememo.analyses.registry import analysis
from tracememo.figures import SERIES, new_figure
from tracememo.store.models import AnalysisResult, Figure, Table, Value, slugify

ID = "llm.eval"


def bootstrap_ci(
    x: np.ndarray, n_bootstrap: int, ci: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean."""
    n = len(x)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


@analysis(
    id=ID,
    inputs=["eval_scores"],
    params={"n_bootstrap": 2000, "ci": 0.95, "seed": 0, "metrics": None, "configs": None},
)
def evaluation(eval_scores: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Mean score and bootstrap CI per (config, metric); a metrics x configs table."""
    df = eval_scores.dropna(subset=["score"])
    metrics = params.get("metrics") or list(dict.fromkeys(df["metric_name"]))
    configs = params.get("configs") or list(dict.fromkeys(df["config"]))
    ci = float(params["ci"])
    rng = np.random.default_rng(int(params["seed"]))
    values: list[Value] = []
    cells: dict[str, dict[str, str]] = {m: {} for m in metrics}
    stats: dict[tuple[str, str], tuple[float, float, float]] = {}
    for c in configs:
        for m in metrics:
            x = df[(df["config"] == c) & (df["metric_name"] == m)]["score"].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            mean = float(x.mean())
            lo, hi = bootstrap_ci(x, int(params["n_bootstrap"]), ci, rng)
            stats[(c, m)] = (mean, lo, hi)
            base = f"{ID}.{slugify(c)}.{slugify(m)}"
            pct = f"{ci:.0%}"
            values += [
                Value(
                    id=f"{base}.mean", value=mean, fmt=".3f", description=f"Mean {m} for config {c}"
                ),
                Value(
                    id=f"{base}.ci_low",
                    value=lo,
                    fmt=".3f",
                    description=f"Lower {pct} bootstrap CI of mean {m}, config {c}",
                ),
                Value(
                    id=f"{base}.ci_high",
                    value=hi,
                    fmt=".3f",
                    description=f"Upper {pct} bootstrap CI of mean {m}, config {c}",
                ),
                Value(
                    id=f"{base}.n",
                    value=int(len(x)),
                    fmt="d",
                    description=f"Number of items scored for {m}, config {c}",
                ),
            ]
            cells[m][c] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
    table_df = pd.DataFrame(
        [{"metric": m, **{c: cells[m].get(c, "") for c in configs}} for m in metrics]
    )
    table = Table(
        id=f"{ID}.metrics",
        dataframe=table_df,
        caption=f"Mean score per metric and configuration with {ci:.0%} bootstrap "
        "confidence intervals.",
    )
    fig = _dot_figure(metrics, configs, stats)
    return AnalysisResult(values=values, figures=[fig], tables=[table])


def _dot_figure(
    metrics: list[str], configs: list[str], stats: dict[tuple[str, str], tuple[float, float, float]]
) -> Figure:
    fig, ax = new_figure(width=6.0, height=3.4)
    y = np.arange(len(metrics))
    k = len(configs)
    offsets = (np.arange(k) - (k - 1) / 2.0) * 0.22
    for i, c in enumerate(configs):
        means = np.array([stats.get((c, m), (np.nan,) * 3)[0] for m in metrics])
        lo = np.array([stats.get((c, m), (np.nan,) * 3)[1] for m in metrics])
        hi = np.array([stats.get((c, m), (np.nan,) * 3)[2] for m in metrics])
        ax.errorbar(
            means,
            y + offsets[i],
            xerr=[means - lo, hi - means],
            fmt="o",
            color=SERIES[i % len(SERIES)],
            markersize=5,
            capsize=2,
            linewidth=1.2,
            label=c,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
    ax.set_yticks(y, metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Mean score (with bootstrap CI)")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower left")
    return Figure(
        id=f"{ID}.means",
        figure=fig,
        caption="Mean evaluation score per metric and configuration; bars show "
        "bootstrap confidence intervals.",
    )
