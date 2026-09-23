"""``drone.loc_err``: localization error of the estimated pose against a reference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tracememo.analyses.registry import analysis
from tracememo.figures import BASELINE, INK_MUTED, SERIES, new_figure
from tracememo.store.models import AnalysisResult, Figure, Value

ID = "drone.loc_err"


def align_to_reference(
    pose: pd.DataFrame, reference: pd.DataFrame, tolerance_s: float
) -> pd.DataFrame:
    """Match each reference sample to the nearest pose sample within ``tolerance_s``.

    Returns a frame with columns ``t_s, x_m, y_m, z_m`` (pose) and
    ``x_ref_m, y_ref_m, z_ref_m`` (reference). Unmatched reference rows are dropped.
    """
    ref = reference[["t_s", "x_m", "y_m", "z_m"]].sort_values("t_s")
    ref = ref.rename(columns={"x_m": "x_ref_m", "y_m": "y_ref_m", "z_m": "z_ref_m"})
    pos = pose[["t_s", "x_m", "y_m", "z_m"]].sort_values("t_s")
    merged = pd.merge_asof(
        ref, pos, on="t_s", direction="nearest", tolerance=tolerance_s, suffixes=("", "")
    )
    return merged.dropna(subset=["x_m"]).reset_index(drop=True)


def error_stats(err_cm: np.ndarray, threshold_cm: float) -> dict[str, float]:
    """Summary statistics of an error series in centimetres."""
    return {
        "mean": float(np.mean(err_cm)),
        "median": float(np.median(err_cm)),
        "p95": float(np.percentile(err_cm, 95)),
        "max": float(np.max(err_cm)),
        "rmse": float(np.sqrt(np.mean(err_cm**2))),
        "pct_under": float(100.0 * np.mean(err_cm < threshold_cm)),
    }


@analysis(
    id=ID,
    inputs=["pose", "nav_target", "beacon"],
    params={"reference": "nav_target", "align_tolerance_s": 0.05, "threshold_cm": 10.0},
)
def localization_error(
    pose: pd.DataFrame, nav_target: pd.DataFrame, beacon: pd.DataFrame, params: dict[str, Any]
) -> AnalysisResult:
    """Horizontal and 3D error between the estimated pose and a reference position.

    ``params["reference"]`` selects ``"nav_target"`` (planned position) or ``"beacon"``
    (external positioning). Positions are in metres; values are reported in centimetres.
    """
    ref_name = params["reference"]
    if ref_name == "nav_target":
        reference = nav_target
    elif ref_name == "beacon":
        reference = beacon
    else:
        raise ValueError(f"{ID}: reference must be 'nav_target' or 'beacon', got {ref_name!r}")
    thr = float(params["threshold_cm"])
    thr_label = f"{thr:g}cm"

    aligned = align_to_reference(pose, reference, float(params["align_tolerance_s"]))
    if aligned.empty:
        raise ValueError(f"{ID}: no pose samples aligned with {ref_name} within tolerance")
    dx = (aligned["x_m"] - aligned["x_ref_m"]).to_numpy()
    dy = (aligned["y_m"] - aligned["y_ref_m"]).to_numpy()
    dz = (aligned["z_m"] - aligned["z_ref_m"]).to_numpy()
    err_h = 100.0 * np.hypot(dx, dy)
    err_3d = 100.0 * np.sqrt(dx**2 + dy**2 + dz**2)
    t = aligned["t_s"].to_numpy()

    values = [
        Value(id=f"{ID}.reference", value=ref_name, description="Reference used for error"),
        Value(
            id=f"{ID}.n_samples",
            value=int(len(aligned)),
            fmt="d",
            description="Number of aligned reference samples",
        ),
    ]
    for suffix, err, label in (("", err_h, "horizontal"), ("_3d", err_3d, "3D")):
        s = error_stats(err, thr)
        values += [
            Value(
                id=f"{ID}.mean{suffix}_cm",
                value=s["mean"],
                unit="cm",
                fmt=".1f",
                description=f"Mean {label} localization error",
            ),
            Value(
                id=f"{ID}.median{suffix}_cm",
                value=s["median"],
                unit="cm",
                fmt=".1f",
                description=f"Median {label} localization error",
            ),
            Value(
                id=f"{ID}.p95{suffix}_cm",
                value=s["p95"],
                unit="cm",
                fmt=".1f",
                description=f"95th percentile {label} localization error",
            ),
            Value(
                id=f"{ID}.max{suffix}_cm",
                value=s["max"],
                unit="cm",
                fmt=".1f",
                description=f"Maximum {label} localization error",
            ),
            Value(
                id=f"{ID}.rmse{suffix}_cm",
                value=s["rmse"],
                unit="cm",
                fmt=".1f",
                description=f"Root-mean-square {label} localization error",
            ),
            Value(
                id=f"{ID}.pct_under_{thr_label}{suffix}",
                value=s["pct_under"],
                unit="%",
                fmt=".1f",
                description=f"Share of samples with {label} error under {thr:g} cm",
            ),
        ]

    figures = [
        _error_vs_time_figure(t, err_h, thr, ref_name),
        _error_cdf_figure(err_h, err_3d, ref_name),
    ]
    return AnalysisResult(values=values, figures=figures)


def _error_vs_time_figure(t: np.ndarray, err_h: np.ndarray, thr: float, ref: str) -> Figure:
    fig, ax = new_figure()
    ax.plot(t, err_h, color=SERIES[0], linewidth=1.2)
    ax.axhline(thr, color=BASELINE, linewidth=0.8)
    ax.text(t[-1], thr, f" {thr:g} cm", color=INK_MUTED, fontsize=8, va="center", ha="left")
    ax.set_xlabel("Time from arming (s)")
    ax.set_ylabel("Horizontal error (cm)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    return Figure(
        id=f"{ID}.error_vs_time",
        caption=f"Horizontal localization error over time, relative to the {ref} position.",
        figure=fig,
    )


def _error_cdf_figure(err_h: np.ndarray, err_3d: np.ndarray, ref: str) -> Figure:
    fig, ax = new_figure()
    for err, label, color in ((err_h, "Horizontal", SERIES[0]), (err_3d, "3D", SERIES[1])):
        x = np.sort(err)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Error (cm)")
    ax.set_ylabel("Fraction of samples")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    return Figure(
        id=f"{ID}.error_cdf",
        caption=f"Empirical CDF of localization error relative to the {ref} position.",
        figure=fig,
    )
