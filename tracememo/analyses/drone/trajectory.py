"""``drone.trajectory``: planned versus actual trajectory figures and path statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tracememo.analyses.registry import analysis
from tracememo.figures import SERIES, new_figure
from tracememo.store.models import AnalysisResult, Figure, Value

ID = "drone.trajectory"


def path_length(xyz: np.ndarray) -> float:
    """Sum of segment lengths of an (n, 3) polyline."""
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


@analysis(id=ID, inputs=["pose", "nav_target"], params={})
def trajectory(
    pose: pd.DataFrame, nav_target: pd.DataFrame, params: dict[str, Any]
) -> AnalysisResult:
    """Top-down and altitude-vs-time plots of planned and estimated trajectories."""
    p = pose.sort_values("t_s")
    n = nav_target.sort_values("t_s")
    p_xyz = p[["x_m", "y_m", "z_m"]].to_numpy()
    n_xyz = n[["x_m", "y_m", "z_m"]].to_numpy()
    duration = float(p["t_s"].iloc[-1] - p["t_s"].iloc[0])

    values = [
        Value(
            id=f"{ID}.duration_s",
            value=duration,
            unit="s",
            fmt=".0f",
            description="Flight duration covered by the pose log",
        ),
        Value(
            id=f"{ID}.planned_length_m",
            value=path_length(n_xyz),
            unit="m",
            fmt=".1f",
            description="Length of the planned path",
        ),
        Value(
            id=f"{ID}.actual_length_m",
            value=path_length(p_xyz),
            unit="m",
            fmt=".1f",
            description="Length of the estimated (flown) path",
        ),
        Value(
            id=f"{ID}.max_altitude_m",
            value=float(p["z_m"].max()),
            unit="m",
            fmt=".2f",
            description="Maximum estimated altitude",
        ),
    ]

    fig, ax = new_figure(width=5.0, height=5.0)
    ax.plot(n["x_m"], n["y_m"], color=SERIES[1], linewidth=1.5, label="Planned")
    ax.plot(p["x_m"], p["y_m"], color=SERIES[0], linewidth=1.0, alpha=0.9, label="Estimated")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, axis="both")
    ax.legend(loc="upper right")
    top_down = Figure(
        id=f"{ID}.top_down",
        figure=fig,
        caption="Top-down view of the planned and estimated trajectories.",
    )

    fig, ax = new_figure()
    ax.plot(n["t_s"], n["z_m"], color=SERIES[1], linewidth=1.5, label="Planned")
    ax.plot(p["t_s"], p["z_m"], color=SERIES[0], linewidth=1.0, alpha=0.9, label="Estimated")
    ax.set_xlabel("Time from arming (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlim(float(p["t_s"].iloc[0]), float(p["t_s"].iloc[-1]))
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    altitude = Figure(
        id=f"{ID}.altitude", figure=fig, caption="Planned and estimated altitude over time."
    )

    return AnalysisResult(values=values, figures=[top_down, altitude])
