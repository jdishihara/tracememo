"""``drone.failures``: detect beacon dropouts, position jumps and drift.

All detection works on the *residual* between the beacon position and the vehicle's own
pose estimate, so a constant beacon-to-pose offset is not an event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tracememo.analyses.drone.loc_err import align_to_reference
from tracememo.analyses.registry import analysis
from tracememo.store.models import AnalysisResult, Table, Value

ID = "drone.failures"
EVENT_TYPES = ("dropout", "jump", "drift")
EVENT_COLUMNS = ["type", "start_s", "end_s", "duration_s", "magnitude", "unit"]


@dataclass(frozen=True)
class Event:
    """One detected failure event."""

    type: str
    start_s: float
    end_s: float
    magnitude: float
    unit: str

    @property
    def duration_s(self) -> float:
        """Event duration."""
        return self.end_s - self.start_s


def detect_dropouts(t: np.ndarray, gap_threshold_s: float) -> list[Event]:
    """Gaps between consecutive beacon samples longer than ``gap_threshold_s``."""
    dt = np.diff(t)
    events = []
    for i in np.flatnonzero(dt > gap_threshold_s):
        events.append(Event("dropout", float(t[i]), float(t[i + 1]), float(dt[i]), "s"))
    return events


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean array as ``(start, end_exclusive)`` index pairs."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def detect_jumps(
    t: np.ndarray,
    offset: np.ndarray,
    residual: np.ndarray,
    jump_threshold_m: float,
    min_samples: int,
) -> tuple[list[Event], np.ndarray]:
    """Step changes in the residual above ``jump_threshold_m`` followed by a sustained offset.

    Returns the events and a mask of samples that belong to a jump.
    """
    step = np.linalg.norm(np.diff(residual, axis=0), axis=1)
    in_jump = np.zeros(len(t), dtype=bool)
    events: list[Event] = []
    i = 0
    while i < len(step):
        if step[i] > jump_threshold_m and not in_jump[i]:
            start = i + 1
            end = start
            while end < len(t) and offset[end] > jump_threshold_m / 2:
                end += 1
            if end - start >= min_samples:
                in_jump[start:end] = True
                mag = float(np.median(offset[start:end]))
                events.append(Event("jump", float(t[start]), float(t[end - 1]), mag, "m"))
                i = end
                continue
        i += 1
    return events, in_jump


def detect_drift(
    t: np.ndarray,
    offset: np.ndarray,
    exclude: np.ndarray,
    drift_threshold_m: float,
    window_samples: int,
) -> list[Event]:
    """Sustained residual offset above ``drift_threshold_m`` (rolling mean over a window)."""
    o = np.where(exclude, 0.0, offset)
    rolling = pd.Series(o).rolling(window_samples, min_periods=1).mean().to_numpy()
    mask = (rolling > drift_threshold_m) & ~exclude
    events = []
    for a, b in _runs(mask):
        if b - a < window_samples:
            continue
        events.append(Event("drift", float(t[a]), float(t[b - 1]), float(rolling[a:b].max()), "m"))
    return events


@analysis(
    id=ID,
    inputs=["pose", "beacon"],
    params={
        "align_tolerance_s": 0.05,
        "gap_threshold_s": 0.5,
        "jump_threshold_m": 0.2,
        "jump_min_samples": 2,
        "drift_threshold_m": 0.1,
        "drift_window_s": 2.0,
    },
    outputs=[f"{ID}.events"],
)
def failures(pose: pd.DataFrame, beacon: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Detect beacon dropouts, jumps and drift; one table row per event, counts as values."""
    b = beacon.sort_values("t_s")
    dropouts = detect_dropouts(b["t_s"].to_numpy(), float(params["gap_threshold_s"]))

    aligned = align_to_reference(pose, b, float(params["align_tolerance_s"]))
    t = aligned["t_s"].to_numpy()
    residual = np.column_stack(
        [aligned["x_ref_m"] - aligned["x_m"], aligned["y_ref_m"] - aligned["y_m"]]
    )
    baseline = np.median(residual, axis=0)
    offset = np.linalg.norm(residual - baseline, axis=1)
    jumps, in_jump = detect_jumps(
        t, offset, residual, float(params["jump_threshold_m"]), int(params["jump_min_samples"])
    )
    rate_hz = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    window = max(1, int(round(float(params["drift_window_s"]) * rate_hz)))
    drifts = detect_drift(t, offset, in_jump, float(params["drift_threshold_m"]), window)

    events = sorted(dropouts + jumps + drifts, key=lambda e: e.start_s)
    df = pd.DataFrame(
        [
            {
                "type": e.type,
                "start_s": e.start_s,
                "end_s": e.end_s,
                "duration_s": e.duration_s,
                "magnitude": e.magnitude,
                "unit": e.unit,
            }
            for e in events
        ],
        columns=EVENT_COLUMNS,
    )
    span = float(b["t_s"].iloc[-1] - b["t_s"].iloc[0]) if len(b) > 1 else 0.0
    gap_total = sum(e.duration_s for e in dropouts)
    values = [
        Value(
            id=f"{ID}.n_events",
            value=len(events),
            fmt="d",
            description="Total number of detected beacon failure events",
        ),
        Value(
            id=f"{ID}.n_dropouts",
            value=len(dropouts),
            fmt="d",
            description="Number of beacon dropouts (gaps above threshold)",
        ),
        Value(
            id=f"{ID}.n_jumps",
            value=len(jumps),
            fmt="d",
            description="Number of beacon position jumps",
        ),
        Value(
            id=f"{ID}.n_drifts",
            value=len(drifts),
            fmt="d",
            description="Number of sustained beacon drift episodes",
        ),
        Value(
            id=f"{ID}.dropout_total_s",
            value=float(gap_total),
            unit="s",
            fmt=".1f",
            description="Total time lost to beacon dropouts",
        ),
        Value(
            id=f"{ID}.availability_pct",
            value=float(100.0 * (1.0 - gap_total / span)) if span > 0 else 100.0,
            unit="%",
            fmt=".1f",
            description="Share of the flight with beacon coverage",
        ),
    ]
    table = Table(
        id=f"{ID}.events", dataframe=df, float_fmt=".2f", caption="Detected beacon failure events."
    )
    return AnalysisResult(values=values, tables=[table])
