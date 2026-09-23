"""Synthetic drone flight: planned square trajectory, lagged true motion, noisy pose estimate
and beacon measurements with injected failure events.

Output tables use the normalized schemas documented in ``docs/schemas.md``:

- ``pose``: ``t_s, x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad``
- ``nav_target``: ``t_s, x_m, y_m, z_m``
- ``beacon``: ``t_s, x_m, y_m, z_m, quality``

``truth.json`` records the configuration, the injected events and localization-error
statistics computed directly from the generator's arrays.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

EventType = Literal["dropout", "jump", "drift"]


@dataclass(frozen=True)
class FailureEvent:
    """An injected beacon failure. ``magnitude_m`` is unused for dropouts."""

    type: EventType
    start_s: float
    duration_s: float
    magnitude_m: float = 0.0

    @property
    def end_s(self) -> float:
        """End time of the event."""
        return self.start_s + self.duration_s


def default_events(duration_s: float) -> tuple[FailureEvent, ...]:
    """Default failure events scaled to the flight duration (none for very short flights)."""
    if duration_s < 40:
        return ()
    return (
        FailureEvent("dropout", start_s=round(0.33 * duration_s, 1), duration_s=3.0),
        FailureEvent("jump", start_s=round(0.58 * duration_s, 1), duration_s=4.0, magnitude_m=0.5),
        FailureEvent(
            "drift",
            start_s=round(0.75 * duration_s, 1),
            duration_s=round(0.125 * duration_s, 1),
            magnitude_m=0.3,
        ),
    )


@dataclass(frozen=True)
class DroneSynthConfig:
    """Parameters of the synthetic flight. All lengths in metres, times in seconds."""

    seed: int = 0
    duration_s: float = 120.0
    rate_hz: float = 10.0
    beacon_every: int = 2
    beacon_jitter_s: float = 0.005
    side_m: float = 4.0
    altitude_m: float = 1.5
    takeoff_s: float = 5.0
    speed_mps: float = 0.5
    lag_tau_s: float = 0.8
    est_noise_xy_m: float = 0.03
    est_noise_z_m: float = 0.02
    beacon_noise_xy_m: float = 0.02
    beacon_noise_z_m: float = 0.03
    threshold_cm: float = 10.0
    events: tuple[FailureEvent, ...] | None = None

    def resolved_events(self) -> tuple[FailureEvent, ...]:
        """Events to inject: explicit ones, or the defaults for this duration."""
        return self.events if self.events is not None else default_events(self.duration_s)


@dataclass
class DroneSynthData:
    """Generated tables plus ground truth."""

    pose: pd.DataFrame
    nav_target: pd.DataFrame
    beacon: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


def planned_position(t: np.ndarray, cfg: DroneSynthConfig) -> np.ndarray:
    """Planned position (n, 3): hover-and-climb during takeoff, then laps of a square."""
    n = len(t)
    xyz = np.zeros((n, 3))
    climbing = t < cfg.takeoff_s
    xyz[climbing, 2] = cfg.altitude_m * t[climbing] / cfg.takeoff_s
    flying = ~climbing
    s = cfg.speed_mps * (t[flying] - cfg.takeoff_s)
    side = cfg.side_m
    leg = (np.floor(s / side) % 4).astype(int)
    frac = np.mod(s, side)
    x = np.select([leg == 0, leg == 1, leg == 2, leg == 3], [frac, side, side - frac, 0.0])
    y = np.select([leg == 0, leg == 1, leg == 2, leg == 3], [0.0, frac, side, side - frac])
    xyz[flying, 0] = x
    xyz[flying, 1] = y
    xyz[flying, 2] = cfg.altitude_m + 0.1 * np.sin(2 * np.pi * (t[flying] - cfg.takeoff_s) / 30.0)
    return xyz


def lagged(planned: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """First-order lag response of the vehicle to the planned trajectory."""
    alpha = dt / tau
    out = np.empty_like(planned)
    out[0] = planned[0]
    for i in range(1, len(planned)):
        out[i] = out[i - 1] + alpha * (planned[i] - out[i - 1])
    return out


def _stats(err_cm: np.ndarray, threshold_cm: float, suffix: str) -> dict[str, float]:
    thr_label = f"{threshold_cm:g}cm"
    sorted_err = np.sort(err_cm)
    return {
        f"mean{suffix}_cm": float(err_cm.sum() / err_cm.size),
        f"median{suffix}_cm": float(np.median(sorted_err)),
        f"p95{suffix}_cm": float(np.percentile(sorted_err, 95)),
        f"max{suffix}_cm": float(sorted_err[-1]),
        f"rmse{suffix}_cm": float(np.sqrt((err_cm * err_cm).sum() / err_cm.size)),
        f"pct_under_{thr_label}{suffix}": float(
            100.0 * (err_cm < threshold_cm).sum() / err_cm.size
        ),
    }


def loc_err_truth(
    pose_xyz: np.ndarray, ref_xyz: np.ndarray, threshold_cm: float
) -> dict[str, float]:
    """Localization-error statistics between matched pose and reference arrays (n, 3)."""
    d = pose_xyz - ref_xyz
    err_h = 100.0 * np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    err_3d = 100.0 * np.sqrt((d**2).sum(axis=1))
    out: dict[str, float] = {"n_samples": int(len(d))}
    out.update(_stats(err_h, threshold_cm, ""))
    out.update(_stats(err_3d, threshold_cm, "_3d"))
    return out


def generate_drone(cfg: DroneSynthConfig | None = None) -> DroneSynthData:
    """Generate a deterministic synthetic flight for ``cfg`` (defaults if None)."""
    cfg = cfg or DroneSynthConfig()
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.rate_hz
    n = int(round(cfg.duration_s * cfg.rate_hz))
    t = np.arange(n) * dt

    planned = planned_position(t, cfg)
    true = lagged(planned, dt, cfg.lag_tau_s)

    est_noise = rng.normal(0.0, 1.0, size=(n, 3)) * [
        cfg.est_noise_xy_m,
        cfg.est_noise_xy_m,
        cfg.est_noise_z_m,
    ]
    pose_xyz = true + est_noise
    vel = np.gradient(planned, dt, axis=0)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    roll = rng.normal(0.0, 0.02, size=n)
    pitch = rng.normal(0.0, 0.02, size=n)
    pose = pd.DataFrame(
        {
            "t_s": t,
            "x_m": pose_xyz[:, 0],
            "y_m": pose_xyz[:, 1],
            "z_m": pose_xyz[:, 2],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
        }
    )
    nav_target = pd.DataFrame(
        {"t_s": t, "x_m": planned[:, 0], "y_m": planned[:, 1], "z_m": planned[:, 2]}
    )

    # Beacon: subsample of the true position with noise, timestamp jitter and events.
    idx = np.arange(0, n, cfg.beacon_every)
    m = len(idx)
    tb = t[idx] + rng.uniform(-cfg.beacon_jitter_s, cfg.beacon_jitter_s, size=m)
    beacon_xyz = true[idx] + rng.normal(0.0, 1.0, size=(m, 3)) * [
        cfg.beacon_noise_xy_m,
        cfg.beacon_noise_xy_m,
        cfg.beacon_noise_z_m,
    ]
    keep = np.ones(m, dtype=bool)
    for ev in cfg.resolved_events():
        window = (tb >= ev.start_s) & (tb < ev.end_s)
        if ev.type == "dropout":
            keep &= ~window
        elif ev.type == "jump":
            beacon_xyz[window, 0] += ev.magnitude_m
        elif ev.type == "drift":
            ramp = (tb[window] - ev.start_s) / ev.duration_s
            beacon_xyz[window, 1] += ev.magnitude_m * ramp
    quality = np.clip(rng.normal(85.0, 5.0, size=m), 0, 100).round().astype(int)
    beacon = pd.DataFrame(
        {
            "t_s": tb[keep],
            "x_m": beacon_xyz[keep, 0],
            "y_m": beacon_xyz[keep, 1],
            "z_m": beacon_xyz[keep, 2],
            "quality": quality[keep],
        }
    )

    def _length(xyz: np.ndarray) -> float:
        return float(np.sqrt((np.diff(xyz, axis=0) ** 2).sum(axis=1)).sum())

    truth: dict[str, Any] = {
        "config": {**asdict(cfg), "events": [asdict(e) for e in cfg.resolved_events()]},
        "events": [asdict(e) for e in cfg.resolved_events()],
        "n_pose": int(n),
        "n_beacon": int(keep.sum()),
        "trajectory": {
            "duration_s": float(t[-1] - t[0]),
            "planned_length_m": _length(planned),
            "actual_length_m": _length(pose_xyz),
            "max_altitude_m": float(pose_xyz[:, 2].max()),
        },
        "loc_err": {
            "nav_target": loc_err_truth(pose_xyz, planned, cfg.threshold_cm),
            "beacon": loc_err_truth(pose_xyz[idx[keep]], beacon_xyz[keep], cfg.threshold_cm),
        },
    }
    return DroneSynthData(pose=pose, nav_target=nav_target, beacon=beacon, truth=truth)


def write_drone(data: DroneSynthData, out_dir: Path) -> list[Path]:
    """Write the tables as Parquet and the truth as JSON; return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ("pose", "nav_target", "beacon"):
        p = out_dir / f"{name}.parquet"
        getattr(data, name).to_parquet(p, index=False)
        written.append(p)
    truth_path = out_dir / "truth.json"
    truth_path.write_text(json.dumps(data.truth, indent=2), encoding="utf-8")
    written.append(truth_path)
    return written
