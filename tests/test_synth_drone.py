"""Synthetic drone generator: determinism, schemas, events and truth."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracememo.synth.drone import DroneSynthConfig, DroneSynthData, generate_drone, write_drone


def test_deterministic_for_seed() -> None:
    a = generate_drone(DroneSynthConfig(seed=3, duration_s=20))
    b = generate_drone(DroneSynthConfig(seed=3, duration_s=20))
    pd.testing.assert_frame_equal(a.pose, b.pose)
    pd.testing.assert_frame_equal(a.beacon, b.beacon)
    assert a.truth == b.truth
    c = generate_drone(DroneSynthConfig(seed=4, duration_s=20))
    assert not a.pose.equals(c.pose)


def test_schemas(synth_data: DroneSynthData) -> None:
    assert list(synth_data.pose.columns) == [
        "t_s",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    ]
    assert list(synth_data.nav_target.columns) == ["t_s", "x_m", "y_m", "z_m"]
    assert list(synth_data.beacon.columns) == ["t_s", "x_m", "y_m", "z_m", "quality"]
    assert synth_data.pose["t_s"].iloc[0] == 0.0
    assert synth_data.pose["t_s"].is_monotonic_increasing


def test_planned_trajectory_stays_in_room(
    synth_data: DroneSynthData, small_cfg: DroneSynthConfig
) -> None:
    nt = synth_data.nav_target
    assert nt["x_m"].between(0, small_cfg.side_m).all()
    assert nt["y_m"].between(0, small_cfg.side_m).all()
    assert nt["z_m"].max() <= small_cfg.altitude_m + 0.1 + 1e-9


def test_events_are_visible_in_beacon(synth_data: DroneSynthData) -> None:
    events = {e["type"]: e for e in synth_data.truth["events"]}
    b = synth_data.beacon
    d = events["dropout"]
    in_gap = b[(b["t_s"] >= d["start_s"]) & (b["t_s"] < d["start_s"] + d["duration_s"])]
    assert in_gap.empty
    j = events["jump"]
    nav = synth_data.nav_target
    win = b[(b["t_s"] >= j["start_s"]) & (b["t_s"] < j["start_s"] + j["duration_s"])]
    ref = nav.iloc[np.searchsorted(nav["t_s"].to_numpy(), win["t_s"].to_numpy())]
    # During the jump the beacon x is offset by ~magnitude beyond lag/noise.
    assert (win["x_m"].to_numpy() - ref["x_m"].to_numpy()).mean() > j["magnitude_m"] * 0.5


def test_truth_stats_keys(synth_data: DroneSynthData) -> None:
    for ref in ("nav_target", "beacon"):
        stats = synth_data.truth["loc_err"][ref]
        for k in (
            "mean_cm",
            "median_cm",
            "p95_cm",
            "max_cm",
            "rmse_cm",
            "pct_under_10cm",
            "mean_3d_cm",
            "rmse_3d_cm",
            "n_samples",
        ):
            assert k in stats
    assert synth_data.truth["loc_err"]["beacon"]["n_samples"] == len(synth_data.beacon)


def test_write_drone(tmp_path: Path, synth_data: DroneSynthData) -> None:
    written = write_drone(synth_data, tmp_path / "out")
    names = sorted(p.name for p in written)
    assert names == [
        "beacon.parquet",
        "flight.bin",
        "marvelmind.csv",
        "nav_target.parquet",
        "pose.parquet",
        "truth.json",
    ]
    assert sorted(p.name for p in write_drone(synth_data, tmp_path / "np", raw_formats=False)) == [
        "beacon.parquet",
        "nav_target.parquet",
        "pose.parquet",
        "truth.json",
    ]
    truth = json.loads((tmp_path / "out" / "truth.json").read_text())
    assert truth["n_pose"] == len(synth_data.pose)
