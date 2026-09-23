"""drone.failures detects the injected dropout, jump and drift events."""

import numpy as np
import pandas as pd
import pytest

from tracememo.analyses.drone.failures import (
    EVENT_COLUMNS,
    ID,
    detect_dropouts,
    detect_jumps,
    failures,
)
from tracememo.analyses.registry import get_analysis
from tracememo.synth.drone import DroneSynthConfig, DroneSynthData, generate_drone

DEFAULTS = get_analysis(ID).params


def test_detects_injected_events(synth_data: DroneSynthData) -> None:
    result = failures(synth_data.pose, synth_data.beacon, DEFAULTS)
    values = {v.id: v.value for v in result.values}
    truth = {e["type"]: e for e in synth_data.truth["events"]}
    assert values[f"{ID}.n_dropouts"] == 1
    assert values[f"{ID}.n_jumps"] == 1
    assert values[f"{ID}.n_drifts"] == 1
    assert values[f"{ID}.n_events"] == 3

    table = result.tables[0]
    assert table.id == f"{ID}.events"
    df: pd.DataFrame = table.dataframe
    assert list(df.columns) == EVENT_COLUMNS
    by_type = {row.type: row for row in df.itertuples()}

    # Dropout: gap starts at the last sample before the gap (within one beacon period).
    d = by_type["dropout"]
    assert d.start_s == pytest.approx(truth["dropout"]["start_s"], abs=0.25)
    assert d.duration_s == pytest.approx(truth["dropout"]["duration_s"], abs=0.25)
    assert d.unit == "s"

    # Jump: starts within one beacon period; magnitude close to the injected offset.
    j = by_type["jump"]
    assert j.start_s == pytest.approx(truth["jump"]["start_s"], abs=0.25)
    assert j.duration_s == pytest.approx(truth["jump"]["duration_s"], abs=0.5)
    assert j.magnitude == pytest.approx(truth["jump"]["magnitude_m"], abs=0.05)
    assert j.unit == "m"

    # Drift: a ramp is flagged once it crosses the threshold; it overlaps the injected window
    # and its magnitude approaches the final bias.
    dr = by_type["drift"]
    t0, t1 = truth["drift"]["start_s"], truth["drift"]["start_s"] + truth["drift"]["duration_s"]
    assert t0 <= dr.start_s <= t1
    assert dr.end_s == pytest.approx(t1, abs=DEFAULTS["drift_window_s"])
    assert dr.magnitude == pytest.approx(truth["drift"]["magnitude_m"], abs=0.06)

    assert values[f"{ID}.dropout_total_s"] == pytest.approx(d.duration_s)
    assert 0 < values[f"{ID}.availability_pct"] < 100


def test_clean_flight_has_no_events() -> None:
    data = generate_drone(DroneSynthConfig(seed=2, duration_s=60, events=()))
    result = failures(data.pose, data.beacon, DEFAULTS)
    values = {v.id: v.value for v in result.values}
    assert values[f"{ID}.n_events"] == 0
    assert values[f"{ID}.availability_pct"] == pytest.approx(100.0)
    assert result.tables[0].dataframe.empty


def test_detect_dropouts_unit() -> None:
    t = np.array([0.0, 0.2, 0.4, 2.0, 2.2])
    events = detect_dropouts(t, gap_threshold_s=0.5)
    assert len(events) == 1
    assert events[0].start_s == 0.4 and events[0].end_s == 2.0
    assert events[0].magnitude == pytest.approx(1.6)


def test_detect_jumps_ignores_single_sample_steps() -> None:
    t = np.arange(10) * 0.2
    residual = np.zeros((10, 2))
    residual[5, 0] = 1.0  # a one-sample spike, not a sustained jump
    offset = np.linalg.norm(residual, axis=1)
    events, mask = detect_jumps(t, offset, residual, jump_threshold_m=0.2, min_samples=2)
    assert events == [] and not mask.any()
    residual[5:9, 0] = 1.0
    offset = np.linalg.norm(residual, axis=1)
    events, mask = detect_jumps(t, offset, residual, jump_threshold_m=0.2, min_samples=2)
    assert len(events) == 1 and mask[5:9].all() and mask.sum() == 4
