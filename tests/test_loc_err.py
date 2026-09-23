"""drone.loc_err reproduces the generator's ground truth (reproduction test)."""

import numpy as np
import pandas as pd
import pytest

from tracememo.analyses.drone.loc_err import ID, align_to_reference, localization_error
from tracememo.synth.drone import DroneSynthData

STAT_KEYS = [
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


@pytest.mark.parametrize("reference", ["nav_target", "beacon"])
def test_values_match_truth(synth_data: DroneSynthData, reference: str) -> None:
    params = {"reference": reference, "align_tolerance_s": 0.05, "threshold_cm": 10.0}
    result = localization_error(synth_data.pose, synth_data.nav_target, synth_data.beacon, params)
    values = {v.id: v for v in result.values}
    truth = synth_data.truth["loc_err"][reference]
    assert values[f"{ID}.n_samples"].value == truth["n_samples"]
    assert values[f"{ID}.reference"].value == reference
    for key in STAT_KEYS:
        assert values[f"{ID}.{key}"].value == pytest.approx(truth[key], abs=1e-9), key
    assert {f.id for f in result.figures} == {f"{ID}.error_vs_time", f"{ID}.error_cdf"}
    for f in result.figures:
        assert f.figure is not None  # saved later by the runner


def test_all_values_have_units_and_descriptions(synth_data: DroneSynthData) -> None:
    result = localization_error(
        synth_data.pose,
        synth_data.nav_target,
        synth_data.beacon,
        {"reference": "beacon", "align_tolerance_s": 0.05, "threshold_cm": 10.0},
    )
    for v in result.values:
        assert v.description
        if isinstance(v.value, float):
            assert v.unit in ("cm", "%")


def test_align_drops_unmatched() -> None:
    pose = pd.DataFrame({"t_s": [0.0, 1.0, 2.0], "x_m": [0.0, 1.0, 2.0], "y_m": 0.0, "z_m": 0.0})
    ref = pd.DataFrame({"t_s": [0.01, 1.5, 2.02], "x_m": [0.0, 0.0, 0.0], "y_m": 0.0, "z_m": 0.0})
    aligned = align_to_reference(pose, ref, tolerance_s=0.05)
    assert len(aligned) == 2
    assert list(aligned["x_m"]) == [0.0, 2.0]


def test_bad_reference_rejected(synth_data: DroneSynthData) -> None:
    with pytest.raises(ValueError, match="reference"):
        localization_error(
            synth_data.pose,
            synth_data.nav_target,
            synth_data.beacon,
            {"reference": "gps", "align_tolerance_s": 0.05, "threshold_cm": 10.0},
        )


def test_threshold_param_changes_id(synth_data: DroneSynthData) -> None:
    result = localization_error(
        synth_data.pose,
        synth_data.nav_target,
        synth_data.beacon,
        {"reference": "beacon", "align_tolerance_s": 0.05, "threshold_cm": 5},
    )
    ids = {v.id for v in result.values}
    assert f"{ID}.pct_under_5cm" in ids
    v = next(v for v in result.values if v.id == f"{ID}.pct_under_5cm")
    assert 0.0 <= float(v.value) <= 100.0
    assert np.isfinite(float(v.value))
