"""drone.trajectory values match generator truth; figures are produced."""

import pytest

from tracememo.analyses.drone.trajectory import ID, path_length, trajectory
from tracememo.synth.drone import DroneSynthData


def test_values_match_truth(synth_data: DroneSynthData) -> None:
    result = trajectory(synth_data.pose, synth_data.nav_target, {})
    values = {v.id: v.value for v in result.values}
    truth = synth_data.truth["trajectory"]
    for key in ("duration_s", "planned_length_m", "actual_length_m", "max_altitude_m"):
        assert values[f"{ID}.{key}"] == pytest.approx(truth[key], rel=1e-9), key
    assert {f.id for f in result.figures} == {f"{ID}.top_down", f"{ID}.altitude"}


def test_path_length_square() -> None:
    import numpy as np

    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    assert path_length(square) == pytest.approx(4.0)
    assert path_length(square[:1]) == 0.0
