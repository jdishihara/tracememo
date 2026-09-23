"""ArduPilot adapter on a committed DataFlash fixture and on generated logs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracememo.adapters import get_adapter
from tracememo.adapters.ardupilot import (
    PROFILES,
    arming_time_us,
    ingest_ardupilot,
    read_messages,
    resolve_options,
)
from tracememo.synth.dataflash import write_dataflash_bin
from tracememo.synth.drone import DroneSynthConfig, DroneSynthData, generate_drone

FIXTURES = Path(__file__).parent / "fixtures"
BIN = FIXTURES / "flight_small.bin"
EXPECTED = json.loads((FIXTURES / "expected_small.json").read_text())


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def test_registered() -> None:
    assert get_adapter("ardupilot") is ingest_ardupilot


def test_fixture_tables_match_expected() -> None:
    tables = {t.id: t for t in ingest_ardupilot(BIN, None, {})}
    assert set(tables) == {"pose", "nav_target", "beacon"}
    for name in ("pose", "nav_target"):
        df = tables[name].df
        exp = EXPECTED[name]
        assert list(df.columns) == exp["columns"]
        assert len(df) == exp["rows"]
        for col, v in exp["first"].items():
            got = float(df[col].iloc[0])
            if col == "yaw_rad":
                got, v = float(_wrap(np.array([got]))[0]), float(_wrap(np.array([v]))[0])
            assert got == pytest.approx(v, abs=2e-4), (name, col)
        assert float(df["x_m"].sum()) == pytest.approx(exp["sum_x_m"], abs=1e-3)
    assert tables["pose"].raw_inputs[0].path == str(BIN)
    assert len(tables["pose"].raw_inputs[0].sha256) == 64


def test_fixture_beacon_from_bcn() -> None:
    beacon = ingest_ardupilot(BIN, ["beacon"], {})[0].df
    assert list(beacon.columns) == ["t_s", "x_m", "y_m", "z_m", "quality"]
    exp = EXPECTED["beacon"]
    # The first beacon sample may sit just before arming (timestamp jitter) and be dropped.
    assert len(beacon) in (exp["rows"], exp["rows"] - 1)
    assert beacon["quality"].dtype.kind == "i"
    assert float(beacon["x_m"].iloc[-1]) == pytest.approx(exp["last"]["x_m"], abs=1e-5)


def test_matches_normalized_tables(synth_data: DroneSynthData, tmp_path: Path) -> None:
    """Round trip: synthetic -> .bin -> adapter reproduces the normalized tables (float32)."""
    path = write_dataflash_bin(synth_data, tmp_path / "flight.bin")
    tables = {t.id: t.df for t in ingest_ardupilot(path, ["pose", "nav_target"], {})}
    for name in ("pose", "nav_target"):
        got, ref = tables[name], getattr(synth_data, name)
        assert len(got) == len(ref)
        np.testing.assert_allclose(got["t_s"], ref["t_s"], atol=1e-6)
        for c in ("x_m", "y_m", "z_m"):
            np.testing.assert_allclose(got[c], ref[c], atol=1e-5)
    np.testing.assert_allclose(
        _wrap(tables["pose"]["yaw_rad"].to_numpy()),
        _wrap(synth_data.pose["yaw_rad"].to_numpy()),
        atol=2e-4,
    )
    assert tables["pose"]["yaw_rad"].between(-np.pi, np.pi).all()


def test_prearm_samples_dropped_and_instance_filtered() -> None:
    frames = read_messages(BIN, ["XKF1", "EV"])
    assert (frames["XKF1"]["C"] == 1).any()  # second EKF core present in the log
    t_arm = arming_time_us(frames, PROFILES["arducopter4"]["arming"])
    assert t_arm == 5_000_000
    n_prearm = int((frames["XKF1"]["TimeUS"] < t_arm).sum())
    assert n_prearm > 0
    pose = ingest_ardupilot(BIN, ["pose"], {})[0].df
    assert (pose["t_s"] >= 0).all()
    assert len(pose) == EXPECTED["pose"]["rows"]  # core 0 only, post-arm only

    keep = ingest_ardupilot(BIN, ["pose"], {"drop_before_arming": False})[0].df
    assert len(keep) == EXPECTED["pose"]["rows"] + n_prearm // 1
    assert keep["t_s"].iloc[0] < 0


def test_mapping_override_and_convert() -> None:
    opts = {
        "mapping": {
            "pose": {
                "sources": [
                    {
                        "message": "XKF1",
                        "instance": {"field": "C", "value": 1},
                        "fields": {"x_m": "PE", "y_m": "PN", "z_m": {"field": "PD", "scale": -1}},
                    }
                ]
            }
        }
    }
    core1 = ingest_ardupilot(BIN, ["pose"], opts)[0].df
    core0 = ingest_ardupilot(BIN, ["pose"], {})[0].df
    assert list(core1.columns) == ["t_s", "x_m", "y_m", "z_m"]
    np.testing.assert_allclose(core1["x_m"], core0["x_m"] + 0.5, atol=1e-5)


def test_no_arming_event_falls_back_to_first_timestamp(tmp_path: Path) -> None:
    data = generate_drone(DroneSynthConfig(seed=1, duration_s=2.0))
    path = write_dataflash_bin(data, tmp_path / "f.bin", prearm_s=0.0)
    pose = ingest_ardupilot(path, ["pose"], {"arming": None})[0].df
    assert pose["t_s"].iloc[0] == pytest.approx(0.0)
    assert len(pose) == len(data.pose)


def test_errors() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_options({"profile": "px4"})
    with pytest.raises(KeyError, match="no mapping"):
        ingest_ardupilot(BIN, ["gps"], {})
    with pytest.raises(ValueError, match="no GPS messages"):
        ingest_ardupilot(
            BIN, ["pose"], {"mapping": {"pose": {"sources": [{"message": "GPS", "fields": {}}]}}}
        )
    with pytest.raises(KeyError, match="no field"):
        ingest_ardupilot(
            BIN,
            ["pose"],
            {"mapping": {"pose": {"sources": [{"message": "XKF1", "fields": {"x_m": "Nope"}}]}}},
        )


def test_multi_source_merge_tolerance() -> None:
    """nav_target joins PSCN/PSCE/PSCD by nearest time; all rows should match."""
    nav = ingest_ardupilot(BIN, ["nav_target"], {})[0].df
    assert not nav.isna().any().any()
    assert len(nav) == EXPECTED["nav_target"]["rows"]
    strict = ingest_ardupilot(BIN, ["nav_target"], {"merge_tolerance_s": 0.0})[0].df
    assert len(strict) == len(nav)  # identical timestamps still match at zero tolerance
    assert isinstance(strict, pd.DataFrame)
