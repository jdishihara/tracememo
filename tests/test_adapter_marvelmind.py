"""Marvelmind CSV adapter on the committed fixture and on hand-written CSVs."""

import json
from pathlib import Path

import numpy as np
import pytest

from tracememo.adapters import get_adapter
from tracememo.adapters.marvelmind import ingest_marvelmind
from tracememo.synth.drone import MARVELMIND_TIME_OFFSET_S

FIXTURES = Path(__file__).parent / "fixtures"
CSV = FIXTURES / "marvelmind_small.csv"
EXPECTED = json.loads((FIXTURES / "expected_small.json").read_text())["beacon"]


def test_registered() -> None:
    assert get_adapter("marvelmind") is ingest_marvelmind


def test_fixture_defaults() -> None:
    (t,) = ingest_marvelmind(CSV, None, {})
    assert t.id == "beacon_raw"
    assert list(t.df.columns) == ["t_s", "x_m", "y_m", "z_m", "quality"]
    assert len(t.df) == EXPECTED["rows"]
    assert t.df["t_s"].iloc[0] == pytest.approx(
        EXPECTED["first"]["t_s"] + MARVELMIND_TIME_OFFSET_S, abs=1e-3
    )
    assert t.df["x_m"].iloc[-1] == pytest.approx(EXPECTED["last"]["x_m"], abs=1e-3)
    assert t.raw_inputs[0].path == str(CSV)


def test_time_offset_and_table_name() -> None:
    (t,) = ingest_marvelmind(
        CSV, ["beacon"], {"table": "beacon", "time_offset_s": -MARVELMIND_TIME_OFFSET_S}
    )
    assert t.id == "beacon"
    assert t.df["t_s"].iloc[0] == pytest.approx(EXPECTED["first"]["t_s"], abs=1e-3)
    with pytest.raises(KeyError, match="produces"):
        ingest_marvelmind(CSV, ["beacon"], {})


def test_custom_columns_units_and_address(tmp_path: Path) -> None:
    csv = tmp_path / "hedge.csv"
    csv.write_text(
        "ts_ms; hedge; x_mm; y_mm; z_mm\n"
        "1000; 5; 1000; 2000; 500\n"
        "1200; 6; 9999; 9999; 9999\n"
        "1400; 5; 1100; 2100; 600\n",
        encoding="utf-8",
    )
    (t,) = ingest_marvelmind(
        csv,
        None,
        {
            "delimiter": ";",
            "columns": {"time": "ts_ms", "x": "x_mm", "y": "y_mm", "z": "z_mm", "address": "hedge"},
            "address": 5,
            "length_unit": "mm",
            "time_unit": "ms",
            "time_origin": "first",
        },
    )
    assert list(t.df.columns) == ["t_s", "x_m", "y_m", "z_m"]  # no quality column configured
    np.testing.assert_allclose(t.df["t_s"], [0.0, 0.4])
    np.testing.assert_allclose(t.df["x_m"], [1.0, 1.1])
    np.testing.assert_allclose(t.df["z_m"], [0.5, 0.6])


def test_missing_column_error(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("Time,X,Y\n1,2,3\n", encoding="utf-8")
    with pytest.raises(KeyError, match="'Z' for z"):
        ingest_marvelmind(csv, None, {})


def test_bad_rows_dropped(tmp_path: Path) -> None:
    csv = tmp_path / "gaps.csv"
    csv.write_text("Time,X,Y,Z\n2,1,1,1\n1,0,0,0\nnan,3,3,3\n", encoding="utf-8")
    (t,) = ingest_marvelmind(csv, None, {})
    assert list(t.df["t_s"]) == [1.0, 2.0]  # sorted, NaN row dropped
