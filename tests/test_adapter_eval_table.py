"""Eval-table adapter: long/wide CSV and JSON inputs."""

from pathlib import Path

import pytest

from tracememo.adapters import get_adapter
from tracememo.adapters.eval_table import OUT_COLUMNS, ingest_eval_table


def test_registered() -> None:
    assert get_adapter("eval_table") is ingest_eval_table


def test_long_csv_default_columns(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    p.write_text("item_id,config,metric_name,score\nq1,a,corr,0.5\nq2,a,corr,\nq1,b,corr,0.9\n")
    (t,) = ingest_eval_table(p, None, {})
    assert t.id == "eval_scores" and list(t.df.columns) == OUT_COLUMNS
    assert len(t.df) == 2  # NaN score dropped
    assert t.df["score"].dtype == float


def test_long_custom_columns(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    p.write_text("qid,run,metric,value\n1,a,corr,0.5\n")
    (t,) = ingest_eval_table(
        p,
        ["scores"],
        {
            "table": "scores",
            "columns": {
                "item_id": "qid",
                "config": "run",
                "metric_name": "metric",
                "score": "value",
            },
        },
    )
    assert t.id == "scores"
    assert t.df.iloc[0].to_dict() == {
        "item_id": "1",
        "config": "a",
        "metric_name": "corr",
        "score": 0.5,
    }


def test_wide_json(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text(
        '{"data": [{"item_id": "q1", "config": "a", "corr": 0.5, "faith": 0.7},'
        ' {"item_id": "q2", "config": "a", "corr": 0.6, "faith": 0.8}]}'
    )
    (t,) = ingest_eval_table(p, None, {"format": "wide"})
    assert len(t.df) == 4
    assert set(t.df["metric_name"]) == {"corr", "faith"}
    (t2,) = ingest_eval_table(p, None, {"format": "wide", "metrics": ["corr"]})
    assert set(t2.df["metric_name"]) == {"corr"}


def test_errors(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    p.write_text("item_id,config,metric_name,score\nq1,a,c,0.5\n")
    with pytest.raises(ValueError, match="format"):
        ingest_eval_table(p, None, {"format": "tall"})
    with pytest.raises(KeyError, match="produces"):
        ingest_eval_table(p, ["other"], {})
    p.write_text("item_id,score\nq1,0.5\n")
    with pytest.raises(KeyError, match="'config' for config"):
        ingest_eval_table(p, None, {})
