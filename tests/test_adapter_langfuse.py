"""Langfuse export adapter: shapes, versions, latency fallback, span types."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tracememo.adapters import get_adapter
from tracememo.adapters.langfuse import build_tables, ingest_langfuse, load_export, trace_version
from tracememo.synth.rag import RagSynthConfig, generate_rag, write_rag

SPAN_COLS = ["trace_id", "span_id", "parent_id", "name", "start", "end", "duration_ms", "metadata"]
TRACE_COLS = ["trace_id", "version", "total_latency_ms", "input", "output"]


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("rag")
    write_rag(generate_rag(RagSynthConfig(n_traces=4, n_items=5)), out)
    return out


def test_registered() -> None:
    assert get_adapter("langfuse") is ingest_langfuse


def test_ingest_synthetic_export(export_dir: Path) -> None:
    tables = {t.id: t for t in ingest_langfuse(export_dir / "traces.json", None, {})}
    spans, traces = tables["spans"].df, tables["traces"].df
    assert list(spans.columns) == SPAN_COLS
    assert list(traces.columns) == TRACE_COLS
    assert len(traces) == 8 and len(spans) == 8 * 5
    assert set(traces["version"]) == {"v1", "v2"}
    assert str(spans["start"].dtype).startswith("datetime64") and "UTC" in str(spans["start"].dtype)
    assert (spans["duration_ms"] > 0).all()
    root = spans[spans["name"] == "pipeline"]
    assert root["parent_id"].isna().all()
    # Root span duration equals the trace latency.
    merged = root.merge(traces, on="trace_id")
    assert (merged["duration_ms"] - merged["total_latency_ms"]).abs().max() < 1e-3
    assert json.loads(spans["metadata"].iloc[0]) in ({"stage": "embed_query"}, {})
    assert tables["spans"].raw_inputs[0].path.endswith("traces.json")


def test_only_requested_tables(export_dir: Path) -> None:
    (t,) = ingest_langfuse(export_dir / "traces.json", ["traces"], {})
    assert t.id == "traces"
    with pytest.raises(KeyError, match="produces"):
        ingest_langfuse(export_dir / "traces.json", ["nope"], {})


def test_export_shapes_and_embedded_observations() -> None:
    trace = {
        "id": "t1",
        "version": "a",
        "observations": [
            {
                "id": "s1",
                "traceId": "t1",
                "type": "SPAN",
                "name": "x",
                "startTime": "2026-01-01T00:00:00Z",
                "endTime": "2026-01-01T00:00:01Z",
            }
        ],
    }
    for obj in ([dict(trace)], {"data": [dict(trace)]}):
        traces, obs = load_export(json.loads(json.dumps(obj)))
        assert len(traces) == 1 and len(obs) == 1 and "observations" not in traces[0]
    with pytest.raises(ValueError, match="unrecognized"):
        load_export({"foo": 1})


def test_latency_fallback_and_span_types() -> None:
    traces = [{"id": "t1", "metadata": {"version": "m1"}, "tags": ["version:tg"]}]
    obs = [
        {
            "id": "s1",
            "traceId": "t1",
            "type": "SPAN",
            "name": "a",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-01T00:00:01Z",
        },
        {
            "id": "s2",
            "traceId": "t1",
            "type": "GENERATION",
            "name": "b",
            "startTime": "2026-01-01T00:00:01Z",
            "endTime": "2026-01-01T00:00:03Z",
        },
        {
            "id": "e1",
            "traceId": "t1",
            "type": "EVENT",
            "name": "evt",
            "startTime": "2026-01-01T00:00:01Z",
            "endTime": "2026-01-01T00:00:01Z",
        },
    ]
    spans, tr = build_tables(traces, obs, version_from="metadata.version")
    assert list(spans["name"]) == ["a", "b"]  # EVENT excluded
    assert tr["version"].iloc[0] == "m1"
    assert tr["total_latency_ms"].iloc[0] == pytest.approx(3000.0)  # from span extent
    _, tr2 = build_tables(traces, obs, version_from="tag:version:")
    assert tr2["version"].iloc[0] == "tg"
    _, tr3 = build_tables(traces, obs, version_from="release")
    assert pd.isna(tr3["version"].iloc[0])
    spans_only, _ = build_tables(traces, obs, span_types=("SPAN",))
    assert list(spans_only["name"]) == ["a"]


def test_trace_version_variants() -> None:
    t = {"version": "v9", "metadata": {"version": "m"}, "tags": ["env:prod", "version:tg"]}
    assert trace_version(t, "version") == "v9"
    assert trace_version(t, "metadata.version") == "m"
    assert trace_version(t, "tag:version:") == "tg"
    assert trace_version({}, "tag:version:") is None
    assert trace_version({}, "metadata.version") is None
