"""Synthetic RAG generator: determinism, export shape, truth."""

import json
from pathlib import Path

import numpy as np

from tracememo.synth.rag import METRICS, STAGES, RagSynthConfig, generate_rag, write_rag


def test_deterministic() -> None:
    a = generate_rag(RagSynthConfig(seed=3, n_traces=5, n_items=6))
    b = generate_rag(RagSynthConfig(seed=3, n_traces=5, n_items=6))
    assert a.traces == b.traces and a.observations == b.observations
    assert a.eval_scores.equals(b.eval_scores)
    c = generate_rag(RagSynthConfig(seed=4, n_traces=5, n_items=6))
    assert a.observations != c.observations


def test_shapes_and_truth() -> None:
    cfg = RagSynthConfig(seed=0, n_traces=10, n_items=8)
    d = generate_rag(cfg)
    assert len(d.traces) == 20
    assert len(d.observations) == 20 * (len(STAGES) + 1)
    assert len(d.eval_scores) == len(cfg.configs) * len(METRICS) * cfg.n_items
    assert d.eval_scores["score"].between(0, 1).all()
    t = d.truth["latency"]
    assert t["v2"]["e2e_median_ms"] < t["v1"]["e2e_median_ms"]
    assert 0 < t["reduction_pct"] < 100
    assert t["v2"]["stages"]["retrieve"]["median_ms"] < t["v1"]["stages"]["retrieve"]["median_ms"]
    for c in cfg.configs:
        for m in METRICS:
            assert abs(d.truth["eval"][c][m]["sample_mean"] - cfg.metric_means[c][m]) < 0.15


def test_spans_are_sequential_and_sum_to_latency() -> None:
    d = generate_rag(RagSynthConfig(n_traces=3))
    by_trace: dict[str, list[dict]] = {}
    for o in d.observations:
        by_trace.setdefault(o["traceId"], []).append(o)
    for t in d.traces:
        obs = by_trace[t["id"]]
        root = next(o for o in obs if o["parentObservationId"] is None)
        stages = [o for o in obs if o["parentObservationId"] == root["id"]]
        assert [o["name"] for o in stages] == list(STAGES)
        for a, b in zip(stages, stages[1:], strict=False):
            assert a["endTime"] == b["startTime"]
        assert stages[0]["startTime"] == root["startTime"] == t["timestamp"]
        assert stages[-1]["endTime"] == root["endTime"]
        assert np.isclose(t["latency"], 0.0) is np.False_


def test_write_rag(tmp_path: Path) -> None:
    d = generate_rag(RagSynthConfig(n_traces=2, n_items=3))
    paths = write_rag(d, tmp_path / "rag")
    assert [p.name for p in paths] == ["traces.json", "eval_scores.csv", "truth.json"]
    obj = json.loads((tmp_path / "rag" / "traces.json").read_text())
    assert set(obj) == {"traces", "observations"}
