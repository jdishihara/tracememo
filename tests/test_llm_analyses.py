"""llm.latency and llm.eval against synthetic truth."""

import numpy as np
import pytest

from tracememo.adapters.eval_table import to_long
from tracememo.adapters.langfuse import build_tables
from tracememo.analyses.llm.eval import bootstrap_ci, evaluation
from tracememo.analyses.llm.latency import latency, stage_order, stage_spans
from tracememo.synth.rag import METRICS, STAGES, RagSynthConfig, RagSynthData, generate_rag


@pytest.fixture(scope="module")
def rag() -> RagSynthData:
    return generate_rag(RagSynthConfig(seed=2, n_traces=30, n_items=25))


@pytest.fixture(scope="module")
def tables(rag: RagSynthData):
    spans, traces = build_tables(rag.traces, rag.observations)
    return spans, traces, to_long(rag.eval_scores, {})


def test_latency_matches_truth(rag: RagSynthData, tables) -> None:
    spans, traces, _ = tables
    res = latency(spans, traces, {"baseline": None, "candidate": None, "stage_names": None})
    v = {x.id: x.value for x in res.values}
    t = rag.truth["latency"]
    for ver in ("v1", "v2"):
        assert v[f"llm.latency.{ver}.n_traces"] == 30
        assert v[f"llm.latency.{ver}.e2e_median_ms"] == pytest.approx(
            t[ver]["e2e_median_ms"], abs=1e-2
        )
        assert v[f"llm.latency.{ver}.e2e_p95_ms"] == pytest.approx(t[ver]["e2e_p95_ms"], abs=1e-2)
        for s in STAGES:
            assert v[f"llm.latency.{ver}.{s}.median_ms"] == pytest.approx(
                t[ver]["stages"][s]["median_ms"], abs=1e-2
            )
            assert v[f"llm.latency.{ver}.{s}.p95_ms"] == pytest.approx(
                t[ver]["stages"][s]["p95_ms"], abs=1e-2
            )
    assert v["llm.latency.reduction_pct"] == pytest.approx(t["reduction_pct"], abs=1e-3)
    assert v["llm.latency.baseline_version"] == "v1" and v["llm.latency.candidate_version"] == "v2"
    table = res.tables[0].dataframe
    assert list(table["stage"]) == list(STAGES)  # ordered by position in the pipeline
    assert list(table.columns) == [
        "stage",
        "v1_median_ms",
        "v1_p95_ms",
        "v2_median_ms",
        "v2_p95_ms",
    ]
    assert {f.id for f in res.figures} == {"llm.latency.stacked"}


def test_stage_selection(tables) -> None:
    spans, _, _ = tables
    leaves = stage_spans(spans, None)
    assert set(leaves["name"]) == set(STAGES)  # root "pipeline" span excluded
    named = stage_spans(spans, ["retrieve"])
    assert set(named["name"]) == {"retrieve"}
    assert stage_order(leaves) == list(STAGES)


def test_latency_version_params(tables) -> None:
    spans, traces, _ = tables
    res = latency(spans, traces, {"baseline": "v2", "candidate": "v1", "stage_names": None})
    v = {x.id: x.value for x in res.values}
    assert v["llm.latency.reduction_pct"] < 0  # v1 is slower than v2
    with pytest.raises(ValueError, match="not in traces"):
        latency(spans, traces, {"baseline": "v7", "candidate": None, "stage_names": None})


def test_eval_matches_truth(rag: RagSynthData, tables) -> None:
    _, _, scores = tables
    params = {"n_bootstrap": 500, "ci": 0.95, "seed": 0, "metrics": None, "configs": None}
    res = evaluation(scores, params)
    v = {x.id: x for x in res.values}
    for c in rag.truth["eval"]:
        for m in METRICS:
            base = f"llm.eval.{c}.{m}"
            truth = rag.truth["eval"][c][m]["sample_mean"]
            assert v[f"{base}.mean"].value == pytest.approx(truth, abs=1e-9)
            assert v[f"{base}.n"].value == 25
            lo, hi = v[f"{base}.ci_low"].value, v[f"{base}.ci_high"].value
            assert lo < truth < hi
            assert 0.01 < hi - lo < 0.3
    table = res.tables[0].dataframe
    assert list(table.columns) == ["metric", "baseline", "rerank", "rerank_hyde"]
    assert list(table["metric"]) == list(METRICS)
    assert table.iloc[0]["baseline"].startswith(
        f"{v['llm.eval.baseline.correctness.mean'].formatted()} ["
    )
    assert {f.id for f in res.figures} == {"llm.eval.means"}
    # Deterministic under the seed.
    again = evaluation(scores, params)
    assert [x.value for x in again.values] == [x.value for x in res.values]


def test_bootstrap_ci_basic() -> None:
    rng = np.random.default_rng(0)
    x = np.full(50, 0.5)
    assert bootstrap_ci(x, 100, 0.95, rng) == (0.5, 0.5)
    x = rng.normal(0.0, 1.0, size=400)
    lo, hi = bootstrap_ci(x, 2000, 0.95, rng)
    assert lo < x.mean() < hi and 0.15 < hi - lo < 0.25
