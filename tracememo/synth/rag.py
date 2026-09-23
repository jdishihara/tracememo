"""Synthetic RAG pipeline traces and evaluation scores with known ground truth.

Writes a Langfuse-style trace export (``traces.json``), a long-format evaluation CSV
(``eval_scores.csv``) and ``truth.json``. Two system versions are generated; the second is
faster in specific stages. Several configurations get evaluation scores with known means.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGES = ("embed_query", "retrieve", "rerank", "generate")
METRICS = (
    "correctness",
    "faithfulness",
    "relevance",
    "retrieval_mrr",
    "entity_precision",
    "entity_recall",
)


@dataclass(frozen=True)
class RagSynthConfig:
    """Parameters of the synthetic RAG data."""

    seed: int = 0
    n_traces: int = 60  # per version
    n_items: int = 40  # evaluation questions
    versions: tuple[str, ...] = ("v1", "v2")
    # Median stage latency in ms per version (lognormal with sigma below).
    stage_median_ms: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "v1": {"embed_query": 40.0, "retrieve": 350.0, "rerank": 600.0, "generate": 1800.0},
            "v2": {"embed_query": 40.0, "retrieve": 120.0, "rerank": 250.0, "generate": 1750.0},
        }
    )
    stage_sigma: float = 0.25
    configs: tuple[str, ...] = ("baseline", "rerank", "rerank_hyde")
    # True mean score per config and metric (scores are Beta-distributed in [0, 1]).
    metric_means: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "baseline": {
                "correctness": 0.62,
                "faithfulness": 0.75,
                "relevance": 0.70,
                "retrieval_mrr": 0.55,
                "entity_precision": 0.68,
                "entity_recall": 0.52,
            },
            "rerank": {
                "correctness": 0.71,
                "faithfulness": 0.80,
                "relevance": 0.78,
                "retrieval_mrr": 0.72,
                "entity_precision": 0.74,
                "entity_recall": 0.60,
            },
            "rerank_hyde": {
                "correctness": 0.74,
                "faithfulness": 0.79,
                "relevance": 0.81,
                "retrieval_mrr": 0.76,
                "entity_precision": 0.75,
                "entity_recall": 0.66,
            },
        }
    )
    score_concentration: float = 12.0  # Beta(a+b); higher = less spread
    start_time: str = "2026-09-01T00:00:00+00:00"


@dataclass
class RagSynthData:
    """Generated export objects plus truth."""

    traces: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    eval_scores: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def generate_rag(cfg: RagSynthConfig | None = None) -> RagSynthData:
    """Generate deterministic synthetic traces and eval scores for ``cfg``."""
    cfg = cfg or RagSynthConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = datetime.fromisoformat(cfg.start_time).astimezone(UTC)

    traces: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    stage_samples: dict[str, dict[str, list[float]]] = {
        v: {s: [] for s in STAGES} for v in cfg.versions
    }
    e2e_samples: dict[str, list[float]] = {v: [] for v in cfg.versions}
    k = 0
    for v in cfg.versions:
        for i in range(cfg.n_traces):
            trace_id = f"tr_{v}_{i:04d}"
            start = t0 + timedelta(seconds=2.0 * k)
            k += 1
            root_id = f"obs_{trace_id}_root"
            cursor = start
            for s in STAGES:
                median = cfg.stage_median_ms[v][s]
                dur_ms = float(median * np.exp(rng.normal(0.0, cfg.stage_sigma)))
                end = cursor + timedelta(milliseconds=dur_ms)
                observations.append(
                    {
                        "id": f"obs_{trace_id}_{s}",
                        "traceId": trace_id,
                        "parentObservationId": root_id,
                        "type": "SPAN",
                        "name": s,
                        "startTime": _iso(cursor),
                        "endTime": _iso(end),
                        "metadata": {"stage": s},
                    }
                )
                stage_samples[v][s].append(dur_ms)
                cursor = end
            total_ms = (cursor - start).total_seconds() * 1000.0
            e2e_samples[v].append(total_ms)
            observations.append(
                {
                    "id": root_id,
                    "traceId": trace_id,
                    "parentObservationId": None,
                    "type": "SPAN",
                    "name": "pipeline",
                    "startTime": _iso(start),
                    "endTime": _iso(cursor),
                    "metadata": {},
                }
            )
            traces.append(
                {
                    "id": trace_id,
                    "name": "rag_query",
                    "timestamp": _iso(start),
                    "version": v,
                    "release": None,
                    "tags": [f"version:{v}"],
                    "metadata": {"version": v, "question_id": f"q{i % cfg.n_items:03d}"},
                    "input": {"question": f"Synthetic question {i % cfg.n_items}?"},
                    "output": {"answer": f"Synthetic answer {i} from {v}."},
                    "latency": total_ms / 1000.0,
                }
            )

    rows = []
    for c in cfg.configs:
        for m in METRICS:
            mean = cfg.metric_means[c][m]
            a = mean * cfg.score_concentration
            b = (1.0 - mean) * cfg.score_concentration
            scores = rng.beta(a, b, size=cfg.n_items)
            for j, sc in enumerate(scores):
                rows.append(
                    {"item_id": f"q{j:03d}", "config": c, "metric_name": m, "score": float(sc)}
                )
    eval_scores = pd.DataFrame(rows)

    latency_truth: dict[str, Any] = {}
    for v in cfg.versions:
        e2e = np.array(e2e_samples[v])
        latency_truth[v] = {
            "e2e_median_ms": float(np.median(e2e)),
            "e2e_p95_ms": float(np.percentile(e2e, 95)),
            "stages": {
                s: {
                    "median_ms": float(np.median(stage_samples[v][s])),
                    "p95_ms": float(np.percentile(stage_samples[v][s], 95)),
                }
                for s in STAGES
            },
        }
    v_a, v_b = cfg.versions[0], cfg.versions[-1]
    latency_truth["reduction_pct"] = float(
        100.0 * (1.0 - latency_truth[v_b]["e2e_median_ms"] / latency_truth[v_a]["e2e_median_ms"])
    )
    eval_truth = {
        c: {
            m: {
                "sample_mean": float(
                    eval_scores[
                        (eval_scores.config == c) & (eval_scores.metric_name == m)
                    ].score.mean()
                ),
                "true_mean": cfg.metric_means[c][m],
            }
            for m in METRICS
        }
        for c in cfg.configs
    }
    truth = {
        "config": asdict(cfg),
        "n_traces": len(traces),
        "n_observations": len(observations),
        "latency": latency_truth,
        "eval": eval_truth,
    }
    return RagSynthData(traces, observations, eval_scores, truth)


def write_rag(data: RagSynthData, out_dir: Path) -> list[Path]:
    """Write ``traces.json`` (Langfuse-style export), ``eval_scores.csv`` and ``truth.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.json"
    traces_path.write_text(
        json.dumps({"traces": data.traces, "observations": data.observations}, indent=1),
        encoding="utf-8",
    )
    eval_path = out_dir / "eval_scores.csv"
    data.eval_scores.to_csv(eval_path, index=False)
    truth_path = out_dir / "truth.json"
    truth_path.write_text(json.dumps(data.truth, indent=2), encoding="utf-8")
    return [traces_path, eval_path, truth_path]
