"""Caching: unchanged analyses are reused; changed params/data/source cause reruns."""

from pathlib import Path

import pandas as pd
import pytest

from tracememo.adapters.normalized import ingest_normalized
from tracememo.analyses import registry
from tracememo.analyses.registry import AnalysisSpec, load_builtin
from tracememo.analyses.runner import run_analyses
from tracememo.store.models import AnalysisResult, Value
from tracememo.store.store import ValueStore
from tracememo.store.tables import TableStore

SELECTIONS = [
    ("drone.loc_err", {"reference": "beacon"}),
    ("drone.trajectory", {}),
    ("drone.failures", {}),
]


def _ingest(synth_dir: Path, build: Path) -> TableStore:
    tables = TableStore(build / "tables")
    for t in ingest_normalized(synth_dir, None, {}):
        tables.put(t.id, t.df, "normalized", t.raw_inputs)
    tables.save()
    return tables


def _run(tables: TableStore, build: Path, previous: ValueStore | None, selections=SELECTIONS):
    store = ValueStore("cache_test")
    prev = previous.to_manifest() if previous else None
    records = run_analyses(selections, tables, store, build, previous=prev)
    return store, {r.id: r.cached for r in records}


@pytest.fixture
def built(tmp_path: Path, synth_dir: Path) -> tuple[TableStore, Path, ValueStore]:
    load_builtin()
    build = tmp_path / "build"
    tables = _ingest(synth_dir, build)
    store, cached = _run(tables, build, None)
    assert cached == {"drone.loc_err": False, "drone.trajectory": False, "drone.failures": False}
    return tables, build, store


def test_everything_cached_on_rerun(built) -> None:
    tables, build, first = built
    second, cached = _run(tables, build, first)
    assert all(cached.values())
    assert second.to_manifest().values == first.to_manifest().values
    # Provenance timestamps are preserved from the original run.
    v = "drone.loc_err.mean_cm"
    assert second.get_value(v).provenance.timestamp == first.get_value(v).provenance.timestamp


def test_param_change_reruns_only_that_analysis(built) -> None:
    tables, build, first = built
    sel = [
        ("drone.loc_err", {"reference": "nav_target"}),
        ("drone.trajectory", {}),
        ("drone.failures", {}),
    ]
    second, cached = _run(tables, build, first, sel)
    assert cached == {"drone.loc_err": False, "drone.trajectory": True, "drone.failures": True}
    assert second.get_value("drone.loc_err.reference").value == "nav_target"


def test_data_change_reruns_dependents_only(built, tmp_path: Path, synth_dir: Path) -> None:
    tables, build, first = built
    # Modify the beacon table only: loc_err (reference=beacon) and failures depend on it,
    # trajectory does not.
    beacon = pd.read_parquet(synth_dir / "beacon.parquet")
    beacon["x_m"] += 0.01
    tables.put("beacon", beacon, "normalized", tables.entries["beacon"].raw_inputs)
    tables.save()
    second, cached = _run(tables, build, first)
    assert cached == {"drone.loc_err": False, "drone.trajectory": True, "drone.failures": False}


def test_source_change_reruns(built) -> None:
    tables, build, first = built
    spec = registry.get_analysis("drone.trajectory")
    changed = AnalysisSpec(
        id=spec.id,
        inputs=spec.inputs,
        outputs=spec.outputs,
        params=spec.params,
        func=spec.func,
        source_sha256="different",
    )
    registry.register_spec(changed)
    try:
        _, cached = _run(tables, build, first)
    finally:
        registry.register_spec(spec)
    assert cached["drone.trajectory"] is False
    assert cached["drone.loc_err"] is True


def test_missing_output_file_forces_rerun(built) -> None:
    tables, build, first = built
    Path(first.get_figure("drone.trajectory.top_down").png_path).unlink()
    _, cached = _run(tables, build, first)
    assert cached["drone.trajectory"] is False
    assert cached["drone.loc_err"] is True


def test_produced_table_feeds_downstream(tmp_path: Path, synth_dir: Path) -> None:
    """An analysis consuming another's output table runs after it and sees its rows."""
    load_builtin()
    build = tmp_path / "build"
    tables = _ingest(synth_dir, build)

    def count_events(**kw) -> AnalysisResult:
        df = kw["drone.failures.events"]
        return AnalysisResult(values=[Value(id="t.n_rows", value=len(df), fmt="d")])

    registry.register_spec(
        AnalysisSpec(
            id="t.consumer",
            inputs=("drone.failures.events",),
            outputs=(),
            func=count_events,
            source_sha256="x",
        )
    )
    try:
        store, cached = _run(tables, build, None, [("t.consumer", {}), ("drone.failures", {})])
        n = store.get_value("drone.failures.n_events").value
        assert store.get_value("t.n_rows").value == n
        store2, cached2 = _run(tables, build, store, [("t.consumer", {}), ("drone.failures", {})])
        assert cached2 == {"drone.failures": True, "t.consumer": True}
    finally:
        registry.unregister("t.consumer")
