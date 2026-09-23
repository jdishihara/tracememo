"""Analysis registry: registration, ordering, source hashes."""

import pytest

from tracememo.analyses import registry
from tracememo.analyses.registry import AnalysisSpec, analysis, order_analyses, source_hash
from tracememo.store.models import AnalysisResult


@pytest.fixture(autouse=True)
def _clean_registry():
    ids = ["t.a", "t.b", "t.c"]
    yield
    for i in ids:
        registry.unregister(i)


def test_decorator_registers_and_hashes() -> None:
    @analysis(id="t.a", inputs=["pose"], params={"k": 1})
    def a(pose, params):
        return AnalysisResult()

    spec = registry.get_analysis("t.a")
    assert spec.inputs == ("pose",)
    assert spec.params == {"k": 1}
    assert spec.source_sha256 == source_hash(a)
    assert len(spec.source_sha256) == 64


def test_order_by_produced_tables() -> None:
    registry.register_spec(AnalysisSpec(id="t.a", inputs=["mid"], outputs=()))
    registry.register_spec(AnalysisSpec(id="t.b", inputs=["pose"], outputs=("mid",)))
    registry.register_spec(AnalysisSpec(id="t.c", inputs=["pose"], outputs=()))
    ordered = [s.id for s in order_analyses(["t.a", "t.c", "t.b"])]
    assert ordered.index("t.b") < ordered.index("t.a")
    assert ordered.index("t.c") < ordered.index("t.b")  # ties keep config order


def test_cycle_detected() -> None:
    registry.register_spec(AnalysisSpec(id="t.a", inputs=["x"], outputs=("y",)))
    registry.register_spec(AnalysisSpec(id="t.b", inputs=["y"], outputs=("x",)))
    with pytest.raises(ValueError, match="cyclic"):
        order_analyses(["t.a", "t.b"])


def test_unknown_analysis() -> None:
    with pytest.raises(KeyError, match="unknown analysis"):
        registry.get_analysis("nope.nope")
