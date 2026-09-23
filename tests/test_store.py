"""ValueStore behaviour and manifest round trip."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tracememo.store import (
    AnalysisResult,
    DuplicateIdError,
    Figure,
    Provenance,
    UnknownIdError,
    Value,
    ValueStore,
)


def _prov() -> Provenance:
    return Provenance(
        analysis_id="x.y", analysis_source_sha256="s", timestamp=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_add_and_get() -> None:
    s = ValueStore("p")
    s.add_result(
        AnalysisResult(values=[Value(id="a.b", value=1)], figures=[Figure(id="a.f", caption="c")])
    )
    assert s.get_value("a.b").value == 1
    assert s.get_figure("a.f").caption == "c"
    assert s.all_ids() == {"a.b", "a.f"}


def test_duplicate_ids_rejected_across_kinds() -> None:
    s = ValueStore("p")
    s.add_value(Value(id="a.b", value=1))
    with pytest.raises(DuplicateIdError):
        s.add_value(Value(id="a.b", value=2))
    with pytest.raises(DuplicateIdError):
        s.add_figure(Figure(id="a.b", caption="c"))


def test_unknown_id() -> None:
    s = ValueStore("p")
    with pytest.raises(UnknownIdError):
        s.get_value("no.such")
    with pytest.raises(UnknownIdError):
        s.explain("no.such")


def test_explain_returns_provenance() -> None:
    s = ValueStore("p")
    s.add_value(Value(id="a.b", value=1, provenance=_prov()))
    assert s.explain("a.b").analysis_id == "x.y"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    s = ValueStore("proj")
    s.add_value(Value(id="a.b", value=1.5, unit="cm", fmt=".1f", provenance=_prov()))
    s.add_figure(Figure(id="a.f", caption="cap", png_path="x.png", provenance=_prov()))
    path = s.save(tmp_path / "build" / "manifest.json")
    loaded = ValueStore.load(path)
    assert loaded.project == "proj"
    assert loaded.get_value("a.b") == s.get_value("a.b")
    assert loaded.get_figure("a.f") == s.get_figure("a.f")
