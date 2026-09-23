"""Value/Figure/Table models and id validation."""

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from tracememo.store.models import Figure, Provenance, Value


def test_value_formatting() -> None:
    assert Value(id="a.b", value=3.14159, fmt=".2f").formatted() == "3.14"
    assert Value(id="a.b", value=42, fmt="d").formatted() == "42"
    assert Value(id="a.b", value="beacon").formatted() == "beacon"


def test_value_unwraps_numpy_scalars() -> None:
    v = Value(id="a.b", value=np.float64(1.5))
    assert isinstance(v.value, float)
    v = Value(id="a.c", value=np.int64(7))
    assert isinstance(v.value, int)


@pytest.mark.parametrize("bad", ["nodots", "Upper.case", "a..b", "a.b-c", ".a.b", "a.b."])
def test_invalid_ids_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        Value(id=bad, value=1)
    with pytest.raises(ValidationError):
        Figure(id=bad, caption="x")


def test_figure_excludes_in_memory_object() -> None:
    f = Figure(id="a.b", caption="c", figure=object())
    assert "figure" not in f.model_dump()


def test_provenance_roundtrip() -> None:
    p = Provenance(
        analysis_id="drone.loc_err",
        analysis_source_sha256="abc",
        params={"k": 1},
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert Provenance.model_validate_json(p.model_dump_json()) == p
