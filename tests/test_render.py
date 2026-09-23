"""Markdown renderer: golden file, unknown ids, provenance links."""

from datetime import UTC, datetime
from pathlib import Path

import jinja2
import pandas as pd
import pytest

from tracememo.report.render import MarkdownRenderer
from tracememo.store import Figure, InputFile, Provenance, Table, UnknownIdError, Value, ValueStore

GOLDEN = Path(__file__).parent / "golden"


def _store(tmp_path: Path) -> ValueStore:
    prov = Provenance(
        analysis_id="demo.stats",
        analysis_source_sha256="0123456789abcdef",
        git_commit="deadbeefcafe",
        git_dirty=False,
        params={"k": 2},
        raw_inputs=[InputFile(path="/data/pose.parquet", sha256="feedface00")],
        timestamp=datetime(2026, 9, 23, 12, 0, tzinfo=UTC),
    )
    s = ValueStore("golden", created=prov.timestamp)
    s.add_value(Value(id="demo.mean_cm", value=7.2549, unit="cm", fmt=".1f", provenance=prov))
    s.add_value(Value(id="demo.pct", value=87.52, unit="%", fmt=".1f", provenance=prov))
    s.add_value(Value(id="demo.n", value=585, fmt="d", provenance=prov))
    s.add_value(Value(id="demo.ref", value="beacon", provenance=prov))
    png = tmp_path / "figures" / "demo.fig.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"")
    s.add_figure(Figure(id="demo.fig", caption="A figure.", png_path=str(png), provenance=prov))
    pq = tmp_path / "tables_out" / "demo.table.parquet"
    pq.parent.mkdir(parents=True)
    pd.DataFrame({"type": ["dropout", "jump"], "magnitude_m": [0.0, 0.5]}).to_parquet(
        pq, index=False
    )
    s.add_table(Table(id="demo.table", caption="A table.", parquet_path=str(pq), provenance=prov))
    return s


TEMPLATE = """# Golden

Mean {{ val("demo.mean_cm") }} / {{ val("demo.mean_cm", with_unit=True) }}
Pct {{ val("demo.pct", with_unit=True) }} N={{ val("demo.n") }} ref={{ val("demo.ref") }}
raw={{ val_raw("demo.mean_cm") }} unit={{ unit("demo.mean_cm") }}

{{ fig("demo.fig") }}

{{ tab("demo.table") }}

{{ provenance_appendix() }}
"""


@pytest.mark.parametrize("links", [False, True])
def test_golden(tmp_path: Path, links: bool) -> None:
    store = _store(tmp_path)
    (tmp_path / "t.md.j2").write_text(TEMPLATE, encoding="utf-8")
    out = MarkdownRenderer(store, tmp_path, provenance_links=links).render(
        tmp_path / "t.md.j2", tmp_path / "out.md"
    )
    got = out.read_text(encoding="utf-8")
    golden = GOLDEN / ("render_links.md" if links else "render_plain.md")
    if not golden.exists():  # first run writes the golden file; review it and commit it
        golden.write_text(got, encoding="utf-8")
    assert got == golden.read_text(encoding="utf-8")


def test_unknown_value_id_fails_loudly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (tmp_path / "t.md.j2").write_text('{{ val("demo.missing") }}', encoding="utf-8")
    with pytest.raises(UnknownIdError):
        MarkdownRenderer(store, tmp_path).render(tmp_path / "t.md.j2", tmp_path / "out.md")


def test_undefined_template_variable_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (tmp_path / "t.md.j2").write_text("{{ nothing_here }}", encoding="utf-8")
    with pytest.raises(jinja2.UndefinedError):
        MarkdownRenderer(store, tmp_path).render(tmp_path / "t.md.j2", tmp_path / "out.md")
