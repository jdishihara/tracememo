"""LaTeX output: values.tex golden file, macros, and compilation when latexmk exists."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from tracememo.report.latex import (
    compile_pdf,
    latexmk_available,
    render_latex,
    value_macros,
    values_tex,
    write_latex_support,
)
from tracememo.report.tables import df_to_latex, latex_escape
from tracememo.store import Figure, Provenance, Table, Value, ValueStore

GOLDEN = Path(__file__).parent / "golden" / "values.tex"


def _store(build: Path) -> ValueStore:
    prov = Provenance(
        analysis_id="demo.stats",
        analysis_source_sha256="0" * 64,
        timestamp=datetime(2026, 9, 23, tzinfo=UTC),
    )
    s = ValueStore("golden")
    s.add_value(Value(id="demo.mean_cm", value=7.2549, unit="cm", fmt=".1f", provenance=prov))
    s.add_value(Value(id="demo.pct", value=87.52, unit="%", fmt=".1f", provenance=prov))
    s.add_value(Value(id="demo.n", value=585, fmt="d", provenance=prov))
    s.add_value(Value(id="demo.ref", value="nav_target", provenance=prov))
    (build / "figures").mkdir(parents=True, exist_ok=True)
    pdf = build / "figures" / "demo.fig.pdf"
    pdf.write_bytes(b"")
    s.add_figure(Figure(id="demo.fig", caption="A figure.", pdf_path=str(pdf), provenance=prov))
    (build / "tables_out").mkdir(parents=True, exist_ok=True)
    tex = build / "tables_out" / "demo.table.tex"
    tex.write_text(df_to_latex(pd.DataFrame({"type": ["a_b"], "v": [0.5]})), encoding="utf-8")
    s.add_table(Table(id="demo.table", caption="A table.", latex_path=str(tex), provenance=prov))
    return s


def test_values_tex_golden(tmp_path: Path) -> None:
    got = values_tex(_store(tmp_path), tmp_path)
    if not GOLDEN.exists():  # first run writes the golden file; review it and commit it
        GOLDEN.write_text(got, encoding="utf-8")
    assert got == GOLDEN.read_text(encoding="utf-8")


def test_value_macros_escape_units_and_strings() -> None:
    v = Value(id="a.b", value=87.52, unit="%", fmt=".1f")
    assert value_macros(v) == [r"\tmdef{val@a.b}{87.5}", r"\tmdef{valu@a.b}{87.5\%}"]
    v = Value(id="a.c", value="nav_target")
    assert value_macros(v)[0] == r"\tmdef{val@a.c}{nav\_target}"
    v = Value(id="a.d", value=2.0, unit="m/s", fmt=".1f")
    assert value_macros(v)[1] == r"\tmdef{valu@a.d}{2.0\,m/s}"


def test_latex_escape() -> None:
    assert latex_escape("50% & a_b #1") == r"50\% \& a\_b \#1"


def test_df_to_latex() -> None:
    out = df_to_latex(pd.DataFrame({"type": ["x"], "v": [1.5]}))
    assert out.startswith(r"\begin{tabular}{lr}")
    assert r"x & 1.5 \\" in out
    assert r"\bottomrule" in out


def test_write_support_files(tmp_path: Path) -> None:
    paths = write_latex_support(_store(tmp_path), tmp_path)
    assert [p.name for p in paths] == ["values.tex", "tracememo.sty"]
    sty = (tmp_path / "tracememo.sty").read_text(encoding="utf-8")
    assert r"\newcommand{\val}" in sty


DOC = r"""\documentclass{article}
\usepackage{tracememo}
\begin{document}
Mean \val{demo.mean_cm} / \valu{demo.mean_cm} / \valu{demo.pct} / \val{demo.n} / \val{demo.ref}.
\tab{demo.table}
\end{document}
"""


@pytest.mark.skipif(not latexmk_available(), reason="latexmk not installed")
def test_compiles_with_latexmk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.figures.clear()  # the golden store's PDF figure is an empty placeholder
    (tmp_path / "doc.tex").write_text(DOC, encoding="utf-8")
    out = render_latex(store, tmp_path / "doc.tex", tmp_path, "report")
    pdf = compile_pdf(out)
    assert pdf.exists() and pdf.stat().st_size > 0


@pytest.mark.skipif(not latexmk_available(), reason="latexmk not installed")
def test_unknown_id_fails_compilation(tmp_path: Path) -> None:
    import subprocess

    store = _store(tmp_path)
    store.figures.clear()
    (tmp_path / "doc.tex").write_text(
        DOC.replace(r"\val{demo.n}", r"\val{demo.missing}"), encoding="utf-8"
    )
    out = render_latex(store, tmp_path / "doc.tex", tmp_path, "report")
    with pytest.raises(subprocess.CalledProcessError):
        compile_pdf(out)
    assert shutil.which("latexmk")
