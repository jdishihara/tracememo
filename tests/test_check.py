"""Deterministic grounding checks: positive and negative cases for each check."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tracememo.check.extract import find_refs, prose_mask
from tracememo.check.numbers import (
    check_comparisons,
    check_raw_numbers,
    check_stale,
    check_text,
    check_unknown_references,
    run_checks,
)
from tracememo.hashing import sha256_file
from tracememo.store.models import Figure, InputFile, Manifest, Provenance, Table, Value

SRC = "a" * 64


def _manifest(tmp_path: Path) -> Manifest:
    raw = tmp_path / "pose.parquet"
    raw.write_bytes(b"pose-bytes")
    prov = Provenance(
        analysis_id="demo.stats",
        analysis_source_sha256=SRC,
        raw_inputs=[InputFile(path=str(raw), sha256=sha256_file(raw))],
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    values = {
        "demo.low_cm": Value(id="demo.low_cm", value=4.6, unit="cm", fmt=".1f", provenance=prov),
        "demo.high_cm": Value(id="demo.high_cm", value=7.3, unit="cm", fmt=".1f", provenance=prov),
        "demo.pct": Value(id="demo.pct", value=87.5, unit="%", fmt=".1f", provenance=prov),
        "demo.ref": Value(id="demo.ref", value="beacon", provenance=prov),
    }
    return Manifest(
        project="t",
        created=prov.timestamp,
        values=values,
        figures={"demo.fig": Figure(id="demo.fig", caption="c", provenance=prov)},
        tables={"demo.tab": Table(id="demo.tab", caption="c", provenance=prov)},
    )


HASHES = {"demo.stats": SRC}


# --- extraction --------------------------------------------------------------------------


def test_find_refs_markdown() -> None:
    text = (
        '{{ val("a.b") }} {{ val("a.c", with_unit=True) }} {{ val_raw(\'a.d\') }} '
        '{{ fig("f.g") }} {{ tab("t.u") }} {{val:p.q}} {{fig:p.r}} {{tab:p.s}} {{ unit("a.b") }}'
    )
    refs = find_refs(text, "markdown")
    assert [(r.kind, r.id) for r in refs] == [
        ("val", "a.b"),
        ("val", "a.c"),
        ("val", "a.d"),
        ("fig", "f.g"),
        ("tab", "t.u"),
        ("val", "p.q"),
        ("fig", "p.r"),
        ("tab", "p.s"),
        ("val", "a.b"),
    ]


def test_find_refs_latex() -> None:
    refs = find_refs(r"\val{a.b} \valu{a.c} \fig{f.g} \tab{t.u} \ref{fig:a.b}", "latex")
    assert [(r.kind, r.id) for r in refs] == [
        ("val", "a.b"),
        ("val", "a.c"),
        ("fig", "f.g"),
        ("tab", "t.u"),
    ]


def test_prose_mask_preserves_lines_and_tokens() -> None:
    text = 'Line one {{ val("a.b") }}.\n{% if x %}\nLine three `code 12`.\n'
    masked, refs = prose_mask(text, "markdown")
    assert len(masked) == len(text)
    assert masked.count("\n") == text.count("\n")
    assert "@R0@" in masked
    assert "12" not in masked and "if x" not in masked


def test_prose_mask_latex_skips_preamble_and_commands() -> None:
    text = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\nValue \\val{a.b} in 2026 % comment 99\n"
        "\\includegraphics[width=0.5\\linewidth]{x.pdf} see Figure~\\ref{fig:1}\n\\end{document}\n"
    )
    masked, refs = prose_mask(text, "latex")
    assert len(masked) == len(text)
    assert "11pt" not in masked and "1in" not in masked and "99" not in masked
    assert "0.5" not in masked and "fig:1" not in masked
    assert "2026" in masked and "@R0@" in masked


# --- check 1: raw numbers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "The mean error was 7.3 cm.",
        "About 10 percent of samples.",
        "Values of 1,234 and 5.",
        "In 2 sensors we saw drift.",
    ],
)
def test_raw_numbers_flagged(line: str) -> None:
    assert check_raw_numbers(line, "f") != []


@pytest.mark.parametrize(
    "line",
    [
        "See Section 2 and Figure 1 and Table 3.1 and Eq. 4.",
        "As shown in [3] and [1, 2].",
        "Data collected in 2024 and 2025.",
        "The 95th percentile and the 3D error.",
        "The p95 latency and the x_1 coordinate.",
        "Mean was @R0@ and median @R1@.",
        "Flight of @R0@ over 10th-order effects.",
    ],
)
def test_raw_numbers_allowed(line: str) -> None:
    assert check_raw_numbers(line, "f") == []


def test_raw_numbers_extra_allow_patterns_and_lines() -> None:
    text = "ok\nISO 9001 rules.\nbad 42 here.\n"
    found = check_raw_numbers(text, "f", allow_patterns=[r"ISO \d+"])
    assert [(f.line, f.message) for f in found] == [
        (3, "raw number '42' in prose; use a value reference")
    ]


def test_raw_numbers_ignore_code_and_headings(tmp_path: Path) -> None:
    text = "## 2 Results\n\n`pct_under_10cm` and ```\n1 2 3\n```\n\n1. item\n2. item\n"
    findings = check_text(text, "f.md.j2", _manifest(tmp_path), source_hashes=HASHES)
    assert [f for f in findings if f.check == "raw_number"] == []


# --- check 2: unknown references ----------------------------------------------------------


def test_unknown_reference(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    text = '{{ val("demo.low_cm") }} {{ val("demo.nope") }} {{ fig("demo.tab") }} {{tab:demo.tab}}'
    refs = find_refs(text, "markdown")
    found = check_unknown_references(refs, m, "f")
    assert [f.ids for f in found] == [["demo.nope"], ["demo.tab"]]  # fig kind mismatch too
    assert all(f.severity == "error" for f in found)


# --- check 3: staleness -------------------------------------------------------------------


def test_stale_clean(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    refs = find_refs('{{ val("demo.low_cm") }}', "markdown")
    assert check_stale(refs, m, "f", HASHES) == []


def test_stale_input_changed(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    (tmp_path / "pose.parquet").write_bytes(b"different")
    refs = find_refs(
        '{{ val("demo.low_cm") }} {{ val("demo.low_cm") }} {{tab:demo.tab}}', "markdown"
    )
    found = check_stale(refs, m, "f", HASHES)
    assert len(found) == 1 and "changed" in found[0].message
    assert found[0].ids == ["demo.low_cm", "demo.tab"] and found[0].line == 1


def test_stale_input_missing(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    (tmp_path / "pose.parquet").unlink()
    found = check_stale(find_refs(r"\val{demo.pct}", "latex"), m, "f", HASHES)
    assert len(found) == 1 and "missing" in found[0].message


def test_stale_source_changed(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    found = check_stale(find_refs("{{fig:demo.fig}}", "markdown"), m, "f", {"demo.stats": "b" * 64})
    assert len(found) == 1 and "source changed" in found[0].message
    found = check_stale(find_refs("{{fig:demo.fig}}", "markdown"), m, "f", {})
    assert len(found) == 1 and "no longer registered" in found[0].message


# --- check 4: comparisons -----------------------------------------------------------------


def _cmp(text: str, tmp_path: Path):
    m = _manifest(tmp_path)
    masked, refs = prose_mask(text, "markdown")
    return check_comparisons(masked, refs, m, "f")


@pytest.mark.parametrize(
    "text",
    [
        'The median {{ val("demo.low_cm") }} was lower than the mean {{ val("demo.high_cm") }}.',
        'Mean {{ val("demo.high_cm") }} exceeds the median {{ val("demo.low_cm") }}.',
        'Error {{ val("demo.high_cm") }} is above {{ val("demo.low_cm") }}.',
        'Error decreased from {{ val("demo.high_cm") }} to {{ val("demo.low_cm") }}.',
        'Error rose from {{ val("demo.low_cm") }} to {{ val("demo.high_cm") }}.',
        'Error fell from {{ val("demo.high_cm", with_unit=True) }}\n'
        'to {{ val("demo.low_cm", with_unit=True) }} overall.',
        'Values {{ val("demo.low_cm") }} and {{ val("demo.high_cm") }} were measured.',
        'Two sentences. {{ val("demo.low_cm") }} was measured. '
        'It is lower than {{ val("demo.high_cm") }}.',
        'Mean {{ val("demo.low_cm") }} was not lower than {{ val("demo.high_cm") }}.',
        'Ref {{ val("demo.ref") }} is lower than {{ val("demo.high_cm") }}.',
    ],
)
def test_comparison_ok_or_skipped(text: str, tmp_path: Path) -> None:
    assert [f for f in _cmp(text, tmp_path) if f.severity == "error"] == []


@pytest.mark.parametrize(
    "text",
    [
        'The median {{ val("demo.low_cm") }} was higher than the mean {{ val("demo.high_cm") }}.',
        'Median {{ val("demo.high_cm") }} is under the mean {{ val("demo.low_cm") }}.',
        'Error increased from {{ val("demo.high_cm") }} to {{ val("demo.low_cm") }}.',
        'Error was reduced from {{ val("demo.low_cm") }} to {{ val("demo.high_cm") }}.',
        'Error {{ val("demo.low_cm") }} exceeds {{ val("demo.low_cm") }}.',
        r"LaTeX: \val{demo.low_cm} is greater than \valu{demo.high_cm}.",
    ],
)
def test_comparison_wrong_direction(text: str, tmp_path: Path) -> None:
    syntax = "latex" if text.startswith("LaTeX") else "markdown"
    m = _manifest(tmp_path)
    masked, refs = prose_mask(text, syntax)
    found = check_comparisons(masked, refs, m, "f")
    assert len(found) == 1 and found[0].severity == "error", text
    assert "claims" in found[0].message


def test_comparison_unit_mismatch_is_warning(tmp_path: Path) -> None:
    found = _cmp('{{ val("demo.pct") }} exceeds {{ val("demo.low_cm") }}.', tmp_path)
    assert len(found) == 1 and found[0].severity == "warning"


# --- driver -------------------------------------------------------------------------------


def test_run_checks_report(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    good = tmp_path / "good.md.j2"
    good.write_text('Mean {{ val("demo.high_cm", with_unit=True) }} in 2026.\n', encoding="utf-8")
    bad = tmp_path / "bad.tex"
    bad.write_text(
        "\\begin{document}\nMean 7.3 cm and \\val{demo.nope}.\n\\end{document}\n", encoding="utf-8"
    )
    report = run_checks([good, bad], m, source_hashes=HASHES)
    assert not report.passed
    assert report.counts() == {"raw_number": 1, "unknown_reference": 1}
    assert all(f.file == str(bad) and f.line == 2 for f in report.findings)
    assert "FAILED" in report.summary()
    clean = run_checks([good], m, source_hashes=HASHES)
    assert clean.passed and "PASSED" in clean.summary()
