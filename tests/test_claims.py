"""LLM claim checker: candidate extraction, prompt rendering, verdict handling."""

import json
from datetime import UTC, datetime
from pathlib import Path

from tracememo.check.claims import check_claims, check_claims_text, extract_claims, render_claim
from tracememo.check.extract import prose_mask
from tracememo.llm import FakeLLM
from tracememo.store.models import Manifest, Provenance, Value


def _manifest() -> Manifest:
    prov = Provenance(
        analysis_id="d.s",
        analysis_source_sha256="0" * 64,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    vals = {
        "d.low_cm": Value(
            id="d.low_cm", value=4.6, unit="cm", fmt=".1f", description="Median", provenance=prov
        ),
        "d.high_cm": Value(
            id="d.high_cm", value=7.3, unit="cm", fmt=".1f", description="Mean", provenance=prov
        ),
        "d.pct": Value(
            id="d.pct",
            value=87.5,
            unit="%",
            fmt=".1f",
            description="Share under threshold",
            provenance=prov,
        ),
    }
    return Manifest(project="t", created=prov.timestamp, values=vals)


TEXT = (
    "The mean {{val:d.high_cm}} was lower than the median {{val:d.low_cm}}.\n"  # covered by check 4
    "Roughly {{val:d.pct}} of samples were within tolerance, which is most of them.\n"  # claim
    "The median error was {{val:d.low_cm}}.\n"  # no claim word
    "Errors were nearly halved relative to the previous flight ({{val:d.high_cm}}).\n"  # claim
    "See {{fig:d.fig}} for more than one view.\n"  # fig only
)


def test_extract_claims_skips_covered_and_plain_sentences() -> None:
    masked, refs = prose_mask(TEXT, "markdown")
    claims = extract_claims(masked, refs)
    assert [c.line for c in claims] == [2, 4]
    assert [r.id for r in claims[0].refs] == ["d.pct"]


def test_render_claim_expands_values() -> None:
    masked, refs = prose_mask(TEXT, "markdown")
    claim = extract_claims(masked, refs)[0]
    rendered = render_claim(claim, refs, _manifest())
    assert "[d.pct = 87.5%]" in rendered and "Share under threshold" in rendered


def test_verdicts_become_warnings_only() -> None:
    fake = FakeLLM(
        responses=[
            json.dumps(
                {
                    "results": [
                        {"index": 0, "verdict": "supported", "reason": "fine"},
                        {
                            "index": 1,
                            "verdict": "unsupported",
                            "reason": "no previous flight value",
                        },
                        {"index": 7, "verdict": "unsupported", "reason": "bogus index"},
                    ]
                }
            )
        ]
    )
    findings = check_claims_text(TEXT, "f.md", _manifest(), fake)
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "claim" and f.severity == "warning" and f.line == 4
    assert f.message == "unsupported: no previous flight value"
    assert f.ids == ["d.high_cm"] and "d.high_cm" in f.text
    assert "Claim 0:" in fake.calls[0].user and "Claim 1:" in fake.calls[0].user


def test_no_claims_means_no_call() -> None:
    fake = FakeLLM()
    assert check_claims_text("Median was {{val:d.low_cm}}.", "f.md", _manifest(), fake) == []
    assert fake.calls == []


def test_check_claims_over_files(tmp_path: Path) -> None:
    f = tmp_path / "a.tex"
    f.write_text(r"\begin{document}Most flights stayed under \valu{d.pct}.\end{document}")
    fake = FakeLLM(
        responses=[json.dumps({"results": [{"index": 0, "verdict": "cant_tell", "reason": "?"}]})]
    )
    out = check_claims([f], _manifest(), fake)
    assert len(out) == 1 and out[0].file == str(f) and out[0].message.startswith("cant_tell")
