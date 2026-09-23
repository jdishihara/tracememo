"""End to end: build the rag_memo example and check report numbers against truth."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tracememo.cli import app

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "rag_memo"
runner = CliRunner()


def test_rag_memo_build(tmp_path: Path) -> None:
    data_dir = tmp_path / "rag"
    res = runner.invoke(
        app, ["synth", "rag", "--out", str(data_dir), "--seed", "0", "--n-traces", "20"]
    )
    assert res.exit_code == 0, res.output
    proj = tmp_path / "rag_memo"
    proj.mkdir()
    for name in ("report.md.j2", "report.tex"):
        shutil.copy(EXAMPLE_DIR / name, proj / name)
    yaml_text = (EXAMPLE_DIR / "project.yaml").read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("../../data/synthetic/rag", str(data_dir))
    yaml_text = yaml_text.replace("compile_pdf: true", "compile_pdf: false")
    (proj / "project.yaml").write_text(yaml_text, encoding="utf-8")

    res = runner.invoke(app, ["build", "--config", str(proj / "project.yaml")])
    assert res.exit_code == 0, res.output
    assert "check PASSED" in res.output
    truth = json.loads((data_dir / "truth.json").read_text())
    report = (proj / "build" / "report.md").read_text(encoding="utf-8")
    assert f"{truth['latency']['reduction_pct']:.1f}% reduction" in report
    assert f"{truth['latency']['v1']['e2e_median_ms']:.0f} ms" in report
    assert f"{truth['eval']['baseline']['correctness']['sample_mean']:.3f}" in report
    assert "| retrieve |" in report and "| retrieval_mrr |" in report
    assert (proj / "build" / "figures" / "llm.latency.stacked.png").exists()
    assert (proj / "build" / "values.tex").exists()
