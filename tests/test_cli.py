"""End-to-end: synth -> build -> report numbers match truth; explain works."""

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from tracememo.cli import app
from tracememo.report.latex import latexmk_available

runner = CliRunner()


def test_synth_and_build(tmp_path: Path, project_dir: Path, synth_dir: Path) -> None:
    cfg = project_dir / "project.yaml"
    res = runner.invoke(app, ["build", "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    build = project_dir / "build"
    assert (build / "manifest.json").exists()
    assert (build / "figures" / "drone.loc_err.error_vs_time.png").exists()
    assert (build / "figures" / "drone.loc_err.error_cdf.pdf").exists()
    assert (build / "figures" / "drone.trajectory.top_down.pdf").exists()
    assert (build / "tables_out" / "drone.failures.events.parquet").exists()
    assert (build / "values.tex").exists() and (build / "report.tex").exists()
    if latexmk_available():
        assert (build / "report.pdf").stat().st_size > 0
    report = (build / "report.md").read_text(encoding="utf-8")
    assert "| dropout |" in report

    truth = json.loads((synth_dir / "truth.json").read_text())["loc_err"]["beacon"]
    assert f"{truth['mean_cm']:.1f} cm" in report
    assert f"{truth['p95_cm']:.1f} cm" in report
    assert f"{truth['pct_under_10cm']:.1f}% of samples" in report
    assert f"Over {truth['n_samples']} aligned samples" in report
    # Every value in the manifest carries provenance pointing at the raw files.
    manifest = json.loads((build / "manifest.json").read_text())
    assert len(manifest["values"]) == 24
    assert set(manifest["analyses"]) == {"drone.loc_err", "drone.trajectory", "drone.failures"}
    for v in manifest["values"].values():
        prov = v["provenance"]
        assert prov["analysis_id"] == v["id"].rsplit(".", 1)[0]
        assert prov["raw_inputs"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) for r in prov["raw_inputs"])
    loc = manifest["values"]["drone.loc_err.mean_cm"]["provenance"]
    assert {Path(r["path"]).name for r in loc["raw_inputs"]} == {
        "pose.parquet",
        "nav_target.parquet",
        "beacon.parquet",
    }


def test_rebuild_is_cached(project_dir: Path) -> None:
    cfg = project_dir / "project.yaml"
    assert runner.invoke(app, ["build", "--config", str(cfg)]).exit_code == 0
    res = runner.invoke(app, ["analyze", "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert res.output.count("cached") == 3 and " ran " not in res.output


def test_explain(project_dir: Path) -> None:
    cfg = project_dir / "project.yaml"
    assert runner.invoke(app, ["build", "--config", str(cfg)]).exit_code == 0
    res = runner.invoke(app, ["explain", "drone.loc_err.mean_cm", "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert "drone.loc_err.mean_cm = " in res.output
    assert '"analysis_id": "drone.loc_err"' in res.output
    res = runner.invoke(app, ["explain", "no.such", "--config", str(cfg)])
    assert res.exit_code == 1
    assert "unknown id" in res.output


def test_synth_cli(tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["synth", "drone", "--out", str(tmp_path / "d"), "--seed", "5", "--duration-s", "10"]
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "d" / "truth.json").exists()
