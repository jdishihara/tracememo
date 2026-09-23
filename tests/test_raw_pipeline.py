"""End to end through the real-data adapters: results match the normalized-data pipeline."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tracememo.cli import app
from tracememo.synth.drone import DroneSynthData

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "drone_memo"
runner = CliRunner()


def test_raw_adapters_reproduce_values(
    tmp_path: Path, synth_dir: Path, project_dir: Path, synth_data: DroneSynthData
) -> None:
    proj = tmp_path / "raw"
    proj.mkdir()
    for name in ("report.md.j2", "report.tex"):
        shutil.copy(EXAMPLE_DIR / name, proj / name)
    yaml_text = (EXAMPLE_DIR / "project_raw.yaml").read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("../../data/synthetic/drone", str(synth_dir))
    yaml_text = yaml_text.replace("compile_pdf: true", "compile_pdf: false")
    (proj / "project.yaml").write_text(yaml_text, encoding="utf-8")

    res = runner.invoke(app, ["build", "--config", str(proj / "project.yaml")])
    assert res.exit_code == 0, res.output
    raw = json.loads((proj / "build_raw" / "manifest.json").read_text())

    assert (
        runner.invoke(app, ["build", "--config", str(project_dir / "project.yaml")]).exit_code == 0
    )
    norm = json.loads((project_dir / "build" / "manifest.json").read_text())

    assert set(raw["values"]) == set(norm["values"])
    truth = synth_data.truth["loc_err"]["beacon"]
    for vid, v in raw["values"].items():
        ref = norm["values"][vid]["value"]
        if isinstance(v["value"], str):
            assert v["value"] == ref
        elif vid.startswith("drone.failures.n_"):
            assert v["value"] == ref  # same events detected
        else:
            # CSV rounding (1 mm) and float32 logging perturb statistics slightly.
            assert v["value"] == pytest.approx(ref, rel=0.02, abs=0.2), vid
    assert raw["values"]["drone.loc_err.mean_cm"]["value"] == pytest.approx(
        truth["mean_cm"], abs=0.1
    )
    adapters = {t["adapter"] for t in raw["input_tables"].values()}
    assert adapters == {"ardupilot", "marvelmind"}
    assert Path(
        raw["values"]["drone.loc_err.mean_cm"]["provenance"]["raw_inputs"][0]["path"]
    ).suffix in (".bin", ".csv")
