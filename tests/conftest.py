"""Shared fixtures: a small synthetic flight and a fully built example project."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tracememo.synth.drone import DroneSynthConfig, DroneSynthData, generate_drone, write_drone

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "drone_memo"


@pytest.fixture(scope="session")
def small_cfg() -> DroneSynthConfig:
    """A 60 s flight: long enough to include all default failure events."""
    return DroneSynthConfig(seed=1, duration_s=60.0)


@pytest.fixture(scope="session")
def synth_data(small_cfg: DroneSynthConfig) -> DroneSynthData:
    return generate_drone(small_cfg)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory, synth_data: DroneSynthData) -> Path:
    out = tmp_path_factory.mktemp("synth") / "drone"
    write_drone(synth_data, out)
    return out


@pytest.fixture(scope="session")
def project_dir(tmp_path_factory: pytest.TempPathFactory, synth_dir: Path) -> Path:
    """A copy of examples/drone_memo pointing at the session's synthetic data."""
    proj = tmp_path_factory.mktemp("proj") / "drone_memo"
    proj.mkdir()
    shutil.copy(EXAMPLE_DIR / "report.md.j2", proj / "report.md.j2")
    shutil.copy(EXAMPLE_DIR / "report.tex", proj / "report.tex")
    yaml_text = (EXAMPLE_DIR / "project.yaml").read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("../../data/synthetic/drone", str(synth_dir))
    (proj / "project.yaml").write_text(yaml_text, encoding="utf-8")
    return proj
