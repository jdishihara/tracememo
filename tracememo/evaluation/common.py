"""Shared helpers: build the example projects in a scratch directory, write results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tracememo.cli import run_analyze, run_ingest, run_render
from tracememo.config import ProjectConfig, load_config
from tracememo.store.store import ValueStore
from tracememo.synth.drone import DroneSynthConfig, generate_drone, write_drone
from tracememo.synth.rag import RagSynthConfig, generate_rag, write_rag

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


def synth_drone(out: Path, seed: int = 0, duration_s: float = 120.0) -> Path:
    """Generate synthetic drone data into ``out``; return the directory."""
    write_drone(generate_drone(DroneSynthConfig(seed=seed, duration_s=duration_s)), out)
    return out


def synth_rag(out: Path, seed: int = 0, n_traces: int = 60) -> Path:
    """Generate synthetic RAG data into ``out``; return the directory."""
    write_rag(generate_rag(RagSynthConfig(seed=seed, n_traces=n_traces)), out)
    return out


def prepare_example(
    example: str, work: Path, data_dir: Path, config_name: str = "project.yaml"
) -> Path:
    """Copy an example project into ``work`` pointing at ``data_dir``; return its config path."""
    src = EXAMPLES / example
    dst = work / f"{example}_{config_name.replace('.yaml', '')}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("report.md.j2", "report.tex"):
        shutil.copy(src / name, dst / name)
    text = (src / config_name).read_text(encoding="utf-8")
    text = text.replace(
        f"../../data/synthetic/{'rag' if example == 'rag_memo' else 'drone'}", str(data_dir)
    )
    text = text.replace("compile_pdf: true", "compile_pdf: false")
    cfg_path = dst / "project.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def build_project(cfg_path: Path, render: bool = False) -> ProjectConfig:
    """Ingest + analyze (+ render) a project without the checker or PDF compilation."""
    cfg = load_config(cfg_path)
    cfg.report.compile_pdf = False
    run_ingest(cfg)
    run_analyze(cfg)
    if render:
        run_render(cfg)
    return cfg


def load_store(cfg: ProjectConfig) -> ValueStore:
    """Load the built manifest of a project."""
    return ValueStore.load(cfg.manifest_path)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a Markdown table with the given column order."""
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def write_results(name: str, data: dict[str, Any], out_dir: Path) -> Path:
    """Write ``<out_dir>/<name>.json``; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
