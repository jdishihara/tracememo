"""tracememo command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tracememo.config import ProjectConfig, load_config

app = typer.Typer(no_args_is_help=True, help="Reports where every number has provenance.")
synth_app = typer.Typer(no_args_is_help=True, help="Generate synthetic data with known truth.")
app.add_typer(synth_app, name="synth")

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Path to project.yaml", show_default=True)
]


def _echo(msg: str) -> None:
    typer.echo(msg)


# -- pipeline steps (importable, used by `build`) ---------------------------------


def run_ingest(cfg: ProjectConfig) -> None:
    """Run every adapter in the config and write ``build/tables``."""
    from tracememo.adapters import get_adapter
    from tracememo.store.tables import TableStore

    tables = TableStore(cfg.build_path / "tables")
    for inp in cfg.inputs:
        adapter = get_adapter(inp.adapter)
        for t in adapter(cfg.resolve(inp.path), inp.tables, inp.options):
            entry = tables.put(t.id, t.df, inp.adapter, t.raw_inputs)
            _echo(f"ingest  {t.id:<16} {entry.rows:>7} rows  {entry.sha256[:8]}  via {inp.adapter}")
    tables.save()


def run_analyze(cfg: ProjectConfig) -> None:
    """Run the configured analyses and write ``build/manifest.json``."""
    from tracememo.analyses.registry import load_builtin
    from tracememo.analyses.runner import run_analyses
    from tracememo.store.store import ValueStore
    from tracememo.store.tables import TableStore

    load_builtin()
    tables = TableStore.load(cfg.build_path / "tables")
    if not tables.entries:
        raise typer.BadParameter("no ingested tables found; run `tracememo ingest` first")
    store = ValueStore(cfg.name)
    previous = None
    if cfg.manifest_path.exists():
        previous = ValueStore.load(cfg.manifest_path).to_manifest()
    selections = [(a.id, a.params) for a in cfg.analyses]
    for rec in run_analyses(
        selections, tables, store, cfg.build_path, git_root=cfg.config_dir, previous=previous
    ):
        state = "cached" if rec.cached else "ran"
        _echo(f"analyze {rec.id:<24} {state:<6} source {rec.source_sha256[:8]}")
    path = store.save(cfg.manifest_path)
    _echo(
        f"wrote {path} ({len(store.values)} values, {len(store.figures)} figures, "
        f"{len(store.tables)} tables)"
    )


def run_render(cfg: ProjectConfig) -> list[Path]:
    """Render the configured report formats from ``build/manifest.json``."""
    from tracememo.report.latex import compile_pdf, latexmk_available, render_latex
    from tracememo.report.render import MarkdownRenderer
    from tracememo.store.store import ValueStore

    store = ValueStore.load(cfg.manifest_path)
    outputs: list[Path] = []
    name = cfg.report.output_name
    for fmt in cfg.report.formats:
        if fmt == "markdown":
            renderer = MarkdownRenderer(
                store, cfg.build_path, provenance_links=cfg.report.provenance_links
            )
            out = renderer.render(
                cfg.resolve(cfg.report.markdown_template), cfg.build_path / f"{name}.md"
            )
            outputs.append(out)
            _echo(f"render  markdown -> {out}")
        elif fmt == "latex":
            out = render_latex(store, cfg.resolve(cfg.report.latex_template), cfg.build_path, name)
            outputs.append(out)
            _echo(f"render  latex    -> {out} (+ values.tex, tracememo.sty)")
            if cfg.report.compile_pdf:
                if latexmk_available():
                    pdf = compile_pdf(out)
                    outputs.append(pdf)
                    _echo(f"render  pdf      -> {pdf}")
                else:
                    _echo("render  pdf      skipped: latexmk not found")
    return outputs


# -- commands ----------------------------------------------------------------------


@app.command()
def ingest(config: ConfigOpt = Path("project.yaml")) -> None:
    """Run adapters: raw files -> normalized tables."""
    run_ingest(load_config(config))


@app.command()
def analyze(config: ConfigOpt = Path("project.yaml")) -> None:
    """Run analyses: normalized tables -> values, figures, tables."""
    run_analyze(load_config(config))


@app.command()
def render(config: ConfigOpt = Path("project.yaml")) -> None:
    """Render the report from build/manifest.json."""
    run_render(load_config(config))


@app.command()
def build(config: ConfigOpt = Path("project.yaml")) -> None:
    """ingest + analyze + render."""
    cfg = load_config(config)
    run_ingest(cfg)
    run_analyze(cfg)
    run_render(cfg)


@app.command()
def explain(
    value_id: Annotated[str, typer.Argument(help="Value, figure or table id")],
    config: ConfigOpt = Path("project.yaml"),
) -> None:
    """Print the full provenance of a value, figure or table."""
    from tracememo.store.store import UnknownIdError, ValueStore

    cfg = load_config(config)
    store = ValueStore.load(cfg.manifest_path)
    try:
        prov = store.explain(value_id)
    except UnknownIdError:
        _echo(f"unknown id {value_id!r}; known ids: {', '.join(sorted(store.all_ids()))}")
        raise typer.Exit(code=1) from None
    if value_id in store.values:
        v = store.values[value_id]
        _echo(f"{v.id} = {v.formatted()}{(' ' + v.unit) if v.unit else ''}  ({v.description})")
    _echo(json.dumps(prov.model_dump(mode="json"), indent=2))


@synth_app.command("drone")
def synth_drone(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path(
        "data/synthetic/drone"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 0,
    duration_s: Annotated[float, typer.Option("--duration-s")] = 120.0,
) -> None:
    """Generate a synthetic drone flight (pose, nav_target, beacon tables + truth.json)."""
    from tracememo.synth.drone import DroneSynthConfig, generate_drone, write_drone

    data = generate_drone(DroneSynthConfig(seed=seed, duration_s=duration_s))
    for p in write_drone(data, out):
        _echo(f"wrote {p}")


if __name__ == "__main__":
    app()
