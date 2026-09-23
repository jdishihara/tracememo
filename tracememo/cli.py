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
        elif fmt == "html":
            from tracememo.report.viewer import render_viewer

            out = render_viewer(
                store,
                cfg.resolve(cfg.report.markdown_template),
                cfg.build_path,
                cfg.build_path / f"{name}.html",
            )
            outputs.append(out)
            _echo(f"render  html     -> {out} (click a number for its provenance)")
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


def check_files(cfg: ProjectConfig, extra: list[Path] | None = None) -> list[Path]:
    """Templates to check: configured report templates plus ``check.files`` and ``extra``."""
    files: list[Path] = []
    if "markdown" in cfg.report.formats:
        files.append(cfg.resolve(cfg.report.markdown_template))
    if "latex" in cfg.report.formats:
        files.append(cfg.resolve(cfg.report.latex_template))
    files += [cfg.resolve(f) for f in cfg.check.files]
    files += [Path(f).resolve() for f in (extra or [])]
    return [f for f in files if f.exists()]


def run_check(cfg: ProjectConfig, extra: list[Path] | None = None, llm: bool | None = None) -> bool:
    """Run the checks, print the report, write ``build/check_report.json``.

    ``llm`` adds the LLM-assisted claim check (default: ``check.llm_claims`` from config).
    """
    from tracememo.check.numbers import run_checks
    from tracememo.store.store import ValueStore

    if not cfg.manifest_path.exists():
        raise typer.BadParameter("no build/manifest.json found; run `tracememo build` first")
    manifest = ValueStore.load(cfg.manifest_path).to_manifest()
    files = check_files(cfg, extra)
    report = run_checks(files, manifest, cfg.check.allow_patterns)
    if cfg.check.llm_claims if llm is None else llm:
        from tracememo.check.claims import check_claims
        from tracememo.llm import make_client

        report.findings += check_claims(files, manifest, make_client(cfg.llm))
        report.findings.sort(key=lambda x: (x.file, x.line, x.check))
    out = cfg.build_path / "check_report.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _echo(report.summary())
    _echo(f"wrote {out}")
    return report.passed


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
    """ingest + analyze + render + check."""
    cfg = load_config(config)
    run_ingest(cfg)
    run_analyze(cfg)
    run_render(cfg)
    if cfg.check.run_in_build and not run_check(cfg):
        raise typer.Exit(code=1)


@app.command()
def check(
    config: ConfigOpt = Path("project.yaml"),
    files: Annotated[
        list[Path] | None, typer.Argument(help="Extra templates or draft fragments to check")
    ] = None,
    llm: Annotated[
        bool, typer.Option("--llm/--no-llm", help="Also run the LLM-assisted claim check")
    ] = False,
) -> None:
    """Run the grounding checks on the report templates. Exits 1 on any error."""
    if not run_check(load_config(config), files, llm=llm or None):
        raise typer.Exit(code=1)


@app.command()
def viewer(config: ConfigOpt = Path("project.yaml")) -> None:
    """Write build/<name>.html: the report with clickable provenance for every number."""
    from tracememo.report.viewer import render_viewer
    from tracememo.store.store import ValueStore

    cfg = load_config(config)
    store = ValueStore.load(cfg.manifest_path)
    out = render_viewer(
        store,
        cfg.resolve(cfg.report.markdown_template),
        cfg.build_path,
        cfg.build_path / f"{cfg.report.output_name}.html",
    )
    _echo(f"wrote {out}")


@app.command()
def draft(
    section: Annotated[str, typer.Option("--section", help="Section name, e.g. results")],
    outline: Annotated[Path, typer.Option("--outline", help="Outline file written by you")],
    config: ConfigOpt = Path("project.yaml"),
    out: Annotated[Path | None, typer.Option("--out", help="Output fragment path")] = None,
    syntax: Annotated[
        str | None, typer.Option("--syntax", help="markdown | latex | placeholder")
    ] = None,
) -> None:
    """Draft a section with the LLM as a live template fragment (placeholders, no digits)."""
    from tracememo.draft.drafter import convert_placeholders, draft_section
    from tracememo.llm import make_client
    from tracememo.store.store import ValueStore

    cfg = load_config(config)
    if not cfg.manifest_path.exists():
        raise typer.BadParameter("no build/manifest.json found; run `tracememo build` first")
    store = ValueStore.load(cfg.manifest_path)
    syntax = syntax or cfg.draft.syntax
    ext = {"markdown": ".md.j2", "latex": ".tex", "placeholder": ".md"}[syntax]
    out_path = out or cfg.resolve(cfg.draft.out_dir) / f"{section}{ext}"
    result = draft_section(
        make_client(cfg.llm),
        store,
        section,
        outline.read_text(encoding="utf-8"),
        max_fix_rounds=cfg.draft.max_fix_rounds,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(convert_placeholders(result.text, syntax), encoding="utf-8")
    raw_path = out_path.with_name(f"{section}.draft.md")
    raw_path.write_text(result.text, encoding="utf-8")
    for f in result.findings:
        _echo(f.format())
    state = "clean" if result.clean else "has checker errors (see above)"
    _echo(f"draft   {section}: {state} after {result.attempts} attempt(s)")
    _echo(f"wrote {out_path} (placeholder form: {raw_path})")


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


@synth_app.command("rag")
def synth_rag(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path(
        "data/synthetic/rag"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 0,
    n_traces: Annotated[int, typer.Option("--n-traces", help="Traces per version")] = 60,
) -> None:
    """Generate synthetic RAG traces (Langfuse-style JSON) and eval scores (CSV) + truth.json."""
    from tracememo.synth.rag import RagSynthConfig, generate_rag, write_rag

    data = generate_rag(RagSynthConfig(seed=seed, n_traces=n_traces))
    for p in write_rag(data, out):
        _echo(f"wrote {p}")


if __name__ == "__main__":
    app()
