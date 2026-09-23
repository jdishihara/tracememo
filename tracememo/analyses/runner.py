"""Run analyses in order, save their outputs, stamp provenance and reuse cached results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.analyses.registry import AnalysisSpec, order_analyses
from tracememo.figures import save_figure
from tracememo.gitinfo import git_info
from tracememo.hashing import hash_dataframe
from tracememo.report.tables import df_to_latex, df_to_markdown
from tracememo.store.models import (
    AnalysisRecord,
    AnalysisResult,
    InputFile,
    Manifest,
    Provenance,
    TableRef,
)
from tracememo.store.store import ValueStore
from tracememo.store.tables import TableStore


@dataclass
class _ProducedTable:
    path: Path
    sha256: str
    raw_inputs: list[InputFile]


def run_analyses(
    selections: list[tuple[str, dict[str, Any]]],
    tables: TableStore,
    store: ValueStore,
    build_dir: Path,
    git_root: Path | None = None,
    previous: Manifest | None = None,
) -> list[AnalysisRecord]:
    """Run the selected analyses and add their results to ``store``.

    ``selections`` is a list of ``(analysis_id, param_overrides)``. Figures are written to
    ``build_dir/figures`` and tables to ``build_dir/tables_out``.

    When ``previous`` (the last manifest) is given, an analysis whose source hash, params
    and input table hashes are unchanged, and whose output files still exist, is not rerun:
    its values, figures and tables are copied from the previous manifest with their
    original provenance, and its record is marked ``cached``.
    """
    build_dir = Path(build_dir)
    overrides = dict(selections)
    specs = order_analyses([sid for sid, _ in selections])
    commit, dirty = git_info(git_root or build_dir)
    store.input_tables = dict(tables.entries)
    produced: dict[str, _ProducedTable] = {}
    records: list[AnalysisRecord] = []

    for spec in specs:
        params = {**spec.params, **overrides.get(spec.id, {})}
        refs, raw = _resolve_refs(spec, tables, produced)

        cached = _cached_result(spec, params, refs, previous)
        if cached is not None:
            for tab in cached.tables:
                if tab.id in spec.outputs and tab.parquet_path:
                    df = pd.read_parquet(tab.parquet_path)
                    produced[tab.id] = _ProducedTable(
                        Path(tab.parquet_path), hash_dataframe(df), raw
                    )
            store.add_result(cached)
            assert previous is not None
            prev_rec = previous.analyses[spec.id]
            record = prev_rec.model_copy(update={"cached": True})
            store.analyses[spec.id] = record
            records.append(record)
            continue

        inputs = _load_inputs(spec, tables, produced)
        result = spec.func(**inputs, params=params)
        now = datetime.now(UTC)
        prov = Provenance(
            analysis_id=spec.id,
            analysis_source_sha256=spec.source_sha256,
            git_commit=commit,
            git_dirty=dirty,
            params=params,
            input_tables=refs,
            raw_inputs=raw,
            timestamp=now,
        )
        _finalize(result, spec, prov, build_dir, produced, raw)
        store.add_result(result)
        record = AnalysisRecord(
            id=spec.id,
            source_sha256=spec.source_sha256,
            params=params,
            input_tables=refs,
            cached=False,
            timestamp=now,
        )
        store.analyses[spec.id] = record
        records.append(record)
    return records


def _finalize(
    result: AnalysisResult,
    spec: AnalysisSpec,
    prov: Provenance,
    build_dir: Path,
    produced: dict[str, _ProducedTable],
    raw: list[InputFile],
) -> None:
    """Save figures/tables to disk, fill in their paths and stamp provenance."""
    for v in result.values:
        v.provenance = prov
    for fig in result.figures:
        if fig.figure is not None:
            pdf, png = save_figure(fig.figure, build_dir / "figures" / fig.id)
            fig.pdf_path, fig.png_path = str(pdf), str(png)
            fig.figure = None
        fig.provenance = prov
    for tab in result.tables:
        if tab.dataframe is not None:
            df: pd.DataFrame = tab.dataframe
            out_dir = build_dir / "tables_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            pq = out_dir / f"{tab.id}.parquet"
            df.to_parquet(pq, index=False)
            tex = out_dir / f"{tab.id}.tex"
            tex.write_text(df_to_latex(df, tab.float_fmt) + "\n", encoding="utf-8")
            md = out_dir / f"{tab.id}.md"
            md.write_text(df_to_markdown(df, tab.float_fmt) + "\n", encoding="utf-8")
            tab.parquet_path, tab.latex_path, tab.markdown_path = str(pq), str(tex), str(md)
            tab.columns = [str(c) for c in df.columns]
            if tab.id in spec.outputs:
                produced[tab.id] = _ProducedTable(pq, hash_dataframe(df), raw)
            tab.dataframe = None
        tab.provenance = prov


def _normalize_params(params: dict[str, Any]) -> Any:
    return json.loads(json.dumps(params, sort_keys=True, default=str))


def _cached_result(
    spec: AnalysisSpec, params: dict[str, Any], refs: list[TableRef], previous: Manifest | None
) -> AnalysisResult | None:
    """Return the previous result for ``spec`` if nothing it depends on changed."""
    if previous is None:
        return None
    rec = previous.analyses.get(spec.id)
    if rec is None or rec.source_sha256 != spec.source_sha256:
        return None
    if _normalize_params(rec.params) != _normalize_params(params):
        return None
    if rec.input_tables != refs:
        return None
    values = [v for v in previous.values.values() if _from(v.provenance, spec.id)]
    figures = [f for f in previous.figures.values() if _from(f.provenance, spec.id)]
    tables = [t for t in previous.tables.values() if _from(t.provenance, spec.id)]
    paths = [f.pdf_path for f in figures] + [f.png_path for f in figures]
    paths += [t.parquet_path for t in tables] + [t.latex_path for t in tables]
    paths += [t.markdown_path for t in tables]
    if any(p is None or not Path(p).exists() for p in paths):
        return None
    return AnalysisResult(
        values=[v.model_copy(deep=True) for v in values],
        figures=[f.model_copy(deep=True) for f in figures],
        tables=[t.model_copy(deep=True) for t in tables],
    )


def _from(prov: Provenance | None, analysis_id: str) -> bool:
    return prov is not None and prov.analysis_id == analysis_id


def _resolve_refs(
    spec: AnalysisSpec, tables: TableStore, produced: dict[str, _ProducedTable]
) -> tuple[list[TableRef], list[InputFile]]:
    """Input table hashes and the union of raw input files, without loading data."""
    refs: list[TableRef] = []
    raw: dict[str, InputFile] = {}
    for name in spec.inputs:
        if name in produced:
            p = produced[name]
            refs.append(TableRef(id=name, sha256=p.sha256))
            for f in p.raw_inputs:
                raw[f.path] = f
        else:
            entry = tables.entries.get(name)
            if entry is None:
                raise KeyError(
                    f"analysis {spec.id!r} needs table {name!r}, which was not ingested "
                    "and is not produced by an earlier analysis"
                )
            refs.append(TableRef(id=name, sha256=entry.sha256))
            for f in entry.raw_inputs:
                raw[f.path] = f
    return refs, sorted(raw.values(), key=lambda f: f.path)


def _load_inputs(
    spec: AnalysisSpec, tables: TableStore, produced: dict[str, _ProducedTable]
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    for name in spec.inputs:
        if name in produced:
            inputs[name] = pd.read_parquet(produced[name].path)
        else:
            inputs[name] = tables.get(name)
    return inputs
