# tracememo

Turn raw experiment output into the analysis, figures, tables and first-draft prose of a
technical report, where **every number in the document is linked to the code and data that
produced it**. See `SPEC.md` for the full design.

Status: milestone 2 of 8 (full drone pipeline: `drone.loc_err`, `drone.trajectory`,
`drone.failures`, caching, provenance, `explain`, Markdown and LaTeX/PDF output).

## Quick start

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/tracememo synth drone --out data/synthetic/drone --seed 0
.venv/bin/tracememo build --config examples/drone_memo/project.yaml
open examples/drone_memo/build/report.md
.venv/bin/tracememo explain drone.loc_err.mean_cm --config examples/drone_memo/project.yaml
```

Numbers in `examples/drone_memo/report.md.j2` are written as `{{ val("drone.loc_err.mean_cm") }}`
and in `report.tex` as `\val{drone.loc_err.mean_cm}` (or `\valu{...}` with unit,
`\fig{...}`, `\tab{...}`). Both are filled from `build/manifest.json`, which records the
provenance of every value. If `latexmk` is installed, `build` also produces `report.pdf`.
Re-running `build` reruns only analyses whose code, parameters or input data changed.

## Development

```
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/pytest
```
