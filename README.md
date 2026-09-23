# tracememo

Turn raw experiment output into the analysis, figures, tables and first-draft prose of a
technical report, where **every number in the document is linked to the code and data that
produced it**. See `SPEC.md` for the full design.

Status: milestone 4 of 8 (drone pipeline, caching, Markdown and LaTeX/PDF output, the
deterministic grounding checker, and ArduPilot / Marvelmind adapters).

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

`examples/drone_memo/project_raw.yaml` runs the same memo from an ArduPilot `.bin` log and a
Marvelmind CSV (synthetic ones are generated alongside the Parquet tables), using the
`ardupilot` and `marvelmind` adapters. The message-to-table mapping is configuration; see the
docstrings in `tracememo/adapters/ardupilot.py` and `marvelmind.py` and `docs/schemas.md`.

`tracememo check` (also run at the end of `build`) flags raw numbers typed into prose,
references to ids that are not in the manifest, values whose input files or analysis code
changed since the last run, and two-value comparisons ("A was lower than B") whose direction
contradicts the stored values. It exits non-zero on any error, for use in CI.

## Development

```
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/pytest
```
