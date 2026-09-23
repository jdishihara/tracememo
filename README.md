# tracememo

Turn raw experiment output into the analysis, figures, tables and first-draft prose of a
technical report, where **every number in the document is linked to the code and data that
produced it**. See `SPEC.md` for the full design.

Status: milestone 6 of 8 (drone and LLM/RAG examples, real-data adapters, caching, Markdown
and LaTeX/PDF output, deterministic grounding checker, LLM drafting and claim checks).

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

The second example, `examples/rag_memo`, covers an LLM pipeline: Langfuse-style trace exports
and per-item evaluation scores go through the `langfuse` and `eval_table` adapters, and the
`llm.latency` and `llm.eval` analyses produce stage/end-to-end latency per version and
bootstrap confidence intervals per metric and configuration:

```
.venv/bin/tracememo synth rag --out data/synthetic/rag --seed 0
.venv/bin/tracememo build --config examples/rag_memo/project.yaml
```

## Drafting with an LLM

```
export ANTHROPIC_API_KEY=...
.venv/bin/tracememo draft --section results --outline outline.md --config examples/drone_memo/project.yaml
.venv/bin/tracememo check --llm --config examples/drone_memo/project.yaml drafts/results.md.j2
```

`draft` gives the model your outline plus the list of available value, figure and table ids
and forbids digits: every quantity must be a `{{val:id}}` placeholder. The result is saved as
a template fragment, so it stays live when the data changes, and is checked immediately; on
errors the findings are sent back once for a corrected draft. `check --llm` additionally asks
the model whether each remaining quantitative sentence is supported by the cited values
(warnings only). The model name comes from `llm.model` in `project.yaml`; prompts live in
`tracememo/draft/prompts/`. Set `TRACEMEMO_FAKE_LLM=responses.json` to dry-run without an API key.

`tracememo check` (also run at the end of `build`) flags raw numbers typed into prose,
references to ids that are not in the manifest, values whose input files or analysis code
changed since the last run, and two-value comparisons ("A was lower than B") whose direction
contradicts the stored values. It exits non-zero on any error, for use in CI.

## Development

```
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/pytest
```
