# tracememo — Project Specification

> Hand this file to Claude Code as the project spec. Save it in the repo root (e.g. as `SPEC.md`, and reference it from `CLAUDE.md`). Build in the milestone order at the bottom, and stop for review at the end of each milestone.

## 1. What this is

`tracememo` is a Python tool that turns raw experiment output (flight logs, LLM pipeline traces, evaluation results) into the analysis, figures, tables, and first-draft prose of a technical report, where **every number in the document is linked to the code and data that produced it**.

The motivating use case is writing NASA Technical Memoranda. Two real examples drive the design:

1. **Drone localization memo.** Raw data: ArduPilot DataFlash flight logs (`.bin`) and Marvelmind ultrasonic positioning exports (CSV). Report needs: localization error statistics, planned-vs-actual trajectory plots, a table of sensor failure events (dropouts, position jumps, drift), and prose describing them.
2. **LLM platform (RAG) memo.** Raw data: LLM pipeline traces (Langfuse-style JSON exports) and evaluation results (per-question metric scores). Report needs: latency breakdown by pipeline stage, before/after comparison across system versions, a table of evaluation metrics (correctness, faithfulness, relevance, retrieval ranking, entity precision/recall) across configurations, and prose describing them.

### The core idea: "live numbers"

A number in the report is never typed as text. It is a reference to a computed value, e.g. `\val{drone.loc_err.mean_cm}` in LaTeX. When the data or analysis changes and the pipeline reruns, every number, table, and figure updates. It becomes impossible to have a figure saying 8 cm next to a sentence saying 10 cm.

### The three guarantees

- **Provenance:** every value, figure, and table records which input files (by content hash), which analysis function, which parameters, and which git commit produced it.
- **Consistency:** the document's numbers come only from the value store.
- **Grounding:** LLM-drafted prose is checked; any number or quantitative claim not backed by the value store is flagged.

## 2. Important constraint: data

Real NASA data may be restricted. **The repo must be fully usable and demonstrable without any NASA data.** Build a synthetic data generator (Section 6) that produces realistic flight logs and LLM traces with known ground truth. Real-data adapters must work on real files, but no real data is ever committed to the repo. Add `data/` to `.gitignore` except `data/synthetic/`.

## 3. Tech stack

- Python 3.11+, managed with `uv` (or `pip` + `pyproject.toml`)
- `pandas` (or `polars`) for tables; store normalized data as Parquet
- `matplotlib` for figures (PDF + PNG output, consistent style file)
- `pymavlink` for parsing ArduPilot DataFlash logs (`DFReader`)
- `jinja2` for report templates
- `typer` for the CLI
- `pydantic` for all schemas (values, provenance, manifests, config)
- `anthropic` Python SDK for the LLM drafting and claim-checking steps. Model name must come from config, not be hard-coded. Default: `claude-sonnet-5`. API key from the `ANTHROPIC_API_KEY` environment variable. Check https://docs.claude.com/en/api/overview for current SDK usage.
- `pytest` for tests; `ruff` for lint/format
- LaTeX output compiles with `latexmk` if installed; Markdown output always works without LaTeX

## 4. Architecture

```
tracememo/
  adapters/        # raw files -> normalized tables
    ardupilot.py
    marvelmind.py
    langfuse.py
    eval_table.py  # generic CSV/JSON of per-item metric scores
  analyses/        # normalized tables -> values, figures, tables
    registry.py    # @analysis decorator + dependency graph
    drone/
    llm/
  store/           # value store + provenance
    models.py      # pydantic: Value, Figure, Table, Provenance
    store.py
  report/
    templates/     # Jinja2 LaTeX + Markdown templates
    render.py
  draft/           # LLM drafting
    drafter.py
    prompts/
  check/           # grounding checker
    numbers.py     # deterministic checks
    claims.py      # LLM-assisted claim checks
  synth/           # synthetic data generators
  cli.py
examples/
  drone_memo/
  rag_memo/
tests/
```

### 4.1 Adapters (ingest)

Each adapter reads raw files and writes normalized Parquet tables with documented schemas. Adapters do no analysis.

- **ArduPilot adapter:** use `pymavlink`'s DataFlash reader. Extract the message types needed for position, attitude, navigation targets, and beacon/external positioning. Message names vary by firmware version, so inspect the actual logs and make the message-to-table mapping a config entry rather than hard-coding it. Output tables such as `pose` (time, x, y, z, roll, pitch, yaw), `nav_target` (time, target x, y, z), `beacon` (time, beacon-derived position, quality fields if present). Normalize time to seconds from arming.
- **Marvelmind adapter:** read the CSV export, output `beacon_raw` (time, x, y, z, quality/validity fields). Column names should be configurable.
- **Langfuse adapter:** read exported trace JSON (do not require a live Langfuse server; optionally support fetching via the Langfuse SDK behind a flag). Output `spans` (trace_id, span_id, parent_id, name, start, end, duration_ms, metadata) and `traces` (trace_id, version/tag, total_latency_ms, input, output).
- **Eval table adapter:** read CSV/JSON where each row is (item_id, config/version, metric_name, score). Output a long-format `eval_scores` table.

Every adapter records the SHA-256 of each input file in its output manifest.

### 4.2 Analyses

An analysis is a Python function registered with a decorator:

```python
@analysis(
    id="drone.loc_err",
    inputs=["pose", "nav_target"],
    params={"align_tolerance_s": 0.05},
)
def localization_error(pose, nav_target, params) -> AnalysisResult:
    ...
    return AnalysisResult(
        values=[
            Value(
                id="drone.loc_err.mean_cm",
                value=...,
                unit="cm",
                fmt=".1f",
                description="Mean horizontal localization error",
            )
        ],
        figures=[...],
        tables=[...],
    )
```

- The registry builds a dependency graph from `inputs` and runs analyses in order.
- **Caching:** an analysis reruns only if its input table hashes, params, or source code hash changed.
- Each result's provenance is filled in automatically by the runner (not by the analysis author).

Required analyses for the examples:

Drone:
- `drone.loc_err`: horizontal and 3D error between estimated position and reference (nav target or beacon position, configurable); mean, median, p95, max, RMSE, and % of samples under 10 cm. Figure: error over time. Figure: error histogram/CDF.
- `drone.trajectory`: planned vs. actual trajectory plot (top-down and altitude vs. time).
- `drone.failures`: detect beacon dropouts (gaps longer than a threshold), position jumps (step change above a threshold), and drift (sustained bias over a window). Table: one row per event (type, start, duration, magnitude). Values: count per type.

LLM:
- `llm.latency`: per-stage latency (median, p95) from spans grouped by span name; end-to-end latency per version. Figure: stacked bar of stage latency per version. Values: end-to-end median per version and % reduction between two named versions.
- `llm.eval`: mean and 95% bootstrap CI of each metric per config. Table: metrics × configs. Values: each cell.

### 4.3 Value store and provenance

- `Value`: id (dotted, unique), value (float/int/str), unit, fmt (format spec), description, provenance.
- `Figure`: id, file paths (PDF + PNG), caption, provenance.
- `Table`: id, dataframe (saved as Parquet + rendered LaTeX/Markdown), caption, provenance.
- `Provenance`: analysis id, analysis source hash, git commit (and dirty flag), params, input table ids + hashes, raw input file hashes, timestamp.
- Everything is written to `build/manifest.json`. This file is the single source of truth for the report.

### 4.4 Report rendering

- The renderer generates `build/values.tex`, which defines one LaTeX macro per value, and a `\val{id}` command that looks it up. Value ids with dots must be handled (e.g. map to a lookup via `\csname`), so authors can write `\val{drone.loc_err.mean_cm}`.
- Also generate `\fig{id}` and `\tab{id}` helpers that insert the figure/table with its caption.
- Markdown output: same templates with `{{ val("drone.loc_err.mean_cm") }}` syntax, rendered by Jinja2.
- Optional (config flag): in the Markdown/HTML output, each value renders as a link/tooltip showing its provenance.
- Include a minimal LaTeX template for a tech-memo-style document (title, abstract, sections). Do **not** attempt to reproduce NASA's official template; make the template swappable.

### 4.5 LLM drafting

Command: `tracememo draft --section results --outline outline.md`

- Input to the model: the section outline written by the user, the list of available values (id, formatted value, unit, description), and figure/table ids with captions.
- **Hard rule in the prompt:** the model may not write any digits for quantities. Every quantity must be written as a placeholder `{{val:id}}`, and figures/tables referenced as `{{fig:id}}` / `{{tab:id}}`. Ordinary words like "two sensors" are allowed but discouraged; numbers are preferred as values.
- Output is saved as a template fragment (not as final text), so it stays live when data changes.
- Draft prompts live in `draft/prompts/` as plain text files, versioned in git.

### 4.6 Grounding checker

Command: `tracememo check` (runs on any template, hand-written or drafted).

Deterministic checks (`check/numbers.py`), required:
1. **Raw number check:** flag any digit sequence in prose that is not inside a placeholder/macro, excluding an allowlist (section numbers, years, citations, figure/table labels, units like "3D"). Report file, line, and text.
2. **Unknown reference check:** flag any `\val{}` / `{{val:}}` id not in the manifest.
3. **Staleness check:** flag values whose provenance points to a raw file hash that no longer matches the file on disk, or to an analysis whose source changed since the last run.
4. **Comparison check:** for simple comparative sentences that contain two value references and a comparison word ("lower than", "increased", "reduced", "exceeds", "under"), verify the direction matches the actual values. Keep a small, tested phrase list; don't try to be exhaustive.

LLM-assisted checks (`check/claims.py`), secondary:
5. For each sentence containing a quantitative or comparative claim that the deterministic checker couldn't verify, ask the model whether the claim is supported by the provided values. Return: supported / unsupported / can't tell, with a reason. These are warnings, never auto-fixes.

Output: a human-readable report in the terminal, plus `build/check_report.json`. Exit code non-zero if any deterministic check fails (so it can run in CI).

### 4.7 CLI

```
tracememo ingest   --config project.yaml        # run adapters
tracememo analyze  --config project.yaml        # run analyses (cached)
tracememo draft    --section <name> --outline <file>
tracememo check    --config project.yaml
tracememo build    --config project.yaml        # ingest + analyze + render (+ check)
tracememo explain  <value_id>                   # print a value's full provenance
tracememo synth    drone|rag --out data/synthetic/... --seed 0
```

`project.yaml` lists raw input paths, adapter configs, which analyses to run, parameter overrides, template path, output formats, and LLM model name.

## 5. Evaluation (this is what makes it a research project, not just a tool)

Write `eval/` scripts that produce numbers for the README:

1. **Reproduction test.** Using synthetic data with known ground truth, verify every computed value matches the truth within tolerance.
2. **Drafting grounding rate.** Draft the results section N times (e.g. 20) under two conditions: (a) placeholders required, (b) model given the same values but allowed to write numbers freely. Measure: % of numbers in the text that are wrong or unsupported, and % of comparison claims with the wrong direction. Report both conditions.
3. **Checker recall.** Take correct drafts and inject errors programmatically (swap a number, flip a comparison word, reference a stale value, invent a statistic). Measure what fraction of injected errors the checker catches, split by error type, and the false-positive rate on clean drafts.
4. **Staleness demo.** Change one synthetic flight log, rerun `build`, and show that exactly the dependent values/figures updated and everything else was cached.

## 6. Synthetic data

- **Drone:** generate a planned trajectory (e.g. waypoint square or figure-eight in a room), simulate actual position = planned + controller lag + noise, simulate beacon measurements with configurable noise, and inject known failure events (dropouts, jumps, drift) at known times. Write the output in the same normalized table format the adapters produce, plus (stretch goal) a Marvelmind-style CSV so the Marvelmind adapter is exercised. Save the ground-truth events and statistics to a JSON file for tests.
- **RAG:** generate traces for two system versions with named stages (e.g. `embed_query`, `retrieve`, `rerank`, `generate`) and per-stage latency distributions, where version 2 is faster in specific stages. Generate eval scores per item for several configs with known means. Write in Langfuse-like export JSON and eval CSV.
- All generators take a seed and are deterministic.

## 7. Testing and quality

- Unit tests for every adapter (using small fixture files), every analysis (against synthetic ground truth), the value store, the renderer (golden-file tests of `values.tex` and rendered Markdown), and every deterministic check (positive and negative cases).
- LLM calls are behind an interface with a fake implementation for tests; no test hits the real API.
- CI via GitHub Actions: ruff, pytest, and `tracememo build` on the synthetic examples.
- Type hints throughout.

## 8. Milestones (build in this order)

1. **Skeleton + synthetic drone data.** Repo setup, pydantic models, value store, synthetic drone generator, `drone.loc_err` analysis, Markdown rendering with live values. Done when `tracememo build` on synthetic data produces a Markdown report whose numbers match ground truth.
2. **Full drone pipeline.** Remaining drone analyses, figures, tables, caching, provenance, `explain`, LaTeX output.
3. **Deterministic checker.** Checks 1–4 with tests.
4. **Real-data adapters.** ArduPilot and Marvelmind adapters, tested on fixture files. (User will validate on their own real logs locally.)
5. **LLM example.** Synthetic RAG data, Langfuse and eval adapters, `llm.latency` and `llm.eval`.
6. **LLM drafting + claim checks.** Drafter with placeholder rule, LLM claim checker.
7. **Evaluation.** Section 5 scripts and results.
8. **Polish.** README with a short demo GIF or screenshots, architecture diagram, evaluation results table, and a "how to add a new adapter/analysis" guide. Optional: small web viewer where clicking a number shows its provenance.

## 9. Non-goals

- Not a general notebook replacement or a full agent that decides what analyses to run. Analyses are written by the user; the LLM only drafts prose and checks claims.
- No live dashboards, no database server, no multi-user features.
- Don't reproduce any official NASA template or include any NASA data.

## 10. Style notes for Claude Code

- Prefer small, well-named modules and pure functions over clever abstractions.
- Every public function has a docstring and type hints.
- When the spec is ambiguous, choose the simpler option and note the decision in `DECISIONS.md`.
- Ask before adding dependencies not listed in Section 3.
