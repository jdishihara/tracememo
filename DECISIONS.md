# Decisions

Choices made where `SPEC.md` was ambiguous or silent. Newest at the bottom.

1. **Package manager: pip + venv, not uv.** `uv` is not installed on the development
   machine; the spec allows `pip` + `pyproject.toml`.
2. **`pyyaml` added as a dependency.** The spec requires `project.yaml` but does not list a
   YAML parser in section 3. `pyyaml` is the smallest standard choice. (`numpy` is also
   imported directly; it is already a transitive dependency of pandas.)
3. **Synthetic data is ingested by a `normalized` adapter.** The spec says the synthetic
   generator writes tables "in the same normalized table format the adapters produce". A
   trivial adapter that reads already-normalized Parquet keeps `ingest` uniform: every
   project goes through the same adapter -> table store path, and input file hashes are
   recorded the same way.
4. **Analyses return unsaved figures and dataframes; the runner saves them.** The spec's
   analysis signature is `f(*inputs, params)` with no output directory. So a `Figure`
   carries a matplotlib figure object and a `Table` carries a dataframe; the runner writes
   PDF/PNG/Parquet, fills in paths, and stamps provenance. Analyses stay pure.
5. **Table hashes are content hashes, not file hashes.** Parquet bytes are not guaranteed
   byte-identical across library versions, so a table's hash is a SHA-256 over its column
   names, dtypes and `pandas.util.hash_pandas_object` row hashes. Raw input files are still
   hashed by bytes.
6. **Alignment direction in `drone.loc_err`.** Each *reference* sample (nav target or
   beacon) is matched to the nearest pose sample within `align_tolerance_s`, so the error
   sample count equals the number of matched reference samples. The reference is usually
   the sparser stream (beacons), which avoids matching several pose rows to one beacon.
7. **Value ids are validated** as dotted lowercase identifiers
   (`^[a-z0-9_]+(\.[a-z0-9_]+)+$`) so they can be mapped to LaTeX `\csname` lookups and
   Markdown anchors without escaping.
8. **Synthetic drone truth JSON records both the injected events and the loc_err
   statistics** for each possible reference, computed directly from the generator's arrays
   (not via the analysis code), so the reproduction test compares two independent code paths.
9. **Time base.** All normalized tables use `t_s` = seconds from arming (float). The
   synthetic generator starts at 0.
10. **Caching key = analysis source hash + params + input table hashes**, compared against the
    previous `build/manifest.json`. A cache hit copies the previous values/figures/tables with
    their original provenance (including the original timestamp) and marks the analysis
    record `cached: true`. Missing output files force a rerun. Stale files from analyses no
    longer in the config are left on disk; the manifest is the source of truth, not the
    directory listing.
11. **Failure detection works on the beacon-minus-pose residual**, centred on its median, so a
    constant offset between the two frames is not an event. A jump is a step in the residual
    above `jump_threshold_m` that stays elevated for at least `jump_min_samples`; a drift is a
    rolling-mean offset above `drift_threshold_m` for at least `drift_window_s` outside jump
    windows. A ramp is therefore reported from the moment it crosses the threshold, not from
    the moment it started; tests allow for this.
12. **Event table has one `magnitude` column plus a `unit` column** (`s` for dropouts, `m` for
    jumps and drift) rather than separate columns per type, matching the spec's
    "(type, start, duration, magnitude)" row shape.
13. **LaTeX documents are plain `.tex`, not Jinja templates.** Authors write `\val{id}`,
    `\valu{id}` (with unit), `\fig{id}` and `\tab{id}`; `tracememo.sty` resolves them via
    `\csname` lookups defined in the generated `values.tex`, so dotted ids need no escaping.
    An unknown id is a LaTeX package error, so the PDF cannot build with a dangling reference.
    Figures are included as `{{path}.pdf}` so dots in file names are handled.
14. **Table Markdown/LaTeX are rendered at analyze time** (`build/tables_out/<id>.md|.tex`)
    alongside the Parquet, so the report renderers only splice files in.
15. **`report.markdown_template` / `report.latex_template`** replace the single `template`
    key so both formats can be produced from one config.
