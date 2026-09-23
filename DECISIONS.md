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
16. **The checker runs on templates, not rendered output.** Markdown/Jinja templates, draft
    fragments (`{{val:id}}`) and LaTeX documents are parsed with length-preserving masking
    so findings carry real line numbers. LaTeX text before `\begin{document}` is ignored.
17. **Raw-number allowlist** (built in): years 1900–2099, numbered cross references
    (Section/Figure/Table/Eq. N), numeric citations `[3]`, ordinals (`95th`), and any
    digit run attached to letters (`3D`, `p95`, `x_1`). Numbers inside code spans, headings,
    Jinja statements/expressions, LaTeX commands and options are not prose. Projects can add
    regexes via `check.allow_patterns`.
18. **Staleness findings are grouped** per (analysis, reason) per file, listing the ids,
    instead of one finding per referenced value.
19. **Comparison check scope**: sentences with exactly two value references and one phrase
    from a fixed list (`lower/less/smaller/shorter than`, `below`, `under`;
    `higher/greater/larger/longer/more than`, `above`, `exceeds`, `over`;
    `decreased/reduced/dropped/fell ... from A to B`; `increased/rose/grew ... from A to B`).
    Negated sentences are skipped (left to the LLM claim checker). Different units produce a
    warning, not an error.
20. **`build` runs the checker last** and exits 1 if it fails (`check.run_in_build: false`
    disables this).
21. **ArduPilot mapping is a config profile.** The built-in `arducopter4` profile maps `XKF1`
    (EKF core 0) to `pose`, `PSCN`/`PSCE`/`PSCD` targets to `nav_target` (joined by nearest
    timestamp) and `BCN` to `beacon`, converting NED to ENU. Any part can be overridden under
    `options.mapping`. Arming time comes from `EV` id 10 by default, with a fallback to the
    first timestamp. The user validates the profile on their own logs; field names in real
    firmware may differ and are meant to be adjusted in `project.yaml`, not in code.
22. **The synthetic generator also writes an ArduPilot-format `.bin` and a Marvelmind-style
    CSV**, so both real-data adapters are tested end to end without any real data. The `.bin`
    writer emits `FMT` records and ArduCopter 4.x-like message layouts, plus a second EKF
    core and pre-arm samples to exercise instance filtering and time normalization. Small
    fixture files generated this way are committed under `tests/fixtures/`.
23. **Marvelmind adapter emits `beacon_raw` by default**; `options.table: beacon` feeds the
    drone analyses directly. Time alignment to the flight log is a user-supplied
    `time_offset_s` (or `time_origin: first`); the adapter does not try to auto-align clocks.
24. **`drone.loc_err` still requires a `beacon` table.** For logs without beacons, set
    `reference: nav_target` and supply `beacon` from the `BCN` mapping or a dummy source; a
    proper optional-input mechanism is deferred until a real case needs it.
25. **Langfuse adapter reads exports; live fetching is a guarded optional path.** The
    `langfuse` SDK is not in the spec's dependency list, so `fetch: true` imports it lazily
    and fails with an install hint. The fetch code follows the v2 SDK (`fetch_traces` /
    `fetch_observations`) and is not covered by tests; exports are the supported path.
26. **Version labels come from `options.version_from`**: a trace field (`version`, `release`),
    `metadata.<key>`, or `tag:<prefix>`. Only `SPAN` and `GENERATION` observations become
    spans by default.
27. **Stage spans default to leaf spans** (those that are not a parent of another span), so a
    root "pipeline" span is not counted as a stage. `stage_names` overrides this.
28. **Version and config names are slugified into value ids** (`rerank+hyde` -> `rerank_hyde`);
    tables keep the original labels.
29. **`llm.eval` uses a percentile bootstrap** of the mean with a fixed seed from params, so
    results are reproducible and cacheable.
30. **Figure/table captions are soft-escaped for LaTeX** (`% _ & #` unless already escaped),
    because a bare `%` in a generated caption silently truncated `values.tex`.
31. **Analysis source hashes cover the whole defining module**, not just the decorated
    function. Hashing only the function missed edits to helpers (found when a figure-helper
    change was reported as a cache hit). Any edit to the module reruns all analyses defined in
    it, which is the safe direction.
