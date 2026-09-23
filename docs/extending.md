# Adding an adapter or an analysis

## A new adapter (raw files -> normalized tables)

An adapter is a function `adapter(path, tables, options) -> list[IngestedTable]` registered
under a name that `project.yaml` refers to. It reads raw files, returns normalized dataframes
with the schemas in [`schemas.md`](schemas.md), and records the SHA-256 of every input file.
It does no analysis.

```python
# tracememo/adapters/mysensor.py
from pathlib import Path
from typing import Any
import pandas as pd
from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile


@register_adapter("mysensor")
def ingest_mysensor(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read my sensor's CSV export into the normalized `beacon` table."""
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {"t_s": df[options.get("time_col", "t")], "x_m": df["x"], "y_m": df["y"], "z_m": df["z"]}
    )
    raw = [InputFile(path=str(path), sha256=sha256_file(path))]
    return [IngestedTable(id=options.get("table", "beacon"), df=out, raw_inputs=raw)]
```

Register it by importing the module in `tracememo/adapters/base.py::get_adapter` (built-ins) or
by importing it from your own code before running the CLI. Then in `project.yaml`:

```yaml
inputs:
  - adapter: mysensor
    path: data/run1.csv
    options: {time_col: timestamp, table: beacon}
```

Checklist: document the output schema; make column names and units options rather than
constants; add a small fixture file under `tests/fixtures/` and a test that checks the output
columns, row count and a couple of values; never commit real data.

## A new analysis (tables -> values, figures, tables)

```python
# tracememo/analyses/drone/hover.py
from typing import Any
import pandas as pd
from tracememo.analyses.registry import analysis
from tracememo.figures import SERIES, new_figure
from tracememo.store.models import AnalysisResult, Figure, Table, Value

ID = "drone.hover"


@analysis(id=ID, inputs=["pose"], params={"window_s": 5.0})
def hover_stability(pose: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Position spread while hovering during the first `window_s` seconds."""
    w = pose[pose["t_s"] < params["window_s"]]
    spread_cm = 100.0 * float(w[["x_m", "y_m"]].std().mean())
    fig, ax = new_figure()
    ax.plot(w["t_s"], w["x_m"], color=SERIES[0])
    ax.set_xlabel("Time from arming (s)")
    ax.set_ylabel("x (m)")
    return AnalysisResult(
        values=[
            Value(
                id=f"{ID}.spread_cm",
                value=spread_cm,
                unit="cm",
                fmt=".1f",
                description="Mean horizontal position spread while hovering",
            )
        ],
        figures=[Figure(id=f"{ID}.x_vs_time", figure=fig, caption="x position during hover.")],
        tables=[Table(id=f"{ID}.samples", dataframe=w.head(), caption="First hover samples.")],
    )
```

Rules the runner relies on:

- `inputs` are table ids: ingested tables, or tables another analysis declares in `outputs`.
  The registry orders analyses so producers run first.
- The function is called as `func(**{input: dataframe}, params=merged_params)`. Defaults in
  the decorator are overridden by `analyses[].params` in `project.yaml`.
- Return unsaved objects: a matplotlib figure on `Figure.figure`, a dataframe on
  `Table.dataframe`. The runner writes PDF/PNG/Parquet/Markdown/LaTeX, fills in paths, and
  stamps provenance. Do not write files yourself.
- Ids are dotted lowercase (`analysis.name.value`); labels that are not valid segments go
  through `slugify()`.
- Give every value a `unit`, a format spec and a description: the drafter shows them to the
  LLM and the renderer uses them.
- Caching keys on the module source hash, params and input table hashes: edits to helpers in
  the same module invalidate the cache too.

Make it discoverable: import the module in `tracememo/analyses/drone/__init__.py` (or add your
package to `BUILTIN_MODULES` in `registry.py`), add it to `project.yaml`, and write a test
against synthetic ground truth (extend `tracememo/synth` and `truth.json` if the generator
does not already know the answer).

## A new report template

Markdown templates are Jinja2 with `val("id")`, `val("id", with_unit=True)`, `val_raw`,
`unit`, `fig`, `tab` and `provenance_appendix()`. LaTeX documents are plain `.tex` files that
`\usepackage{tracememo}` and use `\val`, `\valu`, `\fig`, `\tab`. Both are checked by
`tracememo check`; keep every quantity a reference and the check stays green.
