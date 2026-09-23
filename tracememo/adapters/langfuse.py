"""Langfuse trace export adapter.

Reads a JSON export and writes two tables:

- ``spans``: ``trace_id, span_id, parent_id, name, start, end, duration_ms, metadata``
- ``traces``: ``trace_id, version, total_latency_ms, input, output``

Accepted JSON shapes (the Langfuse API/export field names are used throughout):

- ``{"traces": [...], "observations": [...]}``
- a list (or ``{"data": [...]}``) of traces, each optionally embedding ``observations``

Trace fields: ``id, timestamp, version, release, tags, metadata, input, output, latency``.
Observation fields: ``id, traceId, parentObservationId, type, name, startTime, endTime,
metadata``. Only observations of the types in ``options.span_types`` (default ``SPAN``,
``GENERATION``) become spans.

Options::

    options:
      version_from: version            # trace field; or "metadata.<key>"; or "tag:<prefix>"
      span_types: [SPAN, GENERATION]
      fetch: false                     # true: pull from a Langfuse server via the SDK
                                       # (needs the optional `langfuse` package and
                                       # LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)

With ``fetch: true`` the ``path`` is where the fetched export is saved (and hashed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.adapters.base import IngestedTable, register_adapter
from tracememo.hashing import sha256_file
from tracememo.store.models import InputFile

DEFAULT_SPAN_TYPES = ("SPAN", "GENERATION")


def load_export(obj: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize the accepted JSON shapes into ``(traces, observations)`` lists."""
    if isinstance(obj, dict) and "traces" in obj:
        traces = list(obj["traces"])
        observations = list(obj.get("observations", []))
    elif isinstance(obj, dict) and "data" in obj:
        traces = list(obj["data"])
        observations = []
    elif isinstance(obj, list):
        traces = list(obj)
        observations = []
    else:
        raise ValueError("langfuse adapter: unrecognized export shape")
    for t in traces:
        observations.extend(t.pop("observations", []) or [])
    return traces, observations


def trace_version(trace: dict[str, Any], version_from: str) -> str | None:
    """Extract the version label of a trace according to ``version_from``."""
    if version_from.startswith("metadata."):
        key = version_from.split(".", 1)[1]
        md = trace.get("metadata") or {}
        v = md.get(key)
        return None if v is None else str(v)
    if version_from.startswith("tag:"):
        prefix = version_from[4:]
        for tag in trace.get("tags") or []:
            if str(tag).startswith(prefix):
                return str(tag)[len(prefix) :]
        return None
    v = trace.get(version_from)
    return None if v is None else str(v)


def _json_text(v: Any) -> str | None:
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def build_tables(
    traces: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    version_from: str = "version",
    span_types: tuple[str, ...] = DEFAULT_SPAN_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ``spans`` and ``traces`` frames from export objects."""
    span_rows = []
    for o in observations:
        if str(o.get("type", "SPAN")).upper() not in span_types:
            continue
        span_rows.append(
            {
                "trace_id": o.get("traceId"),
                "span_id": o.get("id"),
                "parent_id": o.get("parentObservationId"),
                "name": o.get("name"),
                "start": o.get("startTime"),
                "end": o.get("endTime"),
                "metadata": _json_text(o.get("metadata")),
            }
        )
    spans = pd.DataFrame(
        span_rows,
        columns=["trace_id", "span_id", "parent_id", "name", "start", "end", "metadata"],
    )
    spans["start"] = pd.to_datetime(spans["start"], utc=True, format="ISO8601")
    spans["end"] = pd.to_datetime(spans["end"], utc=True, format="ISO8601")
    spans["duration_ms"] = (spans["end"] - spans["start"]).dt.total_seconds() * 1000.0
    spans = spans[
        ["trace_id", "span_id", "parent_id", "name", "start", "end", "duration_ms", "metadata"]
    ]
    spans = spans.sort_values(["trace_id", "start"]).reset_index(drop=True)

    span_extent = (
        spans.groupby("trace_id").agg(t0=("start", "min"), t1=("end", "max"))
        if not spans.empty
        else pd.DataFrame(columns=["t0", "t1"])
    )
    trace_rows = []
    for t in traces:
        tid = t.get("id")
        latency = t.get("latency")
        if latency is not None:
            total_ms = float(latency) * 1000.0
        elif tid in span_extent.index:
            ext = span_extent.loc[tid]
            total_ms = (ext["t1"] - ext["t0"]).total_seconds() * 1000.0
        else:
            total_ms = float("nan")
        trace_rows.append(
            {
                "trace_id": tid,
                "version": trace_version(t, version_from),
                "total_latency_ms": total_ms,
                "input": _json_text(t.get("input")),
                "output": _json_text(t.get("output")),
            }
        )
    traces_df = pd.DataFrame(
        trace_rows, columns=["trace_id", "version", "total_latency_ms", "input", "output"]
    )
    return spans, traces_df


def fetch_export(options: dict[str, Any]) -> dict[str, Any]:
    """Pull traces and observations from a Langfuse server with the optional SDK.

    Requires the ``langfuse`` package and the usual ``LANGFUSE_*`` environment variables.
    This path is not exercised by the test suite.
    """
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "langfuse adapter: `fetch: true` needs the optional `langfuse` package "
            "(pip install langfuse)"
        ) from e
    client = Langfuse()
    limit = int(options.get("fetch_limit", 100))
    traces: list[dict[str, Any]] = []
    page = 1
    while True:  # pragma: no cover - network
        batch = client.fetch_traces(page=page, limit=limit)
        items = [t.dict() if hasattr(t, "dict") else dict(t) for t in batch.data]
        traces.extend(items)
        if len(items) < limit:
            break
        page += 1
    observations: list[dict[str, Any]] = []
    for t in traces:  # pragma: no cover - network
        obs = client.fetch_observations(trace_id=t["id"], limit=limit)
        observations.extend(o.dict() if hasattr(o, "dict") else dict(o) for o in obs.data)
    return {"traces": traces, "observations": observations}


@register_adapter("langfuse")
def ingest_langfuse(
    path: Path, tables: list[str] | None, options: dict[str, Any]
) -> list[IngestedTable]:
    """Read a Langfuse export (or fetch one) into ``spans`` and ``traces`` tables."""
    path = Path(path)
    if options.get("fetch"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fetch_export(options), indent=1), encoding="utf-8")
    obj = json.loads(path.read_text(encoding="utf-8"))
    traces, observations = load_export(obj)
    spans, traces_df = build_tables(
        traces,
        observations,
        version_from=options.get("version_from", "version"),
        span_types=tuple(options.get("span_types", DEFAULT_SPAN_TYPES)),
    )
    raw = [InputFile(path=str(path), sha256=sha256_file(path))]
    out = {"spans": spans, "traces": traces_df}
    names = tables or list(out)
    unknown = [n for n in names if n not in out]
    if unknown:
        raise KeyError(f"langfuse adapter produces {list(out)}, not {unknown}")
    return [IngestedTable(id=n, df=out[n], raw_inputs=raw) for n in names]
