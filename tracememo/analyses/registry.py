"""The ``@analysis`` decorator, registry and dependency ordering."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any

from tracememo.hashing import sha256_text
from tracememo.store.models import AnalysisResult

AnalysisFn = Callable[..., AnalysisResult]

BUILTIN_MODULES = ["tracememo.analyses.drone", "tracememo.analyses.llm"]


@dataclass(frozen=True)
class AnalysisSpec:
    """A registered analysis function and its declared interface."""

    id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    func: AnalysisFn = field(default=lambda **_: AnalysisResult(), compare=False)
    source_sha256: str = ""


_REGISTRY: dict[str, AnalysisSpec] = {}


def source_hash(func: Callable[..., Any]) -> str:
    """SHA-256 of the source of the module defining ``func`` (used for caching and staleness).

    The whole module is hashed, not just the decorated function, so that edits to helper
    functions the analysis calls also invalidate cached results.
    """
    module = inspect.getmodule(func)
    try:
        src = inspect.getsource(module) if module is not None else inspect.getsource(func)
    except (OSError, TypeError):
        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            src = repr(func)
    return sha256_text(src)


def analysis(
    id: str,
    inputs: list[str],
    params: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
) -> Callable[[AnalysisFn], AnalysisFn]:
    """Register a function as an analysis.

    ``inputs`` are table ids (ingested tables or tables produced by other analyses via
    ``outputs``). The function is called as ``func(**{input: df}, params=params)``.
    """

    def deco(func: AnalysisFn) -> AnalysisFn:
        if id in _REGISTRY and _REGISTRY[id].func is not func:
            raise ValueError(f"analysis id {id!r} registered twice")
        _REGISTRY[id] = AnalysisSpec(
            id=id,
            inputs=tuple(inputs),
            outputs=tuple(outputs or ()),
            params=dict(params or {}),
            func=func,
            source_sha256=source_hash(func),
        )
        return func

    return deco


def register_spec(spec: AnalysisSpec) -> None:
    """Register a pre-built spec (useful in tests)."""
    _REGISTRY[spec.id] = spec


def unregister(id: str) -> None:
    """Remove an analysis from the registry (useful in tests)."""
    _REGISTRY.pop(id, None)


def get_analysis(id: str) -> AnalysisSpec:
    """Look up a registered analysis by id."""
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"unknown analysis {id!r}; known: {sorted(_REGISTRY)}") from None


def all_analyses() -> dict[str, AnalysisSpec]:
    """All registered analyses."""
    return dict(_REGISTRY)


def load_builtin() -> None:
    """Import the built-in analysis modules so their decorators run."""
    for mod in BUILTIN_MODULES:
        importlib.import_module(mod)


def order_analyses(ids: list[str]) -> list[AnalysisSpec]:
    """Return the selected analyses in dependency order.

    An analysis depends on another when one of its inputs is the other's declared output.
    Ties are broken by the order given in ``ids`` so runs are deterministic.
    """
    specs = [get_analysis(i) for i in ids]
    producers: dict[str, str] = {}
    for s in specs:
        for out in s.outputs:
            if out in producers:
                raise ValueError(f"table {out!r} produced by both {producers[out]} and {s.id}")
            producers[out] = s.id
    position = {s.id: n for n, s in enumerate(specs)}
    ts: TopologicalSorter[str] = TopologicalSorter()
    for s in specs:
        deps = [producers[i] for i in s.inputs if i in producers and producers[i] != s.id]
        ts.add(s.id, *deps)
    try:
        ts.prepare()
    except CycleError as e:
        raise ValueError(f"cyclic analysis dependencies: {e.args[1]}") from None
    ordered: list[str] = []
    while ts.is_active():
        ready = sorted(ts.get_ready(), key=position.__getitem__)
        ordered.extend(ready)
        ts.done(*ready)
    by_id = {s.id: s for s in specs}
    return [by_id[i] for i in ordered]
