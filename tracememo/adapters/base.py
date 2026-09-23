"""Adapter interface and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tracememo.store.models import InputFile


@dataclass
class IngestedTable:
    """A normalized table plus the raw files it was derived from."""

    id: str
    df: pd.DataFrame
    raw_inputs: list[InputFile] = field(default_factory=list)


Adapter = Callable[[Path, list[str] | None, dict[str, Any]], list[IngestedTable]]
"""An adapter is ``adapter(path, tables, options) -> list[IngestedTable]``."""

ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str) -> Callable[[Adapter], Adapter]:
    """Decorator registering an adapter under ``name`` for use in ``project.yaml``."""

    def deco(fn: Adapter) -> Adapter:
        ADAPTERS[name] = fn
        return fn

    return deco


def get_adapter(name: str) -> Adapter:
    """Look up an adapter by name, importing the built-in ones first."""
    import tracememo.adapters.normalized  # noqa: F401  (registers on import)

    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)}") from None
