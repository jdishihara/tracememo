"""Pydantic models for values, figures, tables, provenance and the build manifest."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ID_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


def validate_id(value: str) -> str:
    """Check that an id is a dotted lowercase identifier such as ``drone.loc_err.mean_cm``."""
    if not ID_PATTERN.match(value):
        raise ValueError(
            f"invalid id {value!r}: expected a dotted lowercase identifier "
            "such as 'drone.loc_err.mean_cm'"
        )
    return value


class InputFile(BaseModel):
    """A raw input file and the SHA-256 of its bytes."""

    path: str
    sha256: str


class TableRef(BaseModel):
    """A normalized input table and its content hash."""

    id: str
    sha256: str


class Provenance(BaseModel):
    """Everything needed to say where a value, figure or table came from."""

    analysis_id: str
    analysis_source_sha256: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    input_tables: list[TableRef] = Field(default_factory=list)
    raw_inputs: list[InputFile] = Field(default_factory=list)
    timestamp: datetime


class Value(BaseModel):
    """A single computed quantity that the report refers to by id."""

    id: str
    value: float | int | str
    unit: str | None = None
    fmt: str = ""
    description: str = ""
    provenance: Provenance | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return validate_id(v)

    @field_validator("value", mode="before")
    @classmethod
    def _unwrap_numpy(cls, v: Any) -> Any:
        # numpy scalars expose .item(); convert them to plain Python numbers.
        if hasattr(v, "item") and not isinstance(v, str | int | float):
            return v.item()
        return v

    def formatted(self) -> str:
        """Render the value using its format spec (strings are returned as-is)."""
        if isinstance(self.value, str):
            return self.value
        return format(self.value, self.fmt)


class Figure(BaseModel):
    """A figure produced by an analysis.

    ``figure`` holds an in-memory matplotlib figure while the analysis result is being
    processed; the runner saves it and fills in ``pdf_path``/``png_path``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    caption: str
    pdf_path: str | None = None
    png_path: str | None = None
    provenance: Provenance | None = None
    figure: Any = Field(default=None, exclude=True, repr=False)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return validate_id(v)


class Table(BaseModel):
    """A table produced by an analysis.

    ``dataframe`` holds the in-memory pandas dataframe; the runner saves it as Parquet
    and fills in ``parquet_path``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    caption: str
    parquet_path: str | None = None
    latex_path: str | None = None
    markdown_path: str | None = None
    columns: list[str] = Field(default_factory=list)
    float_fmt: str = ".4g"
    provenance: Provenance | None = None
    dataframe: Any = Field(default=None, exclude=True, repr=False)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return validate_id(v)


class AnalysisResult(BaseModel):
    """What an analysis function returns."""

    values: list[Value] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class TableManifestEntry(BaseModel):
    """A normalized table written by an adapter, with its hash and raw inputs."""

    id: str
    path: str
    sha256: str
    rows: int
    columns: list[str]
    adapter: str
    raw_inputs: list[InputFile] = Field(default_factory=list)


class AnalysisRecord(BaseModel):
    """Record of one analysis run (or cache hit) in the manifest."""

    id: str
    source_sha256: str
    params: dict[str, Any]
    input_tables: list[TableRef]
    cached: bool
    timestamp: datetime


class Manifest(BaseModel):
    """``build/manifest.json``: the single source of truth for a report."""

    project: str
    created: datetime
    input_tables: dict[str, TableManifestEntry] = Field(default_factory=dict)
    analyses: dict[str, AnalysisRecord] = Field(default_factory=dict)
    values: dict[str, Value] = Field(default_factory=dict)
    figures: dict[str, Figure] = Field(default_factory=dict)
    tables: dict[str, Table] = Field(default_factory=dict)
