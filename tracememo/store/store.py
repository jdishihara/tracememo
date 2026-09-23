"""In-memory value store that serializes to and from ``build/manifest.json``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tracememo.store.models import (
    AnalysisRecord,
    AnalysisResult,
    Figure,
    Manifest,
    Provenance,
    Table,
    TableManifestEntry,
    Value,
)


class DuplicateIdError(ValueError):
    """Raised when two analyses produce the same id."""


class UnknownIdError(KeyError):
    """Raised when a referenced id is not in the store."""


class ValueStore:
    """Holds every value, figure and table for a project, keyed by id."""

    def __init__(self, project: str, created: datetime | None = None) -> None:
        self.project = project
        self.created = created or datetime.now(UTC)
        self.values: dict[str, Value] = {}
        self.figures: dict[str, Figure] = {}
        self.tables: dict[str, Table] = {}
        self.input_tables: dict[str, TableManifestEntry] = {}
        self.analyses: dict[str, AnalysisRecord] = {}

    # -- adding ---------------------------------------------------------------

    def _check_new(self, id_: str) -> None:
        if id_ in self.values or id_ in self.figures or id_ in self.tables:
            raise DuplicateIdError(f"id {id_!r} is already in the store")

    def add_value(self, value: Value) -> None:
        """Add a value; raises DuplicateIdError if the id exists."""
        self._check_new(value.id)
        self.values[value.id] = value

    def add_figure(self, figure: Figure) -> None:
        """Add a figure; raises DuplicateIdError if the id exists."""
        self._check_new(figure.id)
        self.figures[figure.id] = figure

    def add_table(self, table: Table) -> None:
        """Add a table; raises DuplicateIdError if the id exists."""
        self._check_new(table.id)
        self.tables[table.id] = table

    def add_result(self, result: AnalysisResult) -> None:
        """Add everything from an analysis result."""
        for v in result.values:
            self.add_value(v)
        for f in result.figures:
            self.add_figure(f)
        for t in result.tables:
            self.add_table(t)

    # -- lookup ---------------------------------------------------------------

    def get_value(self, id_: str) -> Value:
        """Return the value with this id or raise UnknownIdError."""
        try:
            return self.values[id_]
        except KeyError:
            raise UnknownIdError(id_) from None

    def get_figure(self, id_: str) -> Figure:
        """Return the figure with this id or raise UnknownIdError."""
        try:
            return self.figures[id_]
        except KeyError:
            raise UnknownIdError(id_) from None

    def get_table(self, id_: str) -> Table:
        """Return the table with this id or raise UnknownIdError."""
        try:
            return self.tables[id_]
        except KeyError:
            raise UnknownIdError(id_) from None

    def explain(self, id_: str) -> Provenance:
        """Return the provenance of any value, figure or table id."""
        for coll in (self.values, self.figures, self.tables):
            item = coll.get(id_)
            if item is not None:
                if item.provenance is None:
                    raise UnknownIdError(f"{id_} has no provenance recorded")
                return item.provenance
        raise UnknownIdError(id_)

    def all_ids(self) -> set[str]:
        """Every id in the store."""
        return set(self.values) | set(self.figures) | set(self.tables)

    # -- persistence ----------------------------------------------------------

    def to_manifest(self) -> Manifest:
        """Build the manifest model from the store contents."""
        return Manifest(
            project=self.project,
            created=self.created,
            input_tables=dict(self.input_tables),
            analyses=dict(self.analyses),
            values=dict(self.values),
            figures=dict(self.figures),
            tables=dict(self.tables),
        )

    def save(self, path: Path) -> Path:
        """Write ``manifest.json`` and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_manifest().model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> ValueStore:
        """Rebuild a store from a manifest model."""
        store = cls(manifest.project, created=manifest.created)
        store.input_tables = dict(manifest.input_tables)
        store.analyses = dict(manifest.analyses)
        store.values = dict(manifest.values)
        store.figures = dict(manifest.figures)
        store.tables = dict(manifest.tables)
        return store

    @classmethod
    def load(cls, path: Path) -> ValueStore:
        """Read a ``manifest.json`` written by :meth:`save`."""
        manifest = Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
        return cls.from_manifest(manifest)
