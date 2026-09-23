"""Store of normalized input tables written by adapters (``build/tables/``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from tracememo.hashing import hash_dataframe
from tracememo.store.models import InputFile, TableManifestEntry

MANIFEST_NAME = "_tables.json"


class TablesManifest(BaseModel):
    """Serialized index of ingested tables."""

    tables: dict[str, TableManifestEntry] = Field(default_factory=dict)


class TableStore:
    """Reads and writes normalized Parquet tables plus their manifest."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.entries: dict[str, TableManifestEntry] = {}

    @property
    def manifest_path(self) -> Path:
        """Path of the tables manifest file."""
        return self.directory / MANIFEST_NAME

    def put(
        self,
        table_id: str,
        df: pd.DataFrame,
        adapter: str,
        raw_inputs: list[InputFile],
    ) -> TableManifestEntry:
        """Write a table as Parquet and record it in the manifest."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{table_id}.parquet"
        df.to_parquet(path, index=False)
        entry = TableManifestEntry(
            id=table_id,
            path=str(path),
            sha256=hash_dataframe(df),
            rows=len(df),
            columns=[str(c) for c in df.columns],
            adapter=adapter,
            raw_inputs=raw_inputs,
        )
        self.entries[table_id] = entry
        return entry

    def get(self, table_id: str) -> pd.DataFrame:
        """Load a table by id."""
        entry = self.entries.get(table_id)
        if entry is None:
            raise KeyError(f"no ingested table named {table_id!r}; run `tracememo ingest`")
        return pd.read_parquet(entry.path)

    def save(self) -> Path:
        """Write the tables manifest."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            TablesManifest(tables=self.entries).model_dump_json(indent=2), encoding="utf-8"
        )
        return self.manifest_path

    @classmethod
    def load(cls, directory: Path) -> TableStore:
        """Open an existing table store; an empty one if no manifest exists."""
        store = cls(directory)
        if store.manifest_path.exists():
            data = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            store.entries = TablesManifest.model_validate(data).tables
        return store
