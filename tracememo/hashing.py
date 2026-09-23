"""Content hashing helpers used for provenance and caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a string encoded as UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes, streamed in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a content hash of a dataframe: column names, dtypes and row hashes.

    This is stable across runs and independent of Parquet encoding details.
    """
    h = hashlib.sha256()
    h.update(repr(list(map(str, df.columns))).encode("utf-8"))
    h.update(repr([str(d) for d in df.dtypes]).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    h.update(row_hashes.to_numpy().tobytes())
    return h.hexdigest()


def hash_params(params: dict[str, Any]) -> str:
    """Return a hash of a parameter dict, independent of key order."""
    return sha256_text(json.dumps(params, sort_keys=True, default=str))
