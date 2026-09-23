"""Adapters turn raw files into normalized tables. They do no analysis."""

from tracememo.adapters.base import ADAPTERS, Adapter, IngestedTable, get_adapter

__all__ = ["ADAPTERS", "Adapter", "IngestedTable", "get_adapter"]
