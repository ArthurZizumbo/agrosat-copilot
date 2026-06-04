"""Cached loaders with graceful degradation.

Centralizes the reading of CSV, YAML and parquet used by the different
dashboard sections. Each loader returns an empty structure when the
file does not exist or cannot be read, so the sections degrade without
raising a traceback. The Streamlit cache (``@st.cache_data``) avoids
re-reading the same artifact between re-renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import yaml


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pl.DataFrame:
    """Loads a CSV as a ``polars.DataFrame`` with Streamlit cache.

    Args:
        path: Absolute or relative path to the CSV on disk.

    Returns:
        Polars DataFrame. Empty if the read fails or the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_csv(path)
    except (pl.exceptions.ComputeError, OSError, ValueError):
        return pl.DataFrame()


@st.cache_data(show_spinner=False)
def load_yaml(path: Path) -> dict[str, Any]:
    """Loads a YAML as a dictionary with Streamlit cache.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary. Empty if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pl.DataFrame:
    """Loads a parquet as a ``polars.DataFrame`` with Streamlit cache.

    Args:
        path: Absolute or relative path to the parquet on disk.

    Returns:
        Polars DataFrame. Empty if the read fails or the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except (pl.exceptions.ComputeError, OSError, ValueError):
        return pl.DataFrame()


def list_csvs(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """Lists CSVs sorted alphabetically, filtering by glob.

    Args:
        directory: Directory to search in.
        pattern: Filtering glob (default ``*.csv``).

    Returns:
        Sorted list of paths. Empty if the directory does not exist.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))
