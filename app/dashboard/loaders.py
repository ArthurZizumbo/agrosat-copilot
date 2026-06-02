"""Loaders cacheados con degradacion graceful.

Centraliza la lectura de CSV, YAML y parquet usados por las distintas
secciones del dashboard. Cada loader devuelve una estructura vacia cuando el
archivo no existe o no se puede leer, de modo que las secciones degradan sin
levantar traceback. El cache de Streamlit (``@st.cache_data``) evita releer
el mismo artefacto entre re-renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import yaml


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pl.DataFrame:
    """Carga un CSV como ``polars.DataFrame`` con cache de Streamlit.

    Args:
        path: Ruta absoluta o relativa al CSV en disco.

    Returns:
        DataFrame de Polars. Vacio si la lectura falla o el archivo no existe.
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
    """Carga un YAML como diccionario con cache de Streamlit.

    Args:
        path: Ruta al archivo YAML.

    Returns:
        Diccionario parseado. Vacio si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pl.DataFrame:
    """Carga un parquet como ``polars.DataFrame`` con cache de Streamlit.

    Args:
        path: Ruta absoluta o relativa al parquet en disco.

    Returns:
        DataFrame de Polars. Vacio si la lectura falla o el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except (pl.exceptions.ComputeError, OSError, ValueError):
        return pl.DataFrame()


def list_csvs(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """Lista CSVs ordenados alfabeticamente filtrando por glob.

    Args:
        directory: Directorio donde buscar.
        pattern: Glob de filtrado (por defecto ``*.csv``).

    Returns:
        Lista de paths ordenada. Vacia si el directorio no existe.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))
