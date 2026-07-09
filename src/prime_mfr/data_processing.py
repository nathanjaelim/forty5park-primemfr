"""
data_processing.py
==================
Load the Yardi pretraining parquet, parse semi-structured columns
(`unit_mix` JSON), normalize types, and produce a clean DataFrame ready
for feature engineering.

Public entry points
-------------------
load_clean()    -> pd.DataFrame
parse_unit_mix(df) -> pd.DataFrame
encode_booleans(df) -> pd.DataFrame
encode_grades(df)   -> pd.DataFrame
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

import prime_mfr.config as config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUE_TOKENS = {"true", "yes", "y", "t", "1", "True", "TRUE", "Yes"}
_FALSE_TOKENS = {"false", "no", "n", "f", "0", "False", "FALSE", "No"}


def _to_bool(value) -> float:
    """Convert mixed truthy/falsy/NaN values to {0.0, 1.0, NaN}."""
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return np.nan
        return 1.0 if value else 0.0
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return np.nan
    if s in _TRUE_TOKENS:
        return 1.0
    if s in _FALSE_TOKENS:
        return 0.0
    return np.nan


def _safe_load_unit_mix(raw) -> list[dict]:
    """Parse unit_mix into a list of dicts. Tolerates JSON, repr-of-list, NaN."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [d for d in raw if isinstance(d, dict)]
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    # Try strict JSON first (cheapest, most common).
    try:
        out = json.loads(s)
    except (ValueError, TypeError):
        # Fall back to Python literal evaluation for reprs with single quotes.
        try:
            out = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    if isinstance(out, dict):
        return [out]
    if isinstance(out, list):
        return [d for d in out if isinstance(d, dict)]
    return []


# ---------------------------------------------------------------------------
# Parsing semi-structured columns
# ---------------------------------------------------------------------------


def parse_unit_mix(df: pd.DataFrame) -> pd.DataFrame:
    """
    The `unit_mix` column is a list of dicts, one per unit type at the
    property. Each row in `df` has a `unit_type` column identifying which
    floorplan applies. We pull that floorplan's sqft / beds / baths /
    num_units onto the row.

    Skipped if columns already populated (idempotent).
    """
    if {"sqft", "beds", "baths"}.issubset(df.columns) and df[
        ["sqft", "beds", "baths"]
    ].notna().any().all():
        # Already enriched.
        return df

    if "unit_mix" not in df.columns:
        return df

    parsed = df["unit_mix"].apply(_safe_load_unit_mix)

    sqft, beds, baths, n_units = [], [], [], []
    for floorplans, ut in zip(parsed, df["unit_type"].astype(str)):
        match = next((fp for fp in floorplans if str(fp.get("unit_type", "")) == ut), None)
        if match is None and floorplans:
            match = floorplans[0]
        sqft.append(_num(match, "sqft"))
        beds.append(_num(match, "beds"))
        baths.append(_num(match, "baths"))
        n_units.append(_num(match, "num_units"))

    df = df.copy()
    df["sqft"] = sqft
    df["beds"] = beds
    df["baths"] = baths
    df["num_units_subtype"] = n_units
    return df


def _num(d: dict | None, key: str) -> float:
    if d is None:
        return np.nan
    v = d.get(key)
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def encode_booleans(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Map boolean-like columns to {0, 1, NaN} float."""
    cols = list(columns) if columns is not None else list(config.BOOLEAN_FEATURES)
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_to_bool).astype("float32")
    return df


def encode_grades(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Map letter-grade columns to ordinal ints (NaN preserved)."""
    cols = list(columns) if columns is not None else list(config.ORDINAL_GRADE_FEATURES)
    df = df.copy()
    mapping = config.GRADE_TO_ORDINAL
    for c in cols:
        if c not in df.columns:
            continue
        ordinal_col = f"{c}_ord"
        df[ordinal_col] = df[c].map(mapping).astype("float32")
    return df


def coerce_categoricals(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Cast columns to pandas 'category' dtype for LightGBM categorical handling."""
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        # Treat NaN as its own category for LightGBM.
        df[c] = df[c].astype("string").fillna("__missing__").astype("category")
    return df


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load Yardi pretraining data and apply universal cleanup steps:
        * parse unit_mix into sqft/beds/baths/num_units_subtype
        * encode boolean-like flags to floats
        * encode letter grades to ordinals
        * drop rows with no rent target

    Returns a DataFrame ready for feature engineering.
    """
    src = Path(path) if path is not None else config.ENRICHED_PARQUET
    if not src.exists():
        # Fallback to raw parquet if enriched isn't there.
        src = config.RAW_PARQUET
    df = pd.read_parquet(src)

    # Parse unit_mix if not already done.
    df = parse_unit_mix(df)

    # Drop rows missing the target.
    df = df[df[config.TARGET].notna()].copy()
    # Drop obvious bad rents (zero or negative).
    df = df[df[config.TARGET] > 0].copy()

    # Boolean + grade encoding.
    df = encode_booleans(df)
    df = encode_grades(df)

    # Property age (clip negative ages from bad year_built rows).
    if "year_built" in df.columns:
        df["property_age"] = (2026 - df["year_built"]).clip(lower=0).astype("float32")

    # Drop the high-cardinality / leaky / id columns at the end so feature
    # engineering still has access to them (e.g. property_id for grouping).
    df.reset_index(drop=True, inplace=True)
    return df
