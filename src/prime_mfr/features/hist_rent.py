# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""
Historical-rent lag features.
=============================

Reads the 24-month hist-rent panel (Yardi Matrix `042026-hist-rent-12060.parquet`)
and pivots it to one row per (property_id, unit_type) with five lag features
anchored at the training period (2026-03-01):

    hist_rent_lag_1m   - Feb 2026 rent for the same (property, unit_type)
    hist_rent_lag_3m   - Dec 2025
    hist_rent_lag_12m  - Mar 2025
    hist_rent_lag_24m  - Mar 2024
    hist_rent_yoy      - lag_1m / lag_13m - 1   (Feb 2026 / Feb 2025 - 1)

CRITICAL leakage rule: the target month (HIST_RENT_TARGET_PERIOD = 2026-03-01)
must NEVER appear in any feature. The smallest valid lag is 1 month.

The 700k-row panel is read once per process and cached at module scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import prime_mfr.config as config

__all__ = ["build_hist_rent_features", "add_hist_rent_features"]


# Module-level cache so we only read the 700k-row hist parquet once per process.
_HIST_RENT_FEATURES_CACHE: pd.DataFrame | None = None


def build_hist_rent_features() -> pd.DataFrame:
    """Build the (property_id, unit_type) -> lag features lookup table.

    Returns a DataFrame with columns:
        property_id, unit_type, hist_rent_lag_1m, hist_rent_lag_3m,
        hist_rent_lag_12m, hist_rent_lag_24m, hist_rent_yoy
    """
    if not config.HIST_RENT_PATH.exists():
        return pd.DataFrame(
            columns=[
                "property_id",
                "unit_type",
                "hist_rent_lag_1m",
                "hist_rent_lag_3m",
                "hist_rent_lag_12m",
                "hist_rent_lag_24m",
                "hist_rent_yoy",
            ]
        )

    hr = pd.read_parquet(config.HIST_RENT_PATH)
    hr["period"] = pd.to_datetime(hr["period"])
    hr = hr[["property_id", "unit_type", "period", "rent"]].dropna(subset=["rent"])

    target = pd.Timestamp(config.HIST_RENT_TARGET_PERIOD)
    months = {
        "hist_rent_lag_1m": target - pd.DateOffset(months=1),  # Feb 2026
        "hist_rent_lag_3m": target - pd.DateOffset(months=3),  # Dec 2025
        "hist_rent_lag_12m": target - pd.DateOffset(months=12),  # Mar 2025
        "hist_rent_lag_24m": target - pd.DateOffset(months=24),  # Mar 2024
        # auxiliary point used only for yoy denominator (Feb 2025)
        "_lag_13m": target - pd.DateOffset(months=13),
    }

    needed = list(months.values())
    sub = hr[hr["period"].isin(needed)].copy()
    wide = sub.pivot_table(
        index=["property_id", "unit_type"],
        columns="period",
        values="rent",
        aggfunc="first",
    ).reset_index()

    for name, ts in months.items():
        if ts in wide.columns:
            wide[name] = wide[ts].astype("float32")
        else:
            wide[name] = np.nan
        wide.drop(columns=[ts], errors="ignore", inplace=True)

    # YoY: Feb 2026 / Feb 2025 - 1. Guard against zero / null denom.
    denom = wide["_lag_13m"]
    num = wide["hist_rent_lag_1m"]
    safe = denom.where((denom > 0), np.nan)
    wide["hist_rent_yoy"] = (num / safe - 1.0).astype("float32")

    keep = [
        "property_id",
        "unit_type",
        "hist_rent_lag_1m",
        "hist_rent_lag_3m",
        "hist_rent_lag_12m",
        "hist_rent_lag_24m",
        "hist_rent_yoy",
    ]
    return wide[keep]


def add_hist_rent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge hist-rent lag features onto df by (property_id, unit_type).

    Features are at (property, unit_type) granularity, not unit-level. When
    multiple unit-level rows share a (property, unit_type), they get
    identical lag values — the hist series is one representative rent track
    per (property, unit_type).
    """
    if "property_id" not in df.columns or "unit_type" not in df.columns:
        return df

    global _HIST_RENT_FEATURES_CACHE
    if _HIST_RENT_FEATURES_CACHE is None:
        _HIST_RENT_FEATURES_CACHE = build_hist_rent_features()
    feats = _HIST_RENT_FEATURES_CACHE
    if feats.empty:
        return df

    df = df.copy()
    df["property_id"] = df["property_id"].astype(str)
    df["unit_type"] = df["unit_type"].astype(str)
    feats = feats.copy()
    feats["property_id"] = feats["property_id"].astype(str)
    feats["unit_type"] = feats["unit_type"].astype(str)

    return df.merge(feats, on=["property_id", "unit_type"], how="left", validate="many_to_one")
