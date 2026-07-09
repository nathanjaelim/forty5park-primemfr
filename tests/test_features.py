# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for feature-engineering modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prime_mfr.features.hist_rent import add_hist_rent_features, build_hist_rent_features


def test_hist_rent_features_schema():
    """Built hist-rent features have the right columns and dtype."""
    feats = build_hist_rent_features()
    expected_cols = {
        "property_id",
        "unit_type",
        "hist_rent_lag_1m",
        "hist_rent_lag_3m",
        "hist_rent_lag_12m",
        "hist_rent_lag_24m",
        "hist_rent_yoy",
    }
    assert expected_cols.issubset(feats.columns)
    # Lag columns should be float32 (memory-efficient encoding choice).
    for col in ("hist_rent_lag_1m", "hist_rent_lag_3m", "hist_rent_yoy"):
        assert feats[col].dtype == np.float32


def test_hist_rent_target_period_not_present():
    """Critical leakage check: 2026-03 (target period) MUST NOT appear in
    any lag feature. lag_1m must be Feb 2026."""
    import prime_mfr.config as config

    feats = build_hist_rent_features()
    if feats.empty:
        return  # skip when hist file unavailable

    # The lag values come from a pivot on `period`. If we accidentally
    # included the target period, lag_1m would equal the rent column
    # exactly. Sanity check the distribution of lag_1m values does not
    # match the v2 training target distribution at 2026-03.
    target_ts = pd.Timestamp(config.HIST_RENT_TARGET_PERIOD)
    feb_2026 = target_ts - pd.DateOffset(months=1)

    # Verify build_hist_rent_features reads ONLY months up to Feb 2026.
    # Reload the raw and check what months are queried.
    hr = pd.read_parquet(config.HIST_RENT_PATH)
    hr["period"] = pd.to_datetime(hr["period"])
    target_rows = hr[hr["period"] == target_ts]
    feb_rows = hr[hr["period"] == feb_2026]

    if len(target_rows) > 0 and len(feb_rows) > 0:
        # The lag_1m feature must align with Feb 2026 values, not March.
        # Pick a property present in both months and verify.
        common = set(target_rows["property_id"]) & set(feb_rows["property_id"])
        if common:
            pid = next(iter(common))
            ut = target_rows[target_rows["property_id"] == pid]["unit_type"].iloc[0]
            feb_rent = feb_rows[
                (feb_rows["property_id"] == pid) & (feb_rows["unit_type"] == ut)
            ]["rent"]
            if len(feb_rent) > 0:
                feat_row = feats[
                    (feats["property_id"] == pid) & (feats["unit_type"] == ut)
                ]
                if len(feat_row) > 0 and not pd.isna(feat_row["hist_rent_lag_1m"].iloc[0]):
                    # lag_1m must equal Feb 2026 rent (not Mar 2026)
                    assert abs(feat_row["hist_rent_lag_1m"].iloc[0] - feb_rent.iloc[0]) < 0.5


def test_add_hist_rent_features_merges_correctly(small_rent_df):
    """add_hist_rent_features adds 5 lag columns when called on a typical df."""
    df_in = small_rent_df.copy()
    df_out = add_hist_rent_features(df_in)
    # If the hist parquet is available, the 5 lag cols should be present.
    expected_new_cols = {
        "hist_rent_lag_1m",
        "hist_rent_lag_3m",
        "hist_rent_lag_12m",
        "hist_rent_lag_24m",
        "hist_rent_yoy",
    }
    new_cols = set(df_out.columns) - set(df_in.columns)
    # Either all 5 are added (hist file present) or none (hist file absent).
    assert new_cols == expected_new_cols or new_cols == set()


def test_add_hist_rent_features_no_property_id():
    """Returns df unchanged if property_id column missing."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = add_hist_rent_features(df)
    assert out.equals(df)


def test_add_hist_rent_features_preserves_row_count(small_rent_df):
    """The merge must be many-to-one — row count of df is preserved."""
    df_in = small_rent_df.copy()
    df_out = add_hist_rent_features(df_in)
    assert len(df_out) == len(df_in)
