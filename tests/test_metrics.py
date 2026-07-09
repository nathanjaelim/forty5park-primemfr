# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for metric computation and log-target transforms."""

from __future__ import annotations

import numpy as np

from prime_mfr.train import back_transform_to_rent, compute_log_target, compute_metrics


def test_compute_metrics_perfect_prediction():
    """A perfect prediction yields MAE=0, MAPE=0, R^2=1."""
    y_true = np.array([1500.0, 2000.0, 2500.0, 3000.0])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred)
    assert m["MAE"] == 0.0
    assert m["MAPE"] == 0.0
    assert m["MedianAPE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"] == 1.0


def test_compute_metrics_constant_offset():
    """Constant $100 over-prediction → MAE=$100, MAPE matches expected, R²<1."""
    y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y_pred = y_true + 100.0
    m = compute_metrics(y_true, y_pred)
    assert m["MAE"] == 100.0
    # Mean of |100|/y_true * 100 = mean(10, 5, 3.33, 2.5)% = 5.208%
    expected_mape = np.mean(100.0 / y_true * 100)
    assert abs(m["MAPE"] - expected_mape) < 1e-6
    # MedianAPE is the median of the per-row APE distribution.
    expected_medape = np.median(100.0 / y_true * 100)
    assert abs(m["MedianAPE"] - expected_medape) < 1e-6
    assert m["R2"] < 1.0  # imperfect


def test_compute_metrics_keys():
    """Returned dict has all expected keys."""
    m = compute_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    for k in ("MAE", "MAPE", "MedianAPE", "RMSE", "MedianAE", "R2"):
        assert k in m


def test_log_target_roundtrip():
    """log1p(rent) → expm1 should recover the original rent (with clipping)."""
    import pandas as pd

    rent = np.array([800.0, 1500.0, 3000.0, 5000.0])
    df = pd.DataFrame({"rent": rent, "sqft": [600.0, 900.0, 1500.0, 2000.0]})
    log_target = compute_log_target(df, use_psf=False)
    recovered = back_transform_to_rent(log_target, use_psf=False)
    np.testing.assert_allclose(recovered, rent, rtol=1e-6)


def test_back_transform_clips_low_predictions():
    """back_transform_to_rent clips outputs at the $100 floor (sanity guard)."""
    # log(0) maps to expm1(0)=0; clipped to 100.
    very_negative_log = np.array([-10.0, -5.0, 0.0])
    out = back_transform_to_rent(very_negative_log, use_psf=False)
    assert (out >= 100.0).all()
