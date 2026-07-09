# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_rent_df() -> pd.DataFrame:
    """A tiny synthetic dataframe mimicking the schema after data_processing."""
    rng = np.random.default_rng(42)
    n = 24
    return pd.DataFrame(
        {
            "property_id": [f"P{i // 4:03d}" for i in range(n)],
            "unit_type": (["1BR/1.0", "2BR/2.0", "3BR/2.0"] * 8)[:n],
            "rent": rng.uniform(1200, 3500, n).round(0),
            "sqft": rng.uniform(600, 1800, n).round(0),
            "beds": rng.integers(1, 4, n),
            "baths": rng.uniform(1.0, 2.5, n).round(1),
            "year_built": rng.integers(1990, 2024, n),
            "latitude": 33.75 + rng.normal(0, 0.05, n),
            "longitude": -84.39 + rng.normal(0, 0.05, n),
        }
    )
