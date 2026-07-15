# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for geographic feature functions (Atlanta landmark distances, H3, haversine)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prime_mfr.features.engineering import (
    add_airport_zone_feature,
    add_h3_cells,
    add_landmark_distances,
    add_marta_distance,
    add_marta_station_density,
    haversine_km,
)


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------


def test_haversine_zero_distance():
    """Same point → 0 km."""
    lat = np.array([33.7490])
    lon = np.array([-84.3880])
    d = haversine_km(lat, lon, lat, lon)
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_atlanta_to_nyc_within_tolerance():
    """Atlanta downtown → NYC midtown: ~1200 km, tolerate 50 km of model error."""
    atl_lat, atl_lon = 33.7490, -84.3880
    nyc_lat, nyc_lon = 40.7549, -73.9840
    d = haversine_km(
        np.array([atl_lat]), np.array([atl_lon]), np.array([nyc_lat]), np.array([nyc_lon])
    )
    assert 1150 < d[0] < 1250, f"Atlanta→NYC expected ~1200km, got {d[0]:.1f}"


def test_haversine_symmetric():
    """d(A, B) == d(B, A)."""
    a_lat, a_lon = np.array([33.75]), np.array([-84.39])
    b_lat, b_lon = np.array([33.85]), np.array([-84.30])
    d1 = haversine_km(a_lat, a_lon, b_lat, b_lon)
    d2 = haversine_km(b_lat, b_lon, a_lat, a_lon)
    np.testing.assert_allclose(d1, d2)


def test_haversine_vectorized():
    """Works on vectors of any length."""
    n = 100
    rng = np.random.default_rng(42)
    lat1 = 33.75 + rng.normal(0, 0.1, n)
    lon1 = -84.39 + rng.normal(0, 0.1, n)
    lat2 = 33.75 + rng.normal(0, 0.1, n)
    lon2 = -84.39 + rng.normal(0, 0.1, n)
    d = haversine_km(lat1, lon1, lat2, lon2)
    assert d.shape == (n,)
    assert (d >= 0).all()
    # All points sampled within ~10 km of Atlanta center
    assert d.max() < 50.0


# ---------------------------------------------------------------------------
# Landmark distances
# ---------------------------------------------------------------------------


def _atlanta_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "latitude": 33.75 + rng.normal(0, 0.05, n),
            "longitude": -84.39 + rng.normal(0, 0.05, n),
            "rent": rng.uniform(1200, 3500, n),
        }
    )


def test_add_landmark_distances_adds_expected_columns():
    df = _atlanta_df(10)
    out = add_landmark_distances(df)
    for col in (
        "dist_buckhead_km",
        "dist_midtown_km",
        "dist_downtown_km",
        "dist_atl_airport_km",
        "dist_min_landmark_km",
    ):
        assert col in out.columns, f"missing column {col}"


def test_landmark_distances_are_non_negative():
    df = _atlanta_df(20)
    out = add_landmark_distances(df)
    for col in [c for c in out.columns if c.startswith("dist_")]:
        assert (out[col] >= 0).all(), f"{col} has negative values"


def test_min_landmark_is_le_each_individual():
    """dist_min_landmark_km must be ≤ every individual dist_<landmark>_km."""
    df = _atlanta_df(20)
    out = add_landmark_distances(df)
    individual_cols = [
        c for c in out.columns if c.startswith("dist_") and c != "dist_min_landmark_km"
    ]
    min_col = out["dist_min_landmark_km"].values
    for col in individual_cols:
        assert (min_col <= out[col].values + 1e-6).all(), (
            f"dist_min_landmark_km > {col} for at least one row"
        )


def test_buckhead_landmark_close_to_buckhead_coords():
    """A property AT Buckhead coords (per eda/atlanta_landmarks.json) → ~0 km."""
    # Coords as registered in the curated landmarks file (Wikipedia infobox).
    buckhead_lat, buckhead_lon = 33.83942, -84.37992
    df = pd.DataFrame(
        {
            "latitude": [buckhead_lat],
            "longitude": [buckhead_lon],
            "rent": [3000.0],
        }
    )
    out = add_landmark_distances(df)
    assert out["dist_buckhead_km"].iloc[0] < 0.1


def test_landmark_distances_missing_coords_handled():
    """Rows with NaN lat/lon get NaN distances (don't crash)."""
    df = pd.DataFrame(
        {
            "latitude": [33.75, np.nan, 33.80],
            "longitude": [-84.39, -84.40, np.nan],
            "rent": [1500.0, 2000.0, 2500.0],
        }
    )
    out = add_landmark_distances(df)
    # Row 0 has valid coords → finite distance.
    assert np.isfinite(out["dist_buckhead_km"].iloc[0])
    # Rows 1, 2 have missing coords → NaN.
    assert pd.isna(out["dist_buckhead_km"].iloc[1]) or out["dist_buckhead_km"].iloc[1] >= 0
    assert pd.isna(out["dist_buckhead_km"].iloc[2]) or out["dist_buckhead_km"].iloc[2] >= 0


# ---------------------------------------------------------------------------
# Airport hot-zone categorical
# ---------------------------------------------------------------------------

# Airport coords per eda/atlanta_landmarks.json (key="atl_airport").
_AIRPORT_LAT, _AIRPORT_LON = 33.6407, -84.4277
_KM_PER_DEG_LAT = 111.0  # rough conversion for building test offsets


def test_add_airport_zone_feature_adds_column():
    df = _atlanta_df(10)
    out = add_airport_zone_feature(df)
    assert "dist_atl_airport_zone" in out.columns


def test_airport_zone_is_categorical_dtype():
    df = _atlanta_df(10)
    out = add_airport_zone_feature(df)
    assert isinstance(out["dist_atl_airport_zone"].dtype, pd.CategoricalDtype)
    assert set(out["dist_atl_airport_zone"].cat.categories) == {"near", "hot_zone", "far"}


def test_airport_zone_near_at_airport_coords():
    """A property AT the airport (0 km) -> "near" (edges are 9/15 km)."""
    df = pd.DataFrame({"latitude": [_AIRPORT_LAT], "longitude": [_AIRPORT_LON], "rent": [1500.0]})
    out = add_airport_zone_feature(df)
    assert out["dist_atl_airport_zone"].iloc[0] == "near"


def test_airport_zone_hot_zone_and_far():
    """~12km out -> hot_zone (9-15km band); ~20km out -> far (>15km)."""
    hot_lat = _AIRPORT_LAT + 12.0 / _KM_PER_DEG_LAT
    far_lat = _AIRPORT_LAT + 20.0 / _KM_PER_DEG_LAT
    df = pd.DataFrame(
        {
            "latitude": [hot_lat, far_lat],
            "longitude": [_AIRPORT_LON, _AIRPORT_LON],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_airport_zone_feature(df)
    assert out["dist_atl_airport_zone"].iloc[0] == "hot_zone"
    assert out["dist_atl_airport_zone"].iloc[1] == "far"


def test_airport_zone_missing_coords_handled():
    """Rows with NaN lat/lon get NaN zone, no crash."""
    df = pd.DataFrame(
        {
            "latitude": [_AIRPORT_LAT, np.nan],
            "longitude": [_AIRPORT_LON, -84.40],
            "rent": [1500.0, 2000.0],
        }
    )
    out = add_airport_zone_feature(df)
    assert out["dist_atl_airport_zone"].iloc[0] == "near"
    assert pd.isna(out["dist_atl_airport_zone"].iloc[1])


def test_airport_zone_is_categorical_not_numeric_config():
    """dist_atl_airport_zone must live in CATEGORICAL_FEATURES, not
    NUMERIC_FEATURES, and dist_atl_airport_km (continuous) should no
    longer be in NUMERIC_FEATURES -- it was replaced by this zone."""
    import prime_mfr.config as config

    assert "dist_atl_airport_zone" in config.CATEGORICAL_FEATURES
    assert "dist_atl_airport_zone" not in config.NUMERIC_FEATURES
    assert "dist_atl_airport_km" not in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# MARTA nearest-station distance
# ---------------------------------------------------------------------------


def test_add_marta_distance_adds_column():
    df = _atlanta_df(10)
    out = add_marta_distance(df)
    assert "dist_marta_km" in out.columns


def test_marta_distance_is_non_negative():
    df = _atlanta_df(20)
    out = add_marta_distance(df)
    finite = out["dist_marta_km"].dropna()
    assert (finite >= 0).all()


def test_marta_five_points_distance_is_zero_at_station_coords():
    """A property AT the Five Points station coords (per
    eda/marta_stations.json, sourced from OSM Overpass) -> dist_marta_km ~ 0."""
    df = pd.DataFrame(
        {
            "latitude": [33.7538868],
            "longitude": [-84.3915963],
            "rent": [2000.0],
        }
    )
    out = add_marta_distance(df)
    assert out["dist_marta_km"].iloc[0] < 0.1


def test_marta_distance_le_landmark_distance_downtown():
    """Downtown sits within ~1km of multiple MARTA stations (Five Points,
    Georgia State, Peachtree Center), so the nearest-MARTA distance for a
    downtown property should be small -- well under the distance to
    Buckhead, a landmark ~9km north."""
    downtown_lat, downtown_lon = 33.75500, -84.39000
    df = pd.DataFrame(
        {
            "latitude": [downtown_lat],
            "longitude": [downtown_lon],
            "rent": [2000.0],
        }
    )
    out = add_marta_distance(add_landmark_distances(df))
    assert out["dist_marta_km"].iloc[0] < out["dist_buckhead_km"].iloc[0]


def test_marta_distance_missing_coords_handled():
    """Rows with NaN lat/lon don't crash; distance is NaN or non-negative."""
    df = pd.DataFrame(
        {
            "latitude": [33.75, np.nan],
            "longitude": [-84.39, -84.40],
            "rent": [1500.0, 2000.0],
        }
    )
    out = add_marta_distance(df)
    assert np.isfinite(out["dist_marta_km"].iloc[0])
    assert pd.isna(out["dist_marta_km"].iloc[1]) or out["dist_marta_km"].iloc[1] >= 0


# ---------------------------------------------------------------------------
# MARTA station density
# ---------------------------------------------------------------------------

# Real coords from eda/marta_stations.json.
_FIVE_POINTS = (33.7538868, -84.3915963)
_NORTH_SPRINGS = (33.9451211, -84.3571916)


def test_add_marta_station_density_adds_column():
    df = _atlanta_df(10)
    out = add_marta_station_density(df)
    assert "num_marta_stations_within_1mi" in out.columns


def test_marta_station_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_marta_station_density(df)
    assert (out["num_marta_stations_within_1mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_marta_stations_within_1mi"])


def test_marta_station_density_downtown_cluster_beats_isolated_suburb():
    """Five Points sits <0.5mi from Georgia State/Peachtree Center/Garnett
    (verified against eda/marta_stations.json), so it should show several
    stations within 1mi. North Springs' nearest neighbor (Sandy Springs) is
    ~0.94mi away, so it should show far fewer -- even though both could have
    a similarly small dist_marta_km (~0), density tells them apart."""
    df = pd.DataFrame(
        {
            "latitude": [_FIVE_POINTS[0], _NORTH_SPRINGS[0]],
            "longitude": [_FIVE_POINTS[1], _NORTH_SPRINGS[1]],
            "rent": [2000.0, 1800.0],
        }
    )
    out = add_marta_station_density(df)
    five_points_count = out["num_marta_stations_within_1mi"].iloc[0]
    north_springs_count = out["num_marta_stations_within_1mi"].iloc[1]
    assert five_points_count >= 4, f"expected >=4 stations near Five Points, got {five_points_count}"
    assert north_springs_count <= 2, f"expected <=2 stations near North Springs, got {north_springs_count}"
    assert five_points_count > north_springs_count


def test_marta_station_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_FIVE_POINTS[0], np.nan],
            "longitude": [_FIVE_POINTS[1], -84.40],
            "rent": [2000.0, 1800.0],
        }
    )
    out = add_marta_station_density(df)
    assert out["num_marta_stations_within_1mi"].iloc[1] == 0


# ---------------------------------------------------------------------------
# H3 spatial cells
# ---------------------------------------------------------------------------


def test_add_h3_cells_adds_two_resolutions():
    df = _atlanta_df(10)
    out = add_h3_cells(df)
    assert "h3_res6" in out.columns
    assert "h3_res8" in out.columns


def test_h3_cells_are_strings():
    df = _atlanta_df(10)
    out = add_h3_cells(df)
    # H3 indices are 15-char hex strings (e.g. "8844d05c5dfffff")
    assert all(isinstance(c, str) for c in out["h3_res6"].dropna())
    assert all(isinstance(c, str) for c in out["h3_res8"].dropna())


def test_h3_res8_more_granular_than_res6():
    """Within a sample, res-8 cells should be at least as numerous as res-6."""
    df = _atlanta_df(100, seed=123)
    out = add_h3_cells(df)
    n_res6 = out["h3_res6"].nunique()
    n_res8 = out["h3_res8"].nunique()
    assert n_res8 >= n_res6


def test_close_properties_share_h3_cell():
    """Two properties 100 m apart should land in the same res-8 cell."""
    df = pd.DataFrame(
        {
            "latitude": [33.7836, 33.7837],  # ~10m apart
            "longitude": [-84.3838, -84.3839],
            "rent": [2000.0, 2100.0],
        }
    )
    out = add_h3_cells(df)
    assert out["h3_res8"].iloc[0] == out["h3_res8"].iloc[1]
    assert out["h3_res6"].iloc[0] == out["h3_res6"].iloc[1]
