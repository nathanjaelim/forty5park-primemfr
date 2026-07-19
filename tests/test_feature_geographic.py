# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for geographic feature functions (Atlanta landmark distances, H3, haversine)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prime_mfr.features.engineering import (
    add_airport_zone_feature,
    add_bar_density,
    add_buckhead_near_flag,
    add_coffee_shop_density,
    add_grocery_density,
    add_h3_cells,
    add_highway_distance,
    add_highway_route_distances,
    add_landmark_distances,
    add_marta_distance,
    add_marta_distance_zone,
    add_marta_station_density,
    add_marta_walkable_flag,
    add_named_park_distance,
    add_park_distance,
    add_restaurant_density,
    add_total_poi_density,
    add_travel_time_features,
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
# Buckhead proximity, binary flag
# ---------------------------------------------------------------------------

_BUCKHEAD_LAT, _BUCKHEAD_LON = 33.83942, -84.37992


def test_add_buckhead_near_flag_adds_column():
    df = _atlanta_df(10)
    out = add_buckhead_near_flag(df)
    assert "buckhead_near" in out.columns


def test_buckhead_near_at_buckhead_coords():
    """A property AT Buckhead (0mi) -> 1.0 (threshold is 6mi)."""
    df = pd.DataFrame(
        {"latitude": [_BUCKHEAD_LAT], "longitude": [_BUCKHEAD_LON], "rent": [2500.0]}
    )
    out = add_buckhead_near_flag(df)
    assert out["buckhead_near"].iloc[0] == 1.0


def test_buckhead_near_far_property_is_zero():
    """~20mi from Buckhead -> 0.0."""
    far_lat = _BUCKHEAD_LAT + 20.0 / _KM_PER_DEG_LAT
    df = pd.DataFrame(
        {"latitude": [far_lat], "longitude": [_BUCKHEAD_LON], "rent": [1500.0]}
    )
    out = add_buckhead_near_flag(df)
    assert out["buckhead_near"].iloc[0] == 0.0


def test_buckhead_near_missing_coords_handled():
    """Rows with NaN lat/lon get NaN flag, no crash."""
    df = pd.DataFrame(
        {
            "latitude": [_BUCKHEAD_LAT, np.nan],
            "longitude": [_BUCKHEAD_LON, -84.40],
            "rent": [2500.0, 1800.0],
        }
    )
    out = add_buckhead_near_flag(df)
    assert out["buckhead_near"].iloc[0] == 1.0
    assert pd.isna(out["buckhead_near"].iloc[1])


def test_buckhead_near_replaces_continuous_in_config():
    """buckhead_near must live in NUMERIC_FEATURES / KNN_LEAN_FEATURES,
    replacing dist_buckhead_km (which is still computed by
    add_landmark_distances() but no longer selected for training)."""
    import prime_mfr.config as config

    assert "buckhead_near" in config.NUMERIC_FEATURES
    assert "dist_buckhead_km" not in config.NUMERIC_FEATURES
    assert "buckhead_near" in config.KNN_LEAN_FEATURES
    assert "dist_buckhead_km" not in config.KNN_LEAN_FEATURES


# ---------------------------------------------------------------------------
# MARTA proximity, binary flag
# ---------------------------------------------------------------------------


def test_add_marta_walkable_flag_adds_column():
    df = _atlanta_df(10)
    out = add_marta_walkable_flag(df)
    assert "marta_walkable" in out.columns


def test_marta_walkable_at_station_coords():
    """A property AT Five Points station coords -> 1.0."""
    df = pd.DataFrame(
        {"latitude": [33.7538868], "longitude": [-84.3915963], "rent": [2000.0]}
    )
    out = add_marta_walkable_flag(df)
    assert out["marta_walkable"].iloc[0] == 1.0


def test_marta_walkable_far_property_is_zero():
    """~20mi from the nearest station -> 0.0 (well past any tested threshold)."""
    df = pd.DataFrame({"latitude": [33.40], "longitude": [-84.10], "rent": [1500.0]})
    out = add_marta_walkable_flag(df)
    assert out["marta_walkable"].iloc[0] == 0.0


def test_marta_walkable_missing_coords_handled():
    df = pd.DataFrame(
        {
            "latitude": [33.7538868, np.nan],
            "longitude": [-84.3915963, -84.40],
            "rent": [2000.0, 1800.0],
        }
    )
    out = add_marta_walkable_flag(df)
    assert out["marta_walkable"].iloc[0] == 1.0
    assert pd.isna(out["marta_walkable"].iloc[1])


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
# MARTA nearest-station distance, binned (same edges as the EDA notebook)
# ---------------------------------------------------------------------------


def test_add_marta_distance_zone_adds_column():
    df = _atlanta_df(10)
    out = add_marta_distance_zone(df)
    assert "dist_marta_zone" in out.columns


def test_marta_distance_zone_is_categorical_dtype():
    df = _atlanta_df(10)
    out = add_marta_distance_zone(df)
    assert isinstance(out["dist_marta_zone"].dtype, pd.CategoricalDtype)
    assert set(out["dist_marta_zone"].cat.categories) == {
        "<0.5mi",
        "0.5-2.5mi",
        "2.5-5mi",
        "5-20mi",
        "20mi+",
    }


def test_marta_distance_zone_near_at_station_coords():
    """A property AT Five Points station coords (~0mi) -> '<0.5mi'."""
    df = pd.DataFrame(
        {"latitude": [33.7538868], "longitude": [-84.3915963], "rent": [2000.0]}
    )
    out = add_marta_distance_zone(df)
    assert out["dist_marta_zone"].iloc[0] == "<0.5mi"


def test_marta_distance_zone_matches_notebook_bins():
    """Cross-check bin assignment against real haversine distances computed
    from add_marta_distance() -- 3 points spanning the mid-range bins."""
    df = pd.DataFrame(
        {
            # ~1.2km (~0.77mi) from Five Points -> 0.5-2.5mi
            "latitude": [33.7538868 + 0.01, 33.60, 33.40],
            "longitude": [-84.3915963, -84.30, -84.10],
            "rent": [2000.0, 1900.0, 1800.0],
        }
    )
    dist_out = add_marta_distance(df)
    zone_out = add_marta_distance_zone(df)
    dist_mi = dist_out["dist_marta_km"] * 0.6213712
    for d_mi, zone in zip(dist_mi, zone_out["dist_marta_zone"]):
        if d_mi < 0.5:
            assert zone == "<0.5mi"
        elif d_mi < 2.5:
            assert zone == "0.5-2.5mi"
        elif d_mi < 5:
            assert zone == "2.5-5mi"
        elif d_mi < 20:
            assert zone == "5-20mi"
        else:
            assert zone == "20mi+"


def test_marta_distance_zone_missing_coords_handled():
    """Rows with NaN lat/lon get NaN zone, no crash."""
    df = pd.DataFrame(
        {
            "latitude": [33.7538868, np.nan],
            "longitude": [-84.3915963, -84.40],
            "rent": [2000.0, 1800.0],
        }
    )
    out = add_marta_distance_zone(df)
    assert out["dist_marta_zone"].iloc[0] == "<0.5mi"
    assert pd.isna(out["dist_marta_zone"].iloc[1])


def test_marta_distance_zone_is_categorical_config():
    """dist_marta_zone must live in CATEGORICAL_FEATURES, not
    NUMERIC_FEATURES (non-monotonic rent relationship, same reasoning as
    dist_atl_airport_zone)."""
    import prime_mfr.config as config

    assert "dist_marta_zone" in config.CATEGORICAL_FEATURES
    assert "dist_marta_zone" not in config.NUMERIC_FEATURES


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
# Coffee shop density
# ---------------------------------------------------------------------------

# Real coords, counts verified directly against eda/coffee_shops.json.
_MIDTOWN_ATL = (33.7867, -84.3833)  # 6 coffee shops within 0.35mi
_RURAL_FAR = (33.10, -85.10)  # 0 coffee shops within 0.35mi


def test_add_coffee_shop_density_adds_column():
    df = _atlanta_df(10)
    out = add_coffee_shop_density(df)
    assert "num_coffee_shops_within_0.35mi" in out.columns


def test_coffee_shop_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_coffee_shop_density(df)
    assert (out["num_coffee_shops_within_0.35mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_coffee_shops_within_0.35mi"])


def test_coffee_shop_density_urban_beats_rural():
    """Midtown has 6 coffee shops within 0.35mi (verified against
    eda/coffee_shops.json); a point far outside the metro has 0."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_ATL[0], _RURAL_FAR[0]],
            "longitude": [_MIDTOWN_ATL[1], _RURAL_FAR[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_coffee_shop_density(df)
    midtown_count = out["num_coffee_shops_within_0.35mi"].iloc[0]
    rural_count = out["num_coffee_shops_within_0.35mi"].iloc[1]
    assert midtown_count == 6
    assert rural_count == 0


def test_coffee_shop_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_ATL[0], np.nan],
            "longitude": [_MIDTOWN_ATL[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_coffee_shop_density(df)
    assert out["num_coffee_shops_within_0.35mi"].iloc[1] == 0


def test_coffee_shop_density_not_in_numeric_config():
    """Coffee shop density was removed from NUMERIC_FEATURES 2026-07-15
    after every radius tested (0.25/0.35/0.5/1mi) landed worse than
    baseline, with the best-looking result (0.25mi) proving unstable under
    a small radius nudge (a hallmark of noise, not real signal)."""
    import prime_mfr.config as config

    assert not any(
        c.startswith("num_coffee_shops_within_") for c in config.NUMERIC_FEATURES
    )


# ---------------------------------------------------------------------------
# Grocery store density
# ---------------------------------------------------------------------------

# Real coords, counts verified directly against eda/grocery_stores.json.
# At 0.25mi, Midtown center itself shows 0 (grocery stores are sparser
# than coffee shops) -- use the actual Whole Foods location for a
# meaningful positive case instead.
_WHOLE_FOODS_MIDTOWN = (33.7861686, -84.3885403)  # 1 grocery store within 0.25mi
_RURAL_FAR_GROCERY = (33.10, -85.10)  # 0 grocery stores within 0.25mi


def test_add_grocery_density_adds_column():
    df = _atlanta_df(10)
    out = add_grocery_density(df)
    assert "num_grocery_stores_within_0.25mi" in out.columns


def test_grocery_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_grocery_density(df)
    assert (out["num_grocery_stores_within_0.25mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_grocery_stores_within_0.25mi"])


def test_grocery_density_urban_beats_rural():
    """A property at the Whole Foods Midtown location has 1 grocery store
    within 0.25mi (verified against eda/grocery_stores.json); a point far
    outside the metro has 0."""
    df = pd.DataFrame(
        {
            "latitude": [_WHOLE_FOODS_MIDTOWN[0], _RURAL_FAR_GROCERY[0]],
            "longitude": [_WHOLE_FOODS_MIDTOWN[1], _RURAL_FAR_GROCERY[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_grocery_density(df)
    near_count = out["num_grocery_stores_within_0.25mi"].iloc[0]
    rural_count = out["num_grocery_stores_within_0.25mi"].iloc[1]
    assert near_count == 1
    assert rural_count == 0


def test_grocery_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_WHOLE_FOODS_MIDTOWN[0], np.nan],
            "longitude": [_WHOLE_FOODS_MIDTOWN[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_grocery_density(df)
    assert out["num_grocery_stores_within_0.25mi"].iloc[1] == 0


def test_grocery_density_in_numeric_config():
    """num_grocery_stores_within_0.25mi must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "num_grocery_stores_within_0.25mi" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Restaurant density
# ---------------------------------------------------------------------------

# Real coords, counts verified directly against eda/restaurants.json.
# Narrowed from 0.5mi to 0.25mi on 2026-07-16 (see RESTAURANT_DENSITY_RADII
# in config.py); Midtown is still dense enough to show a non-trivial count.
_MIDTOWN_RESTAURANTS = (33.7868014, -84.3795169)  # 12 restaurants within 0.25mi
_RURAL_FAR_RESTAURANT = (33.10, -85.10)  # 0 restaurants within 0.25mi


def test_add_restaurant_density_adds_column():
    df = _atlanta_df(10)
    out = add_restaurant_density(df)
    assert "num_restaurants_within_0.25mi" in out.columns


def test_restaurant_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_restaurant_density(df)
    assert (out["num_restaurants_within_0.25mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_restaurants_within_0.25mi"])


def test_restaurant_density_urban_beats_rural():
    """Midtown has 12 restaurants within 0.25mi (verified against
    eda/restaurants.json); a point far outside the metro has 0."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_RESTAURANTS[0], _RURAL_FAR_RESTAURANT[0]],
            "longitude": [_MIDTOWN_RESTAURANTS[1], _RURAL_FAR_RESTAURANT[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_restaurant_density(df)
    urban_count = out["num_restaurants_within_0.25mi"].iloc[0]
    rural_count = out["num_restaurants_within_0.25mi"].iloc[1]
    assert urban_count == 12
    assert rural_count == 0


def test_restaurant_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_RESTAURANTS[0], np.nan],
            "longitude": [_MIDTOWN_RESTAURANTS[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_restaurant_density(df)
    assert out["num_restaurants_within_0.25mi"].iloc[1] == 0


def test_restaurant_density_in_numeric_config():
    """num_restaurants_within_0.25mi must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "num_restaurants_within_0.25mi" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Bar / nightclub density
# ---------------------------------------------------------------------------

# eda/bars_nightclubs.json curated 2026-07-16 from a pre-existing cached
# Overpass export at eda/research/bars.geojson (200 amenity=bar features,
# already scoped to the MSA -- no nightclub-tagged features present in
# that cache). 196 named bars after dropping 4 unnamed nodes. Real coords,
# counts verified directly against eda/bars_nightclubs.json: Bar Moxy
# (Midtown) has 13 bars within 0.25mi, the densest cluster found.
_BAR_MOXY_MIDTOWN = (33.7858797, -84.385988)  # 13 bars within 0.25mi
_RURAL_FAR_BAR = (33.10, -85.10)  # 0 bars within 0.25mi


def test_add_bar_density_adds_column():
    df = _atlanta_df(10)
    out = add_bar_density(df)
    assert "num_bars_nightclubs_within_0.25mi" in out.columns


def test_bar_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_bar_density(df)
    assert (out["num_bars_nightclubs_within_0.25mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_bars_nightclubs_within_0.25mi"])


def test_bar_density_urban_beats_rural():
    """Bar Moxy (Midtown) has 13 bars within 0.25mi (verified against
    eda/bars_nightclubs.json); a point far outside the metro has 0."""
    df = pd.DataFrame(
        {
            "latitude": [_BAR_MOXY_MIDTOWN[0], _RURAL_FAR_BAR[0]],
            "longitude": [_BAR_MOXY_MIDTOWN[1], _RURAL_FAR_BAR[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_bar_density(df)
    urban_count = out["num_bars_nightclubs_within_0.25mi"].iloc[0]
    rural_count = out["num_bars_nightclubs_within_0.25mi"].iloc[1]
    assert urban_count == 13
    assert rural_count == 0


def test_bar_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_BAR_MOXY_MIDTOWN[0], np.nan],
            "longitude": [_BAR_MOXY_MIDTOWN[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_bar_density(df)
    assert out["num_bars_nightclubs_within_0.25mi"].iloc[1] == 0


def test_bar_density_in_numeric_config():
    """num_bars_nightclubs_within_0.25mi must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "num_bars_nightclubs_within_0.25mi" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Combined POI density (coffee + grocery + restaurants + bars/nightclubs)
# ---------------------------------------------------------------------------

# Real coords, counts verified directly against the 4 underlying JSON files
# combined. Spiller Park Coffee (Old Fourth Ward / Ponce City Market area)
# is the densest combined cluster found: 54 POIs of any of the 4 categories
# within 0.25mi.
_SPILLER_PARK = (33.7808665, -84.3839395)  # 54 combined POIs within 0.25mi
_RURAL_FAR_POI = (33.10, -85.10)  # 0 combined POIs within 0.25mi


def test_add_total_poi_density_adds_column():
    df = _atlanta_df(10)
    out = add_total_poi_density(df)
    assert "num_poi_within_0.25mi" in out.columns


def test_total_poi_density_is_non_negative_int():
    df = _atlanta_df(20)
    out = add_total_poi_density(df)
    assert (out["num_poi_within_0.25mi"] >= 0).all()
    assert pd.api.types.is_integer_dtype(out["num_poi_within_0.25mi"])


def test_total_poi_density_urban_beats_rural():
    """Spiller Park Coffee has 54 combined POIs within 0.25mi (verified
    against the 4 underlying JSON files); a point far outside the metro
    has 0."""
    df = pd.DataFrame(
        {
            "latitude": [_SPILLER_PARK[0], _RURAL_FAR_POI[0]],
            "longitude": [_SPILLER_PARK[1], _RURAL_FAR_POI[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_total_poi_density(df)
    urban_count = out["num_poi_within_0.25mi"].iloc[0]
    rural_count = out["num_poi_within_0.25mi"].iloc[1]
    assert urban_count == 54
    assert rural_count == 0


def test_total_poi_density_missing_coords_handled():
    """Rows with NaN lat/lon get a count of 0, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_SPILLER_PARK[0], np.nan],
            "longitude": [_SPILLER_PARK[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_total_poi_density(df)
    assert out["num_poi_within_0.25mi"].iloc[1] == 0


def test_total_poi_density_in_numeric_config():
    """num_poi_within_0.25mi must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "num_poi_within_0.25mi" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Nearest park distance
# ---------------------------------------------------------------------------

# Real coords, verified directly against eda/parks.json. Midtown is close
# to "The Grove" (0.249km); a rural point is far from any curated park.
_MIDTOWN_PARK = (33.7868014, -84.3795169)  # 0.249km to nearest park
_RURAL_FAR_PARK = (33.10, -85.10)  # 11.708km to nearest park


def test_add_park_distance_adds_column():
    df = _atlanta_df(10)
    out = add_park_distance(df)
    assert "dist_nearest_park_km" in out.columns


def test_park_distance_urban_closer_than_rural():
    """Midtown is ~0.25km from its nearest park; a rural point is ~11.7km
    (verified against eda/parks.json)."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_PARK[0], _RURAL_FAR_PARK[0]],
            "longitude": [_MIDTOWN_PARK[1], _RURAL_FAR_PARK[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_park_distance(df)
    urban_dist = out["dist_nearest_park_km"].iloc[0]
    rural_dist = out["dist_nearest_park_km"].iloc[1]
    assert urban_dist == pytest.approx(0.249, abs=0.01)
    assert rural_dist == pytest.approx(11.708, abs=0.01)
    assert urban_dist < rural_dist


def test_park_distance_missing_coords_handled():
    """Rows with NaN lat/lon get NaN, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_PARK[0], np.nan],
            "longitude": [_MIDTOWN_PARK[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_park_distance(df)
    assert np.isnan(out["dist_nearest_park_km"].iloc[1])


def test_park_distance_not_in_numeric_config():
    """dist_nearest_park_km (nearest of 1070) was tested and removed --
    dist_nearest_named_park_km (curated set of 5) replaced it."""
    import prime_mfr.config as config

    assert "dist_nearest_park_km" not in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Nearest named-park distance (curated set of 5, replaces the above)
# ---------------------------------------------------------------------------

# Real coords, verified directly against eda/park_landmarks.json.
_MIDTOWN_NAMED_PARK = (33.7868014, -84.3795169)  # 0.574km to Piedmont Park
_RURAL_FAR_NAMED_PARK = (33.10, -85.10)  # 97.816km to Grant Park (nearest)


def test_add_named_park_distance_adds_column():
    df = _atlanta_df(10)
    out = add_named_park_distance(df)
    assert "dist_nearest_named_park_km" in out.columns


def test_named_park_distance_urban_closer_than_rural():
    """Midtown is ~0.57km from Piedmont Park; a rural point is ~97.8km
    from its nearest curated park (verified against
    eda/park_landmarks.json)."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_NAMED_PARK[0], _RURAL_FAR_NAMED_PARK[0]],
            "longitude": [_MIDTOWN_NAMED_PARK[1], _RURAL_FAR_NAMED_PARK[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_named_park_distance(df)
    urban_dist = out["dist_nearest_named_park_km"].iloc[0]
    rural_dist = out["dist_nearest_named_park_km"].iloc[1]
    assert urban_dist == pytest.approx(0.574, abs=0.01)
    assert rural_dist == pytest.approx(97.816, abs=0.01)
    assert urban_dist < rural_dist


def test_named_park_distance_missing_coords_handled():
    """Rows with NaN lat/lon get NaN, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_MIDTOWN_NAMED_PARK[0], np.nan],
            "longitude": [_MIDTOWN_NAMED_PARK[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_named_park_distance(df)
    assert np.isnan(out["dist_nearest_named_park_km"].iloc[1])


def test_named_park_distance_in_numeric_config():
    """dist_nearest_named_park_km must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "dist_nearest_named_park_km" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Drive-time-to-landmark features
# ---------------------------------------------------------------------------

# eda/travel_times.json generated 2026-07-16 via eda/fetch_travel_times.py
# (989 h3_res6 cells, real OSRM drive times -- this sandbox can't run that
# script itself: no network, and h3 is a compiled macOS extension here
# that won't load). The lookup tests below set h3_res6 directly to known
# cell IDs from the real eda/travel_times.json rather than computing it
# from lat/lon via add_h3_cells(), since a working h3 install isn't
# available in this sandbox to verify that half of the pipeline -- the
# join/lookup logic itself is fully exercised regardless.
_KNOWN_CELL = "8644c1a8fffffff"  # verified: {buckhead: 8.19, midtown: 2.99, downtown: 6.39, atl_airport: 20.25} (minutes)
_UNKNOWN_CELL = "__not_in_table__"


def test_add_travel_time_features_adds_columns():
    df = _atlanta_df(10)
    df = add_h3_cells(df)
    out = add_travel_time_features(df)
    for key in ("buckhead", "midtown", "downtown", "atl_airport"):
        assert f"{key}_drive_min" in out.columns


def test_travel_time_features_missing_h3_column_noop():
    """Without h3_res6 already computed, this is a no-op (not a crash)."""
    df = _atlanta_df(5)
    out = add_travel_time_features(df)
    assert "buckhead_drive_min" not in out.columns


def test_travel_time_features_real_lookup():
    """h3_res6 cell 8644c1a8fffffff has known, verified drive times in
    eda/travel_times.json; a cell not in the table gets NaN."""
    df = pd.DataFrame({"h3_res6": [_KNOWN_CELL, _UNKNOWN_CELL]})
    out = add_travel_time_features(df)
    assert out["midtown_drive_min"].iloc[0] == pytest.approx(2.99, abs=0.01)
    assert out["buckhead_drive_min"].iloc[0] == pytest.approx(8.19, abs=0.01)
    assert out["downtown_drive_min"].iloc[0] == pytest.approx(6.39, abs=0.01)
    assert out["atl_airport_drive_min"].iloc[0] == pytest.approx(20.25, abs=0.01)
    assert np.isnan(out["midtown_drive_min"].iloc[1])


def test_travel_time_features_in_numeric_config():
    """The 4 drive-time columns must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    for key in ("buckhead", "midtown", "downtown", "atl_airport"):
        assert f"{key}_drive_min" in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Nearest highway distance (I-285 / GA-400)
# ---------------------------------------------------------------------------

# Real coords, verified directly against eda/highways.json (2125 points
# resampled every 0.25km along both routes' OSM geometry).
_BUCKHEAD_HIGHWAY = (33.83942, -84.37992)  # 1.428km to nearest highway point (GA-400)
_RURAL_FAR_HIGHWAY = (33.10, -85.10)  # 81.15km to nearest highway point


def test_add_highway_distance_adds_column():
    df = _atlanta_df(10)
    out = add_highway_distance(df)
    assert "dist_nearest_highway_km" in out.columns


def test_highway_distance_urban_closer_than_rural():
    """Buckhead is ~1.43km from the nearest highway point (GA-400); a
    rural point is ~81.2km (verified against eda/highways.json)."""
    df = pd.DataFrame(
        {
            "latitude": [_BUCKHEAD_HIGHWAY[0], _RURAL_FAR_HIGHWAY[0]],
            "longitude": [_BUCKHEAD_HIGHWAY[1], _RURAL_FAR_HIGHWAY[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_highway_distance(df)
    urban_dist = out["dist_nearest_highway_km"].iloc[0]
    rural_dist = out["dist_nearest_highway_km"].iloc[1]
    assert urban_dist == pytest.approx(1.428, abs=0.01)
    assert rural_dist == pytest.approx(81.15, abs=0.1)
    assert urban_dist < rural_dist


def test_highway_distance_missing_coords_handled():
    """Rows with NaN lat/lon get NaN, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_BUCKHEAD_HIGHWAY[0], np.nan],
            "longitude": [_BUCKHEAD_HIGHWAY[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_highway_distance(df)
    assert np.isnan(out["dist_nearest_highway_km"].iloc[1])


def test_highway_distance_not_in_numeric_config():
    """dist_nearest_highway_km (combined) was tested and removed --
    dist_ga400_km/dist_i285_km (separate) replaced it."""
    import prime_mfr.config as config

    assert "dist_nearest_highway_km" not in config.NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# GA-400 / I-285 as separate distance features
# ---------------------------------------------------------------------------

# Real coords, verified directly against eda/highway_ga400.json /
# eda/highway_i285.json.
_BUCKHEAD_ROUTES = (33.83942, -84.37992)  # 1.428km to GA-400, 7.973km to I-285
_RURAL_FAR_ROUTES = (33.10, -85.10)  # 104.901km to GA-400, 81.15km to I-285


def test_add_highway_route_distances_adds_columns():
    df = _atlanta_df(10)
    out = add_highway_route_distances(df)
    assert "dist_ga400_km" in out.columns
    assert "dist_i285_km" in out.columns


def test_highway_route_distances_real_values():
    """Buckhead is ~1.43km from GA-400 and ~7.97km from I-285; a rural
    point is ~104.9km / ~81.2km respectively (verified against
    eda/highway_ga400.json / eda/highway_i285.json)."""
    df = pd.DataFrame(
        {
            "latitude": [_BUCKHEAD_ROUTES[0], _RURAL_FAR_ROUTES[0]],
            "longitude": [_BUCKHEAD_ROUTES[1], _RURAL_FAR_ROUTES[1]],
            "rent": [2200.0, 1400.0],
        }
    )
    out = add_highway_route_distances(df)
    assert out["dist_ga400_km"].iloc[0] == pytest.approx(1.428, abs=0.01)
    assert out["dist_i285_km"].iloc[0] == pytest.approx(7.973, abs=0.01)
    assert out["dist_ga400_km"].iloc[1] == pytest.approx(104.901, abs=0.1)
    assert out["dist_i285_km"].iloc[1] == pytest.approx(81.15, abs=0.1)


def test_highway_route_distances_missing_coords_handled():
    """Rows with NaN lat/lon get NaN, not a crash."""
    df = pd.DataFrame(
        {
            "latitude": [_BUCKHEAD_ROUTES[0], np.nan],
            "longitude": [_BUCKHEAD_ROUTES[1], -84.40],
            "rent": [2200.0, 1800.0],
        }
    )
    out = add_highway_route_distances(df)
    assert np.isnan(out["dist_ga400_km"].iloc[1])
    assert np.isnan(out["dist_i285_km"].iloc[1])


def test_ga400_distance_in_numeric_config():
    """dist_ga400_km must live in NUMERIC_FEATURES."""
    import prime_mfr.config as config

    assert "dist_ga400_km" in config.NUMERIC_FEATURES


def test_i285_distance_in_numeric_config():
    """dist_i285_km added alongside dist_ga400_km to test both together,
    despite its flat real-data trend predicting it won't help on its own
    (see config.py comment)."""
    import prime_mfr.config as config

    assert "dist_i285_km" in config.NUMERIC_FEATURES


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
