"""
feature_engineering.py
======================
All feature transformations that need awareness of the train/validation
split (target encoding, k-NN aggregates) live here. Pure transformations
(landmark distances, H3 cells) are also collected here so the modeling
script stays focused on the CV loop.

The two functions used by `train.py`:
    * add_static_features(df)        -> df  (no leakage, pre-split)
    * add_oof_features(train_df, valid_df, full_train_df)
                                     -> (train_with, valid_with)

`add_oof_features` recomputes target encodings and k-NN features
*using only the training fold's labels* and writes them onto both
folds.

Helper math:
    haversine_km(lat1, lon1, lat2, lon2)   -> float
    bayesian_target_encode(...)
    compute_oof_knn(...)
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import prime_mfr.config as config

# ---------------------------------------------------------------------------
# Geo math
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6371.0088


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: float,
    lon2: float,
) -> np.ndarray | float:
    """Great-circle distance from (lat1, lon1) to a single point (lat2, lon2) in km."""
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _load_landmarks() -> dict[str, tuple[float, float]]:
    """Read landmark coordinates from the JSON reference file."""
    raw = json.loads(Path(config.LANDMARKS_JSON).read_text())
    out: dict[str, tuple[float, float]] = {}
    for key, info in raw["landmarks"].items():
        out[key] = (float(info["latitude"]), float(info["longitude"]))
    return out


def _load_marta_stations() -> list[tuple[float, float]]:
    """
    Read MARTA rail station coordinates from the JSON reference file
    (eda/marta_stations.json, a list of {name, lat, lon, ...} entries —
    see eda/fetch_marta_stations.py for provenance). Returns an empty
    list (rather than raising) if the file is missing, so the pipeline
    degrades gracefully instead of hard-failing on a POI file that's
    optional relative to the core landmark set.
    """
    path = Path(config.MARTA_STATIONS_JSON)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [(float(s["lat"]), float(s["lon"])) for s in raw]


# ---------------------------------------------------------------------------
# Static features (no target leakage, no fold awareness)
# ---------------------------------------------------------------------------


def add_landmark_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add great-circle distance (km) to each Atlanta landmark."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    df = df.copy()
    landmarks = _load_landmarks()
    cols = []
    for key in config.LANDMARKS:
        lat, lon = landmarks[key]
        col = f"dist_{key}_km"
        df[col] = haversine_km(df["latitude"].values, df["longitude"].values, lat, lon).astype(
            "float32"
        )
        cols.append(col)
    df["dist_min_landmark_km"] = df[cols].min(axis=1).astype("float32")
    return df


def add_airport_zone_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket dist_atl_airport_km into a categorical zone: near / hot_zone / far
    (edges + labels in config.ATL_AIRPORT_ZONE_EDGES / _LABELS).

    Rent vs. distance-to-airport is non-monotonic on this dataset: cheapest
    right next to the airport, most expensive AND most volatile in a
    ~9-15km "hot zone", then it settles back down and flattens past ~15km.
    Encoded as a genuine categorical (not an ordinal int bucket like
    sqft_bucket/age_bucket) because "hot_zone" isn't ordered relative to
    "near"/"far" -- it's a different regime, not a bigger/smaller value.

    Computes dist_atl_airport_km internally from config.LANDMARKS'
    "atl_airport" entry (doesn't require add_landmark_distances() to have
    run first, and doesn't add dist_atl_airport_km itself to the output --
    that column isn't in NUMERIC_FEATURES anymore, replaced by this zone).

    Adds:
        dist_atl_airport_zone : category dtype, one of
                                 config.ATL_AIRPORT_ZONE_LABELS, or NaN if
                                 lat/lon missing or "atl_airport" isn't an
                                 active key in config.LANDMARKS.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    if "atl_airport" not in config.LANDMARKS:
        return df
    df = df.copy()
    landmarks = _load_landmarks()
    airport_lat, airport_lon = landmarks["atl_airport"]
    dist = haversine_km(
        df["latitude"].values, df["longitude"].values, airport_lat, airport_lon
    ).astype("float64")

    edges = np.asarray(config.ATL_AIRPORT_ZONE_EDGES, dtype="float64")
    labels = list(config.ATL_AIRPORT_ZONE_LABELS)
    valid = ~np.isnan(dist)

    zone = np.full(len(df), np.nan, dtype=object)
    idx = np.digitize(dist[valid], bins=edges)  # 0..len(labels)-1
    zone[valid] = np.asarray(labels, dtype=object)[idx]

    df["dist_atl_airport_zone"] = pd.Categorical(zone, categories=labels)
    return df


def add_marta_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add great-circle distance (km) to the nearest MARTA rail station.

    Unlike add_landmark_distances (one column per named landmark), this
    collapses ~37 station coordinates into a single aggregate signal —
    "how transit-adjacent is this property" — mirroring dist_min_landmark_km.
    Adding one column per station was considered and rejected per the
    guidance in docs/geospatial_features.md (35+ near-duplicate distance
    columns cost training time and invite overfitting for no signal gain
    over the single nearest-station distance).

    Adds:
        dist_marta_km : float32, NaN if lat/lon missing or the station
                         reference file (eda/marta_stations.json) is absent.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    df = df.copy()
    stations = _load_marta_stations()
    if not stations:
        df["dist_marta_km"] = np.float32(np.nan)
        return df
    dists = np.stack(
        [
            haversine_km(df["latitude"].values, df["longitude"].values, lat, lon)
            for lat, lon in stations
        ]
    )
    df["dist_marta_km"] = dists.min(axis=0).astype("float32")
    return df


def add_marta_station_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count distinct MARTA rail stations within each radius in
    config.MARTA_DENSITY_RADII.

    Companion to add_marta_distance(): nearest-station distance alone can't
    tell a downtown property (Five Points/Georgia State/Peachtree
    Center/Garnett are all <0.5mi of each other) apart from a suburban one
    near a single isolated station (e.g. North Springs' nearest neighbor is
    ~0.94mi away) -- both could have a similarly small dist_marta_km, but
    the former is in a much more transit-dense area.

    Only 37 stations total, so this uses a direct vectorized haversine
    distance matrix rather than a BallTree (unlike
    add_competitor_count_features, which needs a tree because it's
    property-to-property at much larger N). Rows with missing lat/lon
    naturally get a count of 0 (NaN distances fail every "<= radius"
    comparison), matching the existing convention in
    add_competitor_count_features.

    Adds (one column per radius):
        num_marta_stations_within_{label} : int16
    e.g. num_marta_stations_within_1mi.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    df = df.copy()
    stations = _load_marta_stations()

    if not stations:
        for _, label in config.MARTA_DENSITY_RADII:
            df[f"num_marta_stations_within_{label}"] = np.int16(0)
        return df

    dists_km = np.stack(
        [
            haversine_km(df["latitude"].values, df["longitude"].values, lat, lon)
            for lat, lon in stations
        ]
    )  # shape (n_stations, n_rows)

    for radius_mi, label in config.MARTA_DENSITY_RADII:
        radius_km = radius_mi * 1.609344
        within = dists_km <= radius_km  # NaN comparisons -> False
        df[f"num_marta_stations_within_{label}"] = within.sum(axis=0).astype("int16")

    return df


def add_h3_cells(
    df: pd.DataFrame, resolutions: Iterable[int] = config.H3_RESOLUTIONS
) -> pd.DataFrame:
    """
    Add hierarchical hex cell IDs at each resolution. Uses the `h3` package
    if available; otherwise falls back to a simple lat/lon grid (less ideal
    but lets the pipeline run end-to-end).
    """
    df = df.copy()
    try:
        import h3  # type: ignore

        for res in resolutions:
            col = f"h3_res{res}"
            df[col] = [
                (
                    h3.latlng_to_cell(float(lat), float(lon), res)
                    if pd.notna(lat) and pd.notna(lon)
                    else "__missing__"
                )
                for lat, lon in zip(df["latitude"].values, df["longitude"].values)
            ]
    except (ImportError, AttributeError):
        # Fallback: simple lat/lon grid bucketing. Coarser at lower res.
        for res in resolutions:
            step = max(0.01, 0.5 / (2 ** (res - 5)))  # heuristic
            col = f"h3_res{res}"
            lat_bin = (df["latitude"] / step).round().astype("Int64").astype(str)
            lon_bin = (df["longitude"] / step).round().astype("Int64").astype(str)
            df[col] = (lat_bin + "_" + lon_bin).fillna("__missing__")
    return df


def add_static_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative + ratio interactions between physical attributes that the
    trees don't naturally capture (a single split chooses one variable, not
    a product). All features here are leakage-free (no target, no fold).

    Adds:
      sqft_x_beds              : co-scaling of unit footprint and bedroom count
      sqft_x_buckhead_km       : how a large unit's premium varies with distance
      beds_x_buckhead_km       : bedroom premium varies by distance from Buckhead
      year_x_buckhead_km       : age premium varies by location
      sqft_per_bed             : layout density (smaller per-bed = denser, often urban)
      baths_per_bed            : bathroom-to-bedroom luxury proxy
      age_x_sqft               : age effect scales with size (older + bigger = different
                                 trajectory than older + small)
    """
    df = df.copy()

    def _col(name: str) -> pd.Series | None:
        return df[name] if name in df.columns else None

    sqft = _col("sqft")
    beds = _col("beds")
    baths = _col("baths")
    age = _col("property_age")
    yb = _col("year_built")
    d_buck = _col("dist_buckhead_km")

    if sqft is not None and beds is not None:
        df["sqft_x_beds"] = (sqft.astype("float32") * beds.astype("float32")).astype("float32")
        # sqft per bedroom (layout density). Avoid div-by-zero: clip beds to >=1.
        beds_safe = beds.where(beds > 0, 1.0).astype("float32")
        df["sqft_per_bed"] = (sqft.astype("float32") / beds_safe).astype("float32")

    if baths is not None and beds is not None:
        beds_safe = beds.where(beds > 0, 1.0).astype("float32")
        df["baths_per_bed"] = (baths.astype("float32") / beds_safe).astype("float32")

    if sqft is not None and d_buck is not None:
        df["sqft_x_buckhead_km"] = (sqft.astype("float32") * d_buck.astype("float32")).astype(
            "float32"
        )

    if beds is not None and d_buck is not None:
        df["beds_x_buckhead_km"] = (beds.astype("float32") * d_buck.astype("float32")).astype(
            "float32"
        )

    if yb is not None and d_buck is not None:
        df["year_x_buckhead_km"] = (yb.astype("float32") * d_buck.astype("float32")).astype(
            "float32"
        )

    if age is not None and sqft is not None:
        df["age_x_sqft"] = (age.astype("float32") * sqft.astype("float32")).astype("float32")

    return df


def add_bucket_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add discrete bucket columns used by the hierarchical comparable TE:
        sqft_bucket     : int bin index from SQFT_BUCKET_EDGES (0..len(edges))
        age_bucket      : int bin index from AGE_BUCKET_EDGES
        beds_int        : non-negative int bedroom count (NaN -> -1 sentinel)
        baths_int       : non-negative int bath count (NaN -> -1 sentinel)
    All leakage-free (no target).
    """
    df = df.copy()
    if "sqft" in df.columns:
        df["sqft_bucket"] = np.digitize(
            df["sqft"].fillna(-1.0).to_numpy(), bins=np.asarray(config.SQFT_BUCKET_EDGES)
        ).astype("int16")
    if "property_age" in df.columns:
        df["age_bucket"] = np.digitize(
            df["property_age"].fillna(-1.0).to_numpy(), bins=np.asarray(config.AGE_BUCKET_EDGES)
        ).astype("int16")
    if "beds" in df.columns:
        df["beds_int"] = df["beds"].fillna(-1.0).round().astype("int16")
    if "baths" in df.columns:
        df["baths_int"] = df["baths"].fillna(-1.0).round().astype("int16")
    return df


def add_property_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-property structural aggregates that diagnose product typology.
    All derived from physical attributes (no rent / target / fold), so
    leakage-free and computed once over the full dataset.

    Adds (broadcast to all rows of each property):
        property_n_unit_types       : count of rows in the property data
                                      (proxy for distinct floor plans).
                                      BTR is typically 1; apartments 3-5+.
        property_beds_nunique       : distinct bed counts within property
        property_baths_nunique      : distinct bath counts within property
        property_sqft_range_pct     : (sqft_max - sqft_min) / sqft_med —
                                      narrow range = uniform product (BTR);
                                      wide = mixed apartment building.
        is_btr_likely               : composite binary flag combining
                                      single-unit-type + single-bed +
                                      recent vintage + large sqft.
        btr_likely_score            : continuous 0-4 count of conditions
                                      met (single floor plan, single bed,
                                      year >= BTR_MIN_YEAR, sqft >= MIN).
                                      Lets the booster express partial-BTR
                                      properties (e.g. luxury small-format
                                      developments) on a gradient.

    Cost: one groupby on property_id over ~6.9k rows; negligible.
    """
    if "property_id" not in df.columns:
        return df

    df = df.copy()

    agg = df.groupby("property_id", as_index=False).agg(
        _n_rows=("property_id", "size"),
        _beds_nunique=("beds", "nunique"),
        _baths_nunique=("baths", "nunique"),
        _sqft_min=("sqft", "min"),
        _sqft_max=("sqft", "max"),
        _sqft_med=("sqft", "median"),
        _year_built=("year_built", "first"),
    )
    agg["property_n_unit_types"] = agg["_n_rows"].astype("int16")
    agg["property_beds_nunique"] = agg["_beds_nunique"].astype("int16")
    agg["property_baths_nunique"] = agg["_baths_nunique"].astype("int16")

    sqft_med_safe = agg["_sqft_med"].where(agg["_sqft_med"] > 0, np.nan)
    agg["property_sqft_range_pct"] = ((agg["_sqft_max"] - agg["_sqft_min"]) / sqft_med_safe).astype(
        "float32"
    )

    # Composite BTR flag and score.
    cond_uniform_unit = agg["_n_rows"] <= config.BTR_MAX_UNIT_TYPES
    cond_uniform_bed = agg["_beds_nunique"] <= config.BTR_MAX_BEDS_NUNIQUE
    cond_recent = agg["_year_built"].fillna(0) >= config.BTR_MIN_YEAR
    cond_large = agg["_sqft_med"].fillna(0) >= config.BTR_MIN_SQFT

    agg["btr_likely_score"] = (
        cond_uniform_unit.astype("int8")
        + cond_uniform_bed.astype("int8")
        + cond_recent.astype("int8")
        + cond_large.astype("int8")
    ).astype("float32")
    agg["is_btr_likely"] = (cond_uniform_unit & cond_uniform_bed & cond_recent & cond_large).astype(
        "float32"
    )

    keep = [
        "property_id",
        "property_n_unit_types",
        "property_beds_nunique",
        "property_baths_nunique",
        "property_sqft_range_pct",
        "btr_likely_score",
        "is_btr_likely",
    ]
    df = df.merge(agg[keep], on="property_id", how="left")
    return df


def add_competitor_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row and each radius in `config.COMPETITOR_RADII`, count distinct
    OTHER properties within that radius that have at least one listing
    matching this row's bed count.

    Adds (one column per radius):
        n_competitors_within_{label}_same_beds : int16
    e.g. n_competitors_within_0_5mi_same_beds, _within_1mi_, _within_2mi_.

    Leakage-free (uses only physical attributes; no rent / target).
    Self-property is excluded from the neighbor count.
    """
    needed = {"latitude", "longitude", "beds", "property_id"}
    if not needed.issubset(df.columns):
        return df

    df = df.copy()
    df["__row_idx"] = np.arange(len(df), dtype=np.int64)
    df["__beds_int"] = df["beds"].fillna(-9999).round().astype("int32")

    # Property-level: one row per (property_id, beds_int) with property coords.
    prop_beds = (
        df[["property_id", "latitude", "longitude", "__beds_int"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["property_id", "__beds_int"])
        .reset_index(drop=True)
    )

    # Build one BallTree per bed bucket (reused across radii).
    bed_trees: dict[int, tuple[BallTree, np.ndarray]] = {}
    for bed_val, sub in prop_beds.groupby("__beds_int"):
        if bed_val == -9999:
            continue
        coords_rad = np.radians(sub[["latitude", "longitude"]].values)
        if len(coords_rad) == 0:
            continue
        tree = BallTree(coords_rad, metric="haversine")
        bed_trees[int(bed_val)] = (tree, sub["property_id"].values)

    for radius_mi, label in config.COMPETITOR_RADII:
        radius_rad = radius_mi * 1.609344 / config.EARTH_RADIUS_KM
        counts = np.zeros(len(df), dtype="int32")

        for bed_val, (tree, sub_pids) in bed_trees.items():
            rows = df[df["__beds_int"] == bed_val]
            if len(rows) == 0:
                continue
            q_rad = np.radians(rows[["latitude", "longitude"]].fillna(0).values)
            idx_lists = tree.query_radius(q_rad, r=radius_rad)
            row_pids = rows["property_id"].values
            row_idxs = rows["__row_idx"].values
            for j, neigh in enumerate(idx_lists):
                if len(neigh) == 0:
                    continue
                cnt = int((sub_pids[neigh] != row_pids[j]).sum())
                counts[row_idxs[j]] = cnt

        df[f"n_competitors_within_{label}_same_beds"] = counts.astype("int16")

    df.drop(columns=["__row_idx", "__beds_int"], inplace=True)
    return df


def add_competitor_psf_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OOF: for each row and each radius in `config.COMPETITOR_RADII`, compute
    p25 / p50 / p75 / IQR of training-fold competitor PSF among same-bed
    properties within that radius. Pool is aggregated to one PSF per
    (property_id, beds_int) so that a 5-floorplan apartment contributes one
    neighbor, not five.

    For training rows the row's own property is excluded from its neighbor
    pool. For validation rows the property is (by GroupKFold) not in the
    train pool already.

    Per radius (label `L`), adds:
        competitor_psf_p25_within_{L}
        competitor_psf_p50_within_{L}
        competitor_psf_p75_within_{L}
        competitor_psf_iqr_within_{L}
    """
    radii = config.COMPETITOR_RADII

    def _empty_cols() -> list[str]:
        cols: list[str] = []
        for _, label in radii:
            cols += [
                f"competitor_psf_p25_within_{label}",
                f"competitor_psf_p50_within_{label}",
                f"competitor_psf_p75_within_{label}",
                f"competitor_psf_iqr_within_{label}",
            ]
        return cols

    needed = {"latitude", "longitude", "beds", "property_id", "sqft"}
    if not needed.issubset(train_df.columns):
        for c in _empty_cols():
            train_df[c] = np.nan
            valid_df[c] = np.nan
        return train_df, valid_df

    pool = train_df[
        (train_df["sqft"].fillna(0) > 0)
        & train_df[target].notna()
        & train_df["latitude"].notna()
        & train_df["longitude"].notna()
        & train_df["beds"].notna()
    ].copy()
    if len(pool) == 0:
        for c in _empty_cols():
            train_df[c] = np.nan
            valid_df[c] = np.nan
        return train_df, valid_df

    pool["psf"] = pool[target].astype(np.float64) / pool["sqft"].astype(np.float64)
    pool["beds_int"] = pool["beds"].round().astype("int32")

    prop_psf = pool.groupby(["property_id", "beds_int"], as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        psf=("psf", "median"),
    )

    # One BallTree per bed bucket; reused across radii.
    bed_trees: dict[int, tuple[BallTree, np.ndarray, np.ndarray]] = {}
    for bed_val, sub in prop_psf.groupby("beds_int"):
        coords_rad = np.radians(sub[["latitude", "longitude"]].values)
        if len(coords_rad) == 0:
            continue
        tree = BallTree(coords_rad, metric="haversine")
        bed_trees[int(bed_val)] = (
            tree,
            sub["property_id"].values,
            sub["psf"].values.astype(np.float32),
        )

    def _compute(df: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
        df = df.copy()
        beds_int = df["beds"].fillna(-9999).round().astype("int32").values
        n = len(df)

        # For each radius, query once with the largest radius and sub-mask
        # inside; but BallTree.query_radius doesn't return distances unless
        # asked, and re-querying per radius is fast enough at 6.9k × 3 calls.
        for radius_mi, label in radii:
            radius_rad = radius_mi * 1.609344 / config.EARTH_RADIUS_KM
            p25 = np.full(n, np.nan, dtype="float32")
            p50 = np.full(n, np.nan, dtype="float32")
            p75 = np.full(n, np.nan, dtype="float32")

            if "latitude" not in df.columns:
                df[f"competitor_psf_p25_within_{label}"] = p25
                df[f"competitor_psf_p50_within_{label}"] = p50
                df[f"competitor_psf_p75_within_{label}"] = p75
                df[f"competitor_psf_iqr_within_{label}"] = (p75 - p25).astype("float32")
                continue

            for bed_val, (tree, prop_pids, prop_psfs) in bed_trees.items():
                mask = beds_int == bed_val
                if not mask.any():
                    continue
                row_idx = np.where(mask)[0]
                sub_df = df.iloc[row_idx]
                q_rad = np.radians(sub_df[["latitude", "longitude"]].fillna(0).values)
                idx_lists = tree.query_radius(q_rad, r=radius_rad)
                row_pids = sub_df["property_id"].values
                for j, neigh in enumerate(idx_lists):
                    if len(neigh) == 0:
                        continue
                    if exclude_self:
                        keep_mask = prop_pids[neigh] != row_pids[j]
                        neigh_keep = neigh[keep_mask]
                    else:
                        neigh_keep = neigh
                    if len(neigh_keep) < config.COMPETITOR_MIN_K:
                        continue
                    vals = prop_psfs[neigh_keep]
                    p25[row_idx[j]] = np.percentile(vals, 25)
                    p50[row_idx[j]] = np.percentile(vals, 50)
                    p75[row_idx[j]] = np.percentile(vals, 75)

            df[f"competitor_psf_p25_within_{label}"] = p25
            df[f"competitor_psf_p50_within_{label}"] = p50
            df[f"competitor_psf_p75_within_{label}"] = p75
            df[f"competitor_psf_iqr_within_{label}"] = (p75 - p25).astype("float32")

        return df

    train_df = _compute(train_df, exclude_self=True)
    valid_df = _compute(valid_df, exclude_self=False)
    return train_df, valid_df


def add_within_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row heterogeneity features that describe how this unit-row compares
    to *other rows of the same property*. The dataset has 6,887 unit-rows
    across 2,121 properties (avg 3.25 rows/property, max 10), so 93% of
    rows have at least one peer to compare against.

    Adds (added 2026-05-01):
        unit_sqft_within_prop_z       : z-score of sqft within property
                                        (NaN-safe; single-row props -> 0)
        unit_beds_within_prop_z       : z-score of beds within property
        unit_sqft_within_prop_pct     : percentile rank of sqft within
                                        property (0..1, 0.5 if singleton)
        unit_sqft_to_prop_max_share   : sqft / max(prop_sqft)  in [0..1]
        unit_sqft_to_prop_min_share   : sqft / min(prop_sqft)  >= 1.0
        is_largest_unit_in_prop       : 1.0 if this row has prop max sqft
        is_smallest_unit_in_prop      : 1.0 if this row has prop min sqft
        unit_beds_to_prop_max         : beds / max(prop_beds)

    All leakage-free (no rent / target). Captures the within-building
    quality gradient — top-floor / largest unit typically commands a
    rent premium that's not visible from raw sqft alone, because the
    same nominal sqft means different things in a 600-1500 sqft mix vs
    a uniform 1100 sqft building.
    """
    if "property_id" not in df.columns:
        return df
    df = df.copy()
    g = df.groupby("property_id")

    if "sqft" in df.columns:
        prop_sqft_mean = g["sqft"].transform("mean")
        prop_sqft_std = g["sqft"].transform("std")
        prop_sqft_max = g["sqft"].transform("max")
        prop_sqft_min = g["sqft"].transform("min")
        # z-score: 0 when std is 0 / NaN (single-row property)
        std_safe = prop_sqft_std.where(prop_sqft_std > 1e-9, np.nan)
        df["unit_sqft_within_prop_z"] = (
            ((df["sqft"] - prop_sqft_mean) / std_safe).fillna(0.0).astype("float32")
        )
        # percentile rank within property (0.5 for singletons via fillna)
        df["unit_sqft_within_prop_pct"] = g["sqft"].rank(pct=True).fillna(0.5).astype("float32")
        # share of largest / smallest unit in property
        max_safe = prop_sqft_max.where(prop_sqft_max > 0, np.nan)
        min_safe = prop_sqft_min.where(prop_sqft_min > 0, np.nan)
        df["unit_sqft_to_prop_max_share"] = (df["sqft"] / max_safe).fillna(1.0).astype("float32")
        df["unit_sqft_to_prop_min_share"] = (df["sqft"] / min_safe).fillna(1.0).astype("float32")
        # is largest/smallest in property; both 1.0 for singletons (consistent)
        df["is_largest_unit_in_prop"] = (df["sqft"] >= prop_sqft_max - 1e-6).astype("float32")
        df["is_smallest_unit_in_prop"] = (df["sqft"] <= prop_sqft_min + 1e-6).astype("float32")

    if "beds" in df.columns:
        prop_beds_mean = g["beds"].transform("mean")
        prop_beds_std = g["beds"].transform("std")
        prop_beds_max = g["beds"].transform("max")
        std_safe = prop_beds_std.where(prop_beds_std > 1e-9, np.nan)
        df["unit_beds_within_prop_z"] = (
            ((df["beds"] - prop_beds_mean) / std_safe).fillna(0.0).astype("float32")
        )
        max_safe = prop_beds_max.where(prop_beds_max > 0, np.nan)
        df["unit_beds_to_prop_max"] = (df["beds"] / max_safe).fillna(1.0).astype("float32")

    return df


def add_unit_subtype_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row features derived from `num_units_subtype` — the count of actual
    rental units of THIS unit type within the property. (Distinct from
    `num_units`, the property-wide total.) Mean ~75, range 1..910.

    Captures the "dominant vs boutique" axis of within-property product mix:
    a property with 200 1BR + 5 3BR/penthouse units is economically very
    different from a balanced 100/100, even at similar mean sqft. The
    earlier `add_within_property_features` (now disabled) tried to express
    this with unweighted z-scores and overfit; using the raw subtype count
    grounds the signal in actual market presence rather than row counts.

    Adds (added 2026-05-01, replaces the failed heterogeneity attempt):
        log_num_units_subtype          : log1p of num_units_subtype
        unit_subtype_share             : num_units_subtype / num_units (this
                                         row's share of total property units;
                                         clipped to [0, 1.0])
        is_dominant_unit_type          : 1.0 if this row has property max
                                         num_units_subtype (boutique = 0)
        unit_subtype_count_rank_pct    : rank pct of num_units_subtype within
                                         property (1.0 = most common; for
                                         singletons = 0.5 via fillna)

    Plus the raw num_units_subtype is exposed via NUMERIC_FEATURES.
    All leakage-free (no rent / target).
    """
    if "num_units_subtype" not in df.columns or "property_id" not in df.columns:
        return df
    df = df.copy()

    # Log-scaled count.
    df["log_num_units_subtype"] = np.log1p(
        df["num_units_subtype"].fillna(0.0).clip(lower=0.0)
    ).astype("float32")

    # Subtype share of property total. Cap at 1.0 to handle the rare cases
    # where subtype counts slightly exceed property total (median ratio is
    # 1.00 but max observed was 1.01 due to data-quality rounding).
    if "num_units" in df.columns:
        denom = df["num_units"].where(df["num_units"] > 0, np.nan)
        df["unit_subtype_share"] = (
            (df["num_units_subtype"] / denom).clip(0.0, 1.0).fillna(0.0).astype("float32")
        )
    else:
        df["unit_subtype_share"] = 0.0

    # Dominant unit type within property (max num_units_subtype). Tie-break
    # via >= so multi-tied properties have multiple dominant rows (rare).
    g = df.groupby("property_id")
    prop_max_subtype = g["num_units_subtype"].transform("max")
    df["is_dominant_unit_type"] = (df["num_units_subtype"] >= prop_max_subtype - 1e-6).astype(
        "float32"
    )

    # Rank percentile of num_units_subtype within property. 1.0 = the most
    # common unit type; closer to 0 = rarer/boutique floor plan. Singletons
    # get 0.5 via fillna (no info) for symmetry.
    df["unit_subtype_count_rank_pct"] = (
        g["num_units_subtype"].rank(pct=True).fillna(0.5).astype("float32")
    )

    return df


# ---------------------------------------------------------------------------
# Historical rent lag features
# ---------------------------------------------------------------------------
# Extracted to prime_mfr/features/hist_rent.py (Stage 2 of the architecture
# refactor). Re-exported here for backward compatibility.
from prime_mfr.features.hist_rent import (  # noqa: E402, F401  (re-export)
    add_hist_rent_features,
    build_hist_rent_features as _build_hist_rent_features,
)


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """All transforms that don't depend on the train/test split."""
    df = add_landmark_distances(df)
    df = add_airport_zone_feature(df)
    df = add_marta_distance(df)
    df = add_marta_station_density(df)
    df = add_h3_cells(df)
    df = add_text_features(df)
    df = add_geo_aggregates(df)
    df = add_zscore_deviations(df)
    df = add_static_interactions(df)
    df = add_bucket_keys(df)
    df = add_property_structural_features(df)
    # add_within_property_features (sqft/beds z-scores) was tried 2026-05-01
    # and regressed +$25 per fold (heavy overfit on unweighted heterogeneity).
    # Replaced with add_unit_subtype_features below, which uses actual unit
    # counts per subtype rather than row-counts.
    df = add_unit_subtype_features(df)
    df = add_competitor_count_features(df)
    df = add_hist_rent_features(df)
    return df


# ---------------------------------------------------------------------------
# Text features from property_name and street_address
# ---------------------------------------------------------------------------

_DIR_RE = re.compile(r"\b(NE|NW|SE|SW|N|S|E|W)\b", re.IGNORECASE)
_HOUSE_NUM_RE = re.compile(r"^\s*(\d+)\b")
_SUITE_RE = re.compile(r"\b(suite|ste\.?|#|unit)\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_AMP_RE = re.compile(r"&")
_PHASE_RE = re.compile(r"\b(phase|ii|iii|iv|v)\b", re.IGNORECASE)


def _extract_brand(name: str) -> str:
    """
    Extract a "brand" identifier from a property name.

    Strategy:
      1. Lowercase + strip leading "The".
      2. If the first 1-2 tokens match a known operator name, return that.
      3. Otherwise return the first token (works as a high-cardinality
         categorical that target encoding can compress).
    """
    if not name:
        return "__missing__"
    s = name.lower().strip()
    # Strip leading "the " (and trailing ", the" pattern from "Foo, The").
    s = re.sub(r",\s*the\b", "", s)
    s = re.sub(r"^the\s+", "", s)
    tokens = s.split()
    if not tokens:
        return "__missing__"

    # Try multi-word brand match.
    for brand in config.KNOWN_BRANDS:
        b_tokens = brand.split()
        if len(tokens) >= len(b_tokens) and tokens[: len(b_tokens)] == b_tokens:
            return brand
    return tokens[0]


def _extract_street_type(addr: str) -> str:
    """Pull canonical street suffix (road, drive, parkway, ...) from address."""
    if not addr:
        return "__missing__"
    s = addr.lower()
    # Look at last 4 tokens (suffix may precede directional, e.g. "Road NE").
    tokens = re.findall(r"[a-zA-Z]+", s)
    for tok in reversed(tokens[-4:]):
        if tok in config.STREET_SUFFIX_MAP:
            return config.STREET_SUFFIX_MAP[tok]
    return "__missing__"


def _extract_street_name(addr: str) -> str:
    """
    Strip the leading house number from a street address and return the
    lowercased street name. Used as the key for the address-block target
    encoding (added 2026-05-01): granularity finer than zipcode/sub_market
    but coarse enough that ~45% of properties share a street_name with
    another property in the dataset.

    Examples:
      "100 Ashford Gables Drive"  -> "ashford gables drive"
      "1230 Peachtree St NE"      -> "peachtree st ne"
      "Buckhead Loop"             -> "buckhead loop"   (no house number)
    """
    if not addr:
        return "__missing__"
    s = str(addr).strip().lower()
    # Drop leading number (house num) and any leading "##-##" range.
    s = re.sub(r"^\d+(\s*-\s*\d+)?\s+", "", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "__missing__"


def _extract_dir(addr: str) -> str:
    if not addr:
        return "__missing__"
    m = _DIR_RE.search(addr)
    return m.group(1).upper() if m else "__missing__"


def _extract_house_num(addr: str) -> float:
    if not addr:
        return np.nan
    m = _HOUSE_NUM_RE.match(addr)
    return float(m.group(1)) if m else np.nan


def _matches_any(s: str, tokens: tuple[str, ...]) -> bool:
    return any(tok in s for tok in tokens)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive structured features from `property_name` and `street_address`.
    Designed for Atlanta multifamily naming conventions but generalizes to
    other markets by editing config.KNOWN_BRANDS / ICONIC_STREETS.

    Adds:
      Categoricals: brand, street_type, addr_dir
      Booleans (15): name_kw_premium, name_kw_midmarket, name_kw_older,
        name_kw_conversion, name_kw_phase, name_starts_the, name_has_at,
        name_has_digit, name_has_ampersand, name_claims_premium_subm,
        name_subm_match, addr_is_peachtree, addr_is_iconic,
        addr_has_highway, addr_has_suite
      Numerics: name_len, name_n_words, name_caps_ratio,
        addr_house_num, addr_house_num_log, addr_n_words
    """
    df = df.copy()
    name_raw = df.get("property_name", pd.Series([""] * len(df))).fillna("").astype(str)
    addr_raw = df.get("street_address", pd.Series([""] * len(df))).fillna("").astype(str)
    name_lc = name_raw.str.lower()
    addr_lc = addr_raw.str.lower()

    sub_market_lc = (
        df["sub_market"].fillna("").astype(str).str.lower()
        if "sub_market" in df.columns
        else pd.Series([""] * len(df))
    )

    # ---- Brand + categorical extractions ----
    df["brand"] = name_raw.apply(_extract_brand)
    df["street_type"] = addr_raw.apply(_extract_street_type)
    df["addr_dir"] = addr_raw.apply(_extract_dir)
    # street_name (without house number) is the key for the address-block
    # target encoding (TARGET_ENCODE_COLS produces street_name_te). Cardinality
    # ~1462 over 2121 properties; ~45% of properties share a street_name with
    # another property, so the OOF Bayesian TE has real signal to compress.
    df["street_name"] = addr_raw.apply(_extract_street_name)

    # ---- Property-name keyword flags ----
    df["name_kw_premium"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.NAME_KEYWORDS_PREMIUM))
    ).astype("float32")
    df["name_kw_midmarket"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.NAME_KEYWORDS_MIDMARKET))
    ).astype("float32")
    df["name_kw_older"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.NAME_KEYWORDS_OLDER))
    ).astype("float32")
    df["name_kw_conversion"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.NAME_KEYWORDS_CONVERSION))
    ).astype("float32")
    df["name_kw_phase"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.NAME_KEYWORDS_PHASE))
    ).astype("float32")

    df["name_starts_the"] = name_lc.str.match(r"^the\b").astype("float32")
    df["name_has_at"] = name_lc.str.contains(r"\bat\b", regex=True).astype("float32")
    df["name_has_digit"] = name_raw.apply(lambda s: float(bool(_DIGIT_RE.search(s)))).astype(
        "float32"
    )
    df["name_has_ampersand"] = name_raw.apply(lambda s: float(bool(_AMP_RE.search(s)))).astype(
        "float32"
    )

    # Premium sub-market mention in name + match-vs-actual indicator.
    df["name_claims_premium_subm"] = name_lc.apply(
        lambda s: float(_matches_any(s, config.PREMIUM_SUBM_TOKENS))
    ).astype("float32")

    def _subm_match(name: str, subm: str) -> float:
        if not name or not subm:
            return 0.0
        # Take primary token of sub_market (e.g. "buckhead" from "buckhead - east").
        first = subm.split()[0] if subm else ""
        return float(first and first in name)

    df["name_subm_match"] = [
        _subm_match(n, s) for n, s in zip(name_lc.values, sub_market_lc.values)
    ]
    df["name_subm_match"] = pd.Series(df["name_subm_match"]).astype("float32")

    # ---- Property-name string statistics ----
    df["name_len"] = name_raw.str.len().fillna(0).astype("float32")
    df["name_n_words"] = name_raw.str.split().apply(len).astype("float32")
    df["name_caps_ratio"] = name_raw.apply(
        lambda s: (sum(c.isupper() for c in s) / max(len(s), 1)) if s else 0.0
    ).astype("float32")

    # ---- Address features ----
    df["addr_is_peachtree"] = addr_lc.str.contains("peachtree", regex=False).astype("float32")
    df["addr_is_iconic"] = addr_lc.apply(
        lambda s: float(_matches_any(s, config.ICONIC_STREETS))
    ).astype("float32")
    df["addr_has_highway"] = addr_lc.apply(
        lambda s: float(_matches_any(s, config.HIGHWAY_TOKENS))
    ).astype("float32")
    df["addr_has_suite"] = addr_raw.apply(lambda s: float(bool(_SUITE_RE.search(s)))).astype(
        "float32"
    )

    df["addr_house_num"] = addr_raw.apply(_extract_house_num).astype("float32")
    df["addr_house_num_log"] = np.log1p(df["addr_house_num"].fillna(0.0)).astype("float32")
    df["addr_n_words"] = addr_raw.str.split().apply(len).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Geographic aggregates of physical attributes (no target leakage)
# ---------------------------------------------------------------------------


def _property_level_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to one row per property with mean physical attributes. Used for
    computing geographic aggregates at the property level so multi-unit
    properties don't dominate.
    """
    keep_numeric = [c for c in config.GEO_AGG_NUMERICS if c in df.columns]
    geo_cols = [g for g, _ in config.GEO_AGG_LEVELS if g in df.columns]
    cols = ["property_id"] + geo_cols + keep_numeric
    sub = df[cols].copy()
    agg_dict = {c: "mean" for c in keep_numeric}
    for g in geo_cols:
        agg_dict[g] = "first"
    return sub.groupby("property_id", as_index=False).agg(agg_dict)


def add_geo_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (geo_col, alias) and each numeric attribute, add columns:
        {numeric}_{alias}_mean,  {numeric}_{alias}_median,
        {numeric}_{alias}_std,   {numeric}_{alias}_count

    These describe the typical physical profile of properties in the row's
    geographic neighborhood (zipcode, sub_market, county). They use no rent
    information, so leakage is not a concern and they can be computed once
    on the full dataset.
    """
    if not all(g in df.columns for g, _ in config.GEO_AGG_LEVELS):
        return df

    prop_df = _property_level_attributes(df)
    df = df.copy()

    for geo_col, alias in config.GEO_AGG_LEVELS:
        if geo_col not in prop_df.columns:
            continue
        for num in config.GEO_AGG_NUMERICS:
            if num not in prop_df.columns:
                continue
            grouped = prop_df.groupby(geo_col, observed=True)[num]
            stats = pd.DataFrame(
                {
                    f"{num}_{alias}_mean": grouped.transform("mean"),
                    f"{num}_{alias}_median": grouped.transform("median"),
                    f"{num}_{alias}_std": grouped.transform("std"),
                    f"{num}_{alias}_count": grouped.transform("count"),
                }
            )
            stats[geo_col] = prop_df[geo_col].values
            stats["property_id"] = prop_df["property_id"].values
            # Aggregate stats are property-level; merge back onto unit rows.
            merge_cols = ["property_id"] + [
                c for c in stats.columns if c not in {"property_id", geo_col}
            ]
            df = df.merge(stats[merge_cols], on="property_id", how="left")

    # Cast aggregated columns to float32 for compactness.
    for c in df.columns:
        if any(
            c.endswith(f"_{a}_{s}") for _, a in config.GEO_AGG_LEVELS for s in config.GEO_AGG_STATS
        ):
            df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------------------
# Z-score deviations from neighborhood norms (no target leakage)
# ---------------------------------------------------------------------------


def add_zscore_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (geo_col, alias) and numeric attribute, add a z-score column:
        {numeric}_{alias}_z = (value - geo_mean) / geo_std

    Captures "how unusual is this unit relative to its neighborhood" — a
    pattern Kaggle solutions consistently find useful for tabular regression.
    Requires that geo aggregates have already been added.
    """
    df = df.copy()
    for _geo_col, alias in config.ZSCORE_GEO_LEVELS:
        for num in config.ZSCORE_NUMERICS:
            mean_col = f"{num}_{alias}_mean"
            std_col = f"{num}_{alias}_std"
            if num not in df.columns or mean_col not in df.columns or std_col not in df.columns:
                continue
            std_safe = df[std_col].replace(0.0, np.nan)
            z = (df[num] - df[mean_col]) / std_safe
            df[f"{num}_{alias}_z"] = z.astype("float32")
    return df


# ---------------------------------------------------------------------------
# Out-of-fold target encoding (Bayesian smoothed)
# ---------------------------------------------------------------------------


def bayesian_target_encode(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    column: str,
    target: str,
    smoothing: float = 20.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute smoothed target means for each level of `column` using only
    `train_df`, and apply them to both `train_df` and `valid_df`.

    Returns (train_encoded, valid_encoded) Series aligned to inputs.
    """
    global_mean = train_df[target].mean()
    stats = train_df.groupby(column, observed=True)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )

    train_enc = train_df[column].map(smooth).fillna(global_mean).astype("float32")
    valid_enc = valid_df[column].map(smooth).fillna(global_mean).astype("float32")
    return train_enc, valid_enc


# ---------------------------------------------------------------------------
# Out-of-fold k-NN rent aggregates
# ---------------------------------------------------------------------------


def _property_level_pool(train_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Reduce training rows to one row per property (mean rent + lat/lon)
    so the BallTree contains spatial neighbors at the property level
    rather than the row level (which would bias toward big properties).
    """
    pool = (
        train_df.dropna(subset=["latitude", "longitude", target])
        .groupby("property_id", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            rent_mean=(target, "mean"),
        )
    )
    return pool


def compute_oof_knn(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
    k_list: Iterable[int] = config.KNN_K_LIST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each k in `k_list`, add columns knn{k}_rent_mean and knn{k}_rent_std
    to both `train_df` and `valid_df`. Distances are great-circle on
    (lat, lon).

    For training rows, the property's own rent must be excluded from its
    own neighbor pool; we accomplish this by querying k+1 neighbors and
    masking out matches with the same property_id.

    For validation rows, the property is (by GroupKFold construction)
    not in the train pool, so no exclusion is needed.
    """
    k_list = list(k_list)
    k_max = max(k_list)

    pool = _property_level_pool(train_df, target)
    if len(pool) == 0:
        # Should never happen, but stay defensive.
        for k in k_list:
            for stat in ("mean", "std"):
                train_df[f"knn{k}_rent_{stat}"] = np.nan
                valid_df[f"knn{k}_rent_{stat}"] = np.nan
        return train_df, valid_df

    coords = np.radians(pool[["latitude", "longitude"]].values)
    rents = pool["rent_mean"].values
    pool_pids = pool["property_id"].values

    tree = BallTree(coords, metric="haversine")

    # ---- Train rows: query k_max + 1 neighbors and self-exclude.
    train_df = train_df.copy()
    train_q = np.radians(train_df[["latitude", "longitude"]].fillna(0).values)
    has_geo_train = train_df["latitude"].notna().values & train_df["longitude"].notna().values

    n_query = min(k_max + 1, len(pool))
    _, train_idx = tree.query(train_q, k=n_query)
    train_pids = train_df["property_id"].values

    for k in k_list:
        means = np.full(len(train_df), np.nan, dtype=np.float32)
        stds = np.full(len(train_df), np.nan, dtype=np.float32)
        for i in range(len(train_df)):
            if not has_geo_train[i]:
                continue
            neigh = train_idx[i]
            # Drop self-property matches.
            mask = pool_pids[neigh] != train_pids[i]
            kept = neigh[mask][:k]
            if len(kept) == 0:
                continue
            vals = rents[kept]
            means[i] = vals.mean()
            stds[i] = vals.std() if len(vals) > 1 else 0.0
        train_df[f"knn{k}_rent_mean"] = means
        train_df[f"knn{k}_rent_std"] = stds

    # ---- Validation rows: simple k-NN (no exclusion needed).
    valid_df = valid_df.copy()
    valid_q = np.radians(valid_df[["latitude", "longitude"]].fillna(0).values)
    has_geo_valid = valid_df["latitude"].notna().values & valid_df["longitude"].notna().values

    n_query_v = min(k_max, len(pool))
    _, valid_idx = tree.query(valid_q, k=n_query_v)

    for k in k_list:
        means = np.full(len(valid_df), np.nan, dtype=np.float32)
        stds = np.full(len(valid_df), np.nan, dtype=np.float32)
        for i in range(len(valid_df)):
            if not has_geo_valid[i]:
                continue
            kept = valid_idx[i, :k]
            vals = rents[kept]
            means[i] = vals.mean()
            stds[i] = vals.std() if len(vals) > 1 else 0.0
        valid_df[f"knn{k}_rent_mean"] = means
        valid_df[f"knn{k}_rent_std"] = stds

    return train_df, valid_df


# ---------------------------------------------------------------------------
# Combined OOF feature wrapper
# ---------------------------------------------------------------------------


def add_oof_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_for_te: str,
    target_for_knn: str,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute OOF target encodings + k-NN aggregates + neighborhood $/sqft +
    comparable-rent features. All are computed using only `train_df`'s rents
    so that `valid_df` never sees its own labels.

    Args
    ----
    target_for_te
        Column to use for target encoding (typically log_rent when LOG_TARGET).
    target_for_knn
        Column to use for k-NN neighbor aggregates (raw rent for human-readable
        feature scales; log doesn't help interpretability and trees handle
        either fine).
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    # 1) Target encodings.
    for col in config.TARGET_ENCODE_COLS:
        if col not in train_df.columns:
            continue
        train_enc, valid_enc = bayesian_target_encode(
            train_df, valid_df, column=col, target=target_for_te, smoothing=smoothing
        )
        train_df[f"{col}_te"] = train_enc
        valid_df[f"{col}_te"] = valid_enc

    # 2) k-NN (geographic only, all rent levels).
    train_df, valid_df = compute_oof_knn(train_df, valid_df, target=target_for_knn)

    # 3) Neighborhood median rent_per_sqft -> sqft x nbhd_psf interactions.
    train_df, valid_df = add_nbhd_psf_features(train_df, valid_df, target=target_for_knn)

    # 4) comparable_rent_median (geo + size + bedroom matched, KNN-by-distance).
    train_df, valid_df = add_comparable_rent(train_df, valid_df, target=target_for_knn)

    # 4b) hier_comp_rent_te: Bayesian-smoothed cell encoding on a composite key
    # with hierarchical fallback. Complementary to comparable_rent_median.
    train_df, valid_df = add_hierarchical_comp_te(train_df, valid_df, target=target_for_knn)

    # 4c) Competition features: same-bed neighbor PSF p25 / p75 / IQR within
    # COMPETITOR_RADIUS_MI. Self-property excluded from training pool.
    train_df, valid_df = add_competitor_psf_features(train_df, valid_df, target=target_for_knn)

    # 5) Interactions that REQUIRE the OOF target encodings to exist. These
    # multiply a continuous physical attribute by an OOF-encoded categorical,
    # which trees can't simulate with a single split (each split chooses one
    # variable; an interaction is a product). Computing them on top of *_te
    # columns means the encoded value is itself out-of-fold, so no leakage.
    for col_df in (train_df, valid_df):
        if "sqft" in col_df.columns and "sub_market_te" in col_df.columns:
            col_df["sqft_x_subm_te"] = (
                col_df["sqft"].astype("float32") * col_df["sub_market_te"].astype("float32")
            ).astype("float32")
        if "year_built" in col_df.columns and "zipcode_te" in col_df.columns:
            col_df["year_x_zip_te"] = (
                col_df["year_built"].astype("float32") * col_df["zipcode_te"].astype("float32")
            ).astype("float32")
        if "beds" in col_df.columns and "sub_market_te" in col_df.columns:
            col_df["beds_x_subm_te"] = (
                col_df["beds"].astype("float32") * col_df["sub_market_te"].astype("float32")
            ).astype("float32")

    return train_df, valid_df


# ---------------------------------------------------------------------------
# OOF neighborhood $/sqft and sqft x nbhd_psf interactions
# ---------------------------------------------------------------------------


def add_nbhd_psf_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (geo_col, alias) in NBHD_PSF_LEVELS:
        - compute median rent_per_sqft in the training fold per geo level
        - broadcast as `{alias}_psf_median`
        - multiply by row sqft to get `expected_rent_{alias}` (interaction)

    rent_per_sqft is recomputed from the (unleaky) raw rent + sqft on the
    training rows only. Rows with sqft <= 0 or NaN are skipped.
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    # Build a per-geo median table from training rows with valid sqft + rent.
    psf_train = train_df[(train_df["sqft"].fillna(0) > 0) & train_df[target].notna()].copy()
    psf_train["rent_per_sqft"] = psf_train[target].astype(np.float64) / psf_train["sqft"].astype(
        np.float64
    )
    global_med = float(psf_train["rent_per_sqft"].median()) if len(psf_train) else np.nan

    for geo_col, alias in config.NBHD_PSF_LEVELS:
        if geo_col not in train_df.columns:
            continue
        med = psf_train.groupby(geo_col, observed=True)["rent_per_sqft"].median().astype("float32")
        psf_col = f"{alias}_psf_median"
        train_df[psf_col] = train_df[geo_col].map(med).astype("float32")
        valid_df[psf_col] = valid_df[geo_col].map(med).astype("float32")

        # Fill missing geos with global median.
        if not np.isnan(global_med):
            train_df[psf_col] = train_df[psf_col].fillna(global_med).astype("float32")
            valid_df[psf_col] = valid_df[psf_col].fillna(global_med).astype("float32")

        # Interaction: expected rent if this unit were "average" for its geo.
        exp_col = f"expected_rent_{alias}"
        train_df[exp_col] = (train_df[psf_col] * train_df["sqft"].fillna(0)).astype("float32")
        valid_df[exp_col] = (valid_df[psf_col] * valid_df["sqft"].fillna(0)).astype("float32")

    return train_df, valid_df


# ---------------------------------------------------------------------------
# OOF comparable-rent feature (geo + size + bedroom matched)
# ---------------------------------------------------------------------------


def add_comparable_rent(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each row, find up to K nearest comparable PROPERTIES in the training
    fold matching:
        - same `beds` (rounded to int)
        - sqft within +/- COMPARABLE_SQFT_TOL of row's sqft
        - then nearest by haversine distance on (lat, lon)

    Output column: COMPARABLE_FEATURE = median rent of those comparables.
    Excludes the row's own property_id from training (relevant only for
    training rows, since GroupKFold already isolates valid properties).

    Falls back to broader matches when strict filter yields fewer than K
    candidates: drop sqft filter, then drop beds filter, then nearest in geo.
    """
    K = config.COMPARABLE_K
    SQFT_TOL = config.COMPARABLE_SQFT_TOL
    feat = config.COMPARABLE_FEATURE

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    # Build property-level pool from training rows.
    pool = (
        train_df.dropna(subset=["latitude", "longitude", "sqft", target])
        .groupby("property_id", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            sqft=("sqft", "mean"),
            beds=("beds", "mean"),
            rent_mean=(target, "mean"),
        )
    )
    if len(pool) == 0:
        train_df[feat] = np.nan
        valid_df[feat] = np.nan
        return train_df, valid_df

    pool["beds_int"] = pool["beds"].round().astype("Int64")
    pool_coords = np.radians(pool[["latitude", "longitude"]].values)
    pool_sqft = pool["sqft"].values
    pool_beds = pool["beds_int"].values
    pool_pids = pool["property_id"].values
    pool_rents = pool["rent_mean"].values

    tree_full = BallTree(pool_coords, metric="haversine")

    def _comparable_for_rows(rows: pd.DataFrame, exclude_self: bool) -> np.ndarray:
        n = len(rows)
        out = np.full(n, np.nan, dtype=np.float32)
        if n == 0:
            return out

        row_lat = rows["latitude"].values
        row_lon = rows["longitude"].values
        row_sqft = rows["sqft"].values
        row_beds = rows["beds"].round().astype("Int64").values
        row_pid = rows["property_id"].values

        # Pull a moderate neighbor pool, then filter; fall back if needed.
        # 80 is enough to find K=10 strict matches in dense Atlanta zips.
        n_query = min(80, len(pool))
        query_coords = np.radians(
            np.column_stack(
                [
                    np.where(np.isnan(row_lat), 0.0, row_lat),
                    np.where(np.isnan(row_lon), 0.0, row_lon),
                ]
            )
        )
        _, neighbor_idx = tree_full.query(query_coords, k=n_query)

        for i in range(n):
            if np.isnan(row_lat[i]) or np.isnan(row_lon[i]):
                continue
            cand = neighbor_idx[i]
            # Exclude self property.
            if exclude_self:
                cand = cand[pool_pids[cand] != row_pid[i]]

            # Strict filter: same beds + sqft band.
            sqft_lo, sqft_hi = (
                (row_sqft[i] * (1 - SQFT_TOL), row_sqft[i] * (1 + SQFT_TOL))
                if not np.isnan(row_sqft[i])
                else (None, None)
            )
            beds_i = row_beds[i] if pd.notna(row_beds[i]) else None

            def _filter(cand_, beds_match: bool, sqft_match: bool):
                ok = np.ones(len(cand_), dtype=bool)
                if beds_match and beds_i is not None:
                    ok &= pool_beds[cand_] == beds_i
                if sqft_match and sqft_lo is not None:
                    sq = pool_sqft[cand_]
                    ok &= (sq >= sqft_lo) & (sq <= sqft_hi)
                return cand_[ok]

            kept = _filter(cand, True, True)
            if len(kept) < K:
                kept = _filter(cand, True, False)
            if len(kept) < K:
                kept = _filter(cand, False, True)
            if len(kept) < K:
                kept = cand
            kept = kept[:K]
            if len(kept) == 0:
                continue
            out[i] = np.median(pool_rents[kept])
        return out

    train_df[feat] = _comparable_for_rows(train_df, exclude_self=True)
    valid_df[feat] = _comparable_for_rows(valid_df, exclude_self=False)
    return train_df, valid_df


# ---------------------------------------------------------------------------
# OOF hierarchical comparable target encoding
# ---------------------------------------------------------------------------


def _bayes_cell_mean(
    train_df: pd.DataFrame,
    cell_key_col: str,
    target: str,
    smoothing: float,
) -> tuple[pd.Series, pd.Series, float]:
    """
    Compute Bayesian-smoothed mean of `target` per level of `cell_key_col`
    on `train_df`. Returns (smoothed_mean_series, count_series, global_mean).
    """
    global_mean = float(train_df[target].mean())
    grouped = train_df.groupby(cell_key_col, observed=True)[target]
    means = grouped.mean()
    counts = grouped.count()
    smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return smoothed.astype("float32"), counts.astype("int32"), global_mean


def add_hierarchical_comp_te(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hierarchical comparable target encoding. For each row we look up the
    smoothed mean rent of the row's cell at progressively coarser granularities
    and use the *first level whose count >= HIER_CELL_MIN_COUNT*. Falls back
    to global mean if none qualify.

    Cell hierarchy (finest -> coarsest):
        L1: sub_market | beds_int | baths_int | sqft_bucket | age_bucket
        L2: sub_market | beds_int | baths_int | sqft_bucket
        L3: sub_market | beds_int | baths_int
        L4: sub_market | beds_int
        L5: sub_market

    Why this isn't redundant with comparable_rent_median:
        comparable_rent_median picks K nearest properties by haversine distance
        within a sqft band. This picks a fixed cell with all matching members.
        Cell-encoding is more stable in dense submarkets (less noisy) and
        captures bed/bath x age effects that distance-weighted KNN averages
        out. The two features carry correlated but distinct signal.

    Leakage safety under GroupKFold:
        Even when the held-out property is the only one in a fine cell, the
        coarser fallback cells (L4/L5) span thousands of rows from many
        properties. Smoothing further protects against tiny-cell noise.

    Excludes the row's own property from train-row encoding (within the train
    fold), matching the leave-one-out contract used elsewhere.
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    feat = config.HIER_COMP_TE_FEATURE
    min_count = config.HIER_CELL_MIN_COUNT
    smoothing = config.HIER_CELL_SMOOTHING

    # Required columns; if any are missing fill with global mean and bail.
    required = [
        "sub_market",
        "beds_int",
        "baths_int",
        "sqft_bucket",
        "age_bucket",
        "property_id",
        target,
    ]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        gm = float(train_df[target].mean()) if target in train_df.columns else 0.0
        train_df[feat] = np.float32(gm)
        valid_df[feat] = np.float32(gm)
        return train_df, valid_df

    # Build composite cell keys. Use string concatenation; cardinalities are
    # bounded (~50 submarkets x 5 beds x 5 baths x 7 sqft x 5 age = ~44k cells
    # max, but most empty in practice — actual <1k).
    def _build_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
        sm = df["sub_market"].fillna("__missing__").astype(str)
        bi = df["beds_int"].astype(str)
        bai = df["baths_int"].astype(str)
        sb = df["sqft_bucket"].astype(str)
        ab = df["age_bucket"].astype(str)
        return {
            "L1": sm + "|" + bi + "|" + bai + "|" + sb + "|" + ab,
            "L2": sm + "|" + bi + "|" + bai + "|" + sb,
            "L3": sm + "|" + bi + "|" + bai,
            "L4": sm + "|" + bi,
            "L5": sm,
        }

    train_keys = _build_keys(train_df)
    valid_keys = _build_keys(valid_df)

    # Compute smoothed mean + count per cell at each level using train rents.
    levels: list[tuple[str, pd.Series, pd.Series, float]] = []
    global_mean = float(train_df[target].mean())
    tmp_train = train_df[[target, "property_id"]].copy()
    for lvl in ("L1", "L2", "L3", "L4", "L5"):
        tmp_train["_cell"] = train_keys[lvl].values
        smoothed, counts, _ = _bayes_cell_mean(
            tmp_train, "_cell", target=target, smoothing=smoothing
        )
        levels.append((lvl, smoothed, counts, global_mean))

    # ---- Validation rows: simple lookup with hierarchical fallback ----
    val_out = np.full(len(valid_df), global_mean, dtype=np.float32)
    val_assigned = np.zeros(len(valid_df), dtype=bool)
    for lvl, smoothed, counts, _ in levels:
        keys = valid_keys[lvl].values
        # cell mean and count for each valid row's cell at this level
        cm = pd.Series(keys).map(smoothed).to_numpy(dtype="float32")
        cc = pd.Series(keys).map(counts).fillna(0).to_numpy(dtype="int32")
        ok = (~val_assigned) & np.isfinite(cm) & (cc >= min_count)
        val_out[ok] = cm[ok]
        val_assigned |= ok
        if val_assigned.all():
            break
    valid_df[feat] = val_out

    # ---- Training rows: leave-one-out at L1, fallback otherwise ----
    # For training rows, the row's own rent is in the cell's mean. We approximate
    # leave-one-out by recomputing cell means at L1 *excluding* the row's own
    # property_id. For coarser levels, the row's own contribution is diluted
    # enough by smoothing + many properties that we don't bother (matches the
    # comparable_rent_median treatment).
    tr_out = np.full(len(train_df), global_mean, dtype=np.float32)
    tr_assigned = np.zeros(len(train_df), dtype=bool)

    # L1 leave-one-property-out:
    pids = train_df["property_id"].astype(str).values
    l1_keys = train_keys["L1"].values
    rents = train_df[target].astype("float64").values

    # Build per-cell rent sum, count, and per-(cell, property) sum/count
    # so we can subtract the row's own property contribution.
    cell_df = pd.DataFrame({"cell": l1_keys, "pid": pids, "rent": rents})
    cell_agg = cell_df.groupby("cell")["rent"].agg(["sum", "count"])
    cell_pid_agg = cell_df.groupby(["cell", "pid"])["rent"].agg(["sum", "count"])

    # Vectorized per-row LOO mean at L1.
    cell_sum = cell_df["cell"].map(cell_agg["sum"]).to_numpy(dtype="float64")
    cell_cnt = cell_df["cell"].map(cell_agg["count"]).to_numpy(dtype="int32")
    pair_idx = list(zip(cell_df["cell"].values, cell_df["pid"].values))
    pair_sum = cell_pid_agg["sum"].reindex(pair_idx).to_numpy(dtype="float64")
    pair_cnt = cell_pid_agg["count"].reindex(pair_idx).to_numpy(dtype="int32")

    loo_sum = cell_sum - pair_sum
    loo_cnt = cell_cnt - pair_cnt
    with np.errstate(divide="ignore", invalid="ignore"):
        loo_mean = np.where(loo_cnt > 0, loo_sum / np.maximum(loo_cnt, 1), np.nan)
    loo_smoothed = ((loo_cnt * loo_mean + smoothing * global_mean) / (loo_cnt + smoothing)).astype(
        "float32"
    )
    ok_l1 = np.isfinite(loo_smoothed) & (loo_cnt >= min_count)
    tr_out[ok_l1] = loo_smoothed[ok_l1]
    tr_assigned |= ok_l1

    # Fall back through L2..L5 using the (non-LOO) smoothed means.
    for lvl, smoothed, counts, _ in levels[1:]:
        keys = train_keys[lvl].values
        cm = pd.Series(keys).map(smoothed).to_numpy(dtype="float32")
        cc = pd.Series(keys).map(counts).fillna(0).to_numpy(dtype="int32")
        ok = (~tr_assigned) & np.isfinite(cm) & (cc >= min_count)
        tr_out[ok] = cm[ok]
        tr_assigned |= ok
        if tr_assigned.all():
            break
    train_df[feat] = tr_out

    return train_df, valid_df


# ---------------------------------------------------------------------------
# Final feature column resolution
# ---------------------------------------------------------------------------


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_features, categorical_features) actually present in df,
    including dynamically added _te / knn / _ord / aggregate / z-score
    / comparable-rent columns.
    """
    numeric = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    numeric += [f"{c}_te" for c in config.TARGET_ENCODE_COLS if f"{c}_te" in df.columns]
    numeric += [c for c in config.KNN_FEATURES if c in df.columns]
    numeric += [f"{g}_ord" for g in config.ORDINAL_GRADE_FEATURES if f"{g}_ord" in df.columns]
    numeric += [c for c in config.BOOLEAN_FEATURES if c in df.columns]

    # Text-derived features (property_name + street_address).
    numeric += [c for c in config.TEXT_BOOLEAN_FLAGS if c in df.columns]
    numeric += [c for c in config.TEXT_NUMERIC_FEATURES if c in df.columns]

    # Geo aggregates (mean/median/std/count of physical attrs by geo level).
    for _, alias in config.GEO_AGG_LEVELS:
        for num in config.GEO_AGG_NUMERICS:
            for stat in config.GEO_AGG_STATS:
                col = f"{num}_{alias}_{stat}"
                if col in df.columns:
                    numeric.append(col)

    # Z-score deviations from geo means.
    for _, alias in config.ZSCORE_GEO_LEVELS:
        for num in config.ZSCORE_NUMERICS:
            col = f"{num}_{alias}_z"
            if col in df.columns:
                numeric.append(col)

    # Neighborhood $/sqft + sqft x nbhd_psf interactions (OOF).
    for _, alias in config.NBHD_PSF_LEVELS:
        for col in (f"{alias}_psf_median", f"expected_rent_{alias}"):
            if col in df.columns:
                numeric.append(col)

    # Comparable rent.
    if config.COMPARABLE_FEATURE in df.columns:
        numeric.append(config.COMPARABLE_FEATURE)

    # Hierarchical comparable target encoding (cell-based, OOF).
    if config.HIER_COMP_TE_FEATURE in df.columns:
        numeric.append(config.HIER_COMP_TE_FEATURE)

    # Bucket keys (kept as numeric ordinal features — they encode coarse
    # discretizations of sqft / property_age / beds / baths that already-tuned
    # trees can re-split on if useful).
    for col in ("sqft_bucket", "age_bucket", "beds_int", "baths_int"):
        if col in df.columns:
            numeric.append(col)

    # Static interactions (multiplicative + ratio between physical attrs).
    for col in (
        "sqft_x_beds",
        "sqft_per_bed",
        "baths_per_bed",
        "sqft_x_buckhead_km",
        "beds_x_buckhead_km",
        "year_x_buckhead_km",
        "age_x_sqft",
    ):
        if col in df.columns:
            numeric.append(col)

    # OOF-dependent interactions (continuous attr x target-encoded categorical).
    for col in ("sqft_x_subm_te", "year_x_zip_te", "beds_x_subm_te"):
        if col in df.columns:
            numeric.append(col)

    # Property structural / BTR-typology features (added 2026-05-01).
    # Per-property aggregates that distinguish BTR (single unit type, single
    # bed count, recent, large) from apartment buildings (mixed product).
    for col in (
        "property_n_unit_types",
        "property_beds_nunique",
        "property_baths_nunique",
        "property_sqft_range_pct",
        "btr_likely_score",
        "is_btr_likely",
    ):
        if col in df.columns:
            numeric.append(col)

    # Unit-subtype features (added 2026-05-01). Use actual rental-unit
    # counts (num_units_subtype) to ground within-property heterogeneity
    # in market presence rather than row counts. Replaces the
    # add_within_property_features experiment which heavily overfit.
    for col in (
        "log_num_units_subtype",
        "unit_subtype_share",
        "is_dominant_unit_type",
        "unit_subtype_count_rank_pct",
    ):
        if col in df.columns:
            numeric.append(col)

    # Competition / neighbor-PSF features (added 2026-05-01, expanded to
    # 3 radii: 0.5mi / 1mi / 2mi).
    competition_cols: list[str] = []
    for _, label in config.COMPETITOR_RADII:
        competition_cols.append(f"n_competitors_within_{label}_same_beds")
        for stat in ("p25", "p50", "p75", "iqr"):
            competition_cols.append(f"competitor_psf_{stat}_within_{label}")
    for col in competition_cols:
        if col in df.columns:
            numeric.append(col)

    # De-duplicate while preserving order.
    seen = set()
    numeric = [c for c in numeric if not (c in seen or seen.add(c))]

    categorical = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical
