"""
config.py
=========
Central configuration for the Yardi rent prediction pipeline.

Edit feature lists, paths, and hyperparameter search space here. Other
modules (data_processing, feature_engineering, train) import from this
file so changes propagate automatically.
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels above this file:
#   src/prime_mfr/config.py  ->  parents[2] is the repo root.
PROJECT_DIR: Path = Path(__file__).resolve().parents[2]

# Inputs.
# Switched 2026-05-01 from pretraining.parquet (joined on Anthropic side) to
# v2 = build_pretraining_v2.py output, which joins the new April-2026
# unit-mix-enriched + property-enriched files locally and adds the
# `haystacks_unit_type` column (APARTMENT/TOWNHOUSE/ROWHOUSE/DETACHED_ENTRY/
# DETACHED_LUXURY). Original pretraining.parquet kept on disk for ablation.
RAW_PARQUET: Path = PROJECT_DIR / "pretraining_v2.parquet"
ENRICHED_PARQUET: Path = PROJECT_DIR / "eda" / "pretraining_enriched_v2.parquet"
LANDMARKS_JSON: Path = PROJECT_DIR / "eda" / "atlanta_landmarks.json"
MARTA_STATIONS_JSON: Path = PROJECT_DIR / "eda" / "marta_stations.json"

# Historical rent panel (April-2026 enriched, single-snapshot dump 2026-04-20).
# 708,825 rows × 7 cols at (property_id, unit_type, period) granularity, monthly
# from 2015-01-01 to 2026-03-01. The 2026-03-01 row matches the v2 training
# target rent exactly (verified) -- so lag features must NEVER include the
# 2026-03 observation. lag-1 = Feb 2026 is the closest valid feature.
HIST_RENT_PATH: Path = PROJECT_DIR / "artifacts" / "042026-hist-rent-12060-12060.parquet"
HIST_RENT_TARGET_PERIOD: str = "2026-03-01"  # the v2 training period

# Outputs (created by train.py).
ARTIFACTS_DIR: Path = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_PATH: Path = ARTIFACTS_DIR / "best_params.json"
BEST_CB_PARAMS_PATH: Path = ARTIFACTS_DIR / "best_catboost_params.json"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
OOF_PRED_PATH: Path = ARTIFACTS_DIR / "oof_predictions.parquet"
FEATURE_IMP_PATH: Path = ARTIFACTS_DIR / "feature_importance.csv"
RUN_LOG_PATH: Path = ARTIFACTS_DIR / "run.log"

# ---------------------------------------------------------------------------
# Modeling controls
# ---------------------------------------------------------------------------

TARGET: str = "rent"
LOG_TARGET: bool = True  # train on log1p(rent), invert with expm1
USE_PSF_TARGET: bool = False  # if True, train on log1p(rent / sqft) and
# multiply by sqft on prediction. PSF has
# lower log-variance than raw rent because
# sqft is the dominant scale factor — every
# base learns a tighter target distribution.
GROUP_KEY: str = "property_id"  # GroupKFold splits keep all units of
# a property in the same fold
N_FOLDS: int = 5
RANDOM_STATE: int = 42
N_OPTUNA_TRIALS: int = 30  # 30 is a reasonable industry default
# for ~7k rows (set to 0 to skip tuning)
EARLY_STOPPING_ROUNDS: int = 100
NUM_BOOST_ROUND: int = 4000  # capped by early stopping


# ---------------------------------------------------------------------------
# Target leakage exclusions
# ---------------------------------------------------------------------------

# These columns are property-level rent aggregates that include the target
# row's own rent. Using them at train time would inflate metrics; at inference
# time they would not be available.
LEAKAGE_COLS: tuple[str, ...] = (
    "property_average_rent",
    "property_average_rent_sqft",
)

# Identifiers / textual fields that should not enter the model directly.
ID_COLS: tuple[str, ...] = (
    "property_id",
    "source_property_id",
    "property_name",
    "street_address",
    "website",
    "owner_name",
    "source_name",
    "period",
    "occupancy_date",
    "latest_sale_date",
    "latest_sale_price_date",
    "date_x",
    "date_y",
    "parcel_complete",
    "city",
    "state",
    "market",
    "unit_mix",  # JSON dump - already parsed into sqft/beds/baths
    "census_cbsa_geoid",  # constant (Atlanta CBSA only)
    "census_zcta5_geoid",  # equivalent to zipcode
    "census_tract_geoid",  # mostly null for this dataset
)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Numeric features (model sees as floats).
NUMERIC_FEATURES: list[str] = [
    # Unit-level
    "sqft",
    "beds",
    "baths",
    "unit_garage",
    # Property structure
    "num_units",
    "num_units_subtype",  # added 2026-05-01: # actual rental units of
    # THIS unit type within the property (vs
    # num_units = property-wide total)
    "year_built",
    "min_stories",
    "max_stories",
    "number_of_stories",
    "num_buildings",
    "total_parking_spaces",
    "lot_size_in_acres",
    "lot_size_in_square_feet",
    # Property economics / status (non-leaky)
    "occupancy_rate",
    "latest_sale_price_per_sqft",
    "latest_sale_price_per_unit",
    "latest_sale_price_total",
    # Geography (raw + engineered)
    "latitude",
    "longitude",
    "dist_buckhead_km",
    "dist_midtown_km",
    "dist_atl_airport_km",  # re-added 2026-07-15 alongside
    # dist_atl_airport_zone (categorical, see CATEGORICAL_FEATURES) --
    # now testing continuous + categorical together, same pattern as the
    # earlier dist_marta_km + num_marta_stations_within_1mi combo test.
    # dist_downtown_km / dist_min_landmark_km remain removed (dropped
    # 2026-07-15 keeping only buckhead + midtown from that pair).
    # MARTA features removed 2026-07-15 -- tested three configurations
    # against the $76.38 buckhead+midtown baseline: dist_marta_km alone
    # ($77.16), dist_marta_km + num_marta_stations_within_1mi ($76.80),
    # and num_marta_stations_within_1mi alone ($76.61). All three landed
    # worse than baseline in a consistent (not noise-scattered) ordering,
    # suggesting MARTA proximity is redundant with signal the model
    # already gets from buckhead/midtown distance, H3 cells, and
    # submarket/zip target encoding. add_marta_distance() /
    # add_marta_station_density() are untouched in engineering.py --
    # just not consumed here. Revisit if a future feature set changes
    # this picture.
    # Engineered (added later)
    "property_age",
    # Hist-rent lag features (added 2026-05-01). Per (property_id, unit_type)
    # series anchored to 2026-03 target. lag_1 = Feb 2026, lag_3 = Dec 2025,
    # lag_12 = Mar 2025, lag_24 = Mar 2024. yoy = lag_1 / lag_13 - 1. NaN
    # for series too short or with null at the lag month.
    # NOTE: this is the PRIMARY (repricing) model with hist features enabled.
    # The cold-start variant (no hist features) is trained separately by
    # commenting these out and saving artifacts to *.coldstart.* paths.
    "hist_rent_lag_1m",
    "hist_rent_lag_3m",
    "hist_rent_lag_12m",
    "hist_rent_lag_24m",
    "hist_rent_yoy",
]

# Categorical features (LightGBM native categorical handling).
CATEGORICAL_FEATURES: list[str] = [
    "unit_type",
    "unit_mix_type",
    "haystacks_unit_type",  # added 2026-05-01: finer-grained typology from
    # April-2026 enriched feed. Splits Apartment vs
    # Townhouse into APARTMENT / TOWNHOUSE / ROWHOUSE
    # / DETACHED_ENTRY / DETACHED_LUXURY (the latter
    # 3 are the BTR-style segments).
    "sub_market",
    "zipcode",
    "county",
    "parking_type",
    "garage",
    "h3_res6",
    "h3_res8",
    "brand",  # extracted from property_name first token(s)
    "street_type",  # extracted from street_address suffix
    "addr_dir",  # NE/NW/SE/SW/N/S/E/W
    "dist_atl_airport_zone",  # added 2026-07-15: near/hot_zone/far bucketing
    # of dist_atl_airport_km (see ATL_AIRPORT_ZONE_EDGES above). Categorical,
    # not ordinal, because the relationship is non-monotonic. Only reaches
    # lgbm_l1/cat_q50 -- the KNN trainer doesn't consume CATEGORICAL_FEATURES.
]

# Boolean text-derived flags (will be added to BOOLEAN_FEATURES at module load
# via extend below to avoid duplication).
TEXT_BOOLEAN_FLAGS: list[str] = [
    "name_kw_premium",
    "name_kw_midmarket",
    "name_kw_older",
    "name_kw_conversion",
    "name_kw_phase",
    "name_starts_the",
    "name_has_at",
    "name_has_digit",
    "name_has_ampersand",
    "name_claims_premium_subm",
    "name_subm_match",
    "addr_is_peachtree",
    "addr_is_iconic",
    "addr_has_highway",
    "addr_has_suite",
]

# Numeric text-derived continuous features.
TEXT_NUMERIC_FEATURES: list[str] = [
    "name_len",
    "name_n_words",
    "name_caps_ratio",
    "addr_house_num",
    "addr_house_num_log",
    "addr_n_words",
]

# Boolean / yes-no flags (mapped 0/1 in data_processing).
BOOLEAN_FEATURES: list[str] = [
    "is_yardi_btr",
    "is_hstx_btr",
    "is_yardi_unittype_btr",
    "is_leased_up",
    "is_mixed_use",
    "is_elevator_served",
    "has_controlled_access",
    "has_fitness_center",
    "has_business_center",
    "has_clubhouse",
    "has_garage",
    "has_media_room",
    "has_townhouse",
    "covered_parking",
    "rent_office",
    "wd_hookup",
    "wd_in_unit",
]

# Letter-grade columns are encoded as ordinals (A+=8, A=7, ..., D-=-3, NaN=NaN).
ORDINAL_GRADE_FEATURES: list[str] = [
    "property_quality",
    "location_quality",
]

GRADE_TO_ORDINAL: dict[str, int] = {
    "A+": 8,
    "A": 7,
    "A-": 6,
    "B+": 5,
    "B": 4,
    "B-": 3,
    "C+": 2,
    "C": 1,
    "C-": 0,
    "D+": -1,
    "D": -2,
    "D-": -3,
}

# Out-of-fold target-encoded features (added in feature_engineering).
TARGET_ENCODE_COLS: list[str] = [
    "sub_market",
    "zipcode",
    "h3_res6",
    "h3_res8",
    "unit_type",
    "haystacks_unit_type",  # added 2026-05-01: cardinality=5, all cells very
    # well populated -> Bayesian smoothing should give
    # a stable encoding for KNN bases (which can't
    # consume raw categoricals).
    "brand",  # extracted by text feature pipeline
    "street_type",  # extracted by text feature pipeline
    # "street_name",        # ABLATION: added 2026-05-01 but caused +$15 fold-1 lgbm regression
]

# k-NN aggregation features (one row per property, then merged back).
KNN_K_LIST: list[int] = [5, 10, 20]
KNN_FEATURES: list[str] = [f"knn{k}_rent_mean" for k in KNN_K_LIST] + [
    f"knn{k}_rent_std" for k in KNN_K_LIST
]

# ---------------------------------------------------------------------------
# Geographic aggregates (no target; computed on full data, leakage-safe)
# ---------------------------------------------------------------------------
# For each `geo_col` in GEO_AGG_LEVELS, group properties (deduped to one row
# per property) and compute mean/median/std/count of each physical feature
# in GEO_AGG_NUMERICS. Then merge back onto unit-rows. The resulting columns
# follow the naming convention "{numeric}_{geo}_{stat}", e.g. sqft_zip_mean.
GEO_AGG_LEVELS: list[tuple[str, str]] = [
    ("zipcode", "zip"),
    ("sub_market", "subm"),
    ("county", "cty"),
]
GEO_AGG_NUMERICS: list[str] = ["sqft", "beds", "baths", "property_age", "num_units"]
GEO_AGG_STATS: list[str] = ["mean", "median", "std", "count"]

# Z-score deviation columns (numeric values minus geo mean, divided by geo std).
# Naming: "{numeric}_{geo}_z".
ZSCORE_GEO_LEVELS: list[tuple[str, str]] = GEO_AGG_LEVELS
ZSCORE_NUMERICS: list[str] = ["sqft", "beds", "baths", "property_age", "num_units"]

# Out-of-fold neighborhood $/sqft features.
# For each (geo_col, alias) we compute the median rent_per_sqft within the
# training fold and broadcast to all rows. Then "{alias}_psf_median" times
# the row's sqft becomes "expected_rent_{alias}".
NBHD_PSF_LEVELS: list[tuple[str, str]] = [
    ("zipcode", "zip"),
    ("sub_market", "subm"),
]

# Comparable-rent feature (OOF, geo + size + bedroom matched).
COMPARABLE_K: int = 10  # number of nearest comparables
COMPARABLE_SQFT_TOL: float = 0.25  # ±25% sqft window
COMPARABLE_FEATURE: str = "comparable_rent_median"

# Hierarchical comparable target encoding (OOF, Bayesian-smoothed cell mean).
# Distinct from COMPARABLE_FEATURE: that one is KNN-by-distance among matching
# beds/sqft properties; this one is a *cell-encoding* on a fine composite key
# with hierarchical fallback. Cells span many properties so it's leakage-safe
# under GroupKFold (a held-out property's cell still has plenty of other
# properties to learn from). Captures patterns the KNN-distance approach
# averages out (e.g. 2BR/2BA vs 2BR/1BA at similar sqft and location).
#
# Levels searched in order; the first cell with >= HIER_CELL_MIN_COUNT wins.
# The smoothed mean of that cell becomes the feature value.
HIER_CELL_MIN_COUNT: int = 5
HIER_CELL_SMOOTHING: float = 12.0
HIER_COMP_TE_FEATURE: str = "hier_comp_rent_te"
# sqft buckets (right edges; last is +inf via np.digitize). Tuned to roughly
# equal-population bins on Atlanta MFR data.
SQFT_BUCKET_EDGES: tuple[float, ...] = (700.0, 850.0, 1000.0, 1150.0, 1350.0, 1700.0)
# property_age buckets (right edges).
AGE_BUCKET_EDGES: tuple[float, ...] = (5.0, 15.0, 30.0, 50.0)

# dist_atl_airport_km zone edges (right edges, km). Non-monotonic rent
# pattern: near-airport is cheaper, a "hot zone" 9-15km out is the most
# expensive AND most volatile (fewer, pricier outlier properties), then
# it settles back down and flattens past ~15km. Labels below match these
# 3 bands; used by add_airport_zone_feature() as a genuine categorical
# (not ordinal -- "hot_zone" isn't "more" than "near", it's a different
# regime), so LightGBM/CatBoost can split on it without assuming order.
ATL_AIRPORT_ZONE_EDGES: tuple[float, float] = (9.0, 15.0)
ATL_AIRPORT_ZONE_LABELS: tuple[str, str, str] = ("near", "hot_zone", "far")

# ---------------------------------------------------------------------------
# Property structural / BTR-typology features (added 2026-05-01)
# ---------------------------------------------------------------------------
# BTR (build-to-rent) properties have a sharply different PSF regime
# (~$0.90-$1.10 vs $1.50+ for apartments) and are concentrated in our
# worst-residuals subsegment. Structurally, they tend to have a single
# unit type, a single bed count, recent vintage, and large unit sqft —
# all derivable from the rent dataset itself with no rent/target leakage.
BTR_MIN_YEAR: int = 2015  # year_built >= this is the recent-vintage cut
BTR_MIN_SQFT: float = 1500.0  # median property sqft >= this
BTR_MAX_UNIT_TYPES: int = 1  # n_rows == this means single floor plan
BTR_MAX_BEDS_NUNIQUE: int = 1  # distinct bed counts within property

# ---------------------------------------------------------------------------
# Competition / neighbor-PSF features (added 2026-05-01)
# ---------------------------------------------------------------------------
# For each row we compute, against same-bed competitor PROPERTIES within a
# fixed radius:
#   - n_competitors_within_1mi_same_beds   (static, leakage-free count)
#   - competitor_psf_p25_within_1mi        (OOF: train-only neighbors)
#   - competitor_psf_p75_within_1mi        (OOF)
#   - competitor_psf_iqr_within_1mi        (OOF, derived: p75 - p25)
# Self-property is always excluded from the neighbor pool. Same-bed match
# uses the rounded `beds` integer.
COMPETITOR_RADIUS_MI: float = 1.0  # legacy single-radius constant (kept for compat)
COMPETITOR_MIN_K: int = 3  # need at least this many neighbors to compute pXX
EARTH_RADIUS_KM: float = 6371.0088

# Multi-radius expansion (added 2026-05-01).
# (radius_in_miles, label_used_in_column_name)
# Iter 1: tried [0.5, 1, 2] -> regressed to $197.38; 2mi overlaps with
# zip/submarket features. Pruned to 1mi only + add p50 (median) we missed.
COMPETITOR_RADII: list[tuple[float, str]] = [
    (1.0, "1mi"),
]

# MARTA station-density radii (added 2026-07-15). Counts distinct stations
# within radius -- captures "am I near a multi-station cluster" (e.g.
# downtown: Five Points/Georgia State/Peachtree Center/Garnett are all
# <0.5mi apart) vs. "near one isolated station" (most suburban stations,
# e.g. North Springs' nearest neighbor is ~0.94mi away), which
# dist_marta_km (nearest-only) can't distinguish. Starting with a single
# 1mi radius per the COMPETITOR_RADII lesson above (multi-radius regressed
# there); add more only if this one earns its place.
MARTA_DENSITY_RADII: list[tuple[float, str]] = [
    (1.0, "1mi"),
]

# ---------------------------------------------------------------------------
# Text feature extraction (property_name + street_address)
# ---------------------------------------------------------------------------
#
# Keyword vocabularies tuned to Atlanta multifamily naming conventions
# (validated against actual frequencies in the dataset).

NAME_KEYWORDS_PREMIUM: tuple[str, ...] = (
    "residences",
    "tower",
    "heights",
    "vista",
    "pointe",
    "reserve",
    "plaza",
    "estate",
    "estates",
    "mansion",
    "luxury",
)
NAME_KEYWORDS_MIDMARKET: tuple[str, ...] = (
    "apartments",
    "place",
    "crossing",
    "station",
    "commons",
    "square",
    "village",
    "park",  # "park" is most common positioning word
)
NAME_KEYWORDS_OLDER: tuple[str, ...] = (
    "gardens",
    "manor",
    "arms",
    "court",
    "courts",
)
NAME_KEYWORDS_CONVERSION: tuple[str, ...] = (
    "lofts",
    "mill",
    "foundry",
    "works",
    "factory",
    "studios",
)
NAME_KEYWORDS_PHASE: tuple[str, ...] = (
    "phase",
    " ii",
    " iii",
    " iv",  # leading space avoids matching inside other words
)

# Sub-market names that, if mentioned in property_name, claim a premium location.
PREMIUM_SUBM_TOKENS: tuple[str, ...] = (
    "buckhead",
    "midtown",
    "brookhaven",
    "vinings",
    "sandy springs",
    "inman park",
    "virginia highland",
    "old fourth ward",
    "atlantic station",
    "dunwoody",
    "alpharetta",
)

# Curated Atlanta multifamily operator/brand list. Matches first 1-2 tokens
# of property name. Order matters: longer brand names go first to avoid
# substring conflicts.
KNOWN_BRANDS: tuple[str, ...] = (
    "avalon",
    "avalonbay",
    "camden",
    "cortland",
    "greystar",
    "maa",
    "post",
    "amli",
    "gables",
    "lincoln",
    "windsor",
    "columbia",
    "wood partners",
    "millcreek",
    "mill creek",
    "rangewater",
    "bell partners",
    "bozzuto",
    "trammell crow",
    "hines",
    "highmark",
    "jpi",
    "jlb",
    "atlantic",
    "magnolia",
    "highland",
    "rosemont",
    "wesley",
    "ashley",
    "elevate",
    "avana",
    "everleigh",
    "bexley",
    "walton",
    "retreat",
)

# Iconic street names in Atlanta (case-insensitive substring match).
ICONIC_STREETS: tuple[str, ...] = (
    "peachtree",
    "paces ferry",
    "piedmont",
    "ponce de leon",
    "roswell",
    "lenox",
    "pharr",
    "powers ferry",
    "memorial",
)

# Highway tokens to flag.
HIGHWAY_TOKENS: tuple[str, ...] = (
    "highway",
    "hwy",
    "i-285",
    "i-75",
    "i-85",
    "i-20",
    "ga-400",
    "us-",
    "us ",
)

# Street-suffix canonicalization. Maps abbreviations to canonical form.
STREET_SUFFIX_MAP: dict[str, str] = {
    "st": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "dr": "drive",
    "drive": "drive",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "pkwy": "parkway",
    "parkway": "parkway",
    "hwy": "highway",
    "highway": "highway",
    "ln": "lane",
    "lane": "lane",
    "ct": "court",
    "court": "court",
    "cir": "circle",
    "circle": "circle",
    "pl": "place",
    "place": "place",
    "trl": "trail",
    "trail": "trail",
    "way": "way",
}

# ---------------------------------------------------------------------------
# Stacking ensemble configuration
# ---------------------------------------------------------------------------
# Four heterogeneous base models per fold, blended with an Augmented Ridge meta:
#   * lgbm_l1   -> LightGBM objective="regression_l1"  (MAE-aligned trees,
#                  leaf-wise growth, log-rent target)
#   * cat_q50   -> CatBoost loss="Quantile:alpha=0.5"  (different tree library;
#                  ordered boosting + ordered TS encoding for categoricals,
#                  log-rent target)
#   * knn_geo   -> sklearn KNeighborsRegressor on standardized numeric
#                  features only. Non-tree model family — local averaging
#                  in feature space gives error patterns uncorrelated with
#                  the boosters, which is exactly what stacking rewards.
#   * knn_lean  -> KNN on a tighter geo + size subset. Decorrelates from
#                  knn_geo because the smaller feature space picks different
#                  neighbors.
# Their OOF predictions on log-rent are stacked with three raw context
# features (log_sqft, beds, year_built), the Ridge meta learns blend weights,
# and the final prediction is clip(expm1(meta(log_preds + context)), 100, None).
USE_STACKING: bool = True
QUANTILE_ALPHA: float = 0.5

# Each spec drives one trainer call per fold inside run_cv. The "trainer" key
# selects the function in models.py. The "params" dict is merged onto the
# trainer's defaults; `seed_offset` differentiates seeds across bases so
# bagging/dropout draws differ.
BASE_SPECS: list[dict] = [
    {
        "name": "lgbm_l1",
        "trainer": "lightgbm",
        "params": {"objective": "regression_l1", "metric": "mae"},
        "seed_offset": 0,
    },
    {
        "name": "cat_q50",
        "trainer": "catboost",
        "params": {"loss_function": f"Quantile:alpha={QUANTILE_ALPHA}", "eval_metric": "MAE"},
        "seed_offset": 2000,
    },
    # xgb_huber was tried 2026-04-30 and dropped (meta-weight 0.0 in both PSF
    # and no-PSF runs; lgbm_l1 + cat_q50 already cover the GBM role).
    # Quantile NN was tried and dropped (similar diversity to KNN at 3x cost).
    # Both trainers + their config blocks have been removed; reintroduce by
    # adding the trainer in models.py + a config dict here + a BASE_SPECS entry.
    {
        "name": "knn_geo",
        "trainer": "knn",
        "params": {},  # KNN_PARAMS below carries n_neighbors / weights / metric
        "seed_offset": 5000,
    },
    {
        # Second KNN with a smaller, geo+size-only feature subset. The full
        # numeric feature space (~100 cols) makes high-dimensional Euclidean
        # distance noisy ("curse of dimensionality"); a focused subset puts
        # the metric on what actually matters for substitution: location, size,
        # bedroom count. Two parallel KNNs with different feature subsets are
        # uncorrelated enough to both earn meta weight.
        "name": "knn_lean",
        "trainer": "knn",
        "params": {},  # KNN_LEAN_PARAMS below carries the feature subset
        "seed_offset": 6000,
    },
]

# CatBoost defaults (depth + l2_leaf_reg + lr tuned to the typical "bigger but
# slower than LightGBM" sweet spot). Early stopping handled via od_type/od_wait.
CATBOOST_PARAMS: dict = {
    "iterations": 3000,
    "learning_rate": 0.04,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "min_data_in_leaf": 20,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": False,
    "allow_writing_files": False,
}

# K-Nearest Neighbors defaults. Uses numeric features only (categoricals
# are mostly captured via OOF target encodings injected upstream). Distance
# weighting gives closer neighbors more influence; k=15 is a sane default
# for ~5.5k training rows per fold.
KNN_PARAMS: dict = {
    "n_neighbors": 15,
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,  # Euclidean
    "n_jobs": -1,
}

# Second KNN: a tighter feature subset focused on geography, size, bedroom
# count, and neighborhood $/sqft. Standardized & weighted Euclidean. The point
# is decorrelation from knn_geo, which uses the full numeric matrix.
KNN_LEAN_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "dist_buckhead_km",
    "dist_midtown_km",
    # dist_downtown_km / dist_min_landmark_km removed 2026-07-15, matching
    # NUMERIC_FEATURES above.
    "sqft",
    "beds",
    "baths",
    "property_age",
    # OOF target encodings (added inside prepare_fold) supply locality signal
    # missing from raw lat/lon when neighborhoods cross small distances.
    "sub_market_te",
    "zipcode_te",
    # Neighborhood $/sqft proxies (also OOF, available after prepare_fold).
    "subm_psf_median",
    "zip_psf_median",
]

KNN_LEAN_PARAMS: dict = {
    "n_neighbors": 25,  # larger k smooths over the smaller feature space
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1,
    # `feature_subset` is consumed by the KNN trainer and stripped before the
    # sklearn estimator call. Columns missing from the dataframe are skipped.
    "feature_subset": KNN_LEAN_FEATURES,
}

# H3 resolutions to compute.
H3_RESOLUTIONS: list[int] = [6, 8]

# Atlanta landmark keys (must exist in atlanta_landmarks.json).
LANDMARKS: list[str] = ["buckhead", "midtown", "downtown", "atl_airport"]

# ---------------------------------------------------------------------------
# Hyperparameter search space (LightGBM via Optuna)
# ---------------------------------------------------------------------------
#
# Industry-standard ranges for tabular regression with ~7k rows.
# These are used inside train.py inside the Optuna objective.

LGB_FIXED_PARAMS: dict = {
    "objective": "regression_l1",  # MAE-aligned (quantile=0.5 alternative)
    "metric": "mae",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "feature_pre_filter": False,
    "force_col_wise": True,
}


def lgb_search_space(trial) -> dict:
    """Return a dictionary of LightGBM params sampled by Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 192),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search space (CatBoost via Optuna)
# ---------------------------------------------------------------------------
#
# Industry-standard ranges for CatBoost on small-to-medium tabular regression.
# Loss is fixed to Quantile:alpha=0.5 (MAE-aligned, matches our base spec).

CATBOOST_FIXED_PARAMS: dict = {
    "loss_function": f"Quantile:alpha={QUANTILE_ALPHA}",
    "eval_metric": "MAE",
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": False,
    "allow_writing_files": False,
}


def cb_search_space(trial) -> dict:
    """Return a dictionary of CatBoost params sampled by Optuna.

    Depth + learning rate are pinned to ranges that keep tuning trials inside
    the 45s sandbox window. depth>8 gains little on ~5500-row folds and slows
    each iteration ~4x; lr<0.02 means 800 iterations isn't enough to converge.
    """
    return {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
        "border_count": trial.suggest_int("border_count", 64, 254),
    }
