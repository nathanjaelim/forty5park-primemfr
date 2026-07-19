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
COFFEE_SHOPS_JSON: Path = PROJECT_DIR / "eda" / "coffee_shops.json"
GROCERY_STORES_JSON: Path = PROJECT_DIR / "eda" / "grocery_stores.json"
RESTAURANTS_JSON: Path = PROJECT_DIR / "eda" / "restaurants.json"
BARS_NIGHTCLUBS_JSON: Path = PROJECT_DIR / "eda" / "bars_nightclubs.json"
PARKS_JSON: Path = PROJECT_DIR / "eda" / "parks.json"
PARK_LANDMARKS_JSON: Path = PROJECT_DIR / "eda" / "park_landmarks.json"
TRAVEL_TIMES_JSON: Path = PROJECT_DIR / "eda" / "travel_times.json"
HIGHWAYS_JSON: Path = PROJECT_DIR / "eda" / "highways.json"
HIGHWAY_GA400_JSON: Path = PROJECT_DIR / "eda" / "highway_ga400.json"
HIGHWAY_I285_JSON: Path = PROJECT_DIR / "eda" / "highway_i285.json"

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
    "num_restaurants_within_0.5mi",
    # ^ reverted 2026-07-18: num_dining_grocery_within_0.5mi removed
    # (result never reported), back to the plain $75.35 trio, then
    # re-added restaurant density at 0.5mi per request -- this re-
    # establishes the best confirmed POI config found this session
    # ($74.59, see RESTAURANT_DENSITY_RADII history).
    # dist_ga400_km / ga400_near / marta_walkable /
    # num_marta_stations_within_2mi / num_restaurants_within_0.25mi /
    # num_restaurants_within_0.5mi / num_restaurants_within_1mi /
    # num_grocery_stores_within_0.5mi / num_coffee_shops_within_0.5mi /
    # num_coffee_shops_within_0.25mi / num_bars_nightclubs_within_0.5mi
    # all stay defined, just unused here.
    # Widened restaurant density radius to 1mi 2026-07-18 per request, on
    # top of the $74.59 best config -- untested, checking if the wider-is-
    # better trend (0.25mi -> 0.5mi improved) continues at 1mi.
    # Widened restaurant density radius 0.25mi -> 0.5mi 2026-07-18, per
    # request, on top of the $74.87 best config (0.5mi lost badly under
    # OLD default hyperparams, see RESTAURANT_DENSITY_RADII history).
    # Result: $74.59 -- NEW best MAE this session, -$0.28 vs $74.87 and
    # -$0.76 vs $75.35. Reverses the 0.5mi-radius verdict too, and beats
    # the 0.25mi version -- opposite ranking from the OLD default-
    # hyperparam results (0.25mi beat 0.5mi there). Strong confirmation
    # that hyperparams and features interact enough to flip which radius
    # wins, not just whether a feature helps at all.
    # reverted to base 2026-07-18 -- both marta_walkable+marta_density
    # pairings tried on top of this trio (0.5mi/1mi = $75.47, 1mi/2mi =
    # $76.25) came in worse than the plain $75.35 best config, so back to
    # just the trio + hyperparams retuned for it. dist_ga400_km /
    # ga400_near / marta_walkable / num_marta_stations_within_2mi all
    # stay defined, just unused here.
    # num_restaurants_within_0.25mi added 2026-07-18 on top of the $75.35
    # best config, per request. This exact feature already tested worse
    # than the OLD $75.46 baseline earlier this session ($76.24, +$0.78,
    # see restaurant-density history below) using default hyperparams and
    # the original 3-feature trio -- re-tested here with the trio-tuned
    # hyperparams. Result: $74.87 -- NEW best MAE this session, -$0.48 vs
    # $75.35. Reverses the earlier "restaurant density doesn't help"
    # finding, at least in this hyperparam regime -- consistent with the
    # marta_walkable/density lesson that hyperparams and features interact
    # (a feature that lost with one hyperparam set can win with another).
    # Current best config: baseline trio + num_restaurants_within_0.25mi +
    # hyperparams tuned for the plain trio (not yet re-tuned for THIS
    # 4-feature combo -- a fresh retune here might do even better).
    # marta_walkable(2mi) + marta_density(1mi) on top of this
    # trio, combined with retuned LGBM/CatBoost hyperparams, hit $75.37
    # (best MAE this session, -$0.09 vs the $75.46 default-hyperparam
    # baseline) -- but that wasn't a clean comparison since hyperparams
    # and features changed together. marta_walkable / num_marta_stations_
    # within_1mi removed here (ablation): retuned hyperparams (still
    # loaded from best_params.json / best_catboost_params.json) + this
    # plain baseline trio only, no MARTA, to isolate how much of the
    # $75.37 came from the retune alone vs. the MARTA features.
    # dist_ga400_km / ga400_near / marta_walkable /
    # num_marta_stations_within_1mi all stay defined, just unused here.
    # Ablation result: retuned hyperparams + this plain trio (no MARTA) =
    # $76.90 -- MUCH worse than both $75.37 (retuned + MARTA) and the
    # $75.46 default-hyperparam baseline (+$1.44 vs. default). Conclusion:
    # the retuned hyperparams don't generalize across feature sets -- they
    # were tuned (tuning/lightgbm.py + tuning/catboost.py) against
    # whichever fold cache existed at tuning time, which likely still had
    # marta_walkable/marta_density in it (see each script's docstring
    # caveat: "whatever feature set was used for the most recent CV is
    # what gets tuned"). Same hyperparams, minus those 2 features, blow up
    # by +$1.44. So $75.37 was NOT "retuning helps" -- it's a hyperparam
    # set co-adapted to the MARTA feature combo specifically, and it's
    # actively harmful without those features. Do not treat best_params
    # .json / best_catboost_params.json as a general-purpose improvement
    # until re-tuned against whatever the final feature set actually is.
    # Confirms the hypothesis: after re-running `prep`/`foldprep` on THIS
    # plain baseline trio (no MARTA) and re-tuning fresh against it,
    # MAE = $75.35 -- the best result this session, -$0.11 vs the $75.46
    # default-hyperparam baseline, and better than the $75.37 MARTA+
    # mismatched-retune number too. So: retuning genuinely helps when it's
    # tuned against the feature set actually being used -- the earlier
    # $76.90 disaster was purely a stale-cache mismatch, not evidence
    # against retuning itself. Current best config: plain baseline trio
    # (buckhead + midtown + airport zone) + hyperparams tuned specifically
    # for this feature set.
    # dist_atl_airport_km (continuous) removed again 2026-07-15: tested
    # against the $76.38 baseline in 3 configurations -- zone only $75.46
    # (best, -$0.92), distance only $76.78 (+$0.40), both together $76.97
    # (+$0.59, worse than either alone -- redundant encodings of the same
    # signal rather than complementary). Keeping zone-only
    # (dist_atl_airport_zone, in CATEGORICAL_FEATURES) as the winner.
    # dist_downtown_km / dist_min_landmark_km remain removed (dropped
    # 2026-07-15 keeping only buckhead + midtown from that pair).
    # All MARTA features removed 2026-07-15, reverting to the $75.46
    # winner (buckhead + midtown + airport zone only). Every MARTA
    # encoding and combination tried (distance, density, zone, walkable,
    # and pairings across multiple radii/thresholds) landed at or worse
    # than baseline, with the closest results ($75.54-$75.57) unstable
    # enough under small threshold nudges (a 0.1mi change swung MAE by
    # $0.34) to conclude they were noise, not real signal. Full test
    # history: distance+density $77.44 (+$1.98), all 3 original encodings
    # $76.89 (+$1.43), zone only $77.01 (+$1.55), distance+zone $76.57
    # (+$1.11), density(1mi)+zone $75.66 (+$0.20), density(0.5mi)+zone
    # $76.08 (+$0.62), walkable alone at 0.5/1/2mi
    # ($76.67/$76.27/$76.07), walkable(2mi)+density(1mi) $75.54 (+$0.08),
    # walkable(1mi)+density(2mi) $75.57 (+$0.11), walkable(2.1mi)+
    # density(1mi) $75.88 (+$0.42), walkable(2mi)+density(2mi) pending.
    # add_marta_distance() / add_marta_distance_zone() /
    # add_marta_station_density() / add_marta_walkable_flag() all stay in
    # engineering.py, just unused here.
    # All coffee-shop density radii removed 2026-07-15, reverting to the
    # $75.46 baseline. The 0.25mi -> 0.35mi sanity check settled it: 0.35mi
    # tested $76.74 (+$1.28), a $0.96 swing from the 0.25mi result ($75.78,
    # +$0.32) off a 0.1mi radius change -- the same instability pattern
    # seen with marta_walkable's 2.0mi->2.1mi jump ($0.34 swing), just
    # larger. Full radius history: 0.25mi $75.78 (+$0.32), 0.35mi $76.74
    # (+$1.28), 0.5mi $77.15 (+$1.69), 1mi $76.61 (+$1.15) -- no radius
    # produced a stable, reproducible improvement; the one good-looking
    # result (0.25mi) doesn't hold up under a small nudge, so it was noise,
    # not signal. add_coffee_shop_density() / eda/coffee_shops.json /
    # eda/fetch_coffee_shops.py all stay in place, just unused here.
    # Grocery density removed 2026-07-15, reverting to the $75.46 baseline.
    # Both radii tested worse: 1mi $76.52 (+$1.06), 0.25mi $76.97 (+$1.51,
    # worse still -- as expected, since grocery stores are sparse enough
    # that 0.25mi collapses nearly every property to a count of 0). Third
    # POI category (after MARTA and coffee shops) to land the same way
    # across multiple radii -- consistent evidence that POI density isn't
    # adding signal this model doesn't already get from dist_buckhead_km/
    # dist_midtown_km/submarket/zip/h3 cells. add_grocery_density() /
    # eda/grocery_stores.json / eda/fetch_grocery_stores.py all stay in
    # place, just unused here.
    # Restaurant density removed 2026-07-16, reverting to the $75.46
    # baseline. Both radii tested worse: 0.5mi $76.64 (+$1.18), 0.25mi
    # $76.24 (+$0.78, better than 0.5mi but still worse than baseline).
    # Fourth POI density feature (after MARTA, coffee shops, grocery
    # stores) to land worse than baseline at every radius tried --
    # consistent evidence across 4 POI categories that this model already
    # captures whatever "urban/amenity-rich location" signal these
    # features encode via dist_buckhead_km/dist_midtown_km/submarket/zip/
    # h3 cells. add_restaurant_density() / eda/restaurants.json /
    # eda/fetch_restaurants.py all stay in place, just unused here.
    # Bar/nightclub density within 0.25mi tested 2026-07-16 and removed:
    # $76.30 (+$0.84 vs the $75.46 baseline), worse. Fifth POI density
    # feature (after MARTA, coffee shops, grocery stores, restaurants) to
    # land worse than baseline -- every POI category and radius tried this
    # session lands worse, reinforcing that dist_buckhead_km/
    # dist_midtown_km/submarket/zip/h3 cells already capture whatever
    # "urban/amenity-rich location" signal these features encode.
    # add_bar_density() / eda/bars_nightclubs.json /
    # eda/fetch_bars_nightclubs.py all stay in place, just unused here.
    # Combined POI density within 0.25mi tested 2026-07-16 and removed:
    # $77.00 (+$1.54 vs the $75.46 baseline) -- worse than every individual
    # category (coffee $75.78, restaurants $76.24, bars $76.30, grocery
    # $76.97). Combining the 4 categories into one denser count didn't
    # help; if anything it's the worst POI result yet. Consistent with the
    # target-encoding explanation: h3_res8/sub_market/zipcode already
    # capture hyperlocal rent premiums directly from observed rent, so a
    # POI count (individual or combined) is at best redundant and at worst
    # adds noise the model has to spend capacity on. add_total_poi_density()
    # / eda/coffee_shops.json / eda/grocery_stores.json / eda/restaurants.json
    # / eda/bars_nightclubs.json all stay in place, just unused here.
    # Nearest-park distance tested 2026-07-16 and removed: $77.38 (+$1.92
    # vs the $75.46 baseline) -- the worst result of any feature tried
    # this session, worse even than the combined POI density feature
    # (+$1.54). Despite being conceptually different from the POI density
    # features (a continuous nearest-neighbor distance, not a radius
    # count, and without dist_min_landmark_km's "which one" conflation
    # problem), it landed worse still. Likely the same underlying reason
    # as the others -- h3_res8/sub_market/zipcode already capture whatever
    # "proximity to desirable green space" signal this measures, via
    # observed rent rather than an indirect physical-distance proxy -- but
    # the centroid approximation (a large or oddly-shaped park's centroid
    # can sit meaningfully off from its actual boundary) may add its own
    # extra noise on top of that. add_park_distance() / eda/parks.json /
    # eda/fetch_parks.py all stay in place, just unused here.
    # Nearest named-park distance tested 2026-07-16 and removed: $76.45
    # (+$0.99 vs the $75.46 baseline) -- still worse, though clearly
    # better than the "nearest of 1070" version's +$1.92, confirming the
    # curated-landmark-set idea helped some. Not enough to beat baseline,
    # though. At this point every green-space/POI framing tried this
    # session (individual categories, combined, nearest-of-many, nearest-
    # of-curated-few) has landed worse than dist_buckhead_km/
    # dist_midtown_km/h3_res8/sub_market/zipcode already capturing this
    # signal. add_named_park_distance() / eda/park_landmarks.json all stay
    # in place, just unused here.
    # Drive-time-to-landmark features tested 2026-07-16 and removed: $77.04
    # (+$1.58 vs the $75.46 baseline) -- worse, despite being the first
    # feature this session that wasn't just another haversine proxy for
    # location (real OSRM road-network minutes via h3_res6 lookup against
    # eda/travel_times.json). Likely explanation: h3_res6 is exactly the
    # join key used to compute these drive times, so within a given
    # h3_res6 cell every property gets an IDENTICAL drive-time value --
    # this is really a (coarser) re-encoding of h3_res6 itself, not new
    # information, and h3_res6's own target encoding already captures
    # local rent premiums more directly (from observed rent, not an
    # indirect routing proxy). A per-property or finer-grained lookup
    # (e.g. h3_res8 cells, or the property's own coordinates via a live
    # per-row routing call) might avoid this collapse, but wasn't tried.
    # add_travel_time_features() / eda/travel_times.json /
    # eda/fetch_travel_times.py all stay in place, just unused here.
    # downtown_drive_min tested standalone 2026-07-16 and reverted (see
    # user request to go back to baseline before a result was reported).
    # Nearest-highway distance tested 2026-07-16 and removed: $76.97
    # (+$1.51 vs the $75.46 baseline), worse. Consistent with the real
    # rent-vs-distance scatter plot the user shared: a nearly flat trend
    # line across the whole range (~$1900 near 0 miles drifting to ~$1550
    # at 40+), with high variance concentrated near the highways rather
    # than a distinct level shift -- unlike dist_atl_airport_zone's clear
    # regime change, there's no strong signal here for a continuous (or
    # zone) distance feature to extract; the noise near 0 isn't a
    # different level, just more spread. add_highway_distance() /
    # eda/highways.json / eda/research/highways_overpass_query.txt all
    # stay in place, just unused here.
    # GA-400/I-285 highway distances tested in 3 configurations, all worse
    # than the $75.46 baseline: combined nearest-of-either $76.97 (+$1.51),
    # GA-400+I-285 together $77.56 (+$2.10, worst of any feature this
    # session), GA-400 alone $75.99 (+$0.53, by far the best of the three
    # -- confirms GA-400 carries more real signal than I-285, matching the
    # real scatter plots, but still not enough to clear baseline). Fully
    # reverted 2026-07-16. add_highway_distance() /
    # add_highway_route_distances() / eda/highways.json /
    # eda/highway_ga400.json / eda/highway_i285.json all stay in place,
    # just unused here.
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
    "dist_atl_airport_zone",
    # ^ restored 2026-07-18 -- reverted to the $75.46 baseline (buckhead +
    # midtown in NUMERIC_FEATURES, airport zone here) after the untested
    # GA-400 + MARTA walkable/density combo.
    # dist_marta_zone removed 2026-07-15 to isolate the walkable(1mi) +
    # density(2mi) test. See the MARTA test history comment in
    # NUMERIC_FEATURES above.
    # dist_i285_zone (0-4mi/4-7mi/7mi+ bucketing, isolating the unexplained
    # 4-7mi high-rent-outlier cluster) tested 2026-07-16 and removed:
    # $77.44 (+$1.98 vs the $75.46 baseline), worse -- one of the worst
    # results this session. The 4-7mi cluster (verified to include both
    # Buckhead at 4.95mi and Downtown at 5.92mi -- likely "core Atlanta"
    # broadly, not a real I-285 effect) doesn't hold up as a genuine
    # signal once binned. add_i285_zone_feature() / I285_ZONE_EDGES_MI
    # stay in place, just unused here.
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

# dist_i285_km zone edges (right edges, MILES). Added 2026-07-16 based on
# the real rent-vs-I-285-distance scatter plot (user-shared): trend line
# is nearly flat overall, but there's an unusual cluster of high-rent
# outliers concentrated specifically in a ~4-7mi band (likely capturing
# Buckhead/Sandy Springs riding along the same distance axis, not a real
# I-285 effect). Continuous dist_i285_km already tested worse than
# baseline both alone-with-GA400 and combined ($77.56 and $76.97) -- this
# bins into 0-4mi / 4-7mi / 7mi+ to isolate that specific band as its own
# category, same "genuine categorical, not ordinal" reasoning as
# ATL_AIRPORT_ZONE_EDGES (the 4-7mi band isn't "more" than the other two,
# it's a different, unexplained cluster). Not yet tested.
I285_ZONE_EDGES_MI: tuple[float, float] = (4.0, 7.0)
I285_ZONE_LABELS: tuple[str, str, str] = ("0-4mi", "4-7mi", "7mi_plus")

# dist_marta_km zone edges (right edges, MILES -- matches the original EDA
# notebook's bins exactly: eda/research/Yardi EDA - New Geospatial
# Features.ipynb, cell 22, `bins = [0, 0.5, 2.5, 5, 20, inf]`). That
# notebook found the same kind of non-monotonic pattern as the airport
# zone (rent highest under 0.5mi, dips 2.5-5mi, partially recovers past
# 5mi), so this is built as a genuine categorical like
# dist_atl_airport_zone -- NOT as ordinal int codes, which is what a
# later cell in that notebook did (against its own earlier finding).
# dist_marta_km itself is computed in km; add_marta_distance_zone()
# converts to miles before binning so these edges match the notebook
# 1:1 without unit-converting the constants themselves.
MARTA_ZONE_EDGES_MI: tuple[float, float, float, float] = (0.5, 2.5, 5.0, 20.0)
MARTA_ZONE_LABELS: tuple[str, str, str, str, str] = (
    "<0.5mi",
    "0.5-2.5mi",
    "2.5-5mi",
    "5-20mi",
    "20mi+",
)

# Binary "near a MARTA station" flag threshold (miles). Rationale: the
# Rent-vs-MARTA-distance scatter shows almost all the interesting
# variance (highest rents AND every extreme outlier) packed under ~2mi,
# with a near-flat, noisy mean from ~2mi out to 44mi -- a single
# threshold isolating the near-station cluster may carry more signal per
# parameter than the continuous distance or the 5-bin zone (both of which
# tested worse than baseline; see NUMERIC_FEATURES / CATEGORICAL_FEATURES
# comments). Tested against the $75.46 baseline at 3 thresholds: 1.0mi
# $76.27 (+$0.81), 2.0mi $76.07 (+$0.61, best), 0.5mi $76.67 (+$1.21,
# worst -- too tight, throws away too much of the graded signal). Best
# combo found was walkable(2mi)+density(1mi radius): $75.54 (+$0.08).
# Tried the opposite pairing (walkable 1mi + density 2mi) -- tied at
# $75.57 (+$0.11). Nudged to 2.1mi (still paired with density 1mi) --
# jumped to $75.88 (+$0.42), a large swing from a tiny threshold change.
# Back to 2.0mi on 2026-07-15, now paired with density widened to 2mi too
# (see MARTA_DENSITY_RADII) to check if matching radii is more stable.
# Tightened to 0.5mi on 2026-07-18, paired first with density(1mi) then
# density(2mi) -- individual base-learner MAEs dropped but the stacked MAE
# rose (likely correlated-error / reduced-diversity effect across the 4
# bases, or noise -- see chat). Back to 2.0mi on 2026-07-18, re-paired with
# density(1mi) -- this was the best combo found earlier this session
# ($75.54, +$0.08 vs baseline); paired with retuned hyperparams + this
# trio it hit $75.37 (not a clean comparison, see NUMERIC_FEATURES
# comment). Tightened to 0.5mi again on 2026-07-18, re-paired with
# density(1mi), on top of the $75.35 best config: $75.47 (+$0.12 vs
# $75.35, worse but closest of this round). Widened to 1.0mi 2026-07-18,
# re-paired with density(2mi): $76.25 (+$0.90 vs $75.35, worse still).
# Both used the trio-tuned hyperparams as-is, no fresh retune.
MARTA_WALKABLE_THRESHOLD_MI: float = 1.0

# Binary "near Buckhead" flag threshold (miles). Added 2026-07-15 to
# replace the continuous dist_buckhead_km with a single cutoff, mirroring
# marta_walkable's pattern. Changed 6.0mi -> 8.0mi -> 7.0mi 2026-07-16 per
# request. Untested -- new candidate, not validated against the $75.46
# baseline yet.
BUCKHEAD_NEAR_THRESHOLD_MI: float = 7.0

# Binary "near GA-400" flag threshold (miles). Added 2026-07-18, mirroring
# marta_walkable's pattern. Rationale: the real rent-vs-GA-400-distance
# scatter (shared by the user) showed a clean monotonic decline rather
# than MARTA's near-cluster-then-flat shape, so a single near/far cutoff
# is a rougher approximation here than for MARTA -- worth testing anyway
# since dist_ga400_km alone already beat baseline on one run ($75.99).
# Changed 4.0mi -> 6.0mi 2026-07-18 per request, before any result on
# 4.0mi was reported. Untested -- new candidate, not validated against
# the $75.46 baseline yet.
GA400_NEAR_THRESHOLD_MI: float = 6.0

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
# radius per the COMPETITOR_RADII lesson above (multi-radius regressed
# there); add more only if this one earns its place.
# Tested 0.5mi radius 2026-07-15 (vs. the 1mi default): $76.08 (+$0.62 vs
# the $75.46 baseline), worse than 1mi's $75.66 (+$0.20). Likely too tight
# -- most non-downtown properties collapse to a count of 0 at 0.5mi,
# losing the granularity that made the 1mi version useful.
# Widened to 2.0mi on 2026-07-15 to test alongside a tightened
# marta_walkable(1mi) -- tied the original pairing ($75.57 vs $75.54, no
# real difference). Reverted to 1.0mi, tested with walkable(2.1mi) -- that
# combo jumped to $75.88. Widened to 2.0mi again on 2026-07-15, now paired
# with walkable also at 2.0mi (see MARTA_WALKABLE_THRESHOLD_MI), matching
# both radii to test whether that's more stable than mismatched radii.
# Back to 1.0mi on 2026-07-18, paired with walkable(2mi) -- this was the
# best-performing combo found this session ($75.54, +$0.08 vs baseline).
# Widened to 2.0mi 2026-07-18, paired with a tightened walkable(0.5mi) --
# per-base MAEs dropped but stacked MAE rose (see chat). Back to 1.0mi
# 2026-07-18, re-paired with walkable(2mi) to re-test the original
# best-known combo on top of the current baseline trio -- result never
# reported. Still at 1.0mi, paired with a tightened walkable(0.5mi), on
# top of the $75.35 best config: $75.47 (+$0.12 vs $75.35). Widened to
# 2.0mi 2026-07-18, re-paired with a widened walkable(1mi): $76.25
# (+$0.90 vs $75.35, worse still). Both worse than $75.35, no fresh
# retune done for either.
MARTA_DENSITY_RADII: list[tuple[float, str]] = [
    (2.0, "2mi"),
]

# Coffee shop density radius (added 2026-07-15). Counts distinct coffee
# shops within radius -- a walkability/lifestyle-amenity signal distinct
# from dist_buckhead_km / dist_midtown_km / submarket, since a walkable
# retail strip (e.g. West Midtown, Old Fourth Ward, Grant Park) can carry
# its own premium regardless of distance to those named districts.
# Sourced from eda/coffee_shops.json (299 shops, curated from a cached
# Overpass export at eda/research/cafes.geojson -- see
# eda/fetch_coffee_shops.py). Coffee shops are much denser than MARTA
# stations (299 vs. 37, metro-wide), so starting tighter than MARTA's
# radius: 0.5mi is roughly a 10-minute walk, matching the "amenity within
# walking distance" framing. Single radius per the COMPETITOR_RADII /
# MARTA_DENSITY_RADII lesson (multi-radius regressed for competitors).
# Tested against the $75.46 baseline at 4 radii, all worse than baseline:
# 0.25mi $75.78 (+$0.32, best), 0.35mi $76.74 (+$1.28), 0.5mi $77.15
# (+$1.69), 1mi $76.61 (+$1.15). The 0.25mi->0.35mi jump ($0.96 swing off
# a 0.1mi nudge) showed the same instability pattern as
# marta_walkable's threshold sensitivity -- the promising 0.25mi result
# didn't hold up, so coffee density was removed from NUMERIC_FEATURES
# 2026-07-15 (this radius list itself left as-is, just unused).
# 0.5mi's $77.15 was the worst of the 4 radii tested, under OLD default
# hyperparams. Set to 0.5mi 2026-07-18 per request, re-testing under the
# trio-tuned hyperparams that already reversed the restaurant/grocery
# density verdicts. Result: $75.00 (-$0.35 vs the $75.35 plain trio) --
# ALSO reverses under tuned hyperparams (same pattern as restaurant/
# grocery), though weaker than restaurant's $74.59 or grocery's $74.73.
# Narrowed back to 0.25mi 2026-07-18 per request -- this was coffee's
# best-looking (but unstable, see above) radius under the OLD default
# hyperparams ($75.78); re-testing under the trio-tuned hyperparams --
# result never reported. Back to 0.5mi 2026-07-18 (its best confirmed
# result this session, $75.00) as part of a combined-POI-at-0.5mi test
# (restaurants+bars+cafes+grocery all together, see NUMERIC_FEATURES).
COFFEE_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Grocery store density radius (added 2026-07-15). Counts distinct
# grocery stores / supermarkets within radius -- same walkability/
# lifestyle-amenity motivation as COFFEE_DENSITY_RADII, but grocery is a
# different (more essential, less discretionary) amenity category, so it
# may not share coffee density's failure mode. Sourced from
# eda/grocery_stores.json (680 stores incl. Publix, Kroger, Whole Foods,
# Trader Joe's, etc., curated from a cached Overpass export at
# eda/research/grocery.geojson -- see eda/fetch_grocery_stores.py).
# Tested at 1mi: $76.52 (+$1.06 vs the $75.46 baseline), clearly worse --
# same outcome as coffee shop and MARTA density at every radius tried so
# far. Narrowed to 0.25mi on 2026-07-15 per request (grocery stores are
# much sparser than coffee shops, so most properties will show 0 at this
# radius -- even Midtown center shows 0; only right next to a specific
# store like the Whole Foods at 33.7861686/-84.3885403 would register).
# 0.25mi tested $76.97 (+$1.51 vs the $75.46 baseline), worse still, under
# OLD default hyperparams (see NUMERIC_FEATURES/RESTAURANT_DENSITY_RADII
# history for context). Widened to 0.5mi 2026-07-18 per request, re-testing
# under the trio-tuned hyperparams that already reversed the restaurant-
# density verdict. Result: $74.73 (+$0.14 vs the $74.59 restaurant-0.5mi
# best, but -$0.62 vs the $75.35 plain trio) -- grocery density ALSO
# reverses under the tuned hyperparams, same pattern as restaurants,
# though it doesn't beat restaurant density at the same radius.
GROCERY_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Restaurant density radius (added 2026-07-16). Counts distinct restaurants
# within radius -- same walkability/lifestyle-amenity motivation as
# COFFEE_DENSITY_RADII / GROCERY_DENSITY_RADII, but restaurants are the
# densest POI category curated so far (3638 vs. 299 coffee shops, 680
# grocery stores, 37 MARTA stations, metro-wide), so a "dining scene"
# signal could plausibly separate walkable retail corridors even where
# MARTA/coffee/grocery density did not. Sourced from eda/restaurants.json
# (curated from a cached Overpass export at
# eda/research/restaurants_raw.geojson -- see eda/fetch_restaurants.py;
# note that fetch's raw export was NOT reliably scoped to the MSA by
# Overpass's area filter and required an extra bbox+addr:state safety
# filter, see that script's docstring). 0.5mi tested $76.64 (+$1.18 vs the
# $75.46 baseline), worse -- fourth POI category to land worse than
# baseline (see NUMERIC_FEATURES history comment). Narrowed to 0.25mi
# 2026-07-16 per request, matching coffee shop density's most-promising
# (but ultimately unstable) radius. Re-tested 0.25mi 2026-07-18 on top of
# the trio-tuned-hyperparams $75.35 config: $74.87, new best MAE this
# session (see NUMERIC_FEATURES comment) -- reverses the earlier verdict
# in this hyperparam regime. Widened back to 0.5mi 2026-07-18 per request,
# to re-check the radius that lost badly ($76.64) under the OLD default
# hyperparams: $74.59, new best MAE this session, beats 0.25mi too (see
# NUMERIC_FEATURES comment). Widened further to 1mi 2026-07-18 per
# request, continuing to check if the tuned-hyperparam regime keeps
# favoring wider radii -- result never reported. Back to 0.5mi 2026-07-18
# (its best confirmed result this session, $74.59) as part of a
# combined-POI-at-0.5mi test (restaurants+bars+cafes+grocery all
# together, see NUMERIC_FEATURES).
RESTAURANT_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Bar/nightclub density radius (added 2026-07-16). Counts distinct bars +
# nightclubs within radius -- same walkability/lifestyle-amenity family as
# COFFEE_DENSITY_RADII / GROCERY_DENSITY_RADII / RESTAURANT_DENSITY_RADII,
# but nightlife is a more discretionary/younger-demographic amenity than
# dining or groceries, so it may not fail the same way. Sourced from
# eda/bars_nightclubs.json (196 named bars, curated from a pre-existing
# cached Overpass export at eda/research/bars.geojson -- see
# eda/fetch_bars_nightclubs.py; that cache only has amenity=bar, no
# nightclub-tagged features, and was already scoped to the MSA). Starting
# at 0.25mi per request, matching restaurant/coffee density's narrower
# radius. Tested 2026-07-16 under OLD default hyperparams: $76.30 (+$0.84
# vs the $75.46 baseline), worse, removed from NUMERIC_FEATURES. Widened
# to 0.5mi 2026-07-18 per request, re-testing under the trio-tuned
# hyperparams that already reversed restaurant/grocery density's verdicts.
# Result: $75.41 (+$0.06 vs the $75.35 plain trio, essentially flat/within
# noise) -- the one POI category so far that does NOT clearly reverse
# under the tuned hyperparams (coffee/grocery/restaurant all improved on
# the trio; bars roughly ties it). Consistent with nightlife being a more
# discretionary/less location-premium-correlated amenity than the others.
BAR_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Combined POI density radius (added 2026-07-16). Counts ALL POIs across
# the 4 curated categories (coffee shops + grocery stores + restaurants +
# bars/nightclubs -- 4813 total, see eda/coffee_shops.json /
# eda/grocery_stores.json / eda/restaurants.json /
# eda/bars_nightclubs.json) within radius, in one combined count rather
# than 4 separate columns. Motivation: each category tested worse than
# the $75.46 baseline individually, but each is also fairly sparse
# per-category (e.g. only 196 bars metro-wide) -- combining them into one
# "general amenity density" signal gives a much denser, less noisy count
# (54 POIs at the densest cluster found vs. 13 for bars alone), which
# might behave differently than any single sparse category did. Single
# radius, matching the other POI density features' final radius. Tested
# at 0.25mi 2026-07-16 under OLD default hyperparams: $77.00 (+$1.54 vs
# the $75.46 baseline), worse than every individual category -- see
# NUMERIC_FEATURES history. Widened to 0.5mi 2026-07-18 per request,
# re-testing under the trio-tuned hyperparams that already reversed every
# individual POI category's verdict. Result: $74.95 (-$0.40 vs the $75.35
# plain trio -- beats baseline and beats the 4-separate-columns combo
# ($75.21), but still worse than restaurant density alone ($74.59) or
# grocery alone ($74.73)). So collapsing to one combined count is better
# than keeping 4 separate correlated columns, but still not as good as
# just using the single strongest category alone -- consistent with the
# dilution pattern seen throughout this POI re-test round.
TOTAL_POI_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Dining + grocery combined density radius (added 2026-07-18). Counts
# ONLY restaurants + grocery stores within radius, as one combined column
# -- narrower than TOTAL_POI_DENSITY_RADII (which also includes coffee
# shops and bars). Motivation: restaurant density ($74.59) and grocery
# density ($74.73) were the two strongest individual POI categories this
# session, both clearly ahead of coffee ($75.00) and bars ($75.41) --
# combining just these two, rather than all 4, avoids diluting the signal
# with the two weaker categories the way TOTAL_POI_DENSITY_RADII's full
# combine did ($74.95, worse than restaurant alone). Untested.
DINING_GROCERY_DENSITY_RADII: list[tuple[float, str]] = [
    (0.5, "0.5mi"),
]

# Distance to nearest park (added 2026-07-16). Unlike the POI density
# features above (a count within a radius), this is a continuous
# nearest-neighbor distance, same shape as dist_buckhead_km/
# dist_midtown_km -- one feature, not a family of radii. Motivation
# mirrors "green-space proximity" (distance to Piedmont Park, Beltline
# trail) from the feature-brainstorm slide. Sourced from eda/parks.json
# (1070 named parks, curated from a cached Overpass export at
# eda/research/parks.geojson -- leisure=park polygons/multipolygons;
# see eda/fetch_parks.py). Each park is represented by its polygon
# centroid (computed via the shoelace formula, area-weighted across
# multipolygon parts), not its boundary, so distance to a large or
# oddly-shaped park (a long, thin one especially) can be a rougher
# approximation than for the small POI categories above, which are true
# points. Unlike dist_min_landmark_km (flagged earlier as a bad feature
# because it conflates "which landmark" with "how far" across
# directionally distinct landmarks with different rent relationships),
# parks are a single homogeneous "green space access" category, so a
# nearest-park distance doesn't have that same conflation problem. Tested
# 2026-07-16: $77.38 (+$1.92 vs baseline), the worst result of any feature
# tried this session -- removed. Two likely causes: (1) treats all 1070
# OSM-tagged parks as equally meaningful, including tiny pocket parks and
# HOA green spaces, diluting the signal with noise from insignificant
# ones; (2) the polygon-centroid approximation is roughest for large or
# irregularly shaped parks, which are exactly the ones most likely to be
# genuinely significant.

# Distance to nearest NAMED park (added 2026-07-16), replacing the "nearest
# of 1070" version above. Sourced from eda/park_landmarks.json -- a small
# curated set of 5 well-known, unambiguous, single-polygon intown Atlanta
# parks (Piedmont Park, Grant Park, Centennial Olympic Park, Historic
# Fourth Ward Park, Candler Park), the same "small curated landmark set"
# pattern as config.LANDMARKS (buckhead/midtown/downtown/airport) rather
# than "nearest of many." Deliberately excludes the Beltline (not a single
# leisure=park polygon in OSM -- it's tagged as a route/cycleway, no clean
# single point to anchor on) and Chastain Park (not present in the cached
# eda/research/parks.geojson export under a matching name). Also excludes
# "Freedom Park": 4 same-named entries exist in eda/parks.json, 3 of which
# cluster in intown Atlanta (likely disjoint segments of the real Freedom
# Park/Freedom Parkway corridor) but a 4th sits ~40km north near
# Forsyth/Cherokee county -- a different, unrelated park with a name
# collision. Rather than guess which of the 3 intown segments to use,
# Freedom Park was left out entirely to keep this set unambiguous. Not yet
# tested against the $75.46 baseline.

# Drive-time-to-landmark features (added 2026-07-16). Unlike every
# distance/POI feature tried this session (all haversine-based, all
# landed worse than baseline), this measures actual road-network drive
# time in minutes rather than crow-flies distance -- mechanically
# distinct from dist_buckhead_km/dist_midtown_km, which the model may
# already capture via h3_res8/sub_market/zipcode target encodings. Two
# properties equidistant by straight line from Midtown can have very
# different real commute times depending on highway access, so this could
# carry information the existing distance features structurally can't.
# Sourced from eda/travel_times.json (drive time in minutes from every
# h3_res6 cell centroid in the MSA to each of the 4 config.LANDMARKS
# entries, fetched via OSRM's public routing demo server -- see
# eda/fetch_travel_times.py; requires live network + a working h3 install,
# neither available in the sandbox this was built in, so
# eda/travel_times.json does not exist yet until that script is run
# externally). Joined onto each row via its h3_res6 cell (already computed
# earlier in add_static_features), not a live per-property routing call --
# add_travel_time_features() degrades to NaN columns until the JSON
# exists, same graceful-degradation pattern as the other POI loaders.
# eda/travel_times.json generated 2026-07-16 (989 h3_res6 cells covering
# the MSA bbox via OSRM's public demo server; sanity-checked -- no nulls,
# median ~75min reflecting the large bbox, cells near the urban core show
# single-digit-to-teens minutes as expected, e.g. 2.99min to Midtown for
# a near-Midtown cell). Not yet tested against the $75.46 baseline.

# Distance to nearest highway (added 2026-07-16). Covers I-285 (the
# perimeter loop) and GA-400 -- the two highways named in the original
# feature-brainstorm slide ("convenient at 1mi, noisy at 0.1mi"). Sourced
# from eda/highways.json: both routes' OSM way geometries (663 line
# segments, fetched via eda/research/highways_overpass_query.txt --
# highway=motorway with ref matching "I 285" / "GA 400") resampled to
# points every 0.25km along each line (raw OSM vertices are unevenly
# spaced -- dense on curves, sparse on straightaways -- so resampling at a
# fixed interval avoids biasing "nearest point" distance toward the curvy
# stretches). 2125 resampled points total (1361 on I-285, 764 on GA-400;
# combined line length ~327km, longer than the real one-way route
# distance because OSM maps divided highways as separate lines per
# direction of travel). This starts as a plain continuous distance, same
# shape as dist_buckhead_km -- the slide's own framing suggests the real
# relationship may be non-monotonic (bad right next to it, good a mile
# away, flat further out), which if true would favor a zone-style
# categorical encoding instead, mirroring dist_atl_airport_zone. That
# would need real EDA on the actual rent-vs-highway-distance relationship
# to pick sensible bucket edges (same as how the airport zone edges were
# derived), which wasn't done here -- start with continuous distance and
# revisit as a zone if useful. Tested 2026-07-16: $76.97 (+$1.51 vs
# baseline), worse. The real rent-vs-distance scatter plots (user-shared,
# split by route) explain why: GA-400 shows a genuine, fairly clean
# monotonic decline (~$1900 near the highway down to ~$1000 at 40+ miles),
# but I-285 is essentially flat across its whole range with an unusual
# cluster of high-rent outliers specifically in a 4-7mi band (likely
# capturing Buckhead/Sandy Springs riding along the same axis, not a real
# I-285 effect). Combining both routes into one "nearest highway" distance
# diluted GA-400's real signal with I-285's flat noise -- same
# "collapsing heterogeneous categories destroys signal" problem as
# dist_min_landmark_km earlier this session. add_highway_distance() /
# eda/highways.json stay in place, just unused here.

# Distance to GA-400 and I-285 as SEPARATE features (added 2026-07-16),
# replacing the combined dist_nearest_highway_km above for exactly the
# reason described in that comment -- keeping heterogeneous distances
# separate rather than collapsing them into one nearest-of-both distance,
# the same pattern as dist_buckhead_km/dist_midtown_km staying separate
# rather than merged. Sourced from eda/highway_ga400.json (764 points) /
# eda/highway_i285.json (1361 points) -- same 0.25km resampling as the
# combined version, just split by route before resampling (see
# eda/research/highway_points_ga400.geojson /
# eda/research/highway_points_i285.geojson, filtered from the same raw
# Overpass export by the "ref" tag). Given the real scatter plots, GA-400
# is the one actually worth testing (I-285's flat trend line predicts it
# won't help) -- dist_ga400_km is active below; dist_i285_km is built
# (add_highway_route_distances()) but left out of NUMERIC_FEATURES for
# now given the low expectation, easy to add if useful later.

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
    "dist_buckhead_km",  # buckhead_near swap reverted 2026-07-16, matching
    # NUMERIC_FEATURES above.
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
