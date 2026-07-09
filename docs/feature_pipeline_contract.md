# Feature Pipeline Contract — For Data Engineering Handoff

> **Audience:** Data engineering team taking over feature creation
> **Status:** Living contract — updated when features change
> **Last updated:** June 2026

## Why this document exists

Today, feature engineering happens inside the model package (`src/prime_mfr/features/`). That works while the data scientist owns both feature creation and model training, but it creates two limitations:

1. **Scaling.** Feature engineering on every batch inference call re-runs landmark distances, H3 hashing, and target-encoder joins per request. A data engineering team running this upstream on a warehouse (Snowflake / Databricks / Spark) can pre-compute features once per snapshot.
2. **Ownership.** Data engineers own data pipelines. Data scientists own models. Co-locating feature engineering inside the model package conflates the two.

This document defines the **contract** between the two teams. If a data engineering team produces a parquet conforming to this contract, our model will consume it directly — bypassing the in-process feature pipeline.

## Two consumption modes

```mermaid
flowchart LR
    subgraph RAW["Raw Yardi feed"]
        R1[("property-enriched.parquet")]
        R2[("unit-mix-enriched.parquet")]
        R3[("hist-rent.parquet")]
    end

    subgraph MODE1["Mode 1: In-process feature engineering"]
        F1["prime_mfr.features.<br/>add_static_features"]
        F2["prime_mfr.features.<br/>add_oof_features"]
    end

    subgraph MODE2["Mode 2: Pre-engineered (DE handoff)"]
        DE[(("pre_engineered.parquet<br/>conforms to contract"))]
    end

    PRED[["prime-mfr predict"]]
    OUT[("predictions.parquet")]

    RAW --> F1
    F1 --> F2
    F2 --> PRED

    RAW --> DE
    DE --> PRED

    PRED --> OUT

    style MODE2 fill:#F9B233,color:#000
    style DE fill:#F9B233,color:#000
    style PRED fill:#1E2761,color:#fff
```

**Mode 1 — In-process (today's default).** Raw Yardi feeds flow through the Python feature pipeline at training and inference time. Slow per-row but self-contained.

**Mode 2 — Pre-engineered (DE handoff).** A separate data pipeline (Snowflake SQL / Spark / dbt) produces a parquet conforming to this contract. The model loads it and skips its own feature pipeline. Fast and decoupled.

## The contract — what a pre-engineered parquet must contain

### Granularity

**One row per `(property_id, unit_type)` pair** at the prediction snapshot date. Same granularity as the model's training data.

### Required columns

#### Identity & metadata (passed through; not used as features)

| Column | Type | Description |
|---|---|---|
| `property_id` | str | Yardi-stable property identifier |
| `unit_type` | str | Yardi-canonical unit type code (e.g. `1BR/1.0`, `2BR/2.0`) |
| `period` | date | Snapshot date (typically first-of-month) |

#### Numeric features (31 columns)

These are the feature names referenced in `configs/features/numeric.yaml`. Each must be present and numeric. Missing values are allowed; do not impute.

```
sqft, beds, baths, unit_garage
num_units, num_units_subtype, year_built, min_stories, max_stories,
number_of_stories, num_buildings, total_parking_spaces, lot_size_in_acres, lot_size_in_square_feet
occupancy_rate, latest_sale_price_per_sqft, latest_sale_price_per_unit, latest_sale_price_total
latitude, longitude
dist_buckhead_km, dist_midtown_km, dist_downtown_km, dist_atl_airport_km, dist_min_landmark_km
property_age
hist_rent_lag_1m, hist_rent_lag_3m, hist_rent_lag_12m, hist_rent_lag_24m, hist_rent_yoy
```

#### Categorical features (13 columns)

Each must be a string. Empty / NaN allowed (the model handles unseen levels via target-encoder fallback).

```
unit_type, unit_mix_type, haystacks_unit_type
sub_market, zipcode, county
parking_type, garage
h3_res6, h3_res8
brand, street_type, addr_dir
```

#### Ordinal grade features (2 columns)

Encoded as integers: `A+=8, A=7, A-=6, B+=5, B=4, B-=3, C+=2, C=1, C-=0, D+=-1, D=-2, D-=-3`. NaN allowed.

```
property_quality, location_quality
```

#### Boolean features (16 columns)

Each must be 0/1 integer or NaN.

```
is_yardi_btr, is_hstx_btr, is_yardi_unittype_btr, is_leased_up, is_mixed_use,
is_elevator_served, has_controlled_access, has_fitness_center, has_business_center,
has_clubhouse, has_garage, has_media_room, has_townhouse, covered_parking,
rent_office, wd_hookup, wd_in_unit
```

#### Text-derived features (21 columns)

Extracted from `property_name` and `street_address`. The model expects these to already be derived; if the DE team uses different name/address parsing, they must produce equivalent semantics.

**Booleans (15):**
```
name_kw_premium, name_kw_midmarket, name_kw_older, name_kw_conversion, name_kw_phase,
name_starts_the, name_has_at, name_has_digit, name_has_ampersand,
name_claims_premium_subm, name_subm_match,
addr_is_peachtree, addr_is_iconic, addr_has_highway, addr_has_suite
```

**Numerics (6):**
```
name_len, name_n_words, name_caps_ratio,
addr_house_num, addr_house_num_log, addr_n_words
```

## How to validate a parquet conforms

```python
from prime_mfr.features.contract import validate_pre_engineered

df = pd.read_parquet("your_pre_engineered.parquet")
result = validate_pre_engineered(df)
if not result.is_valid:
    for issue in result.issues:
        print(f"  - {issue}")
```

The validator reports:

- **MISSING_COLUMN** — a required column is absent
- **WRONG_DTYPE** — a column exists but has the wrong type
- **OUT_OF_RANGE** — values violate sanity bounds (e.g. lat outside Atlanta MSA, sqft ≤ 0)
- **EXTRA_COLUMN** — present but not in the contract (warning, not error)

CI gates the DE team's pipeline on this validator.

## Sample / canonical pre-engineered dataset

`tests/fixtures/pre_engineered_sample.parquet` is a 200-row pre-engineered sample. It's the canonical example of what the DE team's output should look like. The model's unit tests use this fixture for inference round-trip validation.

To regenerate it from raw Yardi data:

```bash
uv run python -c "
from prime_mfr import data_processing as dp, features as fe
df = dp.load_clean().head(200).copy()
df = fe.add_static_features(df)
df.to_parquet('tests/fixtures/pre_engineered_sample.parquet', index=False)
"
```

## Sanity bounds

The validator enforces these ranges. Out-of-range values fail the contract check.

| Feature | Min | Max | Reason |
|---|---|---|---|
| `latitude` | 32.5 | 35.0 | Atlanta MSA bounding box |
| `longitude` | -85.5 | -83.5 | Atlanta MSA bounding box |
| `sqft` | 100 | 10000 | Plausible unit size |
| `beds` | 0 | 8 | 0 = studio |
| `baths` | 0.5 | 8.0 | Half-baths allowed |
| `year_built` | 1850 | 2030 | Built or under construction |
| `num_units` | 1 | 5000 | Reasonable property size |
| `occupancy_rate` | 0.0 | 1.0 | Fraction |
| `hist_rent_lag_*` | 100 | 50000 | $ per month |

## DE-team scope vs. model scope

| Owned by Data Engineering | Owned by Model Team |
|---|---|
| **Static feature derivation** — landmark distances, H3 cells, text features, geo aggregates, hist-rent lookups (everything in `add_static_features`) | **OOF target encoding** — these are computed inside the training loop using train-fold means with Bayesian smoothing. At inference, the model applies the saved encoder lookup tables. |
| Schema validation against this contract | Loading the trained model artifact (`models/primary/<version>/`) |
| Snapshot freshness, lineage, observability | Feature column ordering at inference (the model's `feature_schema.json` is authoritative) |
| Backfills for historical snapshots | Versioning and promotion of model artifacts |

## Versioning the contract

This contract is versioned alongside the model. Each model artifact's `manifest.json` records:

```json
{
  "feature_contract_version": "1.0",
  "feature_schema_file": "feature_schema.json"
}
```

A change to the contract (adding/removing/renaming a feature, changing types) bumps the major version and is a coordinated release with the DE team. Backward-incompatible changes require:

1. RFC document
2. Co-deployment plan with the DE team
3. Shadow validation period before cutover

## Open questions for the DE team

1. **Snowflake vs Spark vs Databricks for the materialization pipeline?** This affects the SQL/Python translation of the static feature transforms. Independent of the model.
2. **Refresh cadence.** Monthly (matching Yardi drops) or daily (more freshness but more compute)?
3. **Backfill strategy.** Do we need pre-engineered parquets for past snapshots (for retraining), or only current snapshot (for inference)?
4. **Owner-of-record for H3 cells and landmark coordinates.** Today they live in `eda/atlanta_landmarks.json` and `configs/geographic/`. Should this canonical reference data move to the DE team's repo?
