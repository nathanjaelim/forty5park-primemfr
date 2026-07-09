# Feature Engineering Specification — For Data Engineering Re-implementation

> **Audience:** Data engineers re-implementing the feature pipeline in their stack of choice (Snowflake SQL, dbt, Spark, Databricks, BigQuery, etc.)
> **Purpose:** A complete and unambiguous reference for every feature the model consumes, so that a parquet conforming to this spec can be produced upstream of the model.
> **Companion docs:**
> - `docs/feature_pipeline_contract.md` — the contract (what columns to produce, with sanity bounds)
> - This document — the implementation logic (how each feature is computed)

---

## Reading guide

The model consumes **~200 features** organized into 13 conceptual groups. Each section below covers one group with:

- **Inputs** — which Yardi-source columns the feature depends on
- **Logic** — exact transformation rules
- **Output columns** — names, types, semantics
- **Nuances** — edge cases, ordering dependencies, leakage avoidance, surprising behavior
- **SQL-flavored pseudocode** — for the trickier transforms

The pipeline runs in two phases:

1. **Static features** — computable once over the full dataset; no train/test split required. Most features.
2. **Out-of-fold (OOF) features** — leakage-sensitive; must be recomputed per CV fold or applied via train-frozen lookup tables at inference. Three subgroups: target encoding, KNN aggregates, neighborhood PSF.

For inference (i.e. when DE produces the pre-engineered parquet), the OOF features get applied as **left-joins against frozen lookup tables** that the model artifact ships. DE does not need to re-implement the OOF logic — only the static side. This document still describes them for context.

---

## Input data — the four Yardi feeds

| Feed | Cardinality | Join key | Role |
|---|---|---|---|
| `042026-property-enriched-12060.parquet` | ~2,121 properties | `property_id` / `source_property_id` | Building-level features (lat/lon, amenities, year_built, quality grades) |
| `042026-unit-mix-enriched-12060.parquet` | ~6,887 (property × unit_type) | `source_property_id`, `unit_type` | Floor-plan breakdown (sqft, beds, baths) |
| `042026-rent-12060.parquet` | ~6,891 (property × unit_type) | `property_id`, `unit_type` | Training target (asking rent at snapshot) |
| `042026-hist-rent-12060-12060.parquet` | ~708,825 rows | `property_id`, `unit_type`, `period` | 24-month historical rent panel |

The full identity granularity in production is **(property_id, unit_type, unit_mix_type, unit_garage)** at one `period` (2026-03-01).

Yardi sometimes lists the same `(property, unit_type)` twice with different `unit_mix_type` (Apartment vs Townhouse) or `unit_garage` (0/1). The composite key must include all four columns to be unique. The naive 2-column key is wrong — about 9% of rows are duplicates under that key.

---

## Phase 1 — Static features

### 1. Landmark distances (5 features)

**Goal:** Encode "how close is this property to important Atlanta nodes."

**Inputs:** `latitude`, `longitude` from `property-enriched`, plus the curated landmark file `eda/atlanta_landmarks.json`.

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `dist_buckhead_km` | float32 | Great-circle km to Buckhead (33.83942, -84.37992) |
| `dist_midtown_km` | float32 | km to Midtown Atlanta (33.7868014, -84.3795169) |
| `dist_downtown_km` | float32 | km to Downtown Atlanta |
| `dist_atl_airport_km` | float32 | km to Hartsfield-Jackson |
| `dist_min_landmark_km` | float32 | Element-wise min over the four landmark distances |

**Logic (haversine formula):**

```python
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # Earth's radius in km (WGS84 mean)
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = sin(dlat/2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c
```

**SQL equivalent** (Postgres / Snowflake `RADIANS`, `SIN`, `COS`, `ATAN2`):

```sql
SELECT
  property_id,
  6371.0088 * 2 * ASIN(SQRT(
    POWER(SIN(RADIANS(latitude - 33.83942) / 2), 2) +
    COS(RADIANS(33.83942)) * COS(RADIANS(latitude)) *
    POWER(SIN(RADIANS(longitude - (-84.37992)) / 2), 2)
  )) AS dist_buckhead_km
FROM property_enriched;
```

**Nuances:**
- Coordinates are WGS84 decimal degrees.
- Properties with NaN lat or NaN lon produce NaN distances (don't impute; let the model handle missingness).
- `dist_min_landmark_km` is the row-wise minimum of the four landmark columns — implements a "distance to the nearest important point" feature. Take this row-by-row, not over the whole table.
- The landmark file is editable. If we add a new landmark (`ponce_city_market`), a new `dist_ponce_city_market_km` column appears automatically.
- All outputs are `float32` (memory budget; the model doesn't need float64 precision).

---

### 2. H3 spatial cells (2 features)

**Goal:** Quantize geographic coordinates into discrete, hierarchical "neighborhoods" of fixed size. Trees split categoricals well; lat/lon scatter is harder to learn.

**Inputs:** `latitude`, `longitude`.

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `h3_res6` | str (15-char hex) | H3 cell ID at resolution 6 (~36 km² per cell — submarket-sized) |
| `h3_res8` | str (15-char hex) | H3 cell ID at resolution 8 (~0.7 km² per cell — neighborhood-sized) |

**Logic:** Uber's H3 library (`h3-py`). One call per row per resolution: `h3.latlng_to_cell(lat, lon, resolution)`.

```python
import h3
df["h3_res6"] = df.apply(lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], 6), axis=1)
df["h3_res8"] = df.apply(lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], 8), axis=1)
```

**For Snowflake / BigQuery:** there's no native H3 function. Options:
- A Python UDF wrapping the `h3-py` library (preferred).
- BigQuery has `CARTO`-published H3 functions if you're on that ecosystem.
- A pre-computed lookup table mapping every (rounded lat, rounded lon) to its H3 cell.

**Nuances:**
- The choice of two resolutions matters. Res-6 ≈ submarket; res-8 ≈ block. The model uses res-6 as a "city neighborhood" categorical and res-8 as a finer-grain proxy for "this specific cluster of buildings."
- Missing lat/lon produces the literal string `"__missing__"` (treated as its own category by the model).
- H3 cells are strings, not integers. Don't try to numerically encode them — let downstream target encoding do that.
- Resolutions [6, 8] are the production setting; resolution 7 was tried and added no signal; resolution 9 was too granular and overfit per fold.

---

### 3. Text features from `property_name` and `street_address` (24 features)

**Goal:** Extract structured signal from free-text fields.

**Inputs:** `property_name`, `street_address`, plus curated vocabularies in `src/prime_mfr/config.py` (`NAME_KEYWORDS_*`, `KNOWN_BRANDS`, `ICONIC_STREETS`, `HIGHWAY_TOKENS`, `STREET_SUFFIX_MAP`).

**Output columns:**

| Column | Type | Logic |
|---|---|---|
| **Categoricals (3):** | | |
| `brand` | str | First 1-2 tokens of `property_name` matched against `KNOWN_BRANDS` list (Avalon, Camden, Cortland, ...). Falls back to "first word" for unknown brands. |
| `street_type` | str | Last 4 tokens of `street_address` matched against `STREET_SUFFIX_MAP` (Street, Road, Drive, Boulevard, ...). |
| `addr_dir` | str | Directional token (NE, NW, SE, SW, N, S, E, W) from address. |
| **Boolean flags (15):** | int8 (0/1) | All from substring matches in lowercase. |
| `name_kw_premium` | | `name` contains any of: "residences", "tower", "heights", "vista", "pointe", "reserve", "plaza", "estate", "mansion", "luxury" |
| `name_kw_midmarket` | | "apartments", "place", "crossing", "station", "commons", "square", "village", "park" |
| `name_kw_older` | | "gardens", "manor", "arms", "court", "courts" |
| `name_kw_conversion` | | "lofts", "mill", "foundry", "works", "factory", "studios" |
| `name_kw_phase` | | "phase", " ii", " iii", " iv" (multi-phase developments) |
| `name_starts_the` | | name starts with "the " |
| `name_has_at` | | "@" or " at " in name (e.g. "Lofts at Atlantic Station") |
| `name_has_digit` | | name contains any digit (often street-number-style names) |
| `name_has_ampersand` | | name contains "&" |
| `name_claims_premium_subm` | | name contains any of `PREMIUM_SUBM_TOKENS` ("buckhead", "midtown", "brookhaven", ...) |
| `name_subm_match` | | name token matches the FIRST token of `sub_market` (e.g. "Buckhead" in both) |
| `addr_is_peachtree` | | "peachtree" in address |
| `addr_is_iconic` | | address matches any of `ICONIC_STREETS` |
| `addr_has_highway` | | address matches any of `HIGHWAY_TOKENS` ("i-285", "i-75", "ga-400", ...) |
| `addr_has_suite` | | address contains "suite", "ste", "#", or "unit" |
| **Numerics (6):** | float32 | |
| `name_len` | | Character length of `property_name` |
| `name_n_words` | | Space-split word count |
| `name_caps_ratio` | | Fraction of letter chars that are uppercase |
| `addr_house_num` | | First leading integer in `street_address` (NaN if none) |
| `addr_house_num_log` | | `log1p(addr_house_num)` for scale compression |
| `addr_n_words` | | Word count of address |

**Nuances:**
- All matching is **case-insensitive** (lowercase normalization before matching).
- The vocabularies are tuned to **Atlanta's naming conventions**. Re-implementations for new markets should curate similar vocabularies (or use these as a reasonable starting point).
- `brand` extraction tries longer brand matches first ("Wood Partners" before "Wood") to avoid substring conflicts.
- The "name claims premium submarket" flag detects "marketing aspiration" — when an Eastside building names itself "Midtown" but isn't there. The model uses this as a slight negative correction.
- `_kw_phase` looks for `" ii"` and `" iii"` with **leading space** to avoid matching inside other words.

---

### 4. Geographic aggregates (60 features)

**Goal:** Encode "what does the neighborhood look like" without using rent.

**Inputs:** `sub_market`, `zipcode`, `county`, plus the physical attribute columns: `sqft`, `beds`, `baths`, `property_age`, `num_units`.

**Output columns:** For each `(geo_level, alias)` ∈ `[(zipcode, "zip"), (sub_market, "subm"), (county, "cty")]` and each numeric attribute `n` ∈ `[sqft, beds, baths, property_age, num_units]`, we produce four columns:

| Column pattern | Example | Meaning |
|---|---|---|
| `{n}_{alias}_mean` | `sqft_zip_mean` | Mean of `n` over properties (not unit-rows!) in this geo level |
| `{n}_{alias}_median` | `sqft_zip_median` | Median |
| `{n}_{alias}_std` | `sqft_zip_std` | Standard deviation |
| `{n}_{alias}_count` | `sqft_zip_count` | Count of properties in the geo level |

3 geo levels × 5 numerics × 4 stats = **60 columns**.

**Logic:** Aggregate at the **property level first**, then merge back to unit rows. This ensures a 250-unit Class-A property doesn't dominate the aggregate vs. a 50-unit small property in the same ZIP.

```python
# 1. Reduce to one row per property with mean physical attributes.
property_df = df.groupby("property_id").agg({
    "sqft": "mean", "beds": "mean", "baths": "mean",
    "property_age": "mean", "num_units": "mean",
    "sub_market": "first", "zipcode": "first", "county": "first",
})

# 2. For each (geo_level, attribute), compute group stats.
for geo, alias in [("zipcode", "zip"), ("sub_market", "subm"), ("county", "cty")]:
    for n in ["sqft", "beds", "baths", "property_age", "num_units"]:
        grouped = property_df.groupby(geo)[n]
        property_df[f"{n}_{alias}_mean"]   = grouped.transform("mean")
        property_df[f"{n}_{alias}_median"] = grouped.transform("median")
        property_df[f"{n}_{alias}_std"]    = grouped.transform("std")
        property_df[f"{n}_{alias}_count"]  = grouped.transform("count")

# 3. Merge back onto the unit-level df via property_id.
df = df.merge(property_df, on="property_id", how="left")
```

**SQL pseudocode:**

```sql
WITH property_attrs AS (
  SELECT
    property_id,
    sub_market,
    zipcode,
    county,
    AVG(sqft)         AS sqft_avg,
    AVG(beds)         AS beds_avg,
    AVG(baths)        AS baths_avg,
    AVG(property_age) AS age_avg,
    AVG(num_units)    AS num_units_avg
  FROM unit_level
  GROUP BY 1, 2, 3, 4
),
zip_aggs AS (
  SELECT
    property_id, zipcode,
    AVG(sqft_avg)            OVER (PARTITION BY zipcode) AS sqft_zip_mean,
    MEDIAN(sqft_avg)         OVER (PARTITION BY zipcode) AS sqft_zip_median,
    STDDEV(sqft_avg)         OVER (PARTITION BY zipcode) AS sqft_zip_std,
    COUNT(*)                 OVER (PARTITION BY zipcode) AS sqft_zip_count
  FROM property_attrs
)
-- Repeat the same pattern for submarket-level (alias=subm) and county-level (alias=cty).
SELECT ... FROM unit_level u
LEFT JOIN zip_aggs USING (property_id);
```

**Nuances:**
- **No target/rent involvement** — these are pure structural aggregates. Leakage-free; computed once on the full dataset.
- The property-first aggregation is **important**. Aggregating at the unit-row level overweights properties with many floor plans.
- `count` columns are useful because they tell the model how reliable the local mean is (a `sqft_cty_count = 200` mean is more trustworthy than a 5-property county mean).
- All stat columns are cast to **float32**.

---

### 5. Z-score deviations (15 features)

**Goal:** Encode "how unusual is this unit relative to its neighborhood." A pattern that consistently helps tabular regression on Kaggle.

**Inputs:** Requires geo aggregates (group 4) to already be computed.

**Output columns:** For each `(geo_level, alias)` × `numeric` from the same enumeration as group 4:

| Column pattern | Example | Formula |
|---|---|---|
| `{n}_{alias}_z` | `sqft_zip_z` | `(value - mean) / std` |

**Logic:**

```python
for _, alias in [("zipcode", "zip"), ("sub_market", "subm"), ("county", "cty")]:
    for n in ["sqft", "beds", "baths", "property_age", "num_units"]:
        mean_col, std_col = f"{n}_{alias}_mean", f"{n}_{alias}_std"
        std_safe = df[std_col].replace(0.0, np.nan)        # avoid div-by-zero
        df[f"{n}_{alias}_z"] = ((df[n] - df[mean_col]) / std_safe).astype("float32")
```

**Nuances:**
- **Std-safe pattern:** if std is 0 (single-property county), the z-score is NaN (not Inf). The model handles NaN via LightGBM's missing-bin / CatBoost's NaN feature.
- This is **the most-mentioned feature trick in winning Kaggle solutions for tabular regression**. The intuition: a 1500-sqft unit in a 800-sqft-mean zipcode (z = +2) is a different beast than a 1500-sqft unit in a 1500-sqft-mean zipcode (z = 0).
- 3 geo levels × 5 numerics = **15 z-score columns**.

---

### 6. Static interactions (7 features)

**Goal:** Multiplicative and ratio features that trees don't naturally compose (a single tree split chooses one variable, not a product).

**Inputs:** `sqft`, `beds`, `baths`, `property_age`, `year_built`, `dist_buckhead_km` (group 1 must have run first).

**Output columns:**

| Column | Formula | Why it matters |
|---|---|---|
| `sqft_x_beds` | `sqft * beds` | Co-scaling of unit footprint and bedroom count |
| `sqft_per_bed` | `sqft / max(beds, 1)` | Layout density — smaller per-bed = denser, often urban |
| `baths_per_bed` | `baths / max(beds, 1)` | Bathroom-to-bedroom ratio (luxury proxy) |
| `sqft_x_buckhead_km` | `sqft * dist_buckhead_km` | How a large unit's premium varies with distance |
| `beds_x_buckhead_km` | `beds * dist_buckhead_km` | Bedroom premium varies by distance |
| `year_x_buckhead_km` | `year_built * dist_buckhead_km` | Age premium varies by location |
| `age_x_sqft` | `property_age * sqft` | Age effect scales with size |

**Nuances:**
- Where division can divide by zero, beds is **clipped to 1** (not dropped). Studios get `sqft_per_bed = sqft`.
- All outputs are float32.
- These cross-features were the result of error-segmentation analysis (`error_segmentation.py`) — the team found tree-only models underweighted size-distance interactions. Adding these reduced fold-3 MAE by ~$8.

---

### 7. Bucket keys (4 features)

**Goal:** Provide discrete bin indices for the hierarchical comparable target encoding (group 11 below).

**Inputs:** `sqft`, `property_age`, `beds`, `baths`.

**Output columns:**

| Column | Type | Logic |
|---|---|---|
| `sqft_bucket` | int16 | Bin index from edges `(700, 850, 1000, 1150, 1350, 1700)`. 0 = `< 700`; 6 = `>= 1700`. |
| `age_bucket` | int16 | Bin index from edges `(5, 15, 30, 50)`. 0 = `<5`; 4 = `>=50`. |
| `beds_int` | int16 | `beds` rounded to nearest int. NaN → -1 sentinel. |
| `baths_int` | int16 | `baths` rounded to nearest int. NaN → -1 sentinel. |

**Logic:**

```python
df["sqft_bucket"] = np.digitize(df["sqft"].fillna(-1), bins=[700, 850, 1000, 1150, 1350, 1700])
df["age_bucket"]  = np.digitize(df["property_age"].fillna(-1), bins=[5, 15, 30, 50])
df["beds_int"]    = df["beds"].fillna(-1).round().astype("int16")
df["baths_int"]   = df["baths"].fillna(-1).round().astype("int16")
```

**Nuances:**
- The bucket edges were tuned to give approximately equal-population bins on Atlanta data. **For a new market, re-tune the edges** — Dallas units tend to be bigger; Charlotte smaller.
- `-1` is the sentinel for missing; the hierarchical TE handles it as its own bucket.
- `np.digitize` is right-open: a sqft of exactly 700 lands in bucket 1, not bucket 0.

---

### 8. Property structural / BTR features (6 features)

**Goal:** Identify build-to-rent (BTR) typology — single-family-style rental developments that have a sharply different rent regime than apartments (~$0.90-$1.10 PSF vs $1.50+ for apartments).

**Inputs:** `property_id` (groupby key), plus `beds`, `baths`, `sqft`, `year_built` for the aggregations.

**Output columns** (constant per property, broadcast to all unit-rows):

| Column | Type | Logic |
|---|---|---|
| `property_n_unit_types` | int16 | Count of distinct unit-mix rows for this property. BTR typically 1; apartments 3-5+ |
| `property_beds_nunique` | int16 | Distinct bed counts within property |
| `property_baths_nunique` | int16 | Distinct bath counts within property |
| `property_sqft_range_pct` | float32 | `(sqft_max - sqft_min) / sqft_median` — narrow = uniform product (BTR), wide = mixed |
| `is_btr_likely` | float32 (0/1) | Composite binary: ALL FOUR conditions met |
| `btr_likely_score` | float32 (0..4) | Count of conditions met — partial-BTR gradient |

The four conditions for BTR scoring:

1. `property_n_unit_types <= 1` (single floor plan)
2. `property_beds_nunique <= 1` (single bed count)
3. `year_built >= 2015` (recent vintage; BTR boom started ~2015)
4. `property_sqft_median >= 1500` (large units)

**SQL pseudocode:**

```sql
WITH property_aggs AS (
  SELECT
    property_id,
    COUNT(*)                                                    AS property_n_unit_types,
    COUNT(DISTINCT beds)                                        AS property_beds_nunique,
    COUNT(DISTINCT baths)                                       AS property_baths_nunique,
    (MAX(sqft) - MIN(sqft)) / NULLIF(MEDIAN(sqft), 0)           AS property_sqft_range_pct,
    MAX(year_built)                                             AS yb,
    MEDIAN(sqft)                                                AS sqft_med
  FROM unit_level
  GROUP BY property_id
),
scored AS (
  SELECT
    property_id,
    property_n_unit_types, property_beds_nunique,
    property_baths_nunique, property_sqft_range_pct,
    (CASE WHEN property_n_unit_types <= 1 THEN 1 ELSE 0 END +
     CASE WHEN property_beds_nunique <= 1 THEN 1 ELSE 0 END +
     CASE WHEN yb >= 2015 THEN 1 ELSE 0 END +
     CASE WHEN sqft_med >= 1500 THEN 1 ELSE 0 END
    ) AS btr_likely_score,
    (CASE WHEN property_n_unit_types <= 1
         AND property_beds_nunique <= 1
         AND yb >= 2015
         AND sqft_med >= 1500
       THEN 1 ELSE 0 END
    ) AS is_btr_likely
  FROM property_aggs
)
SELECT u.*, s.* FROM unit_level u LEFT JOIN scored s USING (property_id);
```

**Nuances:**
- The thresholds (year ≥ 2015, sqft ≥ 1500, etc.) are tunable via `config.BTR_*` constants and were validated against Atlanta data. For a new market, **start with these and validate against ground truth** (Yardi's own BTR flags + manual labeling).
- `is_btr_likely` (binary) and `btr_likely_score` (0-4) are BOTH features. The score captures "partial-BTR" properties (e.g., a luxury duplex development that's BTR-like but with multiple floor plans). The model uses both.

---

### 9. Unit-subtype counts (added 2026-05-01) — handled in `add_unit_subtype_features`

**Goal:** Distinguish the actual rental-unit count of THIS floor plan from the property-wide total.

**Inputs:** `num_units` (property-wide from property-enriched), `unit_type`, and the unit-level `num_units` from unit-mix-enriched.

**Output column:**

| Column | Type | Logic |
|---|---|---|
| `num_units_subtype` | float32 | The count of units of THIS unit_type at this property (from unit-mix-enriched.num_units, not property-enriched.num_units) |

**Nuances:**
- Critically distinct from `num_units` which is the property-wide total. A 200-unit property with 50 of unit-type-A and 150 of unit-type-B would have:
  - `num_units = 200` for both rows
  - `num_units_subtype = 50` for the A row, `150` for the B row
- This was a 2026-05-01 addition that produced ~$3 MAE improvement on its own.

---

### 10. Competitor counts (1 feature per radius)

**Goal:** Encode supply density — how many other rental properties of similar size exist within a fixed radius.

**Inputs:** `latitude`, `longitude`, `beds` (rounded), per-property set of beds offered.

**Output columns** (currently single radius):

| Column | Type | Logic |
|---|---|---|
| `n_competitors_within_1mi_same_beds` | int16 | Count of OTHER properties within 1 mile that have at least one unit matching this row's rounded bed count |

**Algorithm:**

1. For each property, compute the set of distinct bed counts it offers.
2. Build a spatial index (BallTree with Haversine metric) on property coordinates.
3. For each row, query all properties within `1 mi = 1.609 km`. Subtract 1 to exclude self.
4. Among those neighbors, count how many have THIS row's bed count in their offered set.

**SQL approach for warehouses:**

```sql
-- Step 1: Distinct beds per property
property_beds AS (
  SELECT DISTINCT property_id, ROUND(beds) AS beds_int
  FROM unit_level
),
-- Step 2: All (focal, neighbor) pairs within 1 mile via haversine
candidate_pairs AS (
  SELECT
    p1.property_id      AS focal_property,
    p2.property_id      AS neighbor_property,
    haversine_km(p1.lat, p1.lon, p2.lat, p2.lon) AS dist_km
  FROM property_locations p1
  JOIN property_locations p2 ON p1.property_id < p2.property_id  -- symmetric
  WHERE haversine_km(p1.lat, p1.lon, p2.lat, p2.lon) <= 1.609
),
-- Step 3: For each focal property + each bed count it offers, count matching neighbors
counts AS (
  SELECT
    cp.focal_property,
    pb.beds_int,
    COUNT(DISTINCT cp.neighbor_property) AS n_competitors
  FROM candidate_pairs cp
  JOIN property_beds pb ON pb.property_id = cp.focal_property
  JOIN property_beds nb ON nb.property_id = cp.neighbor_property AND nb.beds_int = pb.beds_int
  GROUP BY cp.focal_property, pb.beds_int
)
SELECT u.*, c.n_competitors AS n_competitors_within_1mi_same_beds
FROM unit_level u
LEFT JOIN counts c
  ON c.focal_property = u.property_id AND c.beds_int = ROUND(u.beds);
```

**Nuances:**
- **Self always excluded** from the neighbor pool.
- **Same-bed match** uses rounded `beds` integer — a 2.5-bed unit (rare) rounds to 3 and matches 3-bed competitors.
- A 0.5-mile and 2-mile radius were also tried; both regressed. Pruned to 1-mile only.
- Pure structural / supply feature — **no rent data involved → leakage-free**.

---

### 11. Historical rent lags (5 features)

**Goal:** The single most important feature group in the model. Encodes "this unit's recent rent trajectory."

**Inputs:** The historical rent panel (`042026-hist-rent-12060-12060.parquet`), keyed on `(property_id, unit_type, period)`.

**Output columns** (per `(property_id, unit_type)`):

| Column | Type | Logic |
|---|---|---|
| `hist_rent_lag_1m` | float32 | Rent at `period = target_period - 1 month` (Feb 2026) |
| `hist_rent_lag_3m` | float32 | Rent at `target_period - 3 months` (Dec 2025) |
| `hist_rent_lag_12m` | float32 | Rent at `target_period - 12 months` (Mar 2025) |
| `hist_rent_lag_24m` | float32 | Rent at `target_period - 24 months` (Mar 2024) |
| `hist_rent_yoy` | float32 | `lag_1m / lag_13m - 1` (Feb 2026 / Feb 2025 - 1) |

**Algorithm:**

```python
target = pd.Timestamp("2026-03-01")
needed_periods = {
    "hist_rent_lag_1m":  target - pd.DateOffset(months=1),
    "hist_rent_lag_3m":  target - pd.DateOffset(months=3),
    "hist_rent_lag_12m": target - pd.DateOffset(months=12),
    "hist_rent_lag_24m": target - pd.DateOffset(months=24),
    "_lag_13m":          target - pd.DateOffset(months=13),  # for YoY denominator
}

# Pivot: one row per (property_id, unit_type), one column per needed period.
wide = hist_df[hist_df.period.isin(needed_periods.values())] \
        .pivot_table(index=["property_id", "unit_type"], columns="period", values="rent")

# Build the 5 features.
for name, ts in needed_periods.items():
    wide[name] = wide[ts]  # column whose name is the timestamp

wide["hist_rent_yoy"] = wide["hist_rent_lag_1m"] / wide["_lag_13m"] - 1.0
# Drop the helper column.
```

**SQL pseudocode:**

```sql
WITH lag_pivot AS (
  SELECT
    property_id,
    unit_type,
    MAX(CASE WHEN period = DATE '2026-02-01' THEN rent END) AS hist_rent_lag_1m,
    MAX(CASE WHEN period = DATE '2025-12-01' THEN rent END) AS hist_rent_lag_3m,
    MAX(CASE WHEN period = DATE '2025-03-01' THEN rent END) AS hist_rent_lag_12m,
    MAX(CASE WHEN period = DATE '2024-03-01' THEN rent END) AS hist_rent_lag_24m,
    MAX(CASE WHEN period = DATE '2025-02-01' THEN rent END) AS _lag_13m
  FROM hist_rent
  GROUP BY 1, 2
)
SELECT
  *,
  CASE WHEN _lag_13m > 0
    THEN hist_rent_lag_1m / _lag_13m - 1.0
    ELSE NULL
  END AS hist_rent_yoy
FROM lag_pivot;
```

**CRITICAL nuance — leakage avoidance:**

> The target month `2026-03-01` must NEVER appear in any feature. The lag-1m feature is **Feb 2026**, not Mar 2026. If you include Mar 2026 as "lag-0", you're reading the target back to the model and OOF metrics become meaningless.

Practical implementation check: when building the historical panel, filter `WHERE period < '2026-03-01'` before computing any lag.

**Other nuances:**
- Granularity is **(property_id, unit_type)** — not unit-level. When a property has multiple unit-rows with the same unit_type but different unit_mix_type or unit_garage, they all get the same lag values.
- Missing months are NaN. The model handles them. **Do not impute** with a forward-fill — that would invent rent values that weren't observed.
- `hist_rent_yoy` requires BOTH `lag_1m` AND `lag_13m`. If either is null, yoy is null. Guard against zero `lag_13m` (no division by zero).
- The panel is ~700k rows but the pivot reduces to ~6k rows (one per property × unit_type). Memory-cheap.

---

## Phase 2 — Out-of-fold (OOF) features

These are computed **per CV fold during training**, using only training-fold data, then applied to both train and validation rows. At inference, the trained encoders are saved as lookup tables and applied via left-join.

**The DE team does NOT need to re-implement the OOF logic.** These are applied by the model's inference pipeline using saved lookup tables. This section is for context only.

### 12. Bayesian-smoothed target encoding (8 features)

For each of `[sub_market, zipcode, h3_res6, h3_res8, unit_type, haystacks_unit_type, brand, street_type]`:

| Column | Logic |
|---|---|
| `{col}_te` | Smoothed target mean for the category, computed per CV fold |

**Smoothing formula:**

```
encoded(category) = (count * mean + smoothing * global_mean) / (count + smoothing)
```

where `smoothing = 20.0` and `mean` / `count` are computed on the train fold only. Unseen categories at val time fall back to `global_mean`.

**Why smoothing matters:** A submarket with only 3 properties shouldn't dominate. Bayesian smoothing pulls low-count categories toward the global mean.

---

### 13. OOF KNN aggregates, neighborhood PSF, hierarchical comparable TE

A further set of features computed per fold (~25 columns). These are documented in code (`add_oof_features` in `engineering.py`) but not reproduced here since:

1. The DE team doesn't re-implement them.
2. They're frozen as lookup tables in the saved model artifact at register-time.
3. Inference is a join, not a recomputation.

If you're curious: see `bayesian_target_encode`, `compute_oof_knn`, `add_nbhd_psf_features`, `add_comparable_rent`, `add_hierarchical_comp_te` in the source.

---

## Pipeline orchestration — execution order matters

The static-feature pipeline runs the 10 transforms in **this exact order** (some depend on prior steps):

```
1. add_landmark_distances        → needs latitude/longitude
2. add_h3_cells                  → needs latitude/longitude
3. add_text_features             → needs property_name, street_address
4. add_geo_aggregates            → needs sub_market/zipcode/county
5. add_zscore_deviations         ← needs (4) to have run (uses _mean and _std cols)
6. add_static_interactions       ← needs (1) for dist_buckhead_km
7. add_bucket_keys               → independent
8. add_property_structural_features → independent
9. add_unit_subtype_features     → independent
10. add_competitor_count_features → needs latitude/longitude/beds; independent of others
11. add_hist_rent_features       → independent (joins from external hist parquet)
```

If DE re-implements in SQL, this ordering translates to a series of CTEs / temp tables. **Do not parallelize step 5 with step 4** — z-scores need the means/stds.

---

## Validation contract

Once your pipeline produces a parquet, validate it:

```python
import pandas as pd
from prime_mfr.features import validate_pre_engineered, summarize
df = pd.read_parquet("your_pipeline_output.parquet")
result = validate_pre_engineered(df)
print(summarize(result))
```

CI gates the model team's consumption on this validator passing. A failure shouldn't deploy. See `docs/feature_pipeline_contract.md` for the schema contract details.

---

## Reproducibility checklist

Before declaring a re-implementation done:

- [ ] All ~31 numeric features in `configs/features/numeric.yaml` are present and have the right types
- [ ] All categorical / boolean / ordinal_grade features are present
- [ ] Output of `validate_pre_engineered(df)` reports **0 errors** (warnings on extras are fine)
- [ ] Spot-check: 5 random rows have feature values within 0.1% of the Python reference pipeline output (`tests/fixtures/pre_engineered_sample.parquet`)
- [ ] Z-score columns have mean ≈ 0, std ≈ 1 within each geo level (Bayesian smoothing aside)
- [ ] Lag features for known properties reproduce the reference values (don't include the target month)
- [ ] `dist_min_landmark_km` ≤ every individual `dist_<landmark>_km` per row

---

## Open questions for the DE team

1. **H3 implementation.** Snowflake doesn't ship native H3; what's your preferred wrapper? Carto's BigQuery H3 functions? A Python UDF? Pre-computed lookup?
2. **Text feature curation.** The Atlanta-specific vocabularies (KNOWN_BRANDS, ICONIC_STREETS, etc.) live in `src/prime_mfr/config.py`. Should these move to a reference-data table the DE team owns?
3. **Hist-rent panel update cadence.** Are you re-pivoting the lag features monthly when new Yardi snapshots arrive, or maintaining a running materialized view?
4. **Compute footprint.** Roughly: 7k properties × 5 numerics × 3 geo levels × 4 stats = 420k mean/std calculations for geo aggregates. Plus a BallTree spatial query for competitor counts. Manageable in a single dbt run.

---

## Versioning

This spec is versioned alongside the model. The current version is `1.0`. Breaking changes (renaming/removing features, changing types, semantic changes) require a coordinated release with the model team.

When you ship a new feature, add it here in the appropriate section with the same level of detail. Future-you (and future-DE-team) will thank you.
