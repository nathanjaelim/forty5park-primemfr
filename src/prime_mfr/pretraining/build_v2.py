"""
build_pretraining_v2.py
=======================
Rebuild the pretraining table from the April-2026 enriched files:
  artifacts/042026-unit-mix-enriched-12060.parquet  (6887 x 12, unit-level)
  artifacts/042026-property-enriched-12060.parquet  (2121 x 59, property-level)

The new files map cleanly to the existing pipeline schema after a column
rename, with one net-new feature:

    haystacks_unit_type
        APARTMENT (5836) / TOWNHOUSE (902) / ROWHOUSE (97) /
        DETACHED_ENTRY (43) / DETACHED_LUXURY (9)

Renames (unit-mix-enriched -> existing pipeline name):
    unit_type                  -> unit_mix_type
    unit_configuration         -> unit_type
    living_area_square_feet    -> sqft
    number_of_bedrooms         -> beds
    number_of_full_baths       -> baths
    num_units                  -> num_units_subtype
    rent_amount                -> rent
    rent_close_date            -> period
    has_direct_access_garage   -> unit_garage   (bool -> int16 0/1)
    date                       -> date_x

Property-enriched contributes the property-level columns (same schema as
the old pretraining property half) plus its own date column -> date_y.

Output: pretraining_v2.parquet (replaces RAW_PARQUET) +
         eda/pretraining_enriched_v2.parquet (replaces ENRICHED_PARQUET).

Both have the same 70-col schema as the existing enriched parquet plus
the new `haystacks_unit_type` column.

Usage
-----
    python build_pretraining_v2.py
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import pandas as pd

import prime_mfr.config as config

PROJECT = config.PROJECT_DIR
SRC_UNIT_MIX = PROJECT / "artifacts" / "042026-unit-mix-enriched-12060.parquet"
SRC_PROPERTY = PROJECT / "artifacts" / "042026-property-enriched-12060.parquet"
OUT_RAW = PROJECT / "pretraining_v2.parquet"
OUT_ENRICHED = PROJECT / "eda" / "pretraining_enriched_v2.parquet"

# Renames from unit-mix-enriched -> pipeline schema.
UNIT_MIX_RENAMES = {
    "unit_type": "unit_mix_type",
    "unit_configuration": "unit_type",
    "living_area_square_feet": "sqft",
    "number_of_bedrooms": "beds",
    "number_of_full_baths": "baths",
    "num_units": "num_units_subtype",
    "rent_amount": "rent",
    "rent_close_date": "period",
    # has_direct_access_garage handled separately (bool -> int16)
    "date": "date_x",
}


def build() -> pd.DataFrame:
    um = pd.read_parquet(SRC_UNIT_MIX)
    pe = pd.read_parquet(SRC_PROPERTY)

    print(f"unit-mix-enriched : {um.shape}")
    print(f"property-enriched : {pe.shape}")

    # 1. Rename unit-mix columns to pipeline names.
    um = um.rename(columns=UNIT_MIX_RENAMES)

    # 2. has_direct_access_garage (bool) -> unit_garage (int16)
    um["unit_garage"] = um["has_direct_access_garage"].astype("int16")
    um = um.drop(columns=["has_direct_access_garage"])

    # 3. property-enriched: date -> date_y; property_id alias
    pe = pe.rename(columns={"date": "date_y"})

    # Make sure no column-name collisions other than source_property_id.
    overlap = (set(um.columns) & set(pe.columns)) - {"source_property_id"}
    if overlap:
        raise RuntimeError(f"Unexpected column overlap between halves: {overlap}")

    # 4. Inner join on source_property_id (every unit-row should have a property).
    df = um.merge(pe, on="source_property_id", how="left", validate="many_to_one")
    if df["market"].isna().any():
        n_missing = df["market"].isna().sum()
        raise RuntimeError(f"{n_missing} unit rows have no property metadata")

    # 5. property_id alias (existing pipeline uses both names).
    df["property_id"] = df["source_property_id"]

    # 6. Add a placeholder unit_mix column. Existing pipeline drops it via
    #    ID_COLS, but presence is required by load_clean's parse-skip check.
    if "unit_mix" not in df.columns:
        df["unit_mix"] = "[]"

    # 7. Make sure the cast/dtype on the columns we touch matches the prior
    #    enriched schema where it cared.
    df["sqft"] = df["sqft"].astype("float64")
    df["beds"] = df["beds"].astype("float64")
    df["baths"] = df["baths"].astype("float64")
    df["num_units_subtype"] = df["num_units_subtype"].astype("float64")
    df["rent"] = df["rent"].astype("float32")

    # 8. Reorder so the schema reads naturally (unit-side first, then property,
    #    then derived).
    leading = [
        "property_id",
        "source_property_id",
        "unit_mix_type",
        "unit_garage",
        "unit_type",
        "haystacks_unit_type",  # NEW
        "period",
        "rent",
        "date_x",
        "sqft",
        "beds",
        "baths",
        "num_units_subtype",
    ]
    rest = [c for c in df.columns if c not in leading]
    df = df[leading + rest]

    return df


def main():
    df = build()

    print()
    print(f"Output rows : {len(df)}  (was 6891 in old enriched)")
    print(f"Columns     : {len(df.columns)}")
    print(f"Properties  : {df['source_property_id'].nunique()}")
    print()
    print("haystacks_unit_type distribution:")
    print(df["haystacks_unit_type"].value_counts(dropna=False).to_string())
    print()
    print("Null counts on critical cols:")
    for c in [
        "rent",
        "sqft",
        "beds",
        "baths",
        "num_units_subtype",
        "haystacks_unit_type",
        "sub_market",
        "zipcode",
    ]:
        print(f"  {c:<22s}: {df[c].isna().sum()}")

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT_ENRICHED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_RAW, index=False)
    df.to_parquet(OUT_ENRICHED, index=False)
    print()
    print(f"Wrote {OUT_RAW}")
    print(f"Wrote {OUT_ENRICHED}")


if __name__ == "__main__":
    main()
