"""Exact point-in-polygon filter for restaurants against the real Atlanta
MSA county boundaries -- fixes the leakage you get from Overpass's `area`
filter, which is a rasterized (not exact) approximation of admin boundaries
and lets a fraction of points near a county edge leak in/out.

Why this is needed:
    Overpass QL's `area["name"=...]["admin_level"="6"]` builds an internal
    grid-based membership test from the boundary relation. It is fast but
    NOT exact -- points near a county line can be misclassified regardless
    of how the query is written (confirmed: adding boundary=administrative
    did not fix it). The only reliable fix is to do the point-in-polygon
    test yourself against the actual boundary geometry.

Workflow:
    1. Run eda/research/restaurants_overpass_query.txt in Overpass Turbo
       (or via the API) and export the result as GeoJSON to
       eda/research/restaurants_raw.geojson.
    2. Run this script. It downloads the Census TIGER/Line cartographic
       boundary shapefile for US counties (small, ~2-5MB), keeps only the
       29 counties in the Atlanta-Sandy Springs-Roswell, GA MSA (by FIPS
       code, state FIPS 13 = Georgia), unions them into one polygon, and
       keeps only restaurant points that fall exactly inside it.
    3. Writes eda/research/restaurants_in_msa.geojson (filtered) and prints
       how many points were dropped as false positives from step 1.

Requires: geopandas, shapely, requests (pip install geopandas shapely requests)
Needs live network access to download the Census shapefile the first time
(cached locally afterward at eda/research/tiger_cache/).

Usage:
    python eda/filter_restaurants_to_msa.py
"""
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import shape
from shapely.ops import unary_union

# Census cartographic boundary file (500k resolution, small download).
# Update the year if you want a different vintage; county FIPS codes below
# are stable regardless of year.
TIGER_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

CACHE_DIR = Path("eda/research/tiger_cache")
ZIP_PATH = CACHE_DIR / "cb_2022_us_county_500k.zip"
SHP_PATH = CACHE_DIR / "cb_2022_us_county_500k.shp"

RAW_GEOJSON = Path("eda/research/restaurants_raw.geojson")
FILTERED_GEOJSON = Path("eda/research/restaurants_in_msa.geojson")

GEORGIA_STATE_FIPS = "13"

# Atlanta-Sandy Springs-Roswell, GA MSA -- 29 counties (2023 OMB delineation).
# County FIPS codes (3-digit, within Georgia).
ATL_MSA_COUNTY_FIPS = {
    "013",  # Barrow
    "015",  # Bartow
    "035",  # Butts
    "045",  # Carroll
    "057",  # Cherokee
    "063",  # Clayton
    "067",  # Cobb
    "077",  # Coweta
    "085",  # Dawson
    "089",  # DeKalb
    "097",  # Douglas
    "113",  # Fayette
    "117",  # Forsyth
    "121",  # Fulton
    "135",  # Gwinnett
    "143",  # Haralson
    "149",  # Heard
    "151",  # Henry
    "159",  # Jasper
    "171",  # Lamar
    "199",  # Meriwether
    "211",  # Morgan
    "217",  # Newton
    "223",  # Paulding
    "227",  # Pickens
    "231",  # Pike
    "247",  # Rockdale
    "255",  # Spalding
    "297",  # Walton
}


def ensure_tiger_shapefile() -> Path:
    if SHP_PATH.exists():
        return SHP_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TIGER_URL} ...")
    resp = requests.get(TIGER_URL, timeout=120)
    resp.raise_for_status()
    ZIP_PATH.write_bytes(resp.content)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(CACHE_DIR)
    if not SHP_PATH.exists():
        raise SystemExit(f"Expected {SHP_PATH} after extracting zip; check TIGER_URL vintage.")
    return SHP_PATH


def build_msa_polygon():
    shp_path = ensure_tiger_shapefile()
    counties = gpd.read_file(shp_path)
    # TIGER cartographic boundary columns: STATEFP, COUNTYFP, NAME, ...
    msa = counties[
        (counties["STATEFP"] == GEORGIA_STATE_FIPS)
        & (counties["COUNTYFP"].isin(ATL_MSA_COUNTY_FIPS))
    ]
    if len(msa) != len(ATL_MSA_COUNTY_FIPS):
        found = set(msa["COUNTYFP"])
        missing = ATL_MSA_COUNTY_FIPS - found
        raise SystemExit(f"Only matched {len(msa)}/{len(ATL_MSA_COUNTY_FIPS)} counties; missing FIPS: {missing}")
    return unary_union(msa.geometry.values), msa


def main():
    if not RAW_GEOJSON.exists():
        raise SystemExit(
            f"{RAW_GEOJSON} not found. Export the Overpass query result as "
            f"GeoJSON to this path first."
        )

    msa_polygon, msa_counties = build_msa_polygon()
    print(f"Built MSA polygon from {len(msa_counties)} counties.")

    raw = json.loads(RAW_GEOJSON.read_text())
    features = raw["features"]
    print(f"Loaded {len(features)} raw restaurant features.")

    kept, dropped = [], []
    for f in features:
        geom = shape(f["geometry"])
        if msa_polygon.contains(geom):
            kept.append(f)
        else:
            dropped.append(f)

    print(f"Kept {len(kept)} inside the exact MSA boundary; dropped {len(dropped)} false positives.")

    out = {"type": "FeatureCollection", "features": kept}
    FILTERED_GEOJSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {FILTERED_GEOJSON}")


if __name__ == "__main__":
    main()
