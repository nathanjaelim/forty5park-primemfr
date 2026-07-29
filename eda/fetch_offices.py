"""Fetch office POI coordinates from OpenStreetMap Overpass API and write
the curated eda/offices.json consumed by add_office_density() in
src/prime_mfr/features/engineering.py.

Usage:
    uv run python eda/fetch_offices.py

Also writes the raw Overpass response to eda/research/offices.geojson so
the query never has to be re-run just to re-derive the curated file (see
build step at the bottom, which can be re-run standalone against an
existing cached .geojson via --from-cache). As of 2026-07-19,
eda/research/offices.geojson already exists as a pre-fetched
overpass-turbo export (office=* tag, same MSA-ish region) -- --from-cache
was used against that file rather than a live Overpass query.

Uses the OSM `office=*` tag (company/coworking/government/insurance/
financial/etc. -- any value), which tags individual office POIs, not
whole buildings. This is an "employment center" proxy, distinct in intent
from the dining/lifestyle-amenity fetches (fetch_restaurants.py /
fetch_coffee_shops.py / fetch_grocery_stores.py / fetch_bars_nightclubs.py).

Same MSA-scoping caveat as fetch_restaurants.py (see that module's
docstring): Overpass's `area` filter is a rasterized approximation and can
leak results outside the intended boundary, so build_curated() below
applies the same extra bbox + addr:state safety filter used there.

Schema written to eda/offices.json (a JSON list, NOT a dict --
add_office_density() iterates `for o in offices: o["lat"], o["lon"]`):

[
  {
    "name": "Truist Plaza",
    "lat": 33.7590,
    "lon": -84.3880,
    "office": "company",
    "source": "..."
  },
  ...
]
"""
import argparse
import json
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# Atlanta-Sandy Springs-Roswell, GA MSA -- 29 counties (2023 OMB
# delineation). See fetch_restaurants.py's docstring: this area-union
# filter did NOT reliably restrict live Overpass results there, so
# build_curated() applies an additional bbox + addr:state safety filter
# below, matching that fetch.
QUERY = """
[out:json][timeout:180];
area["name"="Georgia"]["admin_level"="4"]["boundary"="administrative"]->.ga;
(
  area["name"="Barrow County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Bartow County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Butts County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Carroll County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Cherokee County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Clayton County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Cobb County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Coweta County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Dawson County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="DeKalb County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Douglas County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Fayette County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Forsyth County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Fulton County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Gwinnett County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Haralson County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Heard County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Henry County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Jasper County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Lamar County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Meriwether County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Morgan County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Newton County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Paulding County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Pickens County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Pike County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Rockdale County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Spalding County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
  area["name"="Walton County"]["admin_level"="6"]["boundary"="administrative"](area.ga);
)->.msa;

(
  node["office"](area.msa);
  way["office"](area.msa);
  relation["office"](area.msa);
);
out center tags;
"""

# Generous bbox around the 29-county MSA -- see module docstring.
BBOX_LAT = (33.00, 34.65)
BBOX_LON = (-85.45, -83.45)

GEOJSON_PATH = Path("eda/research/offices.geojson")
OFFICES_PATH = Path("eda/offices.json")


def fetch_raw() -> dict:
    """Query Overpass (trying mirrors on failure) and return raw JSON."""
    last_err = None
    for url in OVERPASS_MIRRORS:
        try:
            resp = requests.post(url, data={"data": QUERY}, timeout=200)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - try next mirror
            last_err = e
    raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_err}")


def raw_to_geojson(raw: dict) -> dict:
    features = []
    for el in raw.get("elements", []):
        if el.get("type") == "node":
            lon, lat = el["lon"], el["lat"]
        else:
            center = el.get("center")
            if not center:
                continue
            lon, lat = center["lon"], center["lat"]
        features.append(
            {
                "type": "Feature",
                "properties": el.get("tags", {}),
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "id": f"{el['type']}/{el['id']}",
            }
        )
    return {
        "type": "FeatureCollection",
        "generator": "eda/fetch_offices.py",
        "features": features,
    }


def build_curated(geojson: dict, fetched_note: str) -> list[dict]:
    """Bbox + addr:state safety filter (see module docstring), then drop
    unnamed nodes and de-dupe, normalizing schema."""
    seen = set()
    offices = []
    for f in sorted(
        geojson["features"], key=lambda f: f["properties"].get("name") or ""
    ):
        p = f["properties"]
        name = p.get("name")
        if not name:
            continue
        lon, lat = f["geometry"]["coordinates"]
        if not (BBOX_LAT[0] <= lat <= BBOX_LAT[1] and BBOX_LON[0] <= lon <= BBOX_LON[1]):
            continue
        state = p.get("addr:state")
        if state and state != "GA":
            continue
        key = (name, round(lat, 5), round(lon, 5))
        if key in seen:
            continue
        seen.add(key)
        offices.append(
            {
                "name": name,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "office": p.get("office"),
                "source": fetched_note,
            }
        )
    return offices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip the live Overpass query; rebuild offices.json "
        "from the existing eda/research/offices_raw.geojson cache.",
    )
    args = parser.parse_args()

    if args.from_cache:
        if not GEOJSON_PATH.exists():
            raise SystemExit(f"{GEOJSON_PATH} not found; run without --from-cache first.")
        geojson = json.loads(GEOJSON_PATH.read_text())
        note = (
            "OpenStreetMap Overpass API (office=*); bbox+addr:state "
            f"filtered to the Atlanta MSA; rebuilt from cached {GEOJSON_PATH}"
        )
    else:
        raw = fetch_raw()
        geojson = raw_to_geojson(raw)
        GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEOJSON_PATH.write_text(json.dumps(geojson, indent=2))
        print(f"Wrote raw Overpass result to {GEOJSON_PATH} ({len(geojson['features'])} nodes)")
        note = (
            "OpenStreetMap Overpass API (office=*); bbox+addr:state "
            f"filtered to the Atlanta MSA; fetched live via {GEOJSON_PATH.name}"
        )

    offices = build_curated(geojson, note)
    OFFICES_PATH.write_text(json.dumps(offices, indent=2) + "\n")
    print(f"Wrote {len(offices)} curated offices to {OFFICES_PATH}")


if __name__ == "__main__":
    main()
