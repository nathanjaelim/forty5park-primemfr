"""Fetch coffee shop coordinates from OpenStreetMap Overpass API and write
the curated eda/coffee_shops.json consumed by add_coffee_shop_density() in
src/prime_mfr/features/engineering.py.

Usage:
    uv run python eda/fetch_coffee_shops.py

Also writes the raw Overpass response to eda/research/cafes.geojson so the
query never has to be re-run just to re-derive the curated file (see build
step at the bottom, which can be re-run standalone against an existing
cached .geojson via --from-cache).

Schema written to eda/coffee_shops.json (a JSON list, NOT a dict --
add_coffee_shop_density() iterates `for s in shops: s["lat"], s["lon"]`):

[
  {
    "name": "Dancing Goats Coffee Bar",
    "lat": 33.7759237,
    "lon": -84.3032546,
    "source": "..."
  },
  ...
]

Note: 299 shops were already cached under eda/research/cafes.geojson from a
prior Overpass fetch, so eda/coffee_shops.json was built via --from-cache
without needing live network access (this sandbox blocks outbound access to
overpass-api.de). Re-running without --from-cache will pull a fresh set if
network access is available.
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

# NOTE: the area ID below is a placeholder (Atlanta metro OSM relation) and
# has not been live-verified in this sandbox (no network access to
# overpass-api.de). Confirm/adjust it before running without --from-cache.
QUERY = """
[out:json][timeout:60];
node["amenity"="cafe"]["cuisine"="coffee_shop"](area:3600088415);
out;
"""

GEOJSON_PATH = Path("eda/research/cafes.geojson")
SHOPS_PATH = Path("eda/coffee_shops.json")


def fetch_raw() -> dict:
    """Query Overpass (trying mirrors on failure) and return raw JSON."""
    last_err = None
    for url in OVERPASS_MIRRORS:
        try:
            resp = requests.post(url, data={"data": QUERY}, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - try next mirror
            last_err = e
    raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_err}")


def raw_to_geojson(raw: dict) -> dict:
    features = [
        {
            "type": "Feature",
            "properties": el.get("tags", {}),
            "geometry": {
                "type": "Point",
                "coordinates": [el["lon"], el["lat"]],
            },
            "id": f"node/{el['id']}",
        }
        for el in raw.get("elements", [])
        if el.get("type") == "node"
    ]
    return {
        "type": "FeatureCollection",
        "generator": "eda/fetch_coffee_shops.py",
        "features": features,
    }


def build_curated(geojson: dict, fetched_note: str) -> list[dict]:
    """Filter to named cafes (drops the rare unnamed OSM node) and
    normalize schema."""
    shops = []
    for f in sorted(
        geojson["features"], key=lambda f: f["properties"].get("name") or ""
    ):
        p = f["properties"]
        name = p.get("name")
        if not name:
            continue
        lon, lat = f["geometry"]["coordinates"]
        shops.append(
            {
                "name": name,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "source": fetched_note,
            }
        )
    return shops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip the live Overpass query; rebuild coffee_shops.json "
        "from the existing eda/research/cafes.geojson cache.",
    )
    args = parser.parse_args()

    if args.from_cache:
        if not GEOJSON_PATH.exists():
            raise SystemExit(f"{GEOJSON_PATH} not found; run without --from-cache first.")
        geojson = json.loads(GEOJSON_PATH.read_text())
        note = (
            f"OpenStreetMap Overpass API (amenity=cafe, cuisine=coffee_shop); "
            f"rebuilt from cached {GEOJSON_PATH}"
        )
    else:
        raw = fetch_raw()
        geojson = raw_to_geojson(raw)
        GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEOJSON_PATH.write_text(json.dumps(geojson, indent=2))
        print(f"Wrote raw Overpass result to {GEOJSON_PATH} ({len(geojson['features'])} nodes)")
        note = (
            f"OpenStreetMap Overpass API (amenity=cafe, cuisine=coffee_shop); "
            f"fetched live via {GEOJSON_PATH.name}"
        )

    shops = build_curated(geojson, note)
    SHOPS_PATH.write_text(json.dumps(shops, indent=2) + "\n")
    print(f"Wrote {len(shops)} curated coffee shops to {SHOPS_PATH}")


if __name__ == "__main__":
    main()
