"""Fetch MARTA rail station coordinates from OpenStreetMap Overpass API
and write the curated eda/marta_stations.json consumed by
add_marta_distance() in src/prime_mfr/features/engineering.py.

Usage:
    uv run python eda/fetch_marta_stations.py

Also writes the raw Overpass response to eda/research/MARTA_stations.geojson
so the query never has to be re-run just to re-derive the curated file
(see build step at the bottom, which can be re-run standalone against
an existing cached .geojson via --from-cache).

Schema written to eda/marta_stations.json (a JSON list, NOT a dict --
add_marta_distance() iterates `for s in stations: s["lat"], s["lon"]`):

[
  {
    "name": "Five Points",
    "lat": 33.7538868,
    "lon": -84.3915963,
    "line_ref": "S1",       # MARTA's own station code, if OSM has it
    "wikidata": "Q5456094", # for cross-checking against other sources
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

QUERY = """
[out:json][timeout:60];
node["railway"="station"]["network"="MARTA"];
out;
"""

GEOJSON_PATH = Path("eda/research/MARTA_stations.geojson")
STATIONS_PATH = Path("eda/marta_stations.json")


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
        "generator": "eda/fetch_marta_stations.py",
        "features": features,
    }


def build_curated(geojson: dict, fetched_note: str) -> list[dict]:
    """Filter to real rail stations (drops bus_station / park&ride nodes
    that sometimes come back tagged network=MARTA) and normalize schema."""
    rail = [
        f for f in geojson["features"]
        if f["properties"].get("railway") == "station"
    ]
    stations = []
    for f in sorted(rail, key=lambda f: f["properties"].get("name", "")):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        stations.append({
            "name": p.get("name"),
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "line_ref": p.get("railway:ref"),
            "wikidata": p.get("wikidata"),
            "source": fetched_note,
        })
    return stations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Skip the live Overpass query; rebuild marta_stations.json "
             "from the existing eda/research/MARTA_stations.geojson cache.",
    )
    args = parser.parse_args()

    if args.from_cache:
        if not GEOJSON_PATH.exists():
            raise SystemExit(f"{GEOJSON_PATH} not found; run without --from-cache first.")
        geojson = json.loads(GEOJSON_PATH.read_text())
        note = f"OpenStreetMap Overpass API (railway=station, network=MARTA); rebuilt from cached {GEOJSON_PATH}"
    else:
        raw = fetch_raw()
        geojson = raw_to_geojson(raw)
        GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEOJSON_PATH.write_text(json.dumps(geojson, indent=2))
        print(f"Wrote raw Overpass result to {GEOJSON_PATH} ({len(geojson['features'])} nodes)")
        note = f"OpenStreetMap Overpass API (railway=station, network=MARTA); fetched live via {GEOJSON_PATH.name}"

    stations = build_curated(geojson, note)
    STATIONS_PATH.write_text(json.dumps(stations, indent=2) + "\n")
    print(f"Wrote {len(stations)} curated stations to {STATIONS_PATH}")


if __name__ == "__main__":
    main()
