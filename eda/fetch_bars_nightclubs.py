"""Fetch bar/nightclub coordinates from OpenStreetMap Overpass API and
write the curated eda/bars_nightclubs.json consumed by add_bar_density()
in src/prime_mfr/features/engineering.py.

Usage:
    uv run python eda/fetch_bars_nightclubs.py

Also writes the raw Overpass response to
eda/research/bars_nightclubs_raw.geojson so the query never has to be
re-run just to re-derive the curated file (see build step at the bottom,
which can be re-run standalone against an existing cached .geojson via
--from-cache).

Query uses a bbox directly (query defined in
eda/research/bars_nightclubs_overpass_query.txt) rather than a county
area-union, since the area-union approach used for restaurants did NOT
reliably scope results to the MSA in practice (see fetch_restaurants.py's
docstring). The bbox generously covers the 29-county Atlanta MSA
(lat 33.00-34.65, lon -85.45--83.45) -- same box used to safety-filter the
restaurants export -- so build_curated() still applies the addr:state
safety filter as a second check, same as fetch_restaurants.py.

Schema written to eda/bars_nightclubs.json (a JSON list, NOT a dict --
add_bar_density() iterates `for b in bars: b["lat"], b["lon"]`):

[
  {
    "name": "Sister Louisa's Church",
    "lat": 33.7553,
    "lon": -84.3684,
    "amenity": "bar",
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
[out:json][timeout:180][bbox:33.00,-85.45,34.65,-83.45];
(
  node["amenity"="bar"];
  node["amenity"="nightclub"];
  way["amenity"="bar"];
  way["amenity"="nightclub"];
);
out center tags;
"""

# Same bbox as the query above -- used as a second safety filter in
# build_curated(), matching fetch_restaurants.py's pattern.
BBOX_LAT = (33.00, 34.65)
BBOX_LON = (-85.45, -83.45)

GEOJSON_PATH = Path("eda/research/bars.geojson")
BARS_PATH = Path("eda/bars_nightclubs.json")


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
        "generator": "eda/fetch_bars_nightclubs.py",
        "features": features,
    }


def build_curated(geojson: dict, fetched_note: str) -> list[dict]:
    """Bbox + addr:state safety filter (see module docstring), then drop
    unnamed nodes and de-dupe, normalizing schema."""
    seen = set()
    bars = []
    for f in sorted(
        geojson["features"], key=lambda f: f["properties"].get("name") or ""
    ):
        p = f["properties"]
        name = p.get("name")
        if not name:
            continue
        amenity = p.get("amenity")
        if amenity not in ("bar", "nightclub"):
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
        bars.append(
            {
                "name": name,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "amenity": amenity,
                "source": fetched_note,
            }
        )
    return bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip the live Overpass query; rebuild bars_nightclubs.json "
        "from the existing eda/research/bars_nightclubs_raw.geojson cache.",
    )
    args = parser.parse_args()

    if args.from_cache:
        if not GEOJSON_PATH.exists():
            raise SystemExit(f"{GEOJSON_PATH} not found; run without --from-cache first.")
        geojson = json.loads(GEOJSON_PATH.read_text())
        note = (
            "OpenStreetMap Overpass API (amenity=bar / amenity=nightclub); "
            f"bbox+addr:state filtered to the Atlanta MSA; rebuilt from cached {GEOJSON_PATH}"
        )
    else:
        raw = fetch_raw()
        geojson = raw_to_geojson(raw)
        GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEOJSON_PATH.write_text(json.dumps(geojson, indent=2))
        print(f"Wrote raw Overpass result to {GEOJSON_PATH} ({len(geojson['features'])} nodes)")
        note = (
            "OpenStreetMap Overpass API (amenity=bar / amenity=nightclub); "
            f"bbox+addr:state filtered to the Atlanta MSA; fetched live via {GEOJSON_PATH.name}"
        )

    bars = build_curated(geojson, note)
    BARS_PATH.write_text(json.dumps(bars, indent=2) + "\n")
    print(f"Wrote {len(bars)} curated bars/nightclubs to {BARS_PATH}")


if __name__ == "__main__":
    main()
