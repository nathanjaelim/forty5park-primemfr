"""Build the curated eda/parks.json consumed by add_park_distance() in
src/prime_mfr/features/engineering.py, from a cached Overpass export at
eda/research/parks.geojson (leisure=park polygons/multipolygons already
scoped to the Atlanta MSA).

Unlike the other eda/fetch_*.py scripts, this one only supports
--from-cache: parks are areas (Polygon/MultiPolygon), not points, so
there's no live-fetch path here yet (would need the same [bbox:...] query
style as eda/research/bars_nightclubs_overpass_query.txt with
["leisure"="park"] in place of ["amenity"=...]). eda/research/parks.geojson
was provided as a pre-existing cached export.

Each park polygon is reduced to a single representative point: its
centroid, computed via the shoelace formula (exterior ring only, ignoring
holes) and area-weighted across MultiPolygon parts. This is an
approximation -- for a large or oddly-shaped park (long and thin
especially), the centroid can sit meaningfully off from the nearest actual
park boundary. Verified against Piedmont Park: computed centroid
(33.7865787, -84.3733086) is close to its real-world center.

Usage:
    python eda/fetch_parks.py --from-cache

Schema written to eda/parks.json (a JSON list, NOT a dict --
add_park_distance() iterates `for p in parks: p["lat"], p["lon"]`):

[
  {
    "name": "Piedmont Park",
    "lat": 33.7865787,
    "lon": -84.3733086,
    "source": "..."
  },
  ...
]
"""
import argparse
import json
from pathlib import Path

GEOJSON_PATH = Path("eda/research/parks.geojson")
PARKS_PATH = Path("eda/parks.json")


def _ring_centroid_area(ring: list[list[float]]):
    """Shoelace-based centroid + signed area for a simple polygon ring.
    ring: list of [lon, lat]. Returns (cx, cy, abs_area) or None if
    degenerate."""
    n = len(ring)
    if n < 3:
        return None
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area2 *= 0.5
    if abs(area2) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    return cx / (6 * area2), cy / (6 * area2), abs(area2)


def _polygon_centroid(coords: list[list[list[float]]]):
    """GeoJSON Polygon coordinates -- uses the exterior ring only (index 0),
    ignoring any holes."""
    return _ring_centroid_area(coords[0])


def _feature_centroid(geom: dict):
    if geom["type"] == "Polygon":
        return _polygon_centroid(geom["coordinates"])
    if geom["type"] == "MultiPolygon":
        parts = [_polygon_centroid(poly) for poly in geom["coordinates"]]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        total_area = sum(p[2] for p in parts)
        if total_area == 0:
            cx = sum(p[0] for p in parts) / len(parts)
            cy = sum(p[1] for p in parts) / len(parts)
            return cx, cy, 0.0
        cx = sum(p[0] * p[2] for p in parts) / total_area
        cy = sum(p[1] * p[2] for p in parts) / total_area
        return cx, cy, total_area
    return None


def build_curated(geojson: dict, fetched_note: str) -> list[dict]:
    seen = set()
    parks = []
    for f in geojson["features"]:
        p = f["properties"]
        name = p.get("name")
        if not name:
            continue
        centroid = _feature_centroid(f["geometry"])
        if centroid is None:
            continue
        lon, lat, _area = centroid
        key = (name, round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        parks.append(
            {
                "name": name,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "source": fetched_note,
            }
        )
    parks.sort(key=lambda s: s["name"])
    return parks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache",
        action="store_true",
        required=True,
        help="Rebuild parks.json from the existing eda/research/parks.geojson "
        "cache (the only supported mode -- see module docstring).",
    )
    args = parser.parse_args()
    if not args.from_cache:
        raise SystemExit("Only --from-cache is supported; see module docstring.")

    if not GEOJSON_PATH.exists():
        raise SystemExit(f"{GEOJSON_PATH} not found.")
    geojson = json.loads(GEOJSON_PATH.read_text())
    note = (
        "OpenStreetMap Overpass API (leisure=park); polygon centroid "
        f"(shoelace formula) computed from cached {GEOJSON_PATH}"
    )

    parks = build_curated(geojson, note)
    PARKS_PATH.write_text(json.dumps(parks, indent=2) + "\n")
    print(f"Wrote {len(parks)} curated parks to {PARKS_PATH}")


if __name__ == "__main__":
    main()
