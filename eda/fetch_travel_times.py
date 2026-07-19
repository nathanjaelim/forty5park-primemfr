"""Fetch drive-time (minutes) from every h3_res6 cell in the Atlanta MSA to
each of the 4 curated landmarks (Buckhead, Midtown, Downtown, ATL airport),
using OSRM's public routing demo server. Writes the curated
eda/travel_times.json consumed by add_travel_time_features() in
src/prime_mfr/features/engineering.py.

MUST be run in an environment with BOTH live network access AND a working
`h3` install -- this sandbox has neither (no outbound network; h3 exists in
.venv but as a macOS-compiled extension that doesn't load on this Linux
sandbox). Run this from your local machine where stacked_cv.py already
runs successfully (h3 is a dependency there, so it should already work).

Usage:
    uv run python eda/fetch_travel_times.py

Why h3_res6 cells instead of per-property coordinates: there are
thousands of properties, and re-querying a routing API per property
wouldn't generalize to new/unseen listings without hitting the API again
at inference time. Precomputing drive time per h3_res6 cell (a few
thousand cells covering the whole MSA, not thousands of properties) lets
add_travel_time_features() do a zero-network lookup at feature-engineering
time -- any property's h3_res6 (already computed elsewhere in the
pipeline) is the join key.

Why OSRM's public demo server: free, no API key, and its /table endpoint
computes a many-to-many time matrix in one HTTP call (so a batch of ~100
cell centroids against 4 landmarks is a single request). It is a shared
public server not meant for heavy production load -- this script rate-
limits itself (SLEEP_BETWEEN_BATCHES) and this should only need to run
once (results are cached to eda/travel_times.json, and the script resumes
from a partial run via --resume). If you have a Google Maps API key and
want traffic-aware times instead, swap fetch_batch() for a Distance
Matrix API call -- OSRM's default driving profile doesn't model
real-time traffic.

Schema written to eda/travel_times.json (a dict keyed by h3_res6 cell ID
-- add_travel_time_features() does `travel_times[h3_res6]["buckhead_min"]`,
etc., with graceful degradation to NaN for any h3 cell not in the table,
e.g. a property just outside the fetched bbox):

{
  "862b939afffffff": {
    "buckhead_min": 12.4,
    "midtown_min": 8.1,
    "downtown_min": 14.7,
    "atl_airport_min": 22.3
  },
  ...
}
"""
import argparse
import json
import time
from pathlib import Path

import h3
import requests

OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving/{coords}"

# Same generous bbox used for the restaurants/bars POI fetches -- covers
# the 29-county Atlanta MSA (lat 33.00-34.65, lon -85.45--83.45).
BBOX_LAT = (33.00, 34.65)
BBOX_LON = (-85.45, -83.45)
H3_RES = 6

LANDMARKS_JSON = Path("eda/atlanta_landmarks.json")
OUT_PATH = Path("eda/travel_times.json")

BATCH_SIZE = 100  # cell centroids per OSRM /table request
SLEEP_BETWEEN_BATCHES = 1.0  # seconds -- be polite to the shared public server


def load_landmarks() -> dict[str, tuple[float, float]]:
    raw = json.loads(LANDMARKS_JSON.read_text())
    return {
        key: (v["latitude"], v["longitude"])
        for key, v in raw["landmarks"].items()
    }


def enumerate_h3_cells() -> list[str]:
    """All h3_res6 cells whose centroid falls within the MSA bbox."""
    poly = h3.LatLngPoly(
        [
            (BBOX_LAT[0], BBOX_LON[0]),
            (BBOX_LAT[0], BBOX_LON[1]),
            (BBOX_LAT[1], BBOX_LON[1]),
            (BBOX_LAT[1], BBOX_LON[0]),
        ]
    )
    return sorted(h3.polygon_to_cells(poly, H3_RES))


def fetch_batch(cell_centroids: list[tuple[str, float, float]], landmarks: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    """One OSRM /table call: N cell centroids (sources) x 4 landmarks
    (destinations). Returns {cell_id: {landmark_key + "_min": minutes}}."""
    landmark_keys = list(landmarks.keys())
    coords = [f"{lon},{lat}" for _, lat, lon in cell_centroids]
    coords += [f"{lon},{lat}" for lat, lon in landmarks.values()]
    n_sources = len(cell_centroids)
    n_total = len(coords)
    sources = ";".join(str(i) for i in range(n_sources))
    destinations = ";".join(str(i) for i in range(n_sources, n_total))

    url = OSRM_TABLE_URL.format(coords=";".join(coords))
    resp = requests.get(url, params={"sources": sources, "destinations": destinations}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    durations = data["durations"]  # seconds, shape (n_sources, n_landmarks)
    out = {}
    for i, (cell_id, _lat, _lon) in enumerate(cell_centroids):
        row = durations[i]
        out[cell_id] = {
            f"{key}_min": (round(row[j] / 60.0, 2) if row[j] is not None else None)
            for j, key in enumerate(landmark_keys)
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells already present in an existing eda/travel_times.json.",
    )
    args = parser.parse_args()

    landmarks = load_landmarks()
    cells = enumerate_h3_cells()
    print(f"{len(cells)} h3_res{H3_RES} cells covering the MSA bbox")

    results: dict[str, dict[str, float]] = {}
    if args.resume and OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
        print(f"Resuming: {len(results)} cells already cached")

    todo = [c for c in cells if c not in results]
    centroids = [(c, *h3.cell_to_latlng(c)) for c in todo]

    for i in range(0, len(centroids), BATCH_SIZE):
        batch = centroids[i : i + BATCH_SIZE]
        print(f"Batch {i // BATCH_SIZE + 1}/{-(-len(centroids) // BATCH_SIZE)} ({len(batch)} cells)...")
        batch_result = fetch_batch(batch, landmarks)
        results.update(batch_result)
        OUT_PATH.write_text(json.dumps(results, indent=2))  # checkpoint after every batch
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Wrote {len(results)} h3_res{H3_RES} cell travel times to {OUT_PATH}")


if __name__ == "__main__":
    main()
