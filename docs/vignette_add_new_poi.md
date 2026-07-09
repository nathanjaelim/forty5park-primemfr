# Vignette — Adding a New Point of Interest as a Landmark

> **Scenario:** you want to test whether "distance to the nearest MARTA rail station" improves the model, just like `dist_atl_airport_km` (Hartsfield-Jackson) does today.
> **Time:** ~15 minutes of edits + ~3 minutes of retraining.
> **Prerequisite:** you can run the pipeline locally (see [`tutorial_local_setup.md`](tutorial_local_setup.md)).

This vignette is a concrete, fact-checked walk-through. Every file path, function name, and JSON schema below has been verified against the actual code at `src/prime_mfr/features/engineering.py` and `configs/features/numeric.yaml`.

---

## How landmarks work in one paragraph

At training time, the feature engineering step calls `add_landmark_distances(df)` which:

1. Reads `eda/atlanta_landmarks.json` and builds a Python dict of `{key: (lat, lon)}` pairs from the `landmarks` block.
2. Iterates over the list `config.LANDMARKS` (in `src/prime_mfr/config.py`) — **not** over every entry in the JSON — and for each active key computes the Haversine great-circle distance in kilometers from every property to that landmark, storing it as `dist_<key>_km`.
3. Computes `dist_min_landmark_km` as the row-wise minimum across all active-landmark distances.

That means adding a new POI requires editing **three files**: the JSON (add the coordinates), `config.py` (activate the key), and `configs/features/numeric.yaml` (tell the model to consume the new column).

There is no auto-detection. Adding to the JSON alone is silently ignored; adding to `config.LANDMARKS` without adding to the JSON raises `KeyError` at training time.

---

## Worked example — "MARTA Five Points" (Downtown Atlanta rail interchange)

Five Points is the central MARTA transfer station in downtown Atlanta (33.7539° N, -84.3919° W). It's a good candidate for a POI feature: transit-adjacent properties often price differently than transit-isolated ones, especially inside I-285.

### Step 1 — Add the landmark to the JSON

Open `eda/atlanta_landmarks.json` and add an entry under the `"landmarks"` block. The loader only reads `latitude` and `longitude`; the other fields (`name`, `type`, `source`, `notes`) are optional human-readable metadata but strongly recommended for auditability.

```json
{
  "_meta": { ... unchanged ... },
  "landmarks": {
    "buckhead":     { ... unchanged ... },
    "midtown":      { ... unchanged ... },
    "downtown":     { ... unchanged ... },
    "atl_airport":  { ... unchanged ... },

    "marta_five_points": {
      "name": "MARTA Five Points station",
      "type": "transit hub / rail interchange",
      "latitude": 33.7539,
      "longitude": -84.3919,
      "source": "https://en.wikipedia.org/wiki/Five_Points_station",
      "notes": "Central Atlanta MARTA transfer point (Red/Gold/Blue/Green lines). Downtown."
    }
  }
}
```

**Key-naming rules** (learned by adversarial testing):

- Must be a valid Python identifier (lowercase, underscores, no spaces or hyphens) — the key becomes part of a column name, `dist_<key>_km`.
- Case matters. `MARTA_Five_Points` and `marta_five_points` produce different columns.
- No leading digits.
- Must be unique within the `landmarks` block.

### Step 2 — Activate the landmark in `config.py`

Open `src/prime_mfr/config.py` and find the `LANDMARKS` list. Add your new key:

```python
# Atlanta landmark keys (must exist in atlanta_landmarks.json).
LANDMARKS: list[str] = [
    "buckhead",
    "midtown",
    "downtown",
    "atl_airport",
    "marta_five_points",  # <-- NEW
]
```

**Adversarial check:** the key you add here must exactly match the JSON key you added in Step 1. A typo (`marta_five_point` vs `marta_five_points`) will raise `KeyError: 'marta_five_point'` when training starts. If that happens, fix the mismatch in either file.

### Step 3 — Tell the model config to consume the new feature

The training pipeline reads `configs/features/numeric.yaml` to build the feature list. Add `dist_marta_five_points_km` to the `geographic` group:

```yaml
groups:
  # ... other groups ...
  geographic:
    - latitude
    - longitude
    - dist_buckhead_km
    - dist_midtown_km
    - dist_downtown_km
    - dist_atl_airport_km
    - dist_marta_five_points_km   # <-- NEW
    - dist_min_landmark_km
```

**Adversarial check:** the column name here must EXACTLY match the pattern `dist_<key>_km` where `<key>` is your JSON/config key. `dist_marta_five_points_km`, not `dist_marta_5_points_km` or `dist_martaFivePoints_km`. If the YAML says a name that the feature pipeline doesn't produce, the model config validator will warn on load but training won't include the feature.

### Step 4 — Add a test

Open `tests/test_feature_geographic.py` and add a test that verifies your new landmark is picked up correctly. Follow the existing pattern:

```python
def test_marta_five_points_distance_is_zero_at_station_coords():
    """A property AT the Five Points station coords → dist_marta_five_points_km ≈ 0."""
    df = pd.DataFrame({
        "latitude":  [33.7539],
        "longitude": [-84.3919],
        "rent":      [2000.0],
    })
    out = add_landmark_distances(df)
    assert "dist_marta_five_points_km" in out.columns
    assert out["dist_marta_five_points_km"].iloc[0] < 0.1
```

**Also update** `test_add_landmark_distances_adds_expected_columns` (around line 81) to include your new column in the expected set:

```python
def test_add_landmark_distances_adds_expected_columns():
    df = _atlanta_df(10)
    out = add_landmark_distances(df)
    for col in (
        "dist_buckhead_km",
        "dist_midtown_km",
        "dist_downtown_km",
        "dist_atl_airport_km",
        "dist_marta_five_points_km",   # <-- NEW
        "dist_min_landmark_km",
    ):
        assert col in out.columns, f"missing column {col}"
```

Run the tests to confirm nothing broke:

```bash
uv run pytest tests/test_feature_geographic.py -v
```

Expected: your new test passes. All other tests still pass.

### Step 5 — Retrain and measure the impact

Clean the previous training run's outputs, then retrain:

```bash
uv run prime-mfr clean --yes
uv run prime-mfr train --model primary
```

At the end you'll get a fresh `artifacts/metrics.json`. Compare its MAE to the baseline **$76.67**.

**Interpretation guide:**

| Δ MAE | What it means | Recommendation |
|---|---|---|
| < −$1 (better) | The POI adds real signal | Keep it. Consider adding related POIs (e.g., other MARTA stations) |
| −$1 to +$1 | Within noise floor | Marginal. Keep only if you have a strong prior it should matter |
| > +$1 (worse) | The POI hurts | Revert (see Step 6). Cost is likely overfitting on a spurious pattern |

The noise floor on this pipeline is about $1 of MAE — LightGBM/CatBoost aren't perfectly deterministic across runs. Don't over-interpret sub-$1 movements.

### Step 6 — If it doesn't help, revert cleanly

If the retrain shows the new landmark degrades performance:

1. Remove the entry from `configs/features/numeric.yaml` (Step 3)
2. Remove the key from `config.LANDMARKS` in `config.py` (Step 2)
3. Remove the tests you added in Step 4
4. Optionally leave the JSON entry (Step 1) — future you can re-enable without re-looking-up coordinates. If keeping it, add a `notes` field explaining "tested and reverted on YYYY-MM-DD, +$X MAE regression"

Re-run tests + train to confirm you're back to baseline.

---

## What can go wrong (fact-checked failure modes)

Each of these is a real error you'll see if you misstep:

### `KeyError: 'my_new_key'` during training prep

**Cause:** you added the key to `config.LANDMARKS` but forgot to add the entry to `atlanta_landmarks.json`.
**Fix:** add the JSON entry (Step 1). The loader does `landmarks[key]` — a missing key raises KeyError immediately.

### The column `dist_my_new_key_km` appears in the DataFrame but the model doesn't seem to use it

**Cause:** you skipped Step 3 (adding to `configs/features/numeric.yaml`). The static-feature pipeline produced the column, but the model config didn't select it.
**Fix:** add `dist_my_new_key_km` to the `geographic` group in the YAML.
**How to verify:** run `uv run prime-mfr list` — the "Features" column count for `primary` should tick up by 1.

### `Contract validation: FAIL — MISSING_COLUMN` in the validator

**Cause:** you added the column to the YAML but the feature pipeline didn't produce it (Step 1 or 2 skipped).
**Fix:** work backwards. Verify `config.LANDMARKS` contains the key AND `atlanta_landmarks.json` has an entry for it. Rerun `uv run pytest tests/test_feature_contract.py`.

### Tests still pass, retrain runs, but the MAE didn't move at all

**Cause:** the model config change didn't take effect. Possible reasons:
- You edited `configs/features/numeric.yaml` but the running Python process has a cached copy. Fix: exit any Python REPL/notebook and re-run `uv run prime-mfr train`.
- You have a `.pyc` cached copy of the config module. Fix: `uv run prime-mfr clean --yes` and retry (this deletes `__pycache__`).

### Landmark coordinates are wrong (property at "landmark" doesn't get dist ≈ 0)

**Cause:** you typo'd the latitude or longitude, or you swapped them (JSON expects `latitude` first, `longitude` second).
**Fix:** double-check the source (Wikipedia infobox coords are given as `DMS` and `decimal`; use the decimal). Confirm the sign — Atlanta is at negative longitude (west of the Greenwich meridian): -84.something.

---

## Adding several POIs at once — a batch pattern

If you want to add all MARTA rail stations (~35 of them) as landmarks, don't hand-copy each one. The right pattern:

1. Cache the station list once (via OpenStreetMap Overpass API or a curated CSV):

   ```python
   # scripts/fetch_marta_stations.py
   import json
   import requests

   query = """
   [out:json];
   node["railway"="station"]["operator"~"MARTA"];
   out;
   """
   resp = requests.post("https://overpass-api.de/api/interpreter", data=query)
   stations = {
       f"marta_{node['tags']['name'].lower().replace(' ', '_').replace('-', '_')}": {
           "name": node["tags"]["name"],
           "type": "MARTA rail station",
           "latitude": node["lat"],
           "longitude": node["lon"],
           "source": "OpenStreetMap Overpass API",
       }
       for node in resp.json()["elements"]
   }
   print(json.dumps(stations, indent=2))
   ```

2. Merge the output into `atlanta_landmarks.json`.

3. In `config.py`, add all the new keys to `LANDMARKS`.

4. In `configs/features/numeric.yaml`, add each `dist_<key>_km` to the `geographic` group.

**But before you go this route: consider that 35 landmark-distance columns is a lot.** You'll pay in training time (LightGBM has more features to consider at each split) and risk overfitting. A better design is often ONE aggregate feature `dist_nearest_marta_station_km` (the min across all stations) — which you'd implement as a new function alongside `add_landmark_distances`. That's covered in the "Adding a new POI feature" section of [`docs/geospatial_features.md`](geospatial_features.md).

---

## Adversarial checklist (before you claim done)

Run through these before committing:

- [ ] `uv run pytest` — all tests pass, including your new one
- [ ] `uv run prime-mfr list` — `primary` shows one more feature than before
- [ ] `python3 -c "from prime_mfr import config; assert 'marta_five_points' in config.LANDMARKS"` — no assertion error
- [ ] `python3 -c "import prime_mfr.features as fe, pandas as pd; import json; d = pd.DataFrame({'latitude':[33.7539],'longitude':[-84.3919],'rent':[2000]}); out = fe.add_landmark_distances(d); assert 'dist_marta_five_points_km' in out.columns; assert out['dist_marta_five_points_km'].iloc[0] < 0.1"` — no assertion error
- [ ] `uv run prime-mfr train --model primary` — completes without error and reports a stacked MAE within a few dollars of baseline
- [ ] `artifacts/metrics.json` MAE delta documented (either kept or reverted with reason)

If all six checkpoints pass, ship it. If any fail, work backwards from that failure — every one has a well-defined cause listed above.
