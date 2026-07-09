# Results — Atlanta multifamily rent prediction

## Summary

We ship three model variants. All metrics are 5-fold `GroupKFold` OOF on
`property_id` over 6,887 (property, unit_type) rows from the April 2026
Atlanta-MSA snapshot. Target is monthly rent in dollars.

| Variant | Deployment regime | MAE | MAPE | MedAPE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|
| **Primary** | Repricing on units with rent history | **$76.67** | **3.51%** | **1.77%** | $270.46 | 0.869 |
| **Cold-start** | New construction, acquisitions, comp pricing | $191.25 | 10.21% | 7.94% | $338.74 | 0.795 |
| **Graceful — full-hist eval** | Single model, val rows have hist | $89.36 | 4.25% | 2.72% | $267.28 | 0.872 |
| **Graceful — null-hist eval** | Single model, val rows cold | $185.22 | 9.91% | 7.82% | — | 0.794 |

## Recommendation

Use the **two-specialist setup (primary + cold-start)** in production and
route by feature availability:

- If `hist_rent_lag_1m IS NOT NULL` → call primary.
- Else → call cold-start.

This buys ~$13/unit MAE on repricing vs. the unified graceful model, which
is the higher-stakes regime. The graceful model is the right fallback if the
team needs a single endpoint and wants to avoid routing logic.

---

## Architecture

### Bases (level 0)

Four heterogeneous learners on log-rent:

1. `lgbm_l1` — LightGBM with L1 (MAE) objective.
2. `cat_q50` — CatBoost quantile regressor at α=0.50 (median).
3. `knn_geo` — k=15 KNN on the full numeric feature matrix.
4. `knn_lean` — k=15 KNN on geography + size only (diversity from
   ignoring engineered features).

All four were chosen for residual diversity: LightGBM and CatBoost give
sharp tree-based predictions on different splitting algorithms (histogram
vs symmetric/ordered boosting); the two KNN bases provide local smoothing
that the trees underweight. Pairwise Spearman correlation of base OOF
residuals ranges from 0.55 (lgb↔cat) down to 0.31 (lgb↔knn_lean).

### Meta (level 1) — Aug-Ridge

`Ridge(alpha=1.0, positive=False, fit_intercept=True)` fitted on the 4 base
log-predictions **plus three raw context features**: `log_sqft`, `beds`,
`year_built`. The trees compress these structural features into step
functions; a linear meta picks up the residual smooth slopes the bases
under-extract. Validated under GroupKFold OOF as in-sample fit: the gain is
−$4.17 to −$4.23 MAE vs. a preds-only Ridge meta.

`positive=False` is required because `beds` empirically gets a negative
coefficient (−0.018) — it's redundant with sqft and corrects rather than
adds. The base-prediction coefficients stay non-negative in practice.

### Splits

`GroupKFold(n_splits=5)` keyed on `property_id`. All units of a building
land in the same fold. This is the only leakage-safe CV strategy when
property-level features (e.g. neighborhood TE, geo z-scores, competition
features) are in play.

---

## Feature groups

The pipeline (see `feature_engineering.py`) builds ~207 features in seven
layers:

1. **Structural** — beds, baths, sqft, year_built, lot_size, n_units, vintage bucket
2. **Geography** — latitude/longitude, H3 cells at multiple resolutions, distance to 11 curated Atlanta landmarks (`eda/atlanta_landmarks.json`)
3. **Geo aggregates + z-scores** — submarket / ZIP / H3-cell means and z-scores of rent and rent-per-sqft
4. **Within-property heterogeneity** — unit's sqft rank inside its property, share of largest/smallest unit
5. **Competition** — neighbor PSF statistics at 0.5 / 1 / 2 mile radii (using a BallTree)
6. **Target encodings (OOF)** — Bayesian-smoothed mean encoding for submarket, ZIP, H3 cell, owner, submarket×vintage, address-block, hierarchical comparable TE
7. **Hist-rent lags** — `lag_1m`, `lag_3m`, `lag_12m`, `lag_24m`, `yoy` (computed at the (property, unit_type) granularity, with the target month 2026-03 excluded)

Text features from `property_name` and `street_address` (directional tokens,
suite/unit indicators, name↔submarket lexical match) live in a separate
section near the end of the feature builder.

---

## Ablations

### lag_1m alone (is the hist gain a backfill artifact?)

| Variant | OOF MAE | ΔMAE vs primary |
|---|---:|---:|
| Primary (all 5 hist features) | $76.67 | — |
| Primary minus `lag_1m` (4 features) | $103.60 | +$26.93 |
| No hist features (cold-start) | $191.25 | +$114.58 |

Conclusion: the gain is **structural**, not a backfill smoothing artifact.

- The unconditional Pearson correlation between `lag_1m` and the 2026-03
  target rent is 0.94 — Atlanta MFR rents are highly autocorrelated month-over-month.
- Dropping `lag_1m` still leaves $88 of the original $115 hist-feature gain
  intact (lag_3m / lag_12m / lag_24m / yoy contribute 76% of the benefit).
- `lag_1m` is responsible for $27 of the total $115 gain (24%).

We keep `lag_1m` in the primary model. It is legitimate signal, not a
duplicate of the target.

### Why the cold-start model is so much worse

Hist features encode an enormous fraction of the predictable variance. Once
removed, the model has to lean entirely on structural + geographic +
target-encoded features. KNN bases bear more weight in cold mode (knn_geo
gets 0.62 of the meta blend), and the meta intercept moves substantially
(−3.70 vs −0.97 for primary), reflecting the model's reduced confidence.

### Graceful degradation

Trained the LGB and CatBoost bases once on a training set where 30% of rows
had all 5 hist features set to NaN (deterministic per-fold mask, seed
20260501). Predicted on each val twice — once with real hist features
("full" eval, simulates the repricing regime), once with hist set to NaN
("null" eval, simulates cold-start). KNN bases were reused from the primary
state since their feature subsets don't include hist columns.

| Eval mode | Graceful MAE | Specialist MAE | Δ |
|---|---:|---:|---:|
| Full hist (repricing) | $89.36 | $76.67 (primary) | +$12.69 |
| Null hist (cold-start) | $185.22 | $191.25 (cold-start) | **−$6.03** |

Graceful actually **beats** the dedicated cold-start model by $6/unit. The
trees that saw real hist features for 70% of training rows learned the
non-hist split structure better than the no-hist trees did, and the
nullification training gave the meta a "what does my prediction look like
when hist is missing?" signal to calibrate against.

The cost is $12.69/unit on repricing — the random nullification softens the
tree splits on lag features. If you only need one model, this is the
right tradeoff to consider.

---

## Limitations

1. **Single-snapshot evaluation.** Train/val splits are spatial (by
   property), not temporal. The model has not been forward-tested against
   the May 2026 or June 2026 rent panels. Recommend forward-validating on
   the next monthly drop before deploying.
2. **Atlanta-only.** Geo features (landmarks, submarket TEs) are
   Atlanta-specific. Re-targeting to another MSA requires curating new
   landmark coordinates and refitting geo aggregates.
3. **Hist coverage = 86%** of training rows have `lag_1m` populated.
   The remaining 14% (mostly new construction, recent reposition) will
   route to the cold-start model in production, where MAE is 2.5× worse.
   This is the structural ceiling of the cold-start regime, not a bug.
4. **No temporal cross-validation in the OOF.** Our GroupKFold splits by
   property, not by time. The hist features could be exploiting subtle
   leakage if rent inflation correlates with property identity in ways the
   group split doesn't catch. The `lag_1m` ablation rules out the simplest
   form (target backfill), but a time-aware CV would be more conclusive.

---

## Reproducibility

- Random seed: `RANDOM_STATE = 42` in `config.py`
- Hyperparameters: `artifacts/best_params.json` (LightGBM) and
  `artifacts/best_catboost_params.json` (CatBoost), from 50-trial Optuna
  studies (~45 min each on a single machine).
- Source data drops: `artifacts/042026-*.parquet` tracked in this repo.
- Metrics JSONs: one per variant in `artifacts/metrics{,.coldstart,.graceful,.no_lag1m,.with_lag1m}.json`.

To rebuild from clone:

```bash
uv sync                           # or `pip install -e ".[dev]"`
uv run prime-mfr train --model primary
# -> artifacts/metrics.json + artifacts/oof_predictions.parquet
```

The `train` subcommand runs prep + all 5 folds × 4 bases + meta end-to-end. See the README's Quick Start for lower-level control (`prime-mfr-pipeline prep|foldprep|train|meta`).
