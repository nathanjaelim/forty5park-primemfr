"""
train.py
========
Tune and evaluate a LightGBM regressor for Atlanta multifamily rent
prediction with 5-fold GroupKFold cross-validation by property_id.

Usage
-----
    python train.py                # full run with Optuna tuning
    python train.py --n-trials 5   # quick smoke test (5 Optuna trials)
    python train.py --no-tune      # skip Optuna; use sane defaults

Outputs (under ./artifacts):
    best_params.json
    metrics.json
    oof_predictions.parquet
    feature_importance.csv
    run.log

Reported metrics (computed on raw rent scale):
    MAE        - Mean Absolute Error  ($/month)
    MAPE       - Mean Absolute Percentage Error (%)
    MedianAPE  - Median APE (%)        - robust counterpart of MAPE
    RMSE       - Root Mean Squared Error ($/month)
    MedianAE   - Median Absolute Error ($/month)
    R^2        - Coefficient of determination
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import prime_mfr.config as config
import prime_mfr.data_processing as dp
import prime_mfr.features as fe
import prime_mfr.models as md

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rent_lgbm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Target transforms (raw rent <-> log target)
# ---------------------------------------------------------------------------
#
# When config.USE_PSF_TARGET is True we train on log1p(rent / sqft) and convert
# back via expm1(pred) * sqft. Predicting per-square-foot rent has tighter
# variance than raw rent (sqft is the dominant scale factor for $/month), so
# every base learns a more concentrated target. Multiplying by sqft at predict
# time recovers raw $/month for metrics and the meta-learner. When False, the
# target is the legacy log1p(rent).


def compute_log_target(df: pd.DataFrame, use_psf: bool | None = None) -> np.ndarray:
    """Build the log-target column from a dataframe holding rent + sqft."""
    if use_psf is None:
        use_psf = config.USE_PSF_TARGET
    rent = df[config.TARGET].astype(np.float64).to_numpy()
    if use_psf:
        sqft = df["sqft"].astype(np.float64).to_numpy()
        sqft = np.where(np.isfinite(sqft) & (sqft > 0.0), sqft, 1.0)
        return np.log1p(rent / sqft)
    return np.log1p(rent)


def back_transform_to_rent(
    log_pred: np.ndarray,
    sqft: np.ndarray | pd.Series | None = None,
    use_psf: bool | None = None,
) -> np.ndarray:
    """Invert compute_log_target. Returns raw $/month, clipped at $100."""
    if use_psf is None:
        use_psf = config.USE_PSF_TARGET
    log_pred = np.asarray(log_pred, dtype=np.float64)
    if use_psf:
        if sqft is None:
            raise ValueError("USE_PSF_TARGET=True but back_transform_to_rent called without sqft")
        sqft_arr = np.asarray(sqft, dtype=np.float64)
        sqft_arr = np.where(np.isfinite(sqft_arr) & (sqft_arr > 0.0), sqft_arr, 1.0)
        psf = np.expm1(log_pred)
        return np.clip(psf * sqft_arr, 100.0, None)
    return np.clip(np.expm1(log_pred), 100.0, None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    ape = abs_err / np.maximum(np.abs(y_true), 1e-9)

    mae = float(abs_err.mean())
    medae = float(np.median(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(ape.mean() * 100)
    medape = float(np.median(ape) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "MAE": mae,
        "MAPE": mape,
        "MedianAPE": medape,
        "RMSE": rmse,
        "MedianAE": medae,
        "R2": r2,
    }


# ---------------------------------------------------------------------------
# Feature matrix preparation per fold
# ---------------------------------------------------------------------------


def prepare_fold(
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    df: pd.DataFrame,
    target_log: str,
    target_raw: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Build the train / valid feature matrices for one fold:
        1. Slice the dataframe by index.
        2. Add OOF target encodings + k-NN features.
        3. Coerce categoricals.
        4. Resolve final column lists.
    """
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    target_for_te = target_log if config.LOG_TARGET else target_raw

    train_df, valid_df = fe.add_oof_features(
        train_df,
        valid_df,
        target_for_te=target_for_te,
        target_for_knn=target_raw,
    )

    # Categorical coercion (train + valid use the same dtype space).
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in train_df.columns]
    train_df = dp.coerce_categoricals(train_df, cat_cols)
    valid_df = dp.coerce_categoricals(valid_df, cat_cols)
    # Align categories so LightGBM doesn't see new ones in valid.
    for c in cat_cols:
        all_cats = pd.api.types.union_categoricals(
            [train_df[c], valid_df[c]], sort_categories=True
        ).categories
        train_df[c] = pd.Categorical(train_df[c], categories=all_cats)
        valid_df[c] = pd.Categorical(valid_df[c], categories=all_cats)

    numeric, categorical = fe.select_feature_columns(train_df)
    return train_df, valid_df, numeric, categorical


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def train_one_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
) -> tuple[lightgbm.Booster, np.ndarray, dict]:  # noqa: F821
    import lightgbm as lgb

    p = {
        **config.LGB_FIXED_PARAMS,
        **params,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
    }

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_va, label=y_va, categorical_feature=cat_cols, reference=train_set, free_raw_data=False
    )

    booster = lgb.train(
        p,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, pred_va, {"best_iteration": booster.best_iteration}


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------


def run_cv(
    df: pd.DataFrame,
    params: dict,
    logger: logging.Logger,
    num_boost_round: int = config.NUM_BOOST_ROUND,
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS,
    keep_models: bool = False,
    use_stacking: bool | None = None,
    base_specs: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run 5-fold GroupKFold CV with N heterogeneous base models per fold and a
    Ridge meta-learner blending their log-rent OOF predictions.

    Bases are configured via `base_specs` (defaults to config.BASE_SPECS):
        [{"name", "trainer", "params", "seed_offset"}, ...]
    where "trainer" is one of "lightgbm" / "catboost" / "knn"
    (see models.TRAINERS).

    When `use_stacking=False`, only the FIRST base spec is fit (used by Optuna
    tuning, which optimizes the LightGBM L1 base alone).

    Returns dict with keys:
        oof_pred           : primary (stack if on, else first-base) raw-scale OOF
        oof_used           : boolean mask of rows that received a prediction
        oof_raw            : {base_name: raw-scale OOF} for every fitted base
        oof_pred_stack     : stacked OOF (raw scale, or None if not stacking)
        per_fold           : list of dicts with per-base, per-fold metrics
        metrics            : aggregated metrics for the primary prediction
        base_metrics       : {base_name: aggregated metrics}
        stack_metrics      : aggregated stacked metrics (or None)
        meta_coefs         : {"weights": {base_name -> w}, "intercept", "alpha"}
        feature_importance : averaged across folds (LightGBM bases only)
        boosters           : LightGBM boosters per fold if keep_models else None
    """
    if use_stacking is None:
        use_stacking = config.USE_STACKING
    if base_specs is None:
        base_specs = config.BASE_SPECS

    if not use_stacking:
        # Tuning / unstacked baseline path: fit only the first spec.
        active_specs = [base_specs[0]]
    else:
        active_specs = base_specs

    target_raw = config.TARGET
    target_log = "log_rent"
    df = df.copy()
    df[target_log] = compute_log_target(df)

    groups = df[config.GROUP_KEY].astype(str).values
    gkf = GroupKFold(n_splits=config.N_FOLDS)

    base_names = [s["name"] for s in active_specs]
    oof_log = {k: np.full(len(df), np.nan, dtype=np.float64) for k in base_names}
    oof_raw = {k: np.full(len(df), np.nan, dtype=np.float64) for k in base_names}

    per_fold_metrics: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    boosters: list[Any] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        t0 = time.time()
        tr_df, va_df, num_cols, cat_cols = prepare_fold(
            tr_idx, va_idx, df, target_log=target_log, target_raw=target_raw
        )
        feat_cols = num_cols + cat_cols
        X_tr = tr_df[feat_cols]
        X_va = va_df[feat_cols]
        if config.LOG_TARGET:
            y_tr = tr_df[target_log].values
            y_va_train = va_df[target_log].values
        else:
            y_tr = tr_df[target_raw].values
            y_va_train = va_df[target_raw].values
        y_va_raw = va_df[target_raw].values
        va_sqft = va_df["sqft"].values if "sqft" in va_df.columns else None

        fold_record: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
        }

        for spec in active_specs:
            name = spec["name"]
            trainer = md.get_trainer(spec["trainer"])
            # Merge logic: `params` is the LightGBM-shaped param-search dict
            # (tuned hyperparameters like feature_fraction, num_leaves, etc.).
            # Only LightGBM bases accept those keys; CatBoost / KNN have their
            # own param namespaces, so don't leak LGBM keys to them.
            base_params = dict(spec.get("params", {}))
            if spec["trainer"] == "lightgbm":
                spec_params = {**params, **base_params}
            elif spec["trainer"] == "catboost":
                # Merge in tuned CatBoost params (if best_catboost_params.json
                # exists). Order: shared defaults <- tuned <- per-spec overrides.
                cb_tuned: dict = {}
                if config.BEST_CB_PARAMS_PATH.exists():
                    try:
                        cb_tuned = json.loads(config.BEST_CB_PARAMS_PATH.read_text())
                    except Exception:
                        cb_tuned = {}
                spec_params = {**config.CATBOOST_PARAMS, **cb_tuned, **base_params}
            elif spec["trainer"] == "knn":
                # Two parallel KNNs: knn_geo uses the full numeric matrix
                # (config.KNN_PARAMS); knn_lean uses a tighter subset
                # (config.KNN_LEAN_PARAMS, which carries `feature_subset`).
                knn_defaults = (
                    config.KNN_LEAN_PARAMS if spec["name"] == "knn_lean" else config.KNN_PARAMS
                )
                spec_params = {**knn_defaults, **base_params}
            else:
                spec_params = base_params
            seed = config.RANDOM_STATE + fold_idx + int(spec.get("seed_offset", 0))

            t_base = time.time()
            pred_va_log, info = trainer(
                X_tr,
                y_tr,
                X_va,
                y_va_train,
                num_cols=num_cols,
                cat_cols=cat_cols,
                params=spec_params,
                seed=seed,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            base_seconds = round(time.time() - t_base, 1)

            # All trainers predict on the configured log-target (raw log-rent,
            # or log-PSF when USE_PSF_TARGET=True). Convert back to raw $/month.
            log_pred = (
                pred_va_log if config.LOG_TARGET else np.log1p(np.clip(pred_va_log, 0.0, None))
            )
            raw_pred = back_transform_to_rent(log_pred, sqft=va_sqft)
            oof_log[name][va_idx] = log_pred
            oof_raw[name][va_idx] = raw_pred

            m = compute_metrics(y_va_raw, raw_pred)
            fold_record[name] = {
                **m,
                "best_iteration": int(info.get("best_iteration", 0)),
                "wall_seconds": base_seconds,
            }

            # LightGBM importance (only — CatBoost / NN don't share this contract).
            if spec["trainer"] == "lightgbm":
                imp = pd.DataFrame(
                    {
                        "feature": info["feature_names"],
                        f"gain_fold{fold_idx}_{name}": info["feature_importance_gain"],
                    }
                )
                importance_frames.append(imp.set_index("feature"))

        fold_record["wall_seconds"] = round(time.time() - t0, 1)
        per_fold_metrics.append(fold_record)

        # Per-fold log line: one chunk per base.
        chunks = [f"  fold {fold_idx}/{config.N_FOLDS}"]
        for name in base_names:
            r = fold_record[name]
            chunks.append(
                f"{name}: iter={r['best_iteration']:>4d} MAE=${r['MAE']:.2f} "
                f"MAPE={r['MAPE']:.2f}% R2={r['R2']:.3f} ({r['wall_seconds']}s)"
            )
        chunks.append(f"total {fold_record['wall_seconds']}s")
        logger.info(" | ".join(chunks))

    # ---------- Aggregate per-base OOF metrics ----------
    used = ~np.isnan(oof_raw[base_names[0]])
    y_true_raw = df[target_raw].values
    base_metrics = {k: compute_metrics(y_true_raw[used], oof_raw[k][used]) for k in base_names}

    # ---------- Fit stacking meta-learner on log-rent scale ----------
    stack_metrics = None
    meta_coefs: dict | None = None
    oof_stack: np.ndarray | None = None
    if use_stacking and len(base_names) >= 2:
        from sklearn.linear_model import Ridge

        y_true_log = df[target_log].values
        X_meta = np.column_stack([oof_log[k][used] for k in base_names])
        y_meta = y_true_log[used]
        meta = Ridge(alpha=1.0, positive=True, fit_intercept=True, random_state=config.RANDOM_STATE)
        meta.fit(X_meta, y_meta)
        meta_coefs = {
            "weights": dict(zip(base_names, [float(c) for c in meta.coef_])),
            "intercept": float(meta.intercept_),
            "alpha": 1.0,
        }
        log_pred_stack = meta.predict(X_meta)
        oof_stack = np.full(len(df), np.nan, dtype=np.float64)
        stack_sqft = df["sqft"].values[used] if "sqft" in df.columns else None
        oof_stack[used] = back_transform_to_rent(log_pred_stack, sqft=stack_sqft)
        stack_metrics = compute_metrics(y_true_raw[used], oof_stack[used])

    # ---------- Feature importance (averaged across folds; LGBM bases only) ----------
    if importance_frames:
        imp_df = pd.concat(importance_frames, axis=1).fillna(0.0)
        gain_cols = [c for c in imp_df.columns if c.startswith("gain_")]
        imp_df["gain_mean"] = imp_df[gain_cols].mean(axis=1)
        imp_df = imp_df.sort_values("gain_mean", ascending=False).reset_index()
    else:
        imp_df = pd.DataFrame()

    primary_oof = oof_stack if (use_stacking and oof_stack is not None) else oof_raw[base_names[0]]
    primary_metrics = stack_metrics if stack_metrics is not None else base_metrics[base_names[0]]

    return {
        "oof_pred": primary_oof,
        "oof_used": used,
        "oof_raw": oof_raw,  # dict[base_name -> ndarray]
        "oof_pred_stack": oof_stack,
        "per_fold": per_fold_metrics,
        "metrics": primary_metrics,
        "base_metrics": base_metrics,
        "stack_metrics": stack_metrics,
        "meta_coefs": meta_coefs,
        "feature_importance": imp_df,
        "boosters": boosters if keep_models else None,
    }


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def tune_with_optuna(
    df: pd.DataFrame,
    n_trials: int,
    logger: logging.Logger,
    storage_path: Path | None = None,
    study_name: str = "rent_lgbm_tuning",
    n_tune_folds: int = 3,
) -> dict:
    """
    Run Optuna TPE on the LightGBM search space.

    During tuning we use `n_tune_folds` (default 3) GroupKFold splits and a
    smaller boost-round budget, which is industry standard practice for
    tabular regression: it cuts wall time roughly in half versus tuning on
    all 5 folds, and the relative ranking of trials is preserved well enough
    to pick a competitive configuration. The final 5-fold CV is run with the
    best params on the full budget.

    If `storage_path` is provided, the study is persisted to SQLite so trials
    accumulate across separate invocations.
    """
    import optuna
    from optuna.samplers import TPESampler

    # During tuning, run with a tighter early-stopping budget for speed.
    # Tighter than final since tuning only needs to rank trials, not fit
    # the absolute best-iteration model.
    tune_num_boost_round = 800
    tune_early_stopping = 40

    # Subset CV folds for tuning speed.
    df_tune = df

    def objective(trial: optuna.Trial) -> float:
        params = config.lgb_search_space(trial)
        original_n_folds = config.N_FOLDS
        config.N_FOLDS = n_tune_folds
        try:
            result = run_cv(
                df_tune,
                params,
                logger=_NULL_LOGGER,
                num_boost_round=tune_num_boost_round,
                early_stopping_rounds=tune_early_stopping,
                keep_models=False,
                # Tune the L1 base only (it's the search space); stacking adds
                # cost and the meta-weights re-equilibrate after final-fit.
                use_stacking=False,
            )
        finally:
            config.N_FOLDS = original_n_folds
        return result["metrics"]["MAE"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # First, count any prior trials to seed the sampler distinctly per batch.
    prior_trials: list[dict] = []
    if storage_path is not None and Path(storage_path).exists():
        try:
            text = Path(storage_path).read_text()
            if text.strip():
                snapshot = json.loads(text)
                prior_trials = snapshot.get("trials", [])
        except Exception as e:
            logger.warning(f"Could not parse {storage_path}: {e}")

    # Vary the seed by the number of prior trials so successive batches don't
    # redraw the same random startup sequence.
    sampler = TPESampler(
        seed=config.RANDOM_STATE + len(prior_trials),
        n_startup_trials=10,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
    )

    # Replay completed trials via add_trial (does NOT re-run the objective).
    # This is a portable alternative to Optuna's SQLite/Journal storage, which
    # does not work on fuse-mounted volumes that lack POSIX file locking.
    for trial in prior_trials:
        try:
            _add_completed_trial(study, trial["params"], trial["value"])
        except Exception as e:
            logger.warning(f"Skipping prior trial during replay: {e}")

    n_existing = len(study.trials)
    if n_existing:
        logger.info(
            f"Loaded existing study '{study_name}' with {n_existing} prior trials "
            f"(best MAE so far=${study.best_value:.2f})."
        )
    logger.info(
        f"Running {n_trials} new Optuna trials " f"(objective: {n_tune_folds}-fold OOF MAE)..."
    )
    t0 = time.time()

    def _cb(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        logger.info(
            f"  trial {trial.number + 1} | "
            f"MAE=${trial.value:.2f} | best so far=${study.best_value:.2f}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)
    logger.info(
        f"Optuna batch done in {time.time() - t0:.0f}s. "
        f"Total trials in study: {len(study.trials)}. "
        f"Best MAE=${study.best_value:.2f}"
    )

    # Snapshot all completed trials to JSON for resumption next call.
    if storage_path is not None:
        snapshot = {
            "study_name": study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [
                {"params": t.params, "value": float(t.value)}
                for t in study.trials
                if t.value is not None and t.state.name == "COMPLETE"
            ],
        }
        Path(storage_path).write_text(json.dumps(snapshot, indent=2))

    return study.best_params


def _add_completed_trial(study, params: dict, value: float) -> None:
    """Insert a previously-completed trial into the study without re-running."""
    import optuna

    # Build a dummy trial through the search space to extract distributions.
    # We use a FixedTrial to capture distribution metadata.
    fixed = optuna.trial.FixedTrial(params)
    config.lgb_search_space(fixed)
    distributions = {
        k: dist for k, dist in zip(params.keys(), [fixed._distributions[k] for k in params.keys()])
    }
    trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
    study.add_trial(trial)


# Silent logger used for inner CV loops during tuning (we only want trial-level
# log lines, not per-fold log lines for every Optuna trial).
_NULL_LOGGER = logging.getLogger("rent_lgbm_silent")
_NULL_LOGGER.addHandler(logging.NullHandler())
_NULL_LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials",
        type=int,
        default=config.N_OPTUNA_TRIALS,
        help="Number of NEW Optuna trials to run this invocation (0 to skip).",
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip Optuna; use sane default LightGBM params."
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Run tuning trials and exit (skip final 5-fold CV).",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip tuning; load best params from the persistent study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=str(config.ARTIFACTS_DIR / "optuna_study.json"),
        help="Path to JSON snapshot of the Optuna study (resumable across calls).",
    )
    parser.add_argument("--study-name", type=str, default="rent_lgbm_tuning")
    parser.add_argument(
        "--tune-folds",
        type=int,
        default=3,
        help="Number of folds used during tuning (final still uses N_FOLDS).",
    )
    args = parser.parse_args()

    logger = setup_logging(config.RUN_LOG_PATH)

    logger.info("=" * 70)
    logger.info("Atlanta Multifamily Rent Prediction - LightGBM 5-fold CV")
    logger.info("=" * 70)

    # ---------------- Load + preprocess ----------------
    t0 = time.time()
    df = dp.load_clean()
    df = fe.add_static_features(df)
    logger.info(
        f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} cols " f"({time.time() - t0:.1f}s)"
    )
    logger.info(f"Unique properties: {df[config.GROUP_KEY].nunique():,}")
    logger.info(
        f"Rent stats: mean=${df[config.TARGET].mean():.2f}  "
        f"median=${df[config.TARGET].median():.2f}  "
        f"min=${df[config.TARGET].min():.2f}  "
        f"max=${df[config.TARGET].max():.2f}"
    )
    logger.info(f"Log target: {config.LOG_TARGET}")
    logger.info(f"Folds: {config.N_FOLDS} (GroupKFold by {config.GROUP_KEY})")

    # ---------------- Tune ----------------
    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    default_params = {
        "learning_rate": 0.04,
        "num_leaves": 64,
        "max_depth": 8,
        "min_child_samples": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_split_gain": 0.0,
    }

    if args.final_only:
        # Load best params from JSON snapshot.
        try:
            snapshot = json.loads(storage_path.read_text())
            best_params = snapshot["best_params"]
            logger.info(
                f"Loaded best params from snapshot ({len(snapshot.get('trials', []))} trials). "
                f"Best tuning MAE=${snapshot['best_value']:.2f}."
            )
        except Exception as e:
            logger.warning(f"Could not load study ({e}); falling back to defaults.")
            best_params = default_params
    elif args.no_tune or args.n_trials == 0:
        logger.info("Skipping Optuna tuning. Using sane default params.")
        best_params = default_params
    else:
        best_params = tune_with_optuna(
            df,
            n_trials=args.n_trials,
            logger=logger,
            storage_path=storage_path,
            study_name=args.study_name,
            n_tune_folds=args.tune_folds,
        )

    if args.tune_only:
        logger.info("--tune-only set; exiting after tuning.")
        return

    logger.info("Best LightGBM params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    # ---------------- Final 5-fold CV with best params ----------------
    logger.info("-" * 70)
    logger.info(
        f"Final 5-fold CV with tuned params (stacking={'on' if config.USE_STACKING else 'off'})..."
    )
    result = run_cv(df, best_params, logger=logger, keep_models=False)

    metrics = result["metrics"]
    base_metrics = result["base_metrics"]
    stack_metrics = result["stack_metrics"]
    meta_coefs = result["meta_coefs"]

    def _log_metrics_block(title: str, m: dict) -> None:
        logger.info(title)
        logger.info(f"  MAE       : ${m['MAE']:.2f}")
        logger.info(f"  MAPE      : {m['MAPE']:.2f}%")
        logger.info(f"  MedianAPE : {m['MedianAPE']:.2f}%")
        logger.info(f"  RMSE      : ${m['RMSE']:.2f}")
        logger.info(f"  MedianAE  : ${m['MedianAE']:.2f}")
        logger.info(f"  R^2       : {m['R2']:.4f}")

    logger.info("-" * 70)
    logger.info("FINAL 5-FOLD GROUPKFOLD CV METRICS (raw rent scale)")
    logger.info("-" * 70)
    for name, m in base_metrics.items():
        _log_metrics_block(f"[Base {name}]", m)
    if stack_metrics is not None:
        _log_metrics_block("[Stacked (Ridge meta on log-rent)]", stack_metrics)
        logger.info(
            f"  Meta weights: {meta_coefs['weights']}  intercept={meta_coefs['intercept']:.4f}"
        )

    logger.info("-" * 70)
    logger.info("Per-fold breakdown:")
    for m in result["per_fold"]:
        chunks = [f"  fold {m['fold']} | n_va={m['n_valid']:>4d}"]
        for name in base_metrics.keys():
            r = m.get(name)
            if not r:
                continue
            chunks.append(f"{name} MAE=${r['MAE']:.2f} R2={r['R2']:.3f} iter={r['best_iteration']}")
        logger.info(" | ".join(chunks))

    # ---------------- Save artifacts ----------------
    config.BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    config.METRICS_PATH.write_text(
        json.dumps(
            {
                "primary": metrics,
                "base_metrics": base_metrics,
                "stack_metrics": stack_metrics,
                "meta_coefs": meta_coefs,
                "per_fold": result["per_fold"],
                "config": {
                    "n_folds": config.N_FOLDS,
                    "log_target": config.LOG_TARGET,
                    "group_key": config.GROUP_KEY,
                    "n_optuna_trials": (0 if args.no_tune else args.n_trials),
                    "use_stacking": config.USE_STACKING,
                    "quantile_alpha": config.QUANTILE_ALPHA,
                },
            },
            indent=2,
            default=float,
        )
    )

    used = result["oof_used"]
    oof_cols = {
        "row_id": np.arange(len(df))[used],
        "property_id": df[config.GROUP_KEY].values[used],
        "rent_actual": df[config.TARGET].values[used],
        "rent_pred": result["oof_pred"][used],  # primary (stack if on, else first base)
    }
    for name, arr in result["oof_raw"].items():
        oof_cols[f"rent_pred_{name}"] = arr[used]
    if result["oof_pred_stack"] is not None:
        oof_cols["rent_pred_stack"] = result["oof_pred_stack"][used]
    oof_df = pd.DataFrame(oof_cols)
    oof_df.to_parquet(config.OOF_PRED_PATH, index=False)

    imp_df = result["feature_importance"]
    imp_df.to_csv(config.FEATURE_IMP_PATH, index=False)

    logger.info("-" * 70)
    logger.info("Artifacts saved to ./artifacts/:")
    logger.info(f"  best_params.json     -> {config.BEST_PARAMS_PATH.name}")
    logger.info(f"  metrics.json         -> {config.METRICS_PATH.name}")
    logger.info(f"  oof_predictions.parquet  -> {config.OOF_PRED_PATH.name}")
    logger.info(f"  feature_importance.csv   -> {config.FEATURE_IMP_PATH.name}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
