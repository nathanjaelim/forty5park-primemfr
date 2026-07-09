"""
models.py
=========
Single-fold trainer functions for the three base models in the stacking
ensemble. Each trainer takes the train/valid splits already prepared by
`train.prepare_fold` (DataFrames with the right column ordering, categoricals
coerced to pandas categorical dtype, log-rent target) and returns the
log-rent prediction for the validation fold plus a small info dict.

Trainers
--------
train_lightgbm_fold      LightGBM regressor with whichever objective is in
                         params (regression_l1, quantile, etc.)
train_catboost_fold      CatBoost regressor with whichever loss_function is
                         in params (Quantile:alpha=0.5 by default)
Each returns (pred_va_log: np.ndarray, info: dict) where info["best_iteration"]
is the iteration count at which validation peaked (or 0 for non-iterative models).

Reasoning
---------
Stacking gain comes from prediction *decorrelation*, not from any single
base being best. We pick three structurally different learners:

* LightGBM uses leaf-wise growth and gradient-based one-side sampling.
* CatBoost uses ordered boosting + ordered target statistics for cats,
  which give it a meaningfully different bias on small / structured data
  even with similar hyperparameters.
* KNN provides a non-parametric local-averaging signal — its errors live
  in different rows than tree errors, so the meta-learner can extract
  genuinely new information from blending them.
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import numpy as np
import pandas as pd

import prime_mfr.config as config

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def train_lightgbm_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    params: dict,
    seed: int,
    num_boost_round: int = config.NUM_BOOST_ROUND,
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS,
) -> tuple[np.ndarray, dict]:
    """LightGBM single-fold trainer (callers pass log-rent y)."""
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
    info = {
        "best_iteration": int(booster.best_iteration),
        "feature_importance_gain": booster.feature_importance(importance_type="gain"),
        "feature_importance_split": booster.feature_importance(importance_type="split"),
        "feature_names": booster.feature_name(),
    }
    return pred_va, info


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def _catboost_prepare_frame(
    df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    """
    CatBoost needs categorical columns as plain str/int (not pandas Categorical
    with NaN). Build a fresh DataFrame in one pass to avoid fragmentation.
    """
    cols: dict[str, pd.Series] = {}
    for c in num_cols:
        cols[c] = df[c].astype(np.float32).reset_index(drop=True)
    for c in cat_cols:
        s = df[c].astype("string").fillna("__missing__").astype(str)
        cols[c] = s.reset_index(drop=True)
    return pd.DataFrame(cols)


def train_catboost_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    params: dict,
    seed: int,
    num_boost_round: int | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[np.ndarray, dict]:
    """CatBoost single-fold trainer."""
    from catboost import CatBoostRegressor, Pool

    base_params = dict(config.CATBOOST_PARAMS)
    if num_boost_round is not None:
        base_params["iterations"] = num_boost_round
    if early_stopping_rounds is not None:
        base_params["od_wait"] = early_stopping_rounds
    p = {**base_params, **params, "random_seed": seed}

    Xtr = _catboost_prepare_frame(X_tr, num_cols, cat_cols)
    Xva = _catboost_prepare_frame(X_va, num_cols, cat_cols)

    cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]
    pool_tr = Pool(Xtr, label=y_tr, cat_features=cat_idx)
    pool_va = Pool(Xva, label=y_va, cat_features=cat_idx)

    model = CatBoostRegressor(**p)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)

    pred_va = model.predict(Xva)
    info = {
        "best_iteration": int(model.get_best_iteration() or model.tree_count_),
        "tree_count": int(model.tree_count_),
    }
    return np.asarray(pred_va, dtype=np.float64), info


# ---------------------------------------------------------------------------
# K-Nearest Neighbors regressor (numeric features only)
#
# KNN gives the stack a *non-tree* signal: predictions come from local
# averaging in feature space, so its errors live in different rows than
# tree errors and the meta-learner can extract genuinely new information.
# Categoricals are dropped (KNN can't natively handle them), but the OOF
# target encodings injected by feature_engineering already encode the
# important high-cardinality cats (sub_market, zipcode, h3) into numeric
# columns, so KNN still sees that signal.
# ---------------------------------------------------------------------------


def train_knn_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    params: dict,
    seed: int,
    num_boost_round: int | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[np.ndarray, dict]:
    """KNN single-fold trainer (numeric features only, log-rent target).

    If `params["feature_subset"]` is given, the KNN distance is computed over
    only those columns (intersected with the dataframe to skip any that were
    not produced by the feature pipeline). Otherwise the full `num_cols` is
    used.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsRegressor

    p = dict(params)
    feature_subset = p.pop("feature_subset", None)
    # Strip non-sklearn keys.
    for k in ("num_boost_round", "early_stopping_rounds"):
        p.pop(k, None)

    if feature_subset is not None:
        # Intersect requested subset with what the pipeline actually produced
        # this fold. Preserve the requested order.
        cols_used = [c for c in feature_subset if c in X_tr.columns]
        if not cols_used:
            # Fall back to all numerics if nothing matched.
            cols_used = list(num_cols)
    else:
        # Use numeric features only. Most categoricals already have OOF target
        # encodings (e.g. sub_market_te, zipcode_te) that sit in num_cols.
        cols_used = list(num_cols)

    Xt_num = X_tr[cols_used].astype(np.float64).to_numpy()
    Xv_num = X_va[cols_used].astype(np.float64).to_numpy()

    imputer = SimpleImputer(strategy="median")
    Xt_imp = imputer.fit_transform(Xt_num)
    Xv_imp = imputer.transform(Xv_num)

    # Standardize on train fold only, then clip to bound the influence of
    # outliers on the distance metric.
    mu = Xt_imp.mean(axis=0)
    sd = Xt_imp.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Xt_z = np.clip((Xt_imp - mu) / sd, -8.0, 8.0).astype(np.float32)
    Xv_z = np.clip((Xv_imp - mu) / sd, -8.0, 8.0).astype(np.float32)

    model = KNeighborsRegressor(**p)
    model.fit(Xt_z, np.asarray(y_tr, dtype=np.float64))
    pred_va = model.predict(Xv_z)

    info = {
        "best_iteration": int(p.get("n_neighbors", 15)),  # placeholder for uniform reporting
        "n_features": int(Xt_z.shape[1]),
        "k": int(p.get("n_neighbors", 15)),
        "feature_subset_size": int(len(cols_used)),
    }
    return np.asarray(pred_va, dtype=np.float64), info


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

TRAINERS = {
    "lightgbm": train_lightgbm_fold,
    "catboost": train_catboost_fold,
    "knn": train_knn_fold,
}


def get_trainer(name: str):
    """Look up a trainer by spec name (raises KeyError if unknown)."""
    if name not in TRAINERS:
        raise KeyError(f"Unknown trainer: {name}. Available: {sorted(TRAINERS)}")
    return TRAINERS[name]
