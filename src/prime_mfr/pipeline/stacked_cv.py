"""
run_full_stacked_cv.py
======================
Run the full 5-fold stacked CV one fold (or one base-per-fold) at a time,
persisting OOF predictions to disk between calls. This works around the
sandbox 45s bash timeout: feature engineering + training a single fold of
3 heterogeneous bases can exceed 45s, so we split work into smaller jobs.

Subcommands
-----------
    prep
        Build the static-feature df + GroupKFold splits + initialize the
        OOF state pickle. Run once.

    foldprep N
        Run feature_engineering OOF augmentation for fold N (1..5) and
        cache the prepared (X_tr, X_va, num_cols, cat_cols, y_tr, y_va_raw)
        to disk. Required before any `train N <base>` call. Idempotent.

    train N <base_name>
        Train one base (e.g. lgbm_l1, cat_q50, nn_q50) on the cached fold-N
        matrices and save its OOF log+raw predictions back into the state.

    fold N
        Convenience: runs `foldprep N` followed by `train N <name>` for
        every base in config.BASE_SPECS. Use only if all 3 bases fit in
        one bash window (typically does for fold ~12k; may not for full).

    meta
        Aggregate per-base OOF, fit Ridge meta-learner on log-rent,
        report stacked metrics + save artifacts.

    status
        Print which folds + bases have completed (read from state pickle).
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import prime_mfr.config as config
import prime_mfr.data_processing as dp
import prime_mfr.features as fe
import prime_mfr.models as md
import prime_mfr.train as train

SCRATCH_DIR = config.ARTIFACTS_DIR / "stacking_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
DF_PATH = SCRATCH_DIR / "df_with_static.parquet"
OOF_PATH = SCRATCH_DIR / "oof_state.pkl"
SPLITS_PATH = SCRATCH_DIR / "splits.json"


# Per-fold prepared matrices live here (overwritten each foldprep N call).
def _foldprep_path(fold_idx: int) -> Path:
    return SCRATCH_DIR / f"fold_{fold_idx}_prepared.pkl"


def _logger() -> logging.Logger:
    log = logging.getLogger("stack_runner")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    return log


def _load_state() -> dict:
    with open(OOF_PATH, "rb") as f:
        return pickle.load(f)


def _save_state(state: dict) -> None:
    with open(OOF_PATH, "wb") as f:
        pickle.dump(state, f)


def _base_names() -> list[str]:
    return [s["name"] for s in config.BASE_SPECS]


def _spec_by_name(name: str) -> dict:
    for s in config.BASE_SPECS:
        if s["name"] == name:
            return s
    raise KeyError(f"unknown base name '{name}' (known: {_base_names()})")


def _resolve_spec_params(spec: dict, bp: dict) -> dict:
    """Build the trainer-specific param dict, with no leakage of LGBM
    tuned hyperparams into CB/NN namespaces."""
    base = dict(spec.get("params", {}))
    if spec["trainer"] == "lightgbm":
        return {**bp, **base}
    if spec["trainer"] == "catboost":
        # Merge in tuned CatBoost params if a tuning artifact exists.
        # Order: shared defaults <- Optuna best_params <- per-spec overrides.
        cb_tuned: dict = {}
        if config.BEST_CB_PARAMS_PATH.exists():
            try:
                cb_tuned = json.loads(config.BEST_CB_PARAMS_PATH.read_text())
            except Exception:
                cb_tuned = {}
        return {**config.CATBOOST_PARAMS, **cb_tuned, **base}
    if spec["trainer"] == "knn":
        # knn_geo -> full numeric matrix; knn_lean -> tighter feature subset.
        knn_defaults = config.KNN_LEAN_PARAMS if spec["name"] == "knn_lean" else config.KNN_PARAMS
        return {**knn_defaults, **base}
    return base


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_prep() -> None:
    log = _logger()
    t0 = time.time()
    df = dp.load_clean()
    df = fe.add_static_features(df)
    df["log_rent"] = train.compute_log_target(df)
    df.to_parquet(DF_PATH, index=False)
    log.info(f"Wrote {df.shape} static-feature df ({time.time()-t0:.1f}s) -> {DF_PATH.name}")

    groups = df[config.GROUP_KEY].astype(str).values
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    splits = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        splits.append({"train": tr_idx.tolist(), "valid": va_idx.tolist()})
    SPLITS_PATH.write_text(json.dumps({"splits": splits}))
    log.info(f"Wrote splits ({len(splits)}) -> {SPLITS_PATH.name}")

    n = len(df)
    base_names = _base_names()
    state = {
        # OOF predictions per base on log-rent + raw scale.
        "oof_log": {k: np.full(n, np.nan, dtype=np.float64) for k in base_names},
        "oof_raw": {k: np.full(n, np.nan, dtype=np.float64) for k in base_names},
        # Per-fold per-base diagnostics.
        "per_fold": [{} for _ in range(config.N_FOLDS)],
        # LightGBM importance per fold (CB/NN excluded).
        "importance": [None] * config.N_FOLDS,
        # Done bookkeeping for status command.
        "done": {fold: {k: False for k in base_names} for fold in range(1, config.N_FOLDS + 1)},
    }
    _save_state(state)
    log.info(f"Initialized OOF state for bases: {base_names}")


def cmd_foldprep(fold_idx: int) -> None:
    """Run prepare_fold for fold N and cache the prepared matrices."""
    log = _logger()
    df = pd.read_parquet(DF_PATH)
    splits = json.loads(SPLITS_PATH.read_text())["splits"]
    s = splits[fold_idx - 1]
    tr_idx = np.asarray(s["train"], dtype=np.int64)
    va_idx = np.asarray(s["valid"], dtype=np.int64)

    t0 = time.time()
    tr_df, va_df, num_cols, cat_cols = train.prepare_fold(
        tr_idx,
        va_idx,
        df,
        target_log="log_rent",
        target_raw=config.TARGET,
    )
    feat_cols = num_cols + cat_cols
    payload = {
        "fold": fold_idx,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "X_tr": tr_df[feat_cols].copy(),
        "X_va": va_df[feat_cols].copy(),
        "y_tr_log": tr_df["log_rent"].values,
        "y_va_log": va_df["log_rent"].values,
        "y_va_raw": va_df[config.TARGET].values,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_train": int(len(tr_idx)),
        "n_valid": int(len(va_idx)),
    }
    with open(_foldprep_path(fold_idx), "wb") as f:
        pickle.dump(payload, f)
    log.info(
        f"foldprep {fold_idx}/{config.N_FOLDS} | n_tr={len(tr_idx):>5d} n_va={len(va_idx):>4d} "
        f"feat={len(feat_cols)} ({time.time()-t0:.1f}s) -> {_foldprep_path(fold_idx).name}"
    )


def cmd_train(fold_idx: int, base_name: str) -> None:
    """Train one base on the cached fold-N matrices."""
    log = _logger()
    fp = _foldprep_path(fold_idx)
    if not fp.exists():
        raise SystemExit(f"Missing prepared fold {fold_idx} (run `foldprep {fold_idx}` first).")
    with open(fp, "rb") as f:
        prep = pickle.load(f)
    spec = _spec_by_name(base_name)
    state = _load_state()
    bp = json.loads(config.BEST_PARAMS_PATH.read_text()) if config.BEST_PARAMS_PATH.exists() else {}

    seed = config.RANDOM_STATE + fold_idx + int(spec.get("seed_offset", 0))
    spec_params = _resolve_spec_params(spec, bp)
    trainer = md.get_trainer(spec["trainer"])

    t0 = time.time()
    pred_log, info = trainer(
        prep["X_tr"],
        prep["y_tr_log"],
        prep["X_va"],
        prep["y_va_log"],
        num_cols=prep["num_cols"],
        cat_cols=prep["cat_cols"],
        params=spec_params,
        seed=seed,
        num_boost_round=config.NUM_BOOST_ROUND,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    )
    wall = round(time.time() - t0, 1)

    va_sqft = prep["X_va"]["sqft"].values if "sqft" in prep["X_va"].columns else None
    raw_pred = train.back_transform_to_rent(pred_log, sqft=va_sqft)
    va_idx = prep["va_idx"]
    state["oof_log"][base_name][va_idx] = pred_log
    state["oof_raw"][base_name][va_idx] = raw_pred

    m = train.compute_metrics(prep["y_va_raw"], raw_pred)
    state["per_fold"][fold_idx - 1].setdefault("fold", fold_idx)
    state["per_fold"][fold_idx - 1].setdefault("n_train", prep["n_train"])
    state["per_fold"][fold_idx - 1].setdefault("n_valid", prep["n_valid"])
    state["per_fold"][fold_idx - 1][base_name] = {
        **m,
        "best_iteration": int(info.get("best_iteration", 0)),
        "wall_seconds": wall,
    }
    state["done"][fold_idx][base_name] = True

    # LightGBM importance.
    if spec["trainer"] == "lightgbm":
        state["importance"][fold_idx - 1] = pd.DataFrame(
            {
                "feature": info["feature_names"],
                "gain": info["feature_importance_gain"],
                "split": info["feature_importance_split"],
            }
        )

    _save_state(state)
    log.info(
        f"  fold {fold_idx}/{config.N_FOLDS} {base_name:8s} "
        f"iter={info.get('best_iteration', 0):>4d} "
        f"MAE=${m['MAE']:.2f} MAPE={m['MAPE']:.2f}% R2={m['R2']:.3f} ({wall}s)"
    )


def cmd_fold(fold_idx: int) -> None:
    """Convenience: foldprep + train all bases for fold N."""
    cmd_foldprep(fold_idx)
    for name in _base_names():
        cmd_train(fold_idx, name)


def cmd_status() -> None:
    log = _logger()
    if not OOF_PATH.exists():
        log.info("No state yet. Run `prep` first.")
        return
    state = _load_state()
    base_names = _base_names()
    header = "  fold | " + " | ".join(f"{n:>8s}" for n in base_names)
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))
    for fold in range(1, config.N_FOLDS + 1):
        marks = " | ".join(
            f"{('done' if state['done'][fold].get(n, False) else '....'): >8s}" for n in base_names
        )
        log.info(f"  {fold:>4d} | {marks}")


def cmd_meta() -> None:
    log = _logger()
    df = pd.read_parquet(DF_PATH)
    state = _load_state()
    base_names = _base_names()
    target_raw = config.TARGET
    target_log = "log_rent"

    # Use rows where ALL bases have a prediction (i.e. every fold completed for every base).
    used = np.ones(len(df), dtype=bool)
    for k in base_names:
        used &= ~np.isnan(state["oof_raw"][k])
    if not used.any():
        raise SystemExit("No rows have predictions from all bases. Run training first.")

    y_true_raw = df[target_raw].values
    y_true_log = df[target_log].values

    base_metrics = {
        k: train.compute_metrics(y_true_raw[used], state["oof_raw"][k][used]) for k in base_names
    }

    from sklearn.linear_model import Ridge

    # Aug-Ridge meta: 4 base log-preds + raw context features (log_sqft, beds, year_built).
    # Validated under GroupKFold OOF: -$4.17 to -$4.23 vs preds-only baseline.
    # Mechanism: the trees compress year_built / log_sqft into step functions, and
    # a linear meta picks up the residual smooth slopes the bases under-extracted.
    # NOTE: positive=False is required since context coefficients can be negative
    # (e.g. `beds` got -0.0347 in fit -- it's redundant with sqft and corrects rather
    # than adds). The base-pred coefs stay non-negative empirically.
    CONTEXT_COLS = ["log_sqft", "beds", "year_built"]

    df_used = df[used].copy()
    df_used["log_sqft"] = np.log1p(df_used["sqft"]).astype("float32")
    ctx_feats = []
    ctx_avail = []
    for c in CONTEXT_COLS:
        if c not in df_used.columns:
            log.info(f"  (Aug-Ridge: dropping missing context col '{c}')")
            continue
        v = pd.to_numeric(df_used[c], errors="coerce")
        v = v.fillna(v.median()).astype("float32").values
        ctx_feats.append(v)
        ctx_avail.append(c)

    X_base = np.column_stack([state["oof_log"][k][used] for k in base_names])
    X_meta = np.column_stack([X_base, *ctx_feats]) if ctx_feats else X_base
    y_meta = y_true_log[used]
    meta = Ridge(alpha=1.0, positive=False, fit_intercept=True, random_state=config.RANDOM_STATE)
    meta.fit(X_meta, y_meta)

    log_pred_stack = meta.predict(X_meta)
    oof_stack = np.full(len(df), np.nan, dtype=np.float64)
    stack_sqft = df["sqft"].values[used] if "sqft" in df.columns else None
    oof_stack[used] = train.back_transform_to_rent(log_pred_stack, sqft=stack_sqft)
    stack_metrics = train.compute_metrics(y_true_raw[used], oof_stack[used])
    n_base = len(base_names)
    meta_coefs = {
        "weights": dict(zip(base_names, [float(c) for c in meta.coef_[:n_base]])),
        "context": dict(zip(ctx_avail, [float(c) for c in meta.coef_[n_base:]])),
        "intercept": float(meta.intercept_),
        "alpha": 1.0,
        "kind": "aug_ridge",
    }

    # LightGBM-base importance averaging.
    imp_frames = [imp for imp in state["importance"] if isinstance(imp, pd.DataFrame)]
    if imp_frames:
        imp_df = pd.concat(
            [
                f.rename(columns={"gain": f"gain_{i+1}", "split": f"split_{i+1}"}).set_index(
                    "feature"
                )
                for i, f in enumerate(imp_frames)
            ],
            axis=1,
        ).fillna(0.0)
        gain_cols = [c for c in imp_df.columns if c.startswith("gain_")]
        imp_df["gain_mean"] = imp_df[gain_cols].mean(axis=1)
        imp_df = imp_df.sort_values("gain_mean", ascending=False).reset_index()
    else:
        imp_df = pd.DataFrame()

    def _block(title, m):
        log.info(title)
        log.info(f"  MAE       : ${m['MAE']:.2f}")
        log.info(f"  MAPE      : {m['MAPE']:.2f}%")
        log.info(f"  MedianAPE : {m['MedianAPE']:.2f}%")
        log.info(f"  RMSE      : ${m['RMSE']:.2f}")
        log.info(f"  MedianAE  : ${m['MedianAE']:.2f}")
        log.info(f"  R^2       : {m['R2']:.4f}")

    log.info("=" * 60)
    log.info("FULL 5-FOLD GROUPKFOLD CV (3-base stack, fold-batched)")
    log.info("=" * 60)
    for k in base_names:
        _block(f"[Base {k}]", base_metrics[k])
    _block(
        "[Stacked (Aug-Ridge meta on log-rent, base preds + log_sqft/beds/year_built)]",
        stack_metrics,
    )
    log.info(f"Meta base weights: {meta_coefs['weights']}")
    log.info(
        f"Meta context coefs: {meta_coefs['context']}  intercept={meta_coefs['intercept']:.4f}"
    )

    log.info("Per-fold:")
    for m in state["per_fold"]:
        if not m or "fold" not in m:
            continue
        chunks = [f"  fold {m['fold']} | n_va={m['n_valid']:>4d}"]
        for k in base_names:
            r = m.get(k)
            if not r:
                continue
            chunks.append(f"{k} MAE=${r['MAE']:.2f} R2={r['R2']:.3f} iter={r['best_iteration']}")
        log.info(" | ".join(chunks))

    # Save artifacts.
    config.METRICS_PATH.write_text(
        json.dumps(
            {
                "primary": stack_metrics,
                "base_metrics": base_metrics,
                "stack_metrics": stack_metrics,
                "meta_coefs": meta_coefs,
                "per_fold": [dict(m) for m in state["per_fold"] if m and "fold" in m],
                "config": {
                    "n_folds": config.N_FOLDS,
                    "log_target": config.LOG_TARGET,
                    "group_key": config.GROUP_KEY,
                    "use_stacking": True,
                    "quantile_alpha": config.QUANTILE_ALPHA,
                    "num_boost_round": config.NUM_BOOST_ROUND,
                    "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS,
                    "base_specs": [
                        {"name": s["name"], "trainer": s["trainer"]} for s in config.BASE_SPECS
                    ],
                },
            },
            indent=2,
            default=float,
        )
    )
    if not imp_df.empty:
        imp_df.to_csv(config.FEATURE_IMP_PATH, index=False)

    oof_cols = {
        "row_id": np.arange(len(df))[used],
        "property_id": df[config.GROUP_KEY].values[used],
        "rent_actual": y_true_raw[used],
        "rent_pred": oof_stack[used],
    }
    for k in base_names:
        oof_cols[f"rent_pred_{k}"] = state["oof_raw"][k][used]
    oof_cols["rent_pred_stack"] = oof_stack[used]
    pd.DataFrame(oof_cols).to_parquet(config.OOF_PRED_PATH, index=False)
    log.info(f"Wrote artifacts: {config.METRICS_PATH.name}, {config.OOF_PRED_PATH.name}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "prep":
        cmd_prep()
    elif cmd == "foldprep":
        cmd_foldprep(int(sys.argv[2]))
    elif cmd == "train":
        cmd_train(int(sys.argv[2]), sys.argv[3])
    elif cmd == "fold":
        cmd_fold(int(sys.argv[2]))
    elif cmd == "status":
        cmd_status()
    elif cmd == "meta":
        cmd_meta()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
