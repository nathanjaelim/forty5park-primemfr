"""
tune_catboost.py
================
Run Optuna TPE on the CatBoost search space (config.cb_search_space), with
the same JSON-snapshot persistence pattern used for LightGBM in train.py.
Each trial does N_TUNE_FOLDS-fold OOF CV using only the cat_q50 base, with
a tightened iteration / early-stopping budget so trials are cheap.

This is structured for the sandbox 45s bash window: each invocation runs
`--n-trials` new trials and appends them to the persistent study snapshot.

Subcommands
-----------
    python tune_catboost.py prep
        Build the static-feature df and per-fold prepared matrices for the
        first N_TUNE_FOLDS folds. Run once.

    python tune_catboost.py tune --n-trials 5
        Run 5 new Optuna trials on the cached fold matrices. Repeats are
        idempotent (params resampled, study persisted).

    python tune_catboost.py status
        Print the number of trials in the study + best params/value so far.

    python tune_catboost.py best
        Print the best params as JSON so you can save them to artifacts/.
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import prime_mfr.config as config
import prime_mfr.data_processing as dp
import prime_mfr.features as fe
import prime_mfr.models as md
import prime_mfr.train as train

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

SCRATCH_DIR = config.ARTIFACTS_DIR / "tune_cb_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

STUDY_PATH = config.ARTIFACTS_DIR / "optuna_catboost_study.json"
BEST_PARAMS_PATH = config.ARTIFACTS_DIR / "best_catboost_params.json"

N_TUNE_FOLDS = 2  # tune on 2 folds (vs 5 final). Cuts tuning wall time
# roughly in half while still ranking trials reliably
# — TPE only needs relative ordering, not absolute MAE.
TUNE_ITERATIONS = 500  # cap for tuning (full uses 3000 in CATBOOST_PARAMS).
# With depth<=8 and lr>=0.02, this is plenty to rank.
TUNE_OD_WAIT = 30  # tighter early stopping for tuning


def _foldprep_path(fold_idx: int) -> Path:
    return SCRATCH_DIR / f"cbtune_fold_{fold_idx}.pkl"


def _logger() -> logging.Logger:
    log = logging.getLogger("tune_catboost")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    return log


# ---------------------------------------------------------------------------
# prep: build df + per-fold prepared matrices
# ---------------------------------------------------------------------------


def cmd_prep() -> None:
    log = _logger()
    t0 = time.time()
    df = dp.load_clean()
    df = fe.add_static_features(df)
    df["log_rent"] = train.compute_log_target(df)
    log.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols ({time.time()-t0:.1f}s)")

    groups = df[config.GROUP_KEY].astype(str).values
    gkf = GroupKFold(n_splits=config.N_FOLDS)

    # Use the FIRST N_TUNE_FOLDS folds. Folds are deterministic, so reusing
    # the cached prep across calls is safe.
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        if fold_idx > N_TUNE_FOLDS:
            break
        t = time.time()
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
            "va_sqft": va_df["sqft"].values,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
        with open(_foldprep_path(fold_idx), "wb") as f:
            pickle.dump(payload, f)
        log.info(
            f"  cached fold {fold_idx}/{N_TUNE_FOLDS}  "
            f"n_tr={len(tr_idx):>5d} n_va={len(va_idx):>4d} "
            f"feat={len(feat_cols)} ({time.time()-t:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Objective: train cat_q50 on each cached fold, return mean OOF MAE
# ---------------------------------------------------------------------------


def _objective_fn(params_extra: dict, seed_base: int) -> float:
    """Mean MAE across the cached N_TUNE_FOLDS folds."""
    fold_maes: list[float] = []
    for fold_idx in range(1, N_TUNE_FOLDS + 1):
        fp = _foldprep_path(fold_idx)
        if not fp.exists():
            raise SystemExit(f"Missing prepared fold {fold_idx}; run `prep` first.")
        with open(fp, "rb") as f:
            prep = pickle.load(f)

        # Tuned trial params: shared CatBoost defaults +  trial-suggested overrides.
        # Override iterations/od_wait with the tuning budget (cheaper than full).
        spec_params = {
            **config.CATBOOST_FIXED_PARAMS,
            **params_extra,
            "iterations": TUNE_ITERATIONS,
            "od_wait": TUNE_OD_WAIT,
        }
        seed = config.RANDOM_STATE + fold_idx + seed_base
        pred_log, _info = md.train_catboost_fold(
            prep["X_tr"],
            prep["y_tr_log"],
            prep["X_va"],
            prep["y_va_log"],
            num_cols=prep["num_cols"],
            cat_cols=prep["cat_cols"],
            params=spec_params,
            seed=seed,
            num_boost_round=TUNE_ITERATIONS,
            early_stopping_rounds=TUNE_OD_WAIT,
        )
        raw_pred = train.back_transform_to_rent(pred_log, sqft=prep["va_sqft"])
        m = train.compute_metrics(prep["y_va_raw"], raw_pred)
        fold_maes.append(float(m["MAE"]))
    return float(np.mean(fold_maes))


# ---------------------------------------------------------------------------
# tune: run N new Optuna trials and append to JSON snapshot
# ---------------------------------------------------------------------------


def _load_prior_trials() -> list[dict]:
    if not STUDY_PATH.exists():
        return []
    try:
        snap = json.loads(STUDY_PATH.read_text())
        return snap.get("trials", [])
    except Exception:
        return []


def _save_snapshot(study, study_name: str = "rent_cb_tuning") -> None:
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
    STUDY_PATH.write_text(json.dumps(snapshot, indent=2))
    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))


def _add_completed_trial(study, params: dict, value: float) -> None:
    """Insert a completed trial into the study without re-running."""
    import optuna

    fixed = optuna.trial.FixedTrial(params)
    config.cb_search_space(fixed)
    distributions = {k: fixed._distributions[k] for k in params.keys() if k in fixed._distributions}
    trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
    study.add_trial(trial)


def cmd_tune(n_trials: int) -> None:
    import optuna
    from optuna.samplers import TPESampler

    log = _logger()
    prior = _load_prior_trials()
    n_prior = len(prior)
    log.info(f"Loaded {n_prior} prior CatBoost trials.")

    # Vary sampler seed so successive batches don't redraw startup samples.
    sampler = TPESampler(seed=config.RANDOM_STATE + n_prior, n_startup_trials=8)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="rent_cb_tuning")

    for t in prior:
        try:
            _add_completed_trial(study, t["params"], t["value"])
        except Exception as e:
            log.warning(f"Could not replay trial: {e}")

    if study.trials:
        log.info(f"Best so far: MAE=${study.best_value:.2f} after {len(study.trials)} trials.")

    def objective(trial: optuna.Trial) -> float:
        params = config.cb_search_space(trial)
        return _objective_fn(params, seed_base=2000 + trial.number)

    def cb(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        log.info(
            f"  trial {trial.number + 1} | MAE=${trial.value:.2f} | "
            f"best so far=${study.best_value:.2f}"
        )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[cb], show_progress_bar=False)
    log.info(
        f"Done {n_trials} new trials in {time.time()-t0:.0f}s. "
        f"Total: {len(study.trials)}. Best MAE=${study.best_value:.2f}"
    )
    _save_snapshot(study)


def cmd_status() -> None:
    log = _logger()
    if not STUDY_PATH.exists():
        log.info("No study yet. Run `prep` then `tune --n-trials N`.")
        return
    snap = json.loads(STUDY_PATH.read_text())
    log.info(f"Study: {snap.get('study_name')}")
    log.info(f"Trials: {len(snap.get('trials', []))}")
    log.info(f"Best MAE: ${snap.get('best_value', float('nan')):.2f}")
    log.info("Best params:")
    for k, v in snap.get("best_params", {}).items():
        log.info(f"  {k}: {v}")


def cmd_best() -> None:
    if not BEST_PARAMS_PATH.exists():
        print("{}")
        return
    print(BEST_PARAMS_PATH.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prep", "tune", "status", "best"])
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of NEW Optuna trials to run this invocation.",
    )
    args = parser.parse_args()
    if args.cmd == "prep":
        cmd_prep()
    elif args.cmd == "tune":
        cmd_tune(args.n_trials)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "best":
        cmd_best()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
