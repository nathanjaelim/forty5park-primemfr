"""
tune_lightgbm.py
================
Run Optuna TPE on the LightGBM search space (config.lgb_search_space), with
the same JSON-snapshot persistence pattern used for CatBoost in tune_catboost.

Reuses the per-fold prep matrices already cached by run_full_stacked_cv.py
(`artifacts/stacking_scratch/fold_N_prepared.pkl`) so we don't duplicate
feature engineering. This means whatever feature set was used for the most
recent CV is what gets tuned — make sure the feature engineering you want
to optimize for is the one already in the cache.

Subcommands
-----------
    python tune_lightgbm.py status
        Print the number of trials in the study + best params/value so far.

    python tune_lightgbm.py tune --n-trials 10
        Run N new Optuna trials and append to the persistent study snapshot.

    python tune_lightgbm.py reset
        Backup and clear the study snapshot. Use after changing the feature
        set so prior trials don't pollute TPE with stale rankings.

    python tune_lightgbm.py best
        Print best params as JSON.

Tuning budget
-------------
N_TUNE_FOLDS=2 trades final-MAE precision for ~2x speedup. Tuning ranks
trials; absolute MAE only needs to be unbiased between trials.
TUNE_NUM_BOOST_ROUND=2000 with TUNE_ESR=50 keeps each trial under ~10s.
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import prime_mfr.config as config
import prime_mfr.models as md
import prime_mfr.train as train

SCRATCH_DIR = config.ARTIFACTS_DIR / "stacking_scratch"
STUDY_PATH = config.ARTIFACTS_DIR / "optuna_study.json"
BEST_PARAMS_PATH = config.BEST_PARAMS_PATH

N_TUNE_FOLDS = 2
TUNE_NUM_BOOST_ROUND = 1500
TUNE_ESR = 40


def _foldprep_path(fold_idx: int) -> Path:
    return SCRATCH_DIR / f"fold_{fold_idx}_prepared.pkl"


def _logger() -> logging.Logger:
    log = logging.getLogger("tune_lgbm")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    return log


# ---------------------------------------------------------------------------
# Objective: train lgbm_l1 on each cached fold, return mean OOF MAE
# ---------------------------------------------------------------------------


def _objective_fn(params_extra: dict, seed_base: int) -> float:
    """Mean MAE across the cached N_TUNE_FOLDS folds."""
    fold_maes: list[float] = []
    for fold_idx in range(1, N_TUNE_FOLDS + 1):
        fp = _foldprep_path(fold_idx)
        if not fp.exists():
            raise SystemExit(
                f"Missing prepared fold {fold_idx} at {fp}. "
                "Run `python run_full_stacked_cv.py prep && foldprep <N>` first."
            )
        with open(fp, "rb") as f:
            prep = pickle.load(f)

        # Tuned trial params: shared LGB defaults + base spec + trial overrides.
        # Our base lgbm_l1 spec uses objective=regression_l1.
        spec_params = {
            "objective": "regression_l1",
            "metric": "mae",
            **params_extra,
        }
        seed = config.RANDOM_STATE + fold_idx + seed_base
        pred_log, _info = md.train_lightgbm_fold(
            prep["X_tr"],
            prep["y_tr_log"],
            prep["X_va"],
            prep["y_va_log"],
            num_cols=prep["num_cols"],
            cat_cols=prep["cat_cols"],
            params=spec_params,
            seed=seed,
            num_boost_round=TUNE_NUM_BOOST_ROUND,
            early_stopping_rounds=TUNE_ESR,
        )
        raw_pred = train.back_transform_to_rent(pred_log)
        m = train.compute_metrics(prep["y_va_raw"], raw_pred)
        fold_maes.append(float(m["MAE"]))
    return float(np.mean(fold_maes))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _load_prior_trials() -> list[dict]:
    if not STUDY_PATH.exists():
        return []
    try:
        snap = json.loads(STUDY_PATH.read_text())
        return snap.get("trials", [])
    except Exception:
        return []


def _save_snapshot(study, study_name: str = "rent_lgbm_tuning") -> None:
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
    config.lgb_search_space(fixed)
    distributions = {k: fixed._distributions[k] for k in params.keys() if k in fixed._distributions}
    trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
    study.add_trial(trial)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_tune(n_trials: int) -> None:
    import optuna
    from optuna.samplers import TPESampler

    log = _logger()
    prior = _load_prior_trials()
    n_prior = len(prior)
    log.info(f"Loaded {n_prior} prior LGB trials.")

    sampler = TPESampler(seed=config.RANDOM_STATE + n_prior, n_startup_trials=8)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name="rent_lgbm_tuning"
    )

    for t in prior:
        try:
            _add_completed_trial(study, t["params"], t["value"])
        except Exception as e:
            log.warning(f"Could not replay trial: {e}")

    if study.trials:
        log.info(f"Best so far: MAE=${study.best_value:.2f} after {len(study.trials)} trials.")

    def objective(trial: optuna.Trial) -> float:
        params = config.lgb_search_space(trial)
        return _objective_fn(params, seed_base=3000 + trial.number)

    def cb(study_, trial: optuna.FrozenTrial) -> None:
        log.info(
            f"  trial {trial.number + 1} | MAE=${trial.value:.2f} | "
            f"best so far=${study_.best_value:.2f}"
        )
        # Persist after every trial so a sandbox timeout doesn't lose progress.
        try:
            _save_snapshot(study_)
        except Exception as e:
            log.warning(f"snapshot save failed: {e}")

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
        log.info("No study yet. Run `tune --n-trials N`.")
        return
    snap = json.loads(STUDY_PATH.read_text())
    log.info(f"Study: {snap.get('study_name')}")
    log.info(f"Trials: {len(snap.get('trials', []))}")
    log.info(f"Best MAE: ${snap.get('best_value', float('nan')):.2f}")
    log.info("Best params:")
    for k, v in snap.get("best_params", {}).items():
        log.info(f"  {k}: {v}")


def cmd_reset() -> None:
    log = _logger()
    if STUDY_PATH.exists():
        bak = STUDY_PATH.with_suffix(".json.bak")
        shutil.copy2(STUDY_PATH, bak)
        STUDY_PATH.unlink()
        log.info(f"Backed up old study -> {bak.name}")
    if BEST_PARAMS_PATH.exists():
        bak = BEST_PARAMS_PATH.with_suffix(".json.bak")
        shutil.copy2(BEST_PARAMS_PATH, bak)
        BEST_PARAMS_PATH.unlink()
        log.info(f"Backed up old best_params -> {bak.name}")
    log.info("Study reset. Run `tune --n-trials N` for a fresh optimization.")


def cmd_best() -> None:
    if not BEST_PARAMS_PATH.exists():
        print("{}")
        return
    print(BEST_PARAMS_PATH.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["tune", "status", "reset", "best"])
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()
    if args.cmd == "tune":
        cmd_tune(args.n_trials)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "reset":
        cmd_reset()
    elif args.cmd == "best":
        cmd_best()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
