# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr train` — train a model variant.

Default: full ensemble (prep + all folds × all bases + meta-learner).

Filter flags let you scope the run:
  --base BASE            Train ONLY that base learner (across all folds).
                         Skips the meta step; reports per-base OOF metrics.
  --fold N               Train ONLY fold N (all bases). Skips meta.
  --fold N --base BASE   Train ONLY (fold N, base BASE). Smallest unit — dev iteration.

Examples
--------
    # The default — full 4-base stacked ensemble across 5 folds
    prime-mfr train --model primary

    # Just the LightGBM base across all folds; see how it performs alone
    prime-mfr train --model primary --base lgbm_l1

    # Just fold 3 (all 4 bases), skip meta
    prime-mfr train --model primary --fold 3

    # Just fold 3 lgbm_l1 (~15s; fastest dev cycle)
    prime-mfr train --model primary --fold 3 --base lgbm_l1
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from prime_mfr.core import load_model_config, list_available_models
from prime_mfr.pipeline import stacked_cv
from prime_mfr import train as train_mod


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train",
        help="Train a model variant. Default = full ensemble across all folds.",
        description=(
            "Loads configs/models/<name>.yaml and runs the stacked-CV pipeline. "
            "Default = full 4-base stack + meta. Use --base to train a single "
            "learner or --fold to train a single fold."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help=f"Model config name (available: {', '.join(list_available_models())}).",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train only fold N (1-5). Skips meta step. Useful for dev iteration.",
    )
    p.add_argument(
        "--base",
        default=None,
        help=(
            "Train only base learner NAME. Without --fold, trains that base across "
            "ALL folds and reports its standalone OOF metrics (skips meta)."
        ),
    )
    p.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip prep step (reuse cached feature-engineered dataframe + splits).",
    )
    p.set_defaults(func=run)


def _print_header(cfg) -> None:  # noqa: ANN001
    print(f"=> Training model: {cfg.name}")
    print(f"   {cfg.description}")
    print(
        f"   features: {len(cfg.numeric_features)} numeric + "
        f"{len(cfg.categorical_features)} categorical"
    )
    print(f"   bases: {[b.name for b in cfg.bases]}")
    print(f"   meta: {cfg.meta.type} (alpha={cfg.meta.alpha})")
    if cfg.nullification.enabled:
        print(
            f"   nullification: enabled "
            f"({cfg.nullification.fraction:.0%} of training rows, "
            f"{len(cfg.nullification.columns)} cols)"
        )
    print()


def _base_names() -> list[str]:
    from prime_mfr import config

    return [s["name"] for s in config.BASE_SPECS]


def _report_single_base_metrics(base_name: str) -> None:
    """Print standalone OOF metrics for one base across all completed folds."""
    from prime_mfr import config

    state = stacked_cv._load_state()
    df = pd.read_parquet(stacked_cv.DF_PATH)
    mask = ~np.isnan(state["oof_raw"][base_name])
    if not mask.any():
        print(f"NOTE: no OOF predictions for base '{base_name}' yet.", file=sys.stderr)
        return

    y_true = df[config.TARGET].values[mask]
    y_pred = state["oof_raw"][base_name][mask]
    m = train_mod.compute_metrics(y_true, y_pred)

    print()
    print(f"=== Single-base OOF metrics for {base_name} (n_rows={int(mask.sum())}) ===")
    print(f"  MAE       : ${m['MAE']:.2f}")
    print(f"  MAPE      : {m['MAPE']:.2f}%")
    print(f"  MedianAPE : {m['MedianAPE']:.2f}%")
    print(f"  RMSE      : ${m['RMSE']:.2f}")
    print(f"  R^2       : {m['R2']:.4f}")


def run(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    _print_header(cfg)

    if cfg.name != "primary":
        print(
            f"NOTE: model variant '{cfg.name}' is loaded from config but "
            f"running it end-to-end via this CLI requires the ModelConfig-\n"
            f"driven runner (not yet built). For now, temporarily toggle "
            f"hist features in config.py's NUMERIC_FEATURES to run cold_start.",
            file=sys.stderr,
        )
        return 2

    # Validate --base name if given.
    if args.base is not None and args.base not in _base_names():
        print(
            f"ERROR: unknown base '{args.base}'. Available: {_base_names()}",
            file=sys.stderr,
        )
        return 2

    # --- Single (fold, base) — smallest unit -----------------------------
    if args.fold is not None and args.base is not None:
        print(f"=> Training fold {args.fold}, base {args.base} only")
        if not args.skip_prep:
            stacked_cv.cmd_foldprep(args.fold)
        stacked_cv.cmd_train(args.fold, args.base)
        return 0

    # --- Single fold, all bases ------------------------------------------
    if args.fold is not None:
        print(f"=> Training all bases for fold {args.fold}")
        stacked_cv.cmd_fold(args.fold)
        return 0

    # --- Single base, all folds — standalone learner ---------------------
    if args.base is not None:
        print(f"=> Training base '{args.base}' across all {cfg.pipeline.cv_splits} folds")
        if not args.skip_prep:
            print("=> Running prep step (feature engineering + CV splits)")
            stacked_cv.cmd_prep()
        for fold in range(1, cfg.pipeline.cv_splits + 1):
            print(f"=> Fold {fold}/{cfg.pipeline.cv_splits}: foldprep + train {args.base}")
            stacked_cv.cmd_foldprep(fold)
            stacked_cv.cmd_train(fold, args.base)
        _report_single_base_metrics(args.base)
        print()
        print(
            "(Meta step SKIPPED — --base runs report the standalone base performance. "
            "To get the full stacked metrics, run `prime-mfr train --model primary`.)"
        )
        return 0

    # --- Full ensemble (default) -----------------------------------------
    if not args.skip_prep:
        print("=> Running prep step (feature engineering + CV splits)")
        stacked_cv.cmd_prep()
    for fold in range(1, cfg.pipeline.cv_splits + 1):
        print(f"=> Fold {fold}/{cfg.pipeline.cv_splits}")
        stacked_cv.cmd_fold(fold)
    print("=> Fitting meta-learner")
    stacked_cv.cmd_meta()
    return 0
