# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr tune` — Optuna hyperparameter search for a base learner."""

from __future__ import annotations

import argparse

from prime_mfr.tuning import catboost as tune_cb
from prime_mfr.tuning import lightgbm as tune_lgb


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tune",
        help="Optuna hyperparameter search for a base learner.",
    )
    p.add_argument(
        "--base",
        required=True,
        choices=["lightgbm", "catboost"],
        help="Which base learner to tune.",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default 30; 50 for production tuning).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    # The existing tuners read N_OPTUNA_TRIALS from config.py + their own argv.
    # We dispatch by base name; trial count override is left as a future
    # improvement (the existing tuners support --trials via their own argparse).
    if args.base == "lightgbm":
        return tune_lgb.main()
    if args.base == "catboost":
        return tune_cb.main()
    print(f"Unknown base: {args.base}", file=__import__("sys").stderr)
    return 2
