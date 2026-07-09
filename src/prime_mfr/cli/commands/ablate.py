# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr ablate` — train a model variant with a feature ablated."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ablate",
        help="Train a model with a feature removed (ablation study).",
        description="Drop one or more features and re-train, to measure their "
        "contribution. Saves to artifacts/metrics.<feature>.json.",
    )
    p.add_argument("--model", default="primary", help="Base model config name.")
    p.add_argument(
        "--drop",
        nargs="+",
        required=True,
        help="Feature name(s) to drop (e.g. hist_rent_lag_1m).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    print("`prime-mfr ablate` is a Stage-5+ deliverable.")
    print(f"  base model: {args.model}")
    print(f"  drop:       {args.drop}")
    print()
    print("For now, manual ablations: comment the feature out in config.py's")
    print("NUMERIC_FEATURES list, re-prep, re-train, and rename the output.")
    return 2
