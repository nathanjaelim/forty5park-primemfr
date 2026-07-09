# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr predict` — batch inference (stub; the inference layer is not yet built)."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "predict",
        help="Batch inference: predict rents for new units.",
        description="Predict rents for a parquet/csv input file. With --auto, "
        "routes by feature availability (primary if hist_rent_lag_1m is present, "
        "else cold_start). Requires the inference layer, which is not yet built.",
    )
    p.add_argument("--model", default=None, help="Specific model config name.")
    p.add_argument(
        "--auto",
        action="store_true",
        help="Auto-route by feature availability (primary vs cold_start).",
    )
    p.add_argument("--input", required=True, help="Input parquet or CSV file.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    print("`prime-mfr predict` is a Stage-5 deliverable (inference layer).")
    print("Targets:")
    print(f"  input  = {args.input}")
    print(f"  output = {args.output}")
    print(f"  model  = {args.model or 'auto-route'}")
    print()
    print("For now, use the OOF predictions in artifacts/oof_predictions.parquet")
    print("as a reference for the prediction column shape.")
    return 2
