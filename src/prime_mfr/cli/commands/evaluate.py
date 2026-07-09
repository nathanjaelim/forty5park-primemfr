# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr evaluate` — re-fit meta + write metrics from existing OOF state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prime_mfr.core import ARTIFACTS_DIR, load_model_config
from prime_mfr.pipeline import stacked_cv


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "evaluate",
        help="Re-fit meta + write metrics from an existing OOF state.",
    )
    p.add_argument("--model", default="primary", help="Model config name.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    print(f"=> Evaluating model: {cfg.name}")
    if cfg.name != "primary":
        print(
            f"NOTE: variant '{cfg.name}' evaluation requires the ModelConfig-driven runner (not yet built). "
            f"Falling back to primary metrics."
        )
    stacked_cv.cmd_meta()

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(Path(metrics_path).read_text())
        s = metrics.get("stack_metrics", {})
        if s:
            print()
            print("Final OOF stacked metrics:")
            print(f"  MAE       : ${s.get('MAE', float('nan')):.2f}")
            print(f"  MAPE      : {s.get('MAPE', float('nan')):.2f}%")
            print(f"  MedianAPE : {s.get('MedianAPE', float('nan')):.2f}%")
            print(f"  R^2       : {s.get('R2', float('nan')):.4f}")
    return 0
