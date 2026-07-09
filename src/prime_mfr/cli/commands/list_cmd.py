# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr list` — list available model configs."""

from __future__ import annotations

import argparse

from prime_mfr.core import list_available_models, load_model_config


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("list", help="List available model configs.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    names = list_available_models()
    if not names:
        print("No model configs found in configs/models/.")
        return 1
    print(f"{'Name':<14}  {'Features':>8}  {'Bases':>5}  Description")
    print("-" * 78)
    for n in names:
        try:
            cfg = load_model_config(n)
            n_feats = len(cfg.numeric_features) + len(cfg.categorical_features)
            desc = cfg.description[:42]
        except Exception as exc:  # noqa: BLE001
            n_feats, desc = -1, f"<load error: {exc!s}>"
        print(f"{n:<14}  {n_feats:>8}  {len(cfg.bases):>5}  {desc}")
    return 0
