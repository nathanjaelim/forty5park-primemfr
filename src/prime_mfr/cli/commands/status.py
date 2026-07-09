# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""`prime-mfr status` — show training state for a model variant."""

from __future__ import annotations

import argparse

from prime_mfr.pipeline import stacked_cv


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("status", help="Show training state (which folds + bases are done).")
    p.add_argument("--model", default="primary", help="Model config name.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    stacked_cv.cmd_status()
    return 0
