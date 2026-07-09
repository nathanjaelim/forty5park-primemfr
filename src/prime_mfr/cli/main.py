# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""prime-mfr — unified command-line interface.

Usage:
    prime-mfr <command> [options]

Commands:
    train     Train a model variant end-to-end (prep + folds + meta).
    status    Show training state for a model variant.
    evaluate  Re-fit meta + write metrics from an existing OOF state.
    predict   Run batch inference (stub — requires the inference layer).
    ablate    Train a model variant with a feature ablated.
    tune      Run Optuna hyperparameter search for a base learner.
    clean     Remove generated outputs (keeps inputs); enables a fresh re-run.
    list      List available model configs.

Use `prime-mfr <command> --help` for command-specific options.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from prime_mfr import __version__
from prime_mfr.cli.commands import (
    ablate,
    clean,
    evaluate,
    list_cmd,
    predict,
    status,
    train,
    tune,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prime-mfr",
        description="Atlanta multifamily rent prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"prime-mfr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    train.register(sub)
    status.register(sub)
    evaluate.register(sub)
    predict.register(sub)
    ablate.register(sub)
    tune.register(sub)
    clean.register(sub)
    list_cmd.register(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
