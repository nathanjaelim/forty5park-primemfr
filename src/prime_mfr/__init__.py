# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""prime_mfr — Atlanta multifamily rent prediction pipeline.

Public API surface:

    from prime_mfr import config
    from prime_mfr import data_processing, feature_engineering, models, train

Subpackages:

    prime_mfr.pipeline      — production stacked-CV runner
    prime_mfr.tuning        — Optuna search for LightGBM and CatBoost
    prime_mfr.evaluation    — error segmentation and post-hoc analysis
    prime_mfr.pretraining   — Yardi pretraining-table builders

CLI entry points (registered as console_scripts in pyproject.toml):

    prime-mfr            — main stacked-CV runner (prep / foldprep / train / meta)
    prime-mfr-tune-lgb   — LightGBM Optuna tuner
    prime-mfr-tune-cb    — CatBoost Optuna tuner
    prime-mfr-segment    — error segmentation report
    prime-mfr-pretrain   — build pretraining tables from Yardi sources
"""

from prime_mfr.__version__ import __version__

__all__ = ["__version__"]
