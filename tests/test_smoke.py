# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Smoke tests — verify every module imports and key public APIs are present."""

from __future__ import annotations

import pytest


def test_package_imports():
    """The top-level package and __version__ are importable."""
    import prime_mfr

    assert prime_mfr.__version__
    assert isinstance(prime_mfr.__version__, str)


def test_core_subpackage():
    """The new core/ subpackage exposes config loaders and paths."""
    from prime_mfr.core import (
        ARTIFACTS_DIR,
        CONFIGS_DIR,
        PROJECT_DIR,
        ModelConfig,
        PipelineConfig,
        list_available_models,
        load_model_config,
        load_pipeline_config,
    )

    assert PROJECT_DIR.exists()
    assert CONFIGS_DIR.exists()
    assert ARTIFACTS_DIR.exists()


def test_features_subpackage():
    """The features/ subpackage re-exports the public API."""
    import prime_mfr.features as features

    for name in (
        "add_static_features",
        "add_oof_features",
        "select_feature_columns",
        "add_hist_rent_features",
        "build_hist_rent_features",
    ):
        assert hasattr(features, name), f"prime_mfr.features missing {name}"


def test_models_registry():
    """models.get_trainer dispatches on trainer name."""
    from prime_mfr.models import get_trainer, TRAINERS

    assert "lightgbm" in TRAINERS
    assert "catboost" in TRAINERS
    assert "knn" in TRAINERS
    assert callable(get_trainer("lightgbm"))
    with pytest.raises(KeyError):
        get_trainer("nonexistent_trainer")


def test_cli_imports():
    """The new unified CLI and all subcommand modules import cleanly."""
    from prime_mfr.cli import main
    from prime_mfr.cli.commands import ablate, evaluate, list_cmd, predict, status, train, tune

    parser = main._build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"
