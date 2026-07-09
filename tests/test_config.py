# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for the YAML config loader (prime_mfr.core.settings)."""

from __future__ import annotations

import pytest

from prime_mfr.core import list_available_models, load_model_config, load_pipeline_config


def test_list_available_models_returns_three_variants():
    names = list_available_models()
    assert "primary" in names
    assert "cold_start" in names
    assert "graceful" in names


def test_pipeline_default_loads():
    p = load_pipeline_config("default")
    assert p.cv_splits == 5
    assert p.cv_strategy == "group_kfold"
    assert p.cv_group_key == "property_id"
    assert p.cv_random_state == 42
    assert p.target_column == "rent"
    assert p.target_transform == "log1p"


def test_primary_model_config():
    cfg = load_model_config("primary")
    assert cfg.name == "primary"
    assert "hist_rent_lag_1m" in cfg.numeric_features
    assert len(cfg.bases) == 4
    base_names = [b.name for b in cfg.bases]
    assert base_names == ["lgbm_l1", "cat_q50", "knn_geo", "knn_lean"]
    assert cfg.meta.type == "aug_ridge"
    assert cfg.meta.alpha == 1.0
    assert cfg.meta.context_features == ("log_sqft", "beds", "year_built")
    assert cfg.nullification.enabled is False


def test_cold_start_model_drops_hist_features():
    cfg = load_model_config("cold_start")
    assert "hist_rent_lag_1m" not in cfg.numeric_features
    assert "hist_rent_lag_3m" not in cfg.numeric_features
    assert "hist_rent_yoy" not in cfg.numeric_features


def test_graceful_model_has_nullification():
    cfg = load_model_config("graceful")
    assert cfg.nullification.enabled is True
    assert cfg.nullification.fraction == pytest.approx(0.30)
    assert "hist_rent_lag_1m" in cfg.nullification.columns
    assert len(cfg.nullification.columns) == 5


def test_base_spec_merges_hyperparams_and_overrides():
    cfg = load_model_config("primary")
    lgb = next(b for b in cfg.bases if b.name == "lgbm_l1")
    # objective from overrides should win
    assert lgb.params["objective"] == "regression_l1"
    # tuned hyperparam should be present from configs/hyperparams/lightgbm.yaml
    assert "learning_rate" in lgb.params


def test_unknown_model_raises():
    with pytest.raises(FileNotFoundError):
        load_model_config("does_not_exist")
