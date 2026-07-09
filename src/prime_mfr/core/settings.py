# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""YAML config loader for prime_mfr.

Public surface:

    load_pipeline_config(name="default")  -> PipelineConfig
    load_model_config(name="primary")     -> ModelConfig
    load_yaml(path)                        -> dict   (raw)

`ModelConfig` resolves its dependencies (pipeline, hyperparam files, feature
groups) at load time so callers get a fully-materialized config back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prime_mfr.core.paths import ARTIFACTS_DIR, CONFIGS_DIR


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file and return it as a Python dict."""
    p = Path(path)
    if not p.is_absolute():
        p = CONFIGS_DIR / p
    with p.open("r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Settings shared across all model variants."""

    target_column: str
    target_transform: str  # "log1p" | "identity" | "log_psf"
    cv_strategy: str  # "group_kfold"
    cv_splits: int
    cv_group_key: str
    cv_random_state: int
    num_boost_round: int
    early_stopping_rounds: int
    n_optuna_trials: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            target_column=d["target"]["column"],
            target_transform=d["target"]["transform"],
            cv_strategy=d["cv"]["strategy"],
            cv_splits=int(d["cv"]["splits"]),
            cv_group_key=d["cv"]["group_key"],
            cv_random_state=int(d["cv"]["random_state"]),
            num_boost_round=int(d["training"]["num_boost_round"]),
            early_stopping_rounds=int(d["training"]["early_stopping_rounds"]),
            n_optuna_trials=int(d["tuning"]["n_optuna_trials"]),
        )


def load_pipeline_config(name: str = "default") -> PipelineConfig:
    """Load a pipeline config by name (e.g. 'default')."""
    return PipelineConfig.from_dict(load_yaml(f"pipelines/{name}.yaml"))


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def _resolve_numeric_features(group_names: list[str]) -> list[str]:
    """Expand named numeric-feature groups (from features/numeric.yaml) to a
    flat list, deduplicated while preserving order of first appearance."""
    groups = load_yaml("features/numeric.yaml")["groups"]
    seen: set[str] = set()
    out: list[str] = []
    for g in group_names:
        if g not in groups:
            raise KeyError(f"Unknown numeric feature group: {g!r}")
        for feat in groups[g]:
            if feat not in seen:
                seen.add(feat)
                out.append(feat)
    return out


def _load_categorical_block() -> dict[str, list[str]]:
    """All categorical/boolean/text/TE feature lists, keyed by section name."""
    return load_yaml("features/categorical.yaml")


# ---------------------------------------------------------------------------
# Hyperparams
# ---------------------------------------------------------------------------


def _resolve_hyperparams_ref(ref: str) -> dict[str, Any]:
    """Resolve a hyperparams reference like 'lightgbm', 'knn.geo', 'knn.lean'.

    For 'lightgbm' and 'catboost', merges the YAML 'fixed' + 'tuned' blocks
    with the corresponding best_params.json artifact (artifact wins).
    """
    parts = ref.split(".")
    file_name = parts[0]
    raw = load_yaml(f"hyperparams/{file_name}.yaml")
    if len(parts) == 2:
        # nested ref like knn.geo
        return dict(raw[parts[1]])

    # lightgbm / catboost: merge fixed + tuned + artifact override.
    merged: dict[str, Any] = {}
    merged.update(raw.get("fixed", {}))
    merged.update(raw.get("tuned", {}))
    artifact_map = {
        "lightgbm": ARTIFACTS_DIR / "best_params.json",
        "catboost": ARTIFACTS_DIR / "best_catboost_params.json",
    }
    artifact_path = artifact_map.get(file_name)
    if artifact_path and artifact_path.exists():
        try:
            merged.update(json.loads(artifact_path.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return merged


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseSpec:
    """One base learner in a stacked ensemble."""

    name: str
    trainer: str
    hyperparams: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)
    seed_offset: int = 0

    @property
    def params(self) -> dict[str, Any]:
        return {**self.hyperparams, **self.overrides}


@dataclass(frozen=True)
class MetaSpec:
    """Meta-learner configuration."""

    type: str  # "aug_ridge" | "ridge"
    alpha: float
    positive: bool
    fit_intercept: bool
    context_features: tuple[str, ...]


@dataclass(frozen=True)
class NullificationSpec:
    """Random feature nullification settings for graceful variant."""

    enabled: bool
    columns: tuple[str, ...] = ()
    fraction: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """A fully-materialized model variant: features + bases + meta + artifacts."""

    name: str
    description: str
    pipeline: PipelineConfig
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    text_booleans: tuple[str, ...]
    text_numerics: tuple[str, ...]
    booleans: tuple[str, ...]
    ordinal_grade: tuple[str, ...]
    target_encode: tuple[str, ...]
    bases: tuple[BaseSpec, ...]
    meta: MetaSpec
    nullification: NullificationSpec
    artifacts: dict[str, Path]
    raw: dict[str, Any]  # the original YAML, for forward-compat reads


def load_model_config(name: str) -> ModelConfig:
    """Load a model variant config by name (e.g. 'primary', 'cold_start')."""
    raw = load_yaml(f"models/{name}.yaml")
    pipeline = load_pipeline_config(raw.get("pipeline", "default"))

    # Features
    feats = raw["features"]
    numeric = _resolve_numeric_features(feats["include_numeric_groups"])
    cat_block = _load_categorical_block()

    def _opt(key: str) -> tuple[str, ...]:
        return tuple(cat_block.get(key, [])) if feats.get(f"include_{key}", False) else ()

    # Booleans/categorical/etc. use boolean toggles in YAML.
    categorical = tuple(cat_block["categorical"]) if feats.get("include_categorical") else ()
    booleans = tuple(cat_block["booleans"]) if feats.get("include_booleans") else ()
    ordinal_grade = tuple(cat_block["ordinal_grade"]) if feats.get("include_ordinal_grade") else ()
    text_booleans = (
        tuple(cat_block["text_booleans"]) if feats.get("include_text_features") else ()
    )
    text_numerics = (
        tuple(cat_block["text_numerics"]) if feats.get("include_text_features") else ()
    )
    target_encode = (
        tuple(cat_block["target_encode"]) if feats.get("include_target_encoding") else ()
    )

    # Bases
    bases = tuple(
        BaseSpec(
            name=b["name"],
            trainer=b["trainer"],
            hyperparams=_resolve_hyperparams_ref(b["hyperparams"]),
            overrides=dict(b.get("overrides", {})),
            seed_offset=int(b.get("seed_offset", 0)),
        )
        for b in raw["bases"]
    )

    # Meta
    m = raw["meta"]
    meta = MetaSpec(
        type=m["type"],
        alpha=float(m["alpha"]),
        positive=bool(m.get("positive", False)),
        fit_intercept=bool(m.get("fit_intercept", True)),
        context_features=tuple(m.get("context_features", [])),
    )

    # Nullification
    nul = raw.get("training", {}).get("nullification", {"enabled": False})
    nullification = NullificationSpec(
        enabled=bool(nul.get("enabled", False)),
        columns=tuple(nul.get("columns", [])),
        fraction=float(nul.get("fraction", 0.0)),
        seed=int(nul.get("seed", 0)),
    )

    # Artifact paths resolved relative to PROJECT_DIR.
    artifacts = {
        k: (CONFIGS_DIR.parent / v).resolve() for k, v in raw.get("artifacts", {}).items()
    }

    return ModelConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        pipeline=pipeline,
        numeric_features=tuple(numeric),
        categorical_features=categorical,
        text_booleans=text_booleans,
        text_numerics=text_numerics,
        booleans=booleans,
        ordinal_grade=ordinal_grade,
        target_encode=target_encode,
        bases=bases,
        meta=meta,
        nullification=nullification,
        artifacts=artifacts,
        raw=raw,
    )


def list_available_models() -> list[str]:
    """List the model names available in configs/models/."""
    return sorted(p.stem for p in (CONFIGS_DIR / "models").glob("*.yaml"))
