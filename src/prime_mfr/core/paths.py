# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Canonical filesystem anchors for the prime_mfr package.

`PROJECT_DIR` resolves to the repository root regardless of where code is
imported from. All other paths are anchored to it.
"""

from __future__ import annotations

from pathlib import Path

# src/prime_mfr/core/paths.py  ->  parents[3] is the repo root.
PROJECT_DIR: Path = Path(__file__).resolve().parents[3]
SRC_DIR: Path = PROJECT_DIR / "src"
CONFIGS_DIR: Path = PROJECT_DIR / "configs"
ARTIFACTS_DIR: Path = PROJECT_DIR / "artifacts"
EDA_DIR: Path = PROJECT_DIR / "eda"

# Ensure artifacts dir exists (training writes to it).
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
