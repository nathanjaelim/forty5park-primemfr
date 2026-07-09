# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""prime_mfr.features — feature engineering subpackage.

Stage 2 of the architecture refactor introduces this subpackage as the
new home for feature engineering. Domain-specific modules are extracted
incrementally:

  - features.hist_rent     — historical-rent lag features (DONE)
  - features.engineering   — everything else (legacy monolith, slated for split)

Public API surface (unchanged from feature_engineering.py):

  - add_static_features(df)
  - add_oof_features(train_df, valid_df, ...)
  - select_feature_columns(df)
  - add_hist_rent_features(df)
"""

from prime_mfr.features.contract import (
    CONTRACT_VERSION,
    ContractSpec,
    Issue,
    IssueKind,
    IssueSeverity,
    ValidationResult,
    summarize,
    validate_pre_engineered,
)
from prime_mfr.features.engineering import (
    add_geo_aggregates,
    add_h3_cells,
    add_landmark_distances,
    add_oof_features,
    add_static_features,
    add_text_features,
    select_feature_columns,
)
from prime_mfr.features.hist_rent import (
    add_hist_rent_features,
    build_hist_rent_features,
)

__all__ = [
    "CONTRACT_VERSION",
    "ContractSpec",
    "Issue",
    "IssueKind",
    "IssueSeverity",
    "ValidationResult",
    "add_geo_aggregates",
    "add_h3_cells",
    "add_hist_rent_features",
    "add_landmark_distances",
    "add_oof_features",
    "add_static_features",
    "add_text_features",
    "build_hist_rent_features",
    "select_feature_columns",
    "summarize",
    "validate_pre_engineered",
]
