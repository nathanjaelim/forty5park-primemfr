# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Feature-pipeline contract for data-engineering handoff.

Defines the schema a pre-engineered parquet must conform to before the
inference pipeline can consume it. See docs/feature_pipeline_contract.md
for the full spec.

Public API
----------
    validate_pre_engineered(df) -> ValidationResult
        Check a DataFrame against the contract.

    ContractSpec()
        Loadable, programmatic representation of the contract.

    CONTRACT_VERSION
        Current contract version. Bumps on breaking schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from prime_mfr.core import load_yaml

CONTRACT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Issue type
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueKind(str, Enum):
    MISSING_COLUMN = "missing_column"
    WRONG_DTYPE = "wrong_dtype"
    OUT_OF_RANGE = "out_of_range"
    EXTRA_COLUMN = "extra_column"
    NULL_NOT_ALLOWED = "null_not_allowed"
    DUPLICATE_KEY = "duplicate_key"


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    severity: IssueSeverity
    column: str | None
    message: str

    def __str__(self) -> str:
        col = f"[{self.column}] " if self.column else ""
        return f"{self.severity.value.upper():<7s} {self.kind.value:<20s} {col}{self.message}"


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        kind: IssueKind,
        severity: IssueSeverity,
        column: str | None,
        message: str,
    ) -> None:
        self.issues.append(Issue(kind, severity, column, message))


# ---------------------------------------------------------------------------
# Sanity bounds — out-of-range data fails the contract
# ---------------------------------------------------------------------------

# (min, max) inclusive. None = no bound on that side.
NUMERIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    # Geography (Atlanta MSA bounding box, generous)
    "latitude": (32.5, 35.0),
    "longitude": (-85.5, -83.5),
    # Unit size
    "sqft": (100.0, 10000.0),
    "beds": (0.0, 8.0),
    # baths=0 occurs for studios (single-room units), so the floor is 0.
    "baths": (0.0, 8.0),
    # Property
    "year_built": (1850.0, 2030.0),
    "num_units": (1.0, 5000.0),
    "occupancy_rate": (0.0, 1.0),
    # Historical lags
    "hist_rent_lag_1m": (100.0, 50000.0),
    "hist_rent_lag_3m": (100.0, 50000.0),
    "hist_rent_lag_12m": (100.0, 50000.0),
    "hist_rent_lag_24m": (100.0, 50000.0),
}


# ---------------------------------------------------------------------------
# Required schema (resolved from configs/features/*.yaml at module load)
# ---------------------------------------------------------------------------


def _resolve_required_columns() -> dict[str, set[str]]:
    """Resolve the required columns from the YAML feature configs.

    Returns a dict of {section: set of column names}. Used by the validator
    to check what must be present.
    """
    numeric_yaml = load_yaml("features/numeric.yaml")
    cat_yaml = load_yaml("features/categorical.yaml")

    numeric: set[str] = set()
    for group_features in numeric_yaml["groups"].values():
        numeric.update(group_features)

    return {
        "numeric": numeric,
        "categorical": set(cat_yaml.get("categorical", [])),
        "ordinal_grade": set(cat_yaml.get("ordinal_grade", [])),
        "booleans": set(cat_yaml.get("booleans", [])),
        "text_booleans": set(cat_yaml.get("text_booleans", [])),
        "text_numerics": set(cat_yaml.get("text_numerics", [])),
    }


# Identity columns must be present but aren't model features.
# The unit-level composite key is (property_id, unit_type, unit_mix_type, unit_garage):
# Yardi sometimes lists the same (property, unit_type) twice with different
# unit_mix_type (Apartment vs Townhouse) or unit_garage (0/1).
IDENTITY_COLUMNS: set[str] = {
    "property_id",
    "unit_type",
    "unit_mix_type",
    "unit_garage",
    "period",
}
COMPOSITE_KEY: tuple[str, ...] = ("property_id", "unit_type", "unit_mix_type", "unit_garage")


@dataclass
class ContractSpec:
    """Programmatic view of the contract.

    Resolved at construction time from configs/features/*.yaml so the
    contract stays in sync with the model's feature lists.
    """

    version: str = CONTRACT_VERSION
    identity: set[str] = field(default_factory=lambda: set(IDENTITY_COLUMNS))
    sections: dict[str, set[str]] = field(default_factory=_resolve_required_columns)

    @property
    def all_required_columns(self) -> set[str]:
        out: set[str] = set(self.identity)
        for cols in self.sections.values():
            out.update(cols)
        return out

    @property
    def all_numeric_columns(self) -> set[str]:
        return self.sections["numeric"] | self.sections["text_numerics"]

    @property
    def all_categorical_columns(self) -> set[str]:
        return self.sections["categorical"]

    @property
    def all_boolean_columns(self) -> set[str]:
        return self.sections["booleans"] | self.sections["text_booleans"]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _is_string_like_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _is_int_or_bool(s: pd.Series) -> bool:
    return (
        pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_float_dtype(s)  # 0.0/1.0/NaN encoding is allowed
    )


def validate_pre_engineered(
    df: pd.DataFrame,
    *,
    spec: ContractSpec | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Validate a pre-engineered DataFrame against the feature contract.

    Parameters
    ----------
    df : DataFrame to validate.
    spec : Optional override of the contract spec (defaults to current).
    strict : If True, extra columns become errors instead of warnings.

    Returns
    -------
    ValidationResult with a list of Issues. Check `.is_valid` for pass/fail.
    """
    spec = spec or ContractSpec()
    result = ValidationResult()

    actual_cols = set(df.columns)
    required = spec.all_required_columns

    # 1. Required columns present?
    for col in sorted(required - actual_cols):
        section = "identity" if col in spec.identity else _find_section(col, spec)
        result.add(
            IssueKind.MISSING_COLUMN,
            IssueSeverity.ERROR,
            col,
            f"required column missing (section: {section})",
        )

    # 2. Extra columns?
    extras = actual_cols - required
    for col in sorted(extras):
        severity = IssueSeverity.ERROR if strict else IssueSeverity.WARNING
        result.add(
            IssueKind.EXTRA_COLUMN,
            severity,
            col,
            "column not in contract (may be future-feature or DE-pipeline metadata)",
        )

    # 3. dtype checks on columns that ARE present
    for col in spec.all_numeric_columns & actual_cols:
        if not _is_numeric_dtype(df[col]):
            result.add(
                IssueKind.WRONG_DTYPE,
                IssueSeverity.ERROR,
                col,
                f"expected numeric dtype, got {df[col].dtype}",
            )

    for col in spec.all_categorical_columns & actual_cols:
        if not _is_string_like_dtype(df[col]):
            result.add(
                IssueKind.WRONG_DTYPE,
                IssueSeverity.ERROR,
                col,
                f"expected string-like dtype, got {df[col].dtype}",
            )

    for col in spec.all_boolean_columns & actual_cols:
        if not _is_int_or_bool(df[col]):
            result.add(
                IssueKind.WRONG_DTYPE,
                IssueSeverity.ERROR,
                col,
                f"expected bool/int/float (0/1) dtype, got {df[col].dtype}",
            )

    # 4. Out-of-range checks on numeric columns with declared bounds.
    # Skipped if the column isn't numeric (the dtype error already fired above).
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        if not _is_numeric_dtype(df[col]):
            continue  # dtype error already reported; bounds check doesn't apply
        s = df[col].dropna()
        if s.empty:
            continue
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= s < lo
        if hi is not None:
            bad |= s > hi
        n_bad = int(bad.sum())
        if n_bad > 0:
            sample = s[bad].head(3).tolist()
            result.add(
                IssueKind.OUT_OF_RANGE,
                IssueSeverity.ERROR,
                col,
                f"{n_bad} value(s) outside [{lo}, {hi}], sample: {sample}",
            )

    # 5. Identity columns can't be null
    for col in spec.identity & actual_cols:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            result.add(
                IssueKind.NULL_NOT_ALLOWED,
                IssueSeverity.ERROR,
                col,
                f"{n_null} null value(s) in identity column",
            )

    # 6. Composite key (property_id, unit_type, unit_mix_type, unit_garage)
    #    must be unique per snapshot period.
    key_cols = [c for c in COMPOSITE_KEY if c in actual_cols]
    if len(key_cols) >= 2:
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            result.add(
                IssueKind.DUPLICATE_KEY,
                IssueSeverity.ERROR,
                None,
                f"{n_dup} row(s) have duplicate composite key {tuple(key_cols)}",
            )

    return result


def _find_section(col: str, spec: ContractSpec) -> str:
    for section_name, cols in spec.sections.items():
        if col in cols:
            return section_name
    return "unknown"


def summarize(result: ValidationResult, *, max_issues: int = 25) -> str:
    """Format a ValidationResult as a human-readable summary."""
    lines = []
    status = "PASS" if result.is_valid else "FAIL"
    lines.append(f"Contract validation: {status}")
    lines.append(f"  errors:   {len(result.errors)}")
    lines.append(f"  warnings: {len(result.warnings)}")
    if result.issues:
        lines.append("")
        lines.append("Issues:")
        for issue in result.issues[:max_issues]:
            lines.append(f"  {issue}")
        if len(result.issues) > max_issues:
            lines.append(f"  ... and {len(result.issues) - max_issues} more")
    return "\n".join(lines)
