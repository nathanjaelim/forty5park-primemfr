# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.
"""Tests for the feature-pipeline contract validator (DE handoff)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prime_mfr.features import contract as feat_contract
from prime_mfr.features.contract import (
    ContractSpec,
    IssueKind,
    IssueSeverity,
    validate_pre_engineered,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pre_engineered_sample.parquet"


# ---------------------------------------------------------------------------
# Fixture access
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """The canonical pre-engineered DE-handoff sample (200 rows)."""
    if not FIXTURE.exists():
        pytest.skip(
            f"Fixture {FIXTURE.name} not generated; regenerate via the snippet in "
            f"docs/feature_pipeline_contract.md."
        )
    return pd.read_parquet(FIXTURE)


def test_fixture_exists():
    """The canonical pre-engineered fixture is in tests/fixtures/."""
    assert FIXTURE.exists(), (
        f"Missing {FIXTURE}. Regenerate it via the snippet in "
        f"docs/feature_pipeline_contract.md."
    )


# ---------------------------------------------------------------------------
# Contract spec
# ---------------------------------------------------------------------------


def test_contract_spec_resolves_from_yaml():
    """ContractSpec loads required columns from configs/features/*.yaml."""
    spec = ContractSpec()
    assert "sqft" in spec.sections["numeric"]
    assert "hist_rent_lag_1m" in spec.sections["numeric"]
    assert "sub_market" in spec.sections["categorical"]
    assert "has_clubhouse" in spec.sections["booleans"]
    assert "property_quality" in spec.sections["ordinal_grade"]
    # Identity columns are always required
    assert "property_id" in spec.identity
    assert "unit_type" in spec.identity


def test_all_required_columns_includes_identity_and_features():
    spec = ContractSpec()
    required = spec.all_required_columns
    assert "property_id" in required
    assert "sqft" in required
    assert "sub_market" in required
    assert "hist_rent_lag_1m" in required


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_canonical_fixture_passes_validation(sample_df):
    """The canonical fixture is the gold standard — it must validate."""
    result = validate_pre_engineered(sample_df)
    assert result.is_valid, (
        "Canonical fixture should validate. Errors found:\n"
        + "\n".join(f"  {e}" for e in result.errors[:10])
    )
    # Warnings are OK — extra columns from the static pipeline are expected.
    assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Negative tests — each error type
# ---------------------------------------------------------------------------


def test_missing_required_column_is_error(sample_df):
    df = sample_df.drop(columns=["sqft"])
    result = validate_pre_engineered(df)
    assert not result.is_valid
    assert any(
        e.kind == IssueKind.MISSING_COLUMN and e.column == "sqft" for e in result.errors
    )


def test_wrong_dtype_on_numeric_is_error(sample_df):
    df = sample_df.copy()
    df["sqft"] = df["sqft"].astype(str)  # cast to string
    result = validate_pre_engineered(df)
    assert not result.is_valid
    assert any(
        e.kind == IssueKind.WRONG_DTYPE and e.column == "sqft" for e in result.errors
    )


def test_out_of_range_latitude_is_error(sample_df):
    df = sample_df.copy()
    df.loc[df.index[:3], "latitude"] = 40.0  # New York, way outside Atlanta MSA
    result = validate_pre_engineered(df)
    assert not result.is_valid
    err = next(e for e in result.errors if e.column == "latitude")
    assert err.kind == IssueKind.OUT_OF_RANGE
    assert "40.0" in err.message or "[40.0" in err.message


def test_out_of_range_sqft_is_error(sample_df):
    df = sample_df.copy()
    df.loc[df.index[0], "sqft"] = -1.0  # impossible
    result = validate_pre_engineered(df)
    assert not result.is_valid
    assert any(e.column == "sqft" and e.kind == IssueKind.OUT_OF_RANGE for e in result.errors)


def test_duplicate_composite_key_is_error(sample_df):
    df = sample_df.copy()
    # Force a true duplicate of the FIRST row on all 4 key columns.
    dup = df.iloc[0].copy()
    df = pd.concat([df, pd.DataFrame([dup])], ignore_index=True)
    result = validate_pre_engineered(df)
    assert not result.is_valid
    assert any(e.kind == IssueKind.DUPLICATE_KEY for e in result.errors)


def test_null_in_identity_column_is_error(sample_df):
    df = sample_df.copy()
    df.loc[df.index[0], "property_id"] = None
    result = validate_pre_engineered(df)
    assert not result.is_valid
    assert any(
        e.column == "property_id" and e.kind == IssueKind.NULL_NOT_ALLOWED
        for e in result.errors
    )


def test_extra_columns_are_warnings_by_default(sample_df):
    df = sample_df.copy()
    df["some_de_team_metadata"] = "x"
    result = validate_pre_engineered(df, strict=False)
    assert result.is_valid  # warnings don't fail
    assert any(
        i.column == "some_de_team_metadata" and i.severity == IssueSeverity.WARNING
        for i in result.warnings
    )


def test_extra_columns_in_strict_mode_are_errors(sample_df):
    df = sample_df.copy()
    df["some_unexpected_column"] = "x"
    result = validate_pre_engineered(df, strict=True)
    assert not result.is_valid
    assert any(e.column == "some_unexpected_column" for e in result.errors)


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------


def test_summarize_pass(sample_df):
    result = validate_pre_engineered(sample_df)
    text = feat_contract.summarize(result, max_issues=5)
    assert "PASS" in text
    assert "errors:   0" in text


def test_summarize_fail():
    """summarize() shows errors when validation fails."""
    df = pd.DataFrame({"property_id": ["X"]})  # missing everything
    result = validate_pre_engineered(df)
    text = feat_contract.summarize(result, max_issues=3)
    assert "FAIL" in text
    assert "missing_column" in text
