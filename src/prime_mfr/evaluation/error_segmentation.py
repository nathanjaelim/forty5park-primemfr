"""
error_segmentation.py
=====================
Segment OOF prediction errors to find where the remaining MAE concentrates.

Reads:
  artifacts/oof_predictions.parquet
  artifacts/stacking_scratch/df_with_static.parquet (for segment columns)

Writes:
  outputs/error_segmentation_report.md  (markdown report)
  outputs/segment_tables.json           (raw segment metrics for downstream use)

Segments analyzed:
  - rent_quartile
  - beds_int
  - sqft_bucket
  - age_bucket
  - sub_market (top 10 by row count)
  - zipcode (top 15 by row count)
  - year_built (decade buckets)

For each segment we compute:
  count, mean_actual_rent, MAE, MAPE, MedianAPE, bias (mean signed err),
  share of total |error|.

Plus a separate worst-residuals table (top 40 by |error|).
"""

# Copyright (c) 2026 Forty5 Park. All Rights Reserved.
# Proprietary and confidential. See LICENSE.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import prime_mfr.config as config


def _ensure_outputs_dir() -> Path:
    """Output to the project's artifacts/ folder (host-portable)."""
    p = config.ARTIFACTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _segment_metrics(group: pd.DataFrame, total_abs_err: float) -> dict:
    err = group["rent_pred"].to_numpy() - group["rent_actual"].to_numpy()
    abs_err = np.abs(err)
    ape = (
        abs_err
        / np.where(group["rent_actual"].to_numpy() > 0, group["rent_actual"].to_numpy(), np.nan)
        * 100.0
    )
    return {
        "n": int(len(group)),
        "mean_rent": float(group["rent_actual"].mean()),
        "MAE": float(abs_err.mean()),
        "MAPE": float(np.nanmean(ape)),
        "MedianAPE": float(np.nanmedian(ape)),
        "bias": float(err.mean()),
        "share_err_pct": float(abs_err.sum() / max(total_abs_err, 1e-9) * 100.0),
    }


def _table_for(
    df: pd.DataFrame,
    group_col: str,
    total_abs_err: float,
    sort_by: str = "share_err_pct",
    top_n: int | None = None,
) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col, observed=True):
        m = _segment_metrics(sub, total_abs_err)
        m[group_col] = key
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    cols = [group_col, "n", "mean_rent", "MAE", "MAPE", "MedianAPE", "bias", "share_err_pct"]
    out = out[cols].sort_values(sort_by, ascending=False)
    if top_n is not None:
        out = out.head(top_n)
    return out


def _format_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "(no rows)\n"
    out = df.copy()
    fmt = {
        "mean_rent": "${:,.0f}",
        "MAE": "${:,.0f}",
        "MAPE": "{:.2f}%",
        "MedianAPE": "{:.2f}%",
        "bias": "${:+,.0f}",
        "share_err_pct": "{:.1f}%",
    }
    for c, f in fmt.items():
        if c in out.columns:
            out[c] = out[c].map(f.format)
    return out.to_markdown(index=False) + "\n"


def main() -> int:
    out_dir = _ensure_outputs_dir()

    # ---- Load OOF + segment columns ----
    oof = pd.read_parquet(config.ARTIFACTS_DIR / "oof_predictions.parquet")
    df = pd.read_parquet(config.ARTIFACTS_DIR / "stacking_scratch" / "df_with_static.parquet")
    df["row_id"] = np.arange(len(df))

    keep = [
        "row_id",
        "beds_int",
        "baths_int",
        "sqft",
        "sqft_bucket",
        "age_bucket",
        "property_age",
        "sub_market",
        "zipcode",
        "year_built",
        "property_id",
    ]
    df = df[[c for c in keep if c in df.columns]]

    merged = oof.merge(df, on="row_id", how="left", suffixes=("", "_x"))
    if "property_id_x" in merged.columns:
        merged = merged.drop(columns=["property_id_x"])

    # rent_quartile (Q1 lowest, Q4 highest) — useful for seeing if luxury is the pain.
    merged["rent_quartile"] = pd.qcut(
        merged["rent_actual"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_luxury"]
    ).astype(str)

    # year_built decade.
    if "year_built" in merged.columns:
        yb = merged["year_built"].fillna(-1).astype(int)
        merged["year_decade"] = np.where(
            yb < 0,
            "missing",
            (yb // 10 * 10).astype(str) + "s",
        )

    # ---- Aggregate metrics ----
    err = merged["rent_pred"].to_numpy() - merged["rent_actual"].to_numpy()
    abs_err = np.abs(err)
    total_abs_err = float(abs_err.sum())
    overall = {
        "n": int(len(merged)),
        "mean_rent": float(merged["rent_actual"].mean()),
        "MAE": float(abs_err.mean()),
        "MAPE": float(np.mean(abs_err / merged["rent_actual"].to_numpy() * 100.0)),
        "MedianAPE": float(np.median(abs_err / merged["rent_actual"].to_numpy() * 100.0)),
        "bias": float(err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }

    # ---- Per-segment tables ----
    tables: dict[str, pd.DataFrame] = {}
    tables["rent_quartile"] = _table_for(
        merged, "rent_quartile", total_abs_err, sort_by="rent_quartile"
    )
    tables["beds_int"] = _table_for(merged, "beds_int", total_abs_err, sort_by="beds_int")
    if "sqft_bucket" in merged.columns:
        tables["sqft_bucket"] = _table_for(
            merged, "sqft_bucket", total_abs_err, sort_by="sqft_bucket"
        )
    if "age_bucket" in merged.columns:
        tables["age_bucket"] = _table_for(merged, "age_bucket", total_abs_err, sort_by="age_bucket")
    if "year_decade" in merged.columns:
        tables["year_decade"] = _table_for(
            merged, "year_decade", total_abs_err, sort_by="year_decade"
        )
    tables["sub_market_top10_by_share"] = _table_for(merged, "sub_market", total_abs_err, top_n=10)
    tables["zipcode_top15_by_share"] = _table_for(merged, "zipcode", total_abs_err, top_n=15)

    # Worst per-base vs stack — see whether each base would have been worse.
    base_cols = [c for c in oof.columns if c.startswith("rent_pred_") and c != "rent_pred_stack"]
    base_summary = []
    for c in base_cols:
        e = (merged[c] - merged["rent_actual"]).to_numpy()
        base_summary.append(
            {
                "model": c.replace("rent_pred_", ""),
                "MAE": float(np.abs(e).mean()),
                "bias": float(e.mean()),
                "MedianAPE": float(np.median(np.abs(e) / merged["rent_actual"] * 100.0)),
            }
        )
    base_summary.append(
        {
            "model": "STACK",
            "MAE": overall["MAE"],
            "bias": overall["bias"],
            "MedianAPE": overall["MedianAPE"],
        }
    )
    base_df = pd.DataFrame(base_summary)

    # ---- Worst residuals ----
    worst = merged.assign(
        err=err,
        abs_err=abs_err,
        ape=abs_err / merged["rent_actual"] * 100.0,
    ).nlargest(40, "abs_err")[
        [
            "row_id",
            "property_id",
            "sub_market",
            "zipcode",
            "beds_int",
            "baths_int",
            "sqft",
            "year_built",
            "rent_actual",
            "rent_pred",
            "err",
            "abs_err",
            "ape",
        ]
    ]

    # ---- Write report ----
    md = []
    md.append("# Error Segmentation Report\n")
    md.append(
        "Stack: 4-base (lgbm_l1 + cat_q50 + knn_geo + knn_lean) "
        "with hierarchical comparable TE and tuned LightGBM.\n"
    )
    md.append("## Overall metrics\n")
    md.append(f"- n: **{overall['n']:,}**\n")
    md.append(f"- mean rent: **${overall['mean_rent']:,.0f}**\n")
    md.append(f"- MAE: **${overall['MAE']:,.2f}**\n")
    md.append(f"- MAPE: **{overall['MAPE']:.2f}%**\n")
    md.append(f"- MedianAPE: **{overall['MedianAPE']:.2f}%**\n")
    md.append(f"- bias (mean signed err): **${overall['bias']:+,.2f}**\n")
    md.append(f"- RMSE: **${overall['RMSE']:,.2f}**\n")

    md.append("\n## Per-base summary\n")
    md.append(_format_table(base_df))

    for name, tbl in tables.items():
        md.append(f"\n## By {name}\n")
        md.append(_format_table(tbl))

    md.append("\n## Worst 40 residuals (by absolute error)\n")
    w = worst.copy()
    for c in ("rent_actual", "rent_pred", "err", "abs_err"):
        w[c] = w[c].map(lambda v: f"${v:+,.0f}")
    w["ape"] = w["ape"].map(lambda v: f"{v:.1f}%")
    if "sqft" in w.columns:
        w["sqft"] = w["sqft"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
    md.append(w.to_markdown(index=False) + "\n")

    md.append("\n## Diagnostic notes\n")
    md.append(
        "- **share_err_pct** = sum of |error| in segment / total |error|. "
        "Segments contributing >10% deserve investigation; high mean APE + "
        "low share = noise; high MAE + high share = systematic miss.\n"
    )
    md.append("- **bias** > 0 means the model over-predicts; < 0 means it under-predicts.\n")
    md.append(
        "- A bias |·| > 10% of segment mean rent indicates a structural under/over-prediction "
        "(suggests adding a feature for that segment).\n"
    )

    report_path = out_dir / "error_segmentation_report.md"
    report_path.write_text("".join(md))

    # Also dump raw tables as JSON for downstream tooling.
    json_payload = {
        "overall": overall,
        "tables": {k: v.to_dict(orient="records") for k, v in tables.items()},
        "base_summary": base_df.to_dict(orient="records"),
        "worst_40": worst.to_dict(orient="records"),
    }
    (out_dir / "segment_tables.json").write_text(json.dumps(json_payload, indent=2, default=str))

    print(f"Report written to {report_path}")
    print(f"Tables JSON written to {out_dir / 'segment_tables.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
