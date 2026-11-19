"""Cohort backtest report generator for as-of 2011-09-10 using src/clv.py and persisted CSVs."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cie.config import AppConfig, get_project_root, load_config
from src.clv import (
    calibration_by_predicted_decile,
    compute_slice_error_table,
)


def build_backtest_report(
    config: AppConfig | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate model-agnostic 2011-09-10 backtest report from persisted CSVs and src/clv.py.

    Uses persisted CSVs directly for all point/CI metrics and whale concentration so zero
    recomputation drift can occur, while calling canonical src/clv.py functions for
    decile calibration and sub-cohort slice diagnostics.
    """
    if config is None:
        config = load_config()
    if root is None:
        root = get_project_root()

    clv_dir = root / "data" / "processed" / "clv"
    feat_path = config.pipeline.paths.resolve("features_test", root)
    preds_path = config.pipeline.paths.resolve("clv_test_predictions", root)

    feats_df = pd.read_parquet(feat_path)
    preds_df = pd.read_parquet(preds_path)

    expansion_factor = float(config.clv.challengers.calibrated_run_rate.expansion_factor)
    preds_df = preds_df.copy()
    preds_df["pred_calibrated_run_rate"] = preds_df["pred_run_rate"] * expansion_factor

    # 1. Load persisted metrics with 95% CIs (exact source of truth)
    tweedie_rr_ci_df = pd.read_csv(clv_dir / "test_metrics_with_95ci.csv")
    cal_rr_ci_df = pd.read_csv(clv_dir / "calibrated_run_rate_test_metrics_with_95ci.csv")
    whale_conc_df = pd.read_csv(clv_dir / "test_whale_concentration.csv")
    whale_sens_df = pd.read_csv(clv_dir / "test_whale_sensitivity.csv")
    seg_slices_df = pd.read_csv(clv_dir / "test_segment_error_slices.csv")

    # 2. Model-agnostic decile calibration across all 3 models using src.clv
    decile_frames = []
    model_cols = {
        "champion_tweedie": "pred_frozen_tweedie",
        "calibrated_run_rate": "pred_calibrated_run_rate",
        "heuristic_run_rate": "pred_run_rate",
    }
    for model_label, col in model_cols.items():
        cal = calibration_by_predicted_decile(preds_df["target"], preds_df[col]).reset_index()
        cal.insert(0, "model", model_label)
        decile_frames.append(cal)
    decile_df = pd.concat(decile_frames, ignore_index=True)

    # 3. Whale concentration & Top-5 actual vs Top-5 squared error coincidence
    top5_actual = preds_df.sort_values("target", ascending=False).head(5)
    top5_actual_ids = [int(x) for x in top5_actual.index]

    coincidence_details: dict[str, Any] = {}
    for model_label, col in model_cols.items():
        sq_err = (preds_df[col] - preds_df["target"]) ** 2
        top5_sq = sq_err.sort_values(ascending=False).head(5)
        top5_sq_ids = [int(x) for x in top5_sq.index]
        total_sse = float(sq_err.sum())
        top5_sse = float(top5_sq.sum())
        coincidence_details[model_label] = {
            "top5_by_actual_target_ids": top5_actual_ids,
            "top5_by_squared_error_ids": top5_sq_ids,
            "coincide_as_sets": set(top5_actual_ids) == set(top5_sq_ids),
            "coincide_in_order": top5_actual_ids == top5_sq_ids,
            "total_sse": total_sse,
            "top5_sq_err_sse": top5_sse,
            "top5_sq_err_share_pct": (top5_sse / total_sse * 100.0) if total_sse > 0 else 0.0,
        }

    # 4. Additional slice evaluations using original src/clv.compute_slice_error_table
    joined = preds_df.join(feats_df[["is_single_invoice", "is_uk"]], how="left")
    joined["whale_group"] = [
        "Top-5 Actual Whales" if int(cid) in top5_actual_ids else "Core Non-Whales (n=3219)"
        for cid in joined.index
    ]
    joined["purchase_cadence"] = joined["is_single_invoice"].map(
        {1: "Single-Invoice", 0: "Repeat Buyer"}
    )
    joined["geography"] = joined["is_uk"].map({1: "UK", 0: "Export (Non-UK)"})

    slice_frames = []
    for dim_name, group_col in [
        ("whale_vs_non_whale", "whale_group"),
        ("repeat_vs_single_invoice", "purchase_cadence"),
        ("uk_vs_export", "geography"),
    ]:
        # 1. Champion (Tweedie) + Heuristic Run-Rate baseline summary
        tbl = compute_slice_error_table(
            joined,
            group_col=group_col,
            y_true_col="target",
            y_pred_col="pred_frozen_tweedie",
            baseline_col="pred_run_rate",
        )
        # 2. Full metric suite for Calibrated Run-Rate using canonical src.clv.compute_slice_error_table
        cal_tbl = compute_slice_error_table(
            joined,
            group_col=group_col,
            y_true_col="target",
            y_pred_col="pred_calibrated_run_rate",
            baseline_col=None,
            sort_by=None,
        )
        # 3. Remaining Run-Rate columns (medae, spearman, normalized_gini, pred_sum_gbp, pred_mean_gbp, mean_bias)
        rr_tbl = compute_slice_error_table(
            joined,
            group_col=group_col,
            y_true_col="target",
            y_pred_col="pred_run_rate",
            baseline_col=None,
            sort_by=None,
        )
        model_metric_cols = [
            "pred_sum_gbp",
            "pred_mean_gbp",
            "total_rev_error_pct",
            "mean_bias_pred_minus_true",
            "mae",
            "medae",
            "rmse",
            "spearman",
            "normalized_gini",
        ]
        for col_name in model_metric_cols:
            out_col = (
                "cal_rr_rev_err_pct" if col_name == "total_rev_error_pct" else f"cal_rr_{col_name}"
            )
            tbl[out_col] = cal_tbl.loc[tbl.index, col_name]

        for col_name in [
            "pred_sum_gbp",
            "pred_mean_gbp",
            "mean_bias_pred_minus_true",
            "medae",
            "spearman",
            "normalized_gini",
        ]:
            tbl[f"run_rate_{col_name}"] = rr_tbl.loc[tbl.index, col_name]

        tbl = tbl.reset_index()
        tbl.insert(0, "slice_dimension", dim_name)
        slice_frames.append(tbl)
    extra_slices_df = pd.concat(slice_frames, ignore_index=True)

    # Assemble JSON report
    report: dict[str, Any] = {
        "cohort_as_of": "2011-09-10",
        "n_customers": int(len(preds_df)),
        "label_lag_note": (
            "Ground-truth 90-day spending labels arrive 90 days after cutoff [as_of, as_of + 90d). "
            "Feature and prediction drift serve as leading indicators, while cohort backtesting "
            "serves as the 90-day lagging indicator."
        ),
        "persisted_metrics_with_95ci": {
            "tweedie_and_run_rate": tweedie_rr_ci_df.to_dict(orient="records"),
            "calibrated_run_rate": cal_rr_ci_df.to_dict(orient="records"),
        },
        "whale_concentration": {
            "persisted_csv_rows": whale_conc_df.to_dict(orient="records"),
            "top5_actual_vs_squared_error_coincidence": coincidence_details,
        },
        "whale_sensitivity_persisted": whale_sens_df.to_dict(orient="records"),
        "decile_calibration": decile_df.to_dict(orient="records"),
        "segment_slices_persisted": seg_slices_df.to_dict(orient="records"),
        "cohort_slices": extra_slices_df.to_dict(orient="records"),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(
            output_dir / "backtest_report_2011-09-10.json", "w", encoding="utf-8", newline="\n"
        ) as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        decile_df.to_csv(
            output_dir / "backtest_decile_calibration.csv", index=False, lineterminator="\n"
        )
        extra_slices_df.to_csv(
            output_dir / "backtest_cohort_slices.csv", index=False, lineterminator="\n"
        )

    return report
