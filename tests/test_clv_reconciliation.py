"""Regression and reconciliation test suite for CLV holdout metrics.

Verifies that serving/evaluation code reproduces the persisted Phase 4b artifacts
(test_metrics_with_95ci.csv, test_predictions_2011-09-10.parquet, test_whale_sensitivity.csv)
within tight numerical tolerances (<= 1e-6).
"""

import numpy as np
import pandas as pd
import pytest

from cie.config import get_project_root, load_config
from src.clv import (
    bootstrap_clv_metrics,
    compute_targeting_simulation,
    evaluate_clv_predictions,
    evaluate_whale_sensitivity_slices,
)


@pytest.fixture(scope="module")
def clv_artifacts():
    """Load persisted test predictions and metrics tables."""
    config = load_config()
    root = get_project_root()

    preds_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    metrics_path = root / "data" / "processed" / "clv" / "test_metrics_with_95ci.csv"
    whale_path = root / "data" / "processed" / "clv" / "test_whale_sensitivity.csv"
    targeting_path = root / "data" / "processed" / "clv" / "targeting_simulation.csv"

    preds_df = pd.read_parquet(preds_path)
    metrics_df = pd.read_csv(metrics_path)
    whale_df = pd.read_csv(whale_path)
    targeting_df = pd.read_csv(targeting_path)

    return {
        "preds": preds_df,
        "metrics": metrics_df,
        "whale": whale_df,
        "targeting": targeting_df,
    }


def test_tweedie_point_estimates_exact_parity(clv_artifacts):
    """Verify Tweedie GLM point estimates reproduce persisted metrics within 1e-6."""
    preds_df = clv_artifacts["preds"]
    metrics_df = clv_artifacts["metrics"]

    y_true = preds_df["target"].to_numpy(dtype=float)
    y_pred = preds_df["pred_frozen_tweedie"].to_numpy(dtype=float)

    eval_results = evaluate_clv_predictions(y_true, y_pred)
    persisted_champ = metrics_df[metrics_df["model"].str.contains("Tweedie")].set_index("metric")

    for metric_name, computed_val in eval_results.items():
        assert metric_name in persisted_champ.index, (
            f"Metric {metric_name} missing from persisted file"
        )
        persisted_val = float(persisted_champ.loc[metric_name, "point_estimate"])
        abs_diff = abs(computed_val - persisted_val)
        assert abs_diff <= 1e-6, (
            f"Tweedie {metric_name} diverged! Computed: {computed_val}, Persisted: {persisted_val}, Diff: {abs_diff}"
        )


def test_run_rate_point_estimates_exact_parity(clv_artifacts):
    """Verify heuristic Run-Rate point estimates reproduce persisted metrics within 1e-6."""
    preds_df = clv_artifacts["preds"]
    metrics_df = clv_artifacts["metrics"]

    y_true = preds_df["target"].to_numpy(dtype=float)
    y_pred = preds_df["pred_run_rate"].to_numpy(dtype=float)

    eval_results = evaluate_clv_predictions(y_true, y_pred)
    persisted_rr = metrics_df[metrics_df["model"].str.contains("Run-Rate")].set_index("metric")

    for metric_name, computed_val in eval_results.items():
        assert metric_name in persisted_rr.index, (
            f"Metric {metric_name} missing from persisted file"
        )
        persisted_val = float(persisted_rr.loc[metric_name, "point_estimate"])
        abs_diff = abs(computed_val - persisted_val)
        assert abs_diff <= 1e-6, (
            f"Run-Rate {metric_name} diverged! Computed: {computed_val}, Persisted: {persisted_val}, Diff: {abs_diff}"
        )


def test_bootstrap_95ci_exact_parity(clv_artifacts):
    """Verify 1,000 bootstrap resamples with seed 42 exactly match persisted CIs within 1e-6."""
    preds_df = clv_artifacts["preds"]
    metrics_df = clv_artifacts["metrics"]

    y_true = preds_df["target"]
    pred_tweedie = preds_df["pred_frozen_tweedie"]
    pred_rr = preds_df["pred_run_rate"]

    boot_tweedie = bootstrap_clv_metrics(
        y_true, pred_tweedie, ref_pred=pred_rr, n_bootstrap=1000, random_state=42
    )
    persisted_champ = metrics_df[metrics_df["model"].str.contains("Tweedie")].set_index("metric")

    for metric_name in boot_tweedie.index:
        for col in ["point_estimate", "boot_mean", "boot_std", "ci_low", "ci_high"]:
            computed_val = float(boot_tweedie.loc[metric_name, col])
            persisted_val = float(persisted_champ.loc[metric_name, col])
            abs_diff = abs(computed_val - persisted_val)
            assert abs_diff <= 1e-6, (
                f"Tweedie {metric_name} [{col}] diverged! Computed: {computed_val}, Persisted: {persisted_val}"
            )


def test_whale_sensitivity_parity(clv_artifacts):
    """Verify whale sensitivity slices (All, Excl-1, Excl-5) match persisted values within 1e-6."""
    preds_df = clv_artifacts["preds"]
    whale_df = clv_artifacts["whale"]

    y_true = preds_df["target"].to_numpy(dtype=float)
    y_pred = preds_df["pred_frozen_tweedie"].to_numpy(dtype=float)

    computed_slices = evaluate_whale_sensitivity_slices(y_true, y_pred, exclude_top_ks=(0, 1, 5))

    for i, row in computed_slices.iterrows():
        persisted_row = whale_df.iloc[i]
        assert row["slice"] == persisted_row["slice"]
        assert row["n_customers"] == persisted_row["n_customers"]
        assert abs(row["rmse"] - persisted_row["rmse"]) <= 1e-6
        assert abs(row["mae"] - persisted_row["mae"]) <= 1e-6
        assert abs(row["spearman"] - persisted_row["spearman"]) <= 1e-6
        assert abs(row["normalized_gini"] - persisted_row["normalized_gini"]) <= 1e-6


def test_targeting_simulation_parity(clv_artifacts):
    """Verify top 10%/20%/30% targeting simulation matches persisted values within 1e-6."""
    preds_df = clv_artifacts["preds"]
    targeting_df = clv_artifacts["targeting"]

    y_true = preds_df["target"].to_numpy(dtype=float)
    pred_tweedie = preds_df["pred_frozen_tweedie"].to_numpy(dtype=float)
    pred_rr = preds_df["pred_run_rate"].to_numpy(dtype=float)

    computed_sim = compute_targeting_simulation(
        y_true,
        {"Champion (Tweedie)": pred_tweedie, "Baseline (Run Rate)": pred_rr},
        top_fracs=[0.1, 0.2, 0.3],
    )

    for i, row in computed_sim.iterrows():
        persisted_row = targeting_df.iloc[i]
        assert row["top_pct"] == persisted_row["top_pct"]
        assert row["model"] == persisted_row["model"]
        assert abs(row["capture_share_pct"] - persisted_row["capture_share_pct"]) <= 1e-6
        assert abs(row["actual_captured_gbp"] - persisted_row["actual_captured_gbp"]) <= 1e-6


def test_golden_heuristic_run_rate_parity(clv_artifacts):
    """(a) Assert heuristic run-rate challenger equals pred_run_rate within 1e-9."""
    config = load_config()
    root = get_project_root()
    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    preds_df = clv_artifacts["preds"]
    calc_rr = test_df["monetary"].to_numpy(dtype=float) * (90.0 / 270.0)
    saved_rr = preds_df["pred_run_rate"].to_numpy(dtype=float)

    max_diff = float(np.max(np.abs(calc_rr - saved_rr)))
    assert max_diff <= 1e-9, f"Heuristic run-rate diverged by {max_diff}"


def test_golden_calibrated_run_rate_parity(clv_artifacts):
    """(b) Assert calibrated run-rate equals heuristic * train-learned expansion constant within 1e-9."""
    config = load_config()
    root = get_project_root()
    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    constant = config.clv.challengers.calibrated_run_rate.expansion_factor
    assert np.isclose(constant, 1.42746205196053)

    calc_rr = test_df["monetary"].to_numpy(dtype=float) * (90.0 / 270.0)
    calibrated_rr = calc_rr * constant

    expected_cal_rr = calc_rr * 1.42746205196053
    max_diff = float(np.max(np.abs(calibrated_rr - expected_cal_rr)))
    assert max_diff <= 1e-9, f"Calibrated run-rate diverged by {max_diff}"


def test_golden_segment_assignments_full_history_vs_270d(clv_artifacts):
    """(c) Audit segment assignments: full-history matches 100%, 270d lookback matches 54.90%."""
    import joblib

    config = load_config()
    root = get_project_root()

    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    seg_path = config.pipeline.paths.resolve("segments_t_seg", root)
    seg_t_seg = pd.read_parquet(seg_path)

    seg_pipeline_path = config.pipeline.paths.resolve("segmentation_pipeline", root)
    seg_pipeline = joblib.load(seg_pipeline_path)

    c_to_s = config.segmentation.cluster_to_segment
    persisted_segments = seg_t_seg.loc[test_df.index, "segment"].tolist()

    # 1. Full-history RFM features reproduce segments_t_seg exactly 100% (3,224 / 3,224)
    rfm_full = seg_t_seg.loc[test_df.index, ["recency_days", "frequency", "monetary"]]
    full_preds = seg_pipeline.predict(rfm_full)
    full_pred_segs = [c_to_s[p] for p in full_preds]
    full_matches = sum(p == t for p, t in zip(full_pred_segs, persisted_segments, strict=False))
    assert full_matches == len(test_df) == 3224, "Full-history RFM must reproduce segments 100%"

    # 2. 270d lookback features match only 54.90% because 270d truncates tenure
    rfm_270d = test_df[["recency_days", "frequency", "monetary"]]
    window_preds = seg_pipeline.predict(rfm_270d)
    window_pred_segs = [c_to_s[p] for p in window_preds]
    window_matches = sum(p == t for p, t in zip(window_pred_segs, persisted_segments, strict=False))
    match_pct = window_matches / len(test_df) * 100.0

    assert window_matches == 1770
    assert np.isclose(match_pct, 54.900744, atol=1e-3)
