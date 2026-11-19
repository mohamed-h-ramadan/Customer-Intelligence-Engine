"""Regression and smoke tests for Step 8 Phase A backtest and monitoring report generation."""

import json
import sys

import numpy as np
import pandas as pd

from cie.config import get_project_root
from cie.monitoring.backtest import build_backtest_report


def test_backtest_report_matches_persisted_csvs():
    """Assert backtest_report_2011-09-10.json numbers exactly match persisted CLV CSVs."""
    root = get_project_root()
    mon_dir = root / "data" / "processed" / "monitoring"
    clv_dir = root / "data" / "processed" / "clv"

    report_path = mon_dir / "backtest_report_2011-09-10.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # 1. Tweedie + Run-Rate CIs match test_metrics_with_95ci.csv
    exp_tweedie_df = pd.read_csv(clv_dir / "test_metrics_with_95ci.csv")
    rep_tweedie_df = pd.DataFrame(report["persisted_metrics_with_95ci"]["tweedie_and_run_rate"])
    pd.testing.assert_frame_equal(rep_tweedie_df, exp_tweedie_df)

    # 2. Calibrated Run-Rate CIs match calibrated_run_rate_test_metrics_with_95ci.csv
    exp_cal_df = pd.read_csv(clv_dir / "calibrated_run_rate_test_metrics_with_95ci.csv")
    rep_cal_df = pd.DataFrame(report["persisted_metrics_with_95ci"]["calibrated_run_rate"])
    pd.testing.assert_frame_equal(rep_cal_df, exp_cal_df)

    # 3. Whale concentration matches test_whale_concentration.csv & coincidence flags
    exp_whale_df = pd.read_csv(clv_dir / "test_whale_concentration.csv")
    rep_whale_df = pd.DataFrame(report["whale_concentration"]["persisted_csv_rows"])
    pd.testing.assert_frame_equal(rep_whale_df, exp_whale_df)

    coinc = report["whale_concentration"]["top5_actual_vs_squared_error_coincidence"]
    assert coinc["champion_tweedie"]["top5_by_actual_target_ids"] == [
        18102,
        17450,
        14646,
        14911,
        14096,
    ]
    assert coinc["champion_tweedie"]["top5_by_squared_error_ids"] == [
        18102,
        17450,
        14096,
        14646,
        14911,
    ]
    assert coinc["champion_tweedie"]["coincide_as_sets"] is True
    assert coinc["champion_tweedie"]["coincide_in_order"] is False

    # 4. Whale sensitivity & Segment slices match persisted CSVs
    exp_sens_df = pd.read_csv(clv_dir / "test_whale_sensitivity.csv")
    rep_sens_df = pd.DataFrame(report["whale_sensitivity_persisted"])
    pd.testing.assert_frame_equal(rep_sens_df, exp_sens_df)

    exp_seg_df = pd.read_csv(clv_dir / "test_segment_error_slices.csv")
    rep_seg_df = pd.DataFrame(report["segment_slices_persisted"])
    pd.testing.assert_frame_equal(rep_seg_df, exp_seg_df)

    # 5. Cohort slices include symmetric Calibrated Run-Rate and Heuristic Run-Rate columns
    slices_df = pd.read_csv(mon_dir / "backtest_cohort_slices.csv")
    assert set(slices_df["slice_dimension"]) == {
        "whale_vs_non_whale",
        "repeat_vs_single_invoice",
        "uk_vs_export",
    }
    for prefix in ["", "cal_rr_", "run_rate_"]:
        for mcol in [
            "pred_sum_gbp",
            "pred_mean_gbp",
            "mae",
            "medae",
            "rmse",
            "spearman",
            "normalized_gini",
        ]:
            col_name = f"{prefix}{mcol}"
            assert col_name in slices_df.columns, f"Missing slice column {col_name}"
    assert "cal_rr_rev_err_pct" in slices_df.columns
    assert "run_rate_rev_err_pct" in slices_df.columns

    # Verify Core Non-Whales slice matches persisted tweedie_rmse_excl_top5_gbp
    core_row = slices_df[slices_df["slice"] == "Core Non-Whales (n=3219)"].iloc[0]
    persisted_excl5 = float(
        exp_whale_df.loc[exp_whale_df["metric"] == "tweedie_rmse_excl_top5_gbp", "value"].iloc[0]
    )
    assert np.isclose(core_row["rmse"], persisted_excl5, atol=1e-6)


def test_prediction_and_segment_mix_drift_artifacts():
    """Verify prediction drift covers all 3 models and segment mix sums to 100%."""
    root = get_project_root()
    mon_dir = root / "data" / "processed" / "monitoring"

    pred_df = pd.read_csv(mon_dir / "prediction_drift_report.csv")
    assert set(pred_df["model_column"]) == {
        "champion_tweedie",
        "calibrated_run_rate",
        "heuristic_run_rate",
    }
    assert (pred_df["psi"] < 0.10).all()

    seg_df = pd.read_csv(mon_dir / "segment_mix_drift.csv")
    assert len(seg_df) == 5
    assert np.isclose(seg_df["ref_share_pct"].sum(), 100.0, atol=1e-6)
    assert np.isclose(seg_df["cur_share_pct"].sum(), 100.0, atol=1e-6)


def test_monitoring_report_generation_smoke(tmp_path):
    """Smoke test full Phase A monitoring runner writing artifacts to an isolated directory."""
    root = get_project_root()
    scripts_dir = str(root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from run_monitoring_phase_a import run_phase_a

    results = run_phase_a(root=root, output_dir=tmp_path, rebuild_deploy_features=False)
    assert len(results["feature_drift"]) == 100
    assert len(results["prediction_drift"]) == 3

    expected_files = [
        "feature_drift_report.csv",
        "prediction_drift_report.csv",
        "segment_mix_drift.csv",
        "backtest_report_2011-09-10.json",
        "backtest_decile_calibration.csv",
        "backtest_cohort_slices.csv",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).is_file(), f"Missing generated monitoring output: {fname}"

    # Verify build_backtest_report also runs cleanly standalone
    bt = build_backtest_report(root=root, output_dir=tmp_path)
    assert bt["cohort_as_of"] == "2011-09-10"
    assert bt["n_customers"] == 3224
