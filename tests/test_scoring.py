"""Unit, property, and golden-file verification tests for cie.core.scoring."""

import numpy as np
import pandas as pd
import pytest

from cie.config import get_project_root
from cie.core.scoring import score_customers, score_customers_batch


@pytest.fixture
def sample_test_features():
    """Load real holdout test features."""
    root = get_project_root()
    test_path = root / "data" / "processed" / "features" / "features_test_2011-09-10.parquet"
    return pd.read_parquet(test_path)


def test_golden_file_test_predictions_parity(sample_test_features):
    """Assert serving code reproduces saved test predictions within 1e-9 tolerance across all 3,224 rows."""
    root = get_project_root()
    golden_path = root / "data" / "processed" / "clv" / "test_predictions_2011-09-10.parquet"
    golden_df = pd.read_parquet(golden_path)

    # Score full test cohort using framework-agnostic serving code
    responses = score_customers(sample_test_features, as_of="2011-09-10", lookback_days=270)
    assert len(responses) == len(golden_df) == 3224

    champion_preds = np.array([r.expected_clv for r in responses])
    run_rate_preds = np.array([r.challengers.run_rate for r in responses])

    max_champ_diff = float(np.max(np.abs(champion_preds - golden_df["pred_frozen_tweedie"].values)))
    max_rr_diff = float(np.max(np.abs(run_rate_preds - golden_df["pred_run_rate"].values)))

    assert max_champ_diff <= 1e-9, f"Champion serving diverged: max diff {max_champ_diff}"
    assert max_rr_diff <= 1e-9, f"Run-rate serving diverged: max diff {max_rr_diff}"


def test_scoring_single_customer_response_structure(sample_test_features):
    """Assert individual scoring response schema and challenger details."""
    single_df = sample_test_features.head(1).copy()
    responses = score_customers(single_df, as_of="2011-09-10", lookback_days=270)

    assert len(responses) == 1
    resp = responses[0]

    assert resp.customer_id == single_df.index[0]
    assert resp.expected_clv > 0.0
    assert resp.model_version == "v1.0-frozen-2010-09-10"
    assert resp.segment in [
        "VIP Champions",
        "Loyal Regulars",
        "Promising Recent Buyers",
        "At-Risk Repeaters",
        "Hibernating / Lost",
    ]
    assert len(resp.crm_action) > 10

    # Challengers presence
    assert resp.challengers.run_rate >= 0.0
    assert resp.challengers.calibrated_run_rate > resp.challengers.run_rate
    assert round(resp.challengers.train_mean, 2) == 752.81
    assert round(resp.challengers.train_median, 2) == 222.08


def test_regime_warnings_on_out_of_range_and_season(sample_test_features):
    """Assert explicit warnings are generated for out-of-range features and out-of-season cutoffs."""
    # Test 1: Feature outside training maximum range (monetary = 500,000 > max 237k)
    extreme_df = sample_test_features.head(1).copy()
    extreme_df["monetary"] = 500000.0

    res_extreme = score_customers(extreme_df, as_of="2011-09-10", lookback_days=270)
    assert len(res_extreme[0].warnings) >= 1
    assert any("exceeds validated training maximum" in w for w in res_extreme[0].warnings)

    # Test 2: Out of season cutoff (April, month 4, outside Q4 [9, 10, 11, 12])
    res_season = score_customers(
        sample_test_features.head(1), as_of="2011-04-15", lookback_days=270
    )
    assert any("outside validated Q4 seasonal regime" in w for w in res_season[0].warnings)

    # Test 3: Wrong lookback (180 days instead of 270)
    res_lookback = score_customers(
        sample_test_features.head(1), as_of="2011-09-10", lookback_days=180
    )
    assert any("is outside the validated 270-day window" in w for w in res_lookback[0].warnings)


def test_structural_nan_contract(sample_test_features):
    """Assert structural NaN contract on mean_interpurchase_days."""
    # Repeat buyer with NaN mean_interpurchase_days must raise ValueError
    bad_df = sample_test_features.head(1).copy()
    bad_df["is_single_invoice"] = 0
    bad_df["mean_interpurchase_days"] = np.nan

    with pytest.raises(ValueError, match="null mean_interpurchase_days"):
        score_customers(bad_df)

    # Single-invoice buyer with NaN mean_interpurchase_days must succeed cleanly
    single_inv_df = sample_test_features.head(1).copy()
    single_inv_df["is_single_invoice"] = 1
    single_inv_df["mean_interpurchase_days"] = np.nan

    res = score_customers(single_inv_df, as_of="2011-09-10", lookback_days=270)
    assert len(res) == 1
    assert res[0].expected_clv > 0.0


def test_missing_feature_column_raises(sample_test_features):
    """Assert missing required feature raises descriptive ValueError."""
    incomplete_df = sample_test_features.drop(columns=["frequency"]).head(2)
    with pytest.raises(ValueError, match="missing required feature columns"):
        score_customers(incomplete_df)


def test_batch_scoring_aggregation(sample_test_features):
    """Assert score_customers_batch returns correct totals and warning counts."""
    subset = sample_test_features.head(5).copy()
    batch_res = score_customers_batch(subset, as_of="2011-09-10", lookback_days=270)

    assert batch_res.total_customers == 5
    assert len(batch_res.scored_customers) == 5
    assert batch_res.total_expected_clv > 0.0
    assert batch_res.model_version == "v1.0-frozen-2010-09-10"
