"""Known-answer tests for PSI/KS drift metrics and 2011-12-10 270d feature provenance."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from cie.config import get_project_root
from cie.monitoring.drift import (
    classify_psi_status,
    compute_categorical_psi,
    compute_feature_drift_table,
    compute_ks,
    compute_psi,
)
from src.features import FEATURE_COLUMNS


def test_psi_known_answer_continuous_and_binary():
    """Verify PSI equals 0.0 on identical inputs and matches hand-computed shifted values."""
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert compute_psi(base, base, n_bins=10) == pytest.approx(0.0, abs=1e-12)

    # Shifted 2-bin example: edges [-inf, 5.5, inf] -> exp [0.5, 0.5], act [0.2, 0.8]
    # PSI = (0.2 - 0.5)*ln(0.2/0.5) + (0.8 - 0.5)*ln(0.8/0.5) = 0.3 * ln(4)
    shifted = np.array([1.0, 2.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    expected_psi = 0.3 * math.log(4.0)
    assert compute_psi(base, shifted, n_bins=2) == pytest.approx(expected_psi, abs=1e-9)

    # Binary feature {0, 1} exact category comparison: exp=[0.8, 0.2], act=[0.4, 0.6]
    # PSI = (0.4 - 0.8)*ln(0.4/0.8) + (0.6 - 0.2)*ln(0.6/0.2) = 0.4 * ln(6)
    bin_exp = np.array([0, 0, 0, 0, 1])
    bin_act = np.array([0, 0, 1, 1, 1])
    assert compute_psi(bin_exp, bin_exp) == pytest.approx(0.0, abs=1e-12)
    assert compute_psi(bin_exp, bin_act) == pytest.approx(0.4 * math.log(6.0), abs=1e-9)

    # Empty array raises explicit ValueError
    with pytest.raises(ValueError, match="empty"):
        compute_psi(np.array([np.nan]), bin_act)


def test_ks_known_answer_and_categorical_psi():
    """Verify two-sample KS and categorical PSI on exact toy distributions."""
    sample = np.array([1.0, 2.0, 3.0])
    stat_same, p_same = compute_ks(sample, sample)
    assert stat_same == pytest.approx(0.0)
    assert p_same == pytest.approx(1.0)

    disjoint = np.array([10.0, 20.0, 30.0])
    stat_diff, p_diff = compute_ks(sample, disjoint)
    assert stat_diff == pytest.approx(1.0)
    assert p_diff == pytest.approx(0.1)

    cat_a = pd.Series(["A", "A", "B"])
    cat_b = pd.Series(["A", "B", "B"])
    assert compute_categorical_psi(cat_a, cat_a) == pytest.approx(0.0, abs=1e-12)
    # p=[2/3, 1/3], q=[1/3, 2/3] -> (1/3 - 2/3)*ln(0.5) + (2/3 - 1/3)*ln(2) = (2/3)*ln(2)
    assert compute_categorical_psi(cat_a, cat_b) == pytest.approx(
        (2.0 / 3.0) * math.log(2.0), abs=1e-9
    )

    assert classify_psi_status(0.05) == "OK"
    assert classify_psi_status(0.10) == "WARN"
    assert classify_psi_status(0.25) == "ALERT"


def test_deploy_270d_features_artifact_and_provenance():
    """Verify persisted 2011-12-10 270d feature artifact schema, bounds, and provenance SHA."""
    root = get_project_root()
    feat_path = root / "data" / "processed" / "features" / "features_deploy_2011-12-10_270d.parquet"
    prov_path = (
        root
        / "data"
        / "processed"
        / "monitoring"
        / "features_deploy_2011-12-10_270d_provenance.json"
    )
    assert feat_path.is_file()
    assert prov_path.is_file()

    df = pd.read_parquet(feat_path)
    prov = json.loads(prov_path.read_text(encoding="utf-8"))

    assert df.index.name == "Customer ID"
    assert list(df.columns) == FEATURE_COLUMNS
    assert prov["as_of"] == "2011-12-10"
    assert prov["lookback_days"] == 270
    assert prov["window_start"] == "2011-03-15"
    assert prov["n_customers"] == len(df)
    assert (df["recency_days"] >= 0).all() and (df["recency_days"] <= 270).all()

    computed_sha = hashlib.sha256(feat_path.read_bytes()).hexdigest()
    assert computed_sha == prov["output_sha256"]


def test_feature_drift_table_structure():
    """Verify compute_feature_drift_table returns expected columns and Bonferroni bounds."""
    root = get_project_root()
    test_df = pd.read_parquet(
        root / "data" / "processed" / "features" / "features_test_2011-09-10.parquet"
    )
    deploy_df = pd.read_parquet(
        root / "data" / "processed" / "features" / "features_deploy_2011-12-10_270d.parquet"
    )
    tbl = compute_feature_drift_table(test_df, deploy_df, "test", "deploy")
    assert len(tbl) == 25
    assert (tbl["ks_bonferroni_p"] >= tbl["ks_pvalue"]).all()
    assert (tbl["ks_bonferroni_p"] <= 1.0).all()


def test_phase3_train_vs_test_drift_exact_parity():
    """Verify test_2011-09-10 vs train_2010-09-10 in feature_drift_report.csv matches Phase 3 (0/25 flagged, max PSI 0.0199)."""
    root = get_project_root()
    drift_path = root / "data" / "processed" / "monitoring" / "feature_drift_report.csv"
    df = pd.read_csv(drift_path)

    sub = df[df["comparison"] == "test_2011-09-10 vs train_2010-09-10"].reset_index(drop=True)
    assert len(sub) == 25

    # Phase 3 flagging rule: PSI >= 0.10 or (KS >= 0.05 and p < 0.01) -> 0 / 25 flagged
    phase3_flagged = (sub["psi"] >= 0.10) | ((sub["ks_stat"] >= 0.05) & (sub["ks_pvalue"] < 0.01))
    assert int(phase3_flagged.sum()) == 0
    assert (sub["psi_status"] == "OK").all()

    # Exact max PSI: 0.0199 (tenure_days_in_window)
    max_row = sub.loc[sub["psi"].idxmax()]
    assert max_row["feature"] == "tenure_days_in_window"
    assert round(float(max_row["psi"]), 4) == pytest.approx(0.0199, abs=1e-4)
