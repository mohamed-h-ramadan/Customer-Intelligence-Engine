"""Feature, prediction, and segment-mix drift calculation utilities for CLV monitoring.

Reuses the Phase 3 Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov
methodology from Notebooks/3_cutoff_design_&_feature_engineering.ipynb.

Initial PSI thresholds (conventional starting points to be tuned empirically):
  - PSI < 0.10        -> OK
  - 0.10 <= PSI < 0.25 -> WARN
  - PSI >= 0.25       -> ALERT
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import FEATURE_COLUMNS

NUMERIC_FEATURE_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c != "country"]

PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# Hard-coded domain hypothesis mapping (not empirically verified from ground-truth targets)
SEASONAL_REGIME_FEATURES: dict[str, str] = {
    "tenure_days_in_window": (
        "seasonal/regime hypothesis: 596 first-time buyers since 2011-09-10 and 425 reactivated "
        "buyers pull lower-tail tenure (10th pct 52.4d -> 35.4d)"
    ),
    "spend_last_90d_ratio": (
        "seasonal/regime hypothesis: trailing 90d at 2011-12-10 sits on Q4 peak (Sep-Dec) "
        "rather than summer (Jun-Sep), lifting mean 90d spend share from ~34-35% to ~50%"
    ),
    "recency_days": (
        "seasonal/regime hypothesis: active buyers shop more recently during Q4 peak "
        "(mean recency compresses from ~86d to ~70d)"
    ),
    "invoices_last_90d": (
        "seasonal/regime hypothesis: trailing 90d covers Q4 shopping frequency surge "
        "(mean 1.22 -> 1.67)"
    ),
    "spend_last_90d": ("seasonal/regime hypothesis: trailing 90d overlaps Q4 seasonal spend peak"),
    "invoices_last_60d": ("seasonal/regime hypothesis: trailing 60d overlaps Oct-Dec peak"),
    "spend_last_60d": ("seasonal/regime hypothesis: trailing 60d overlaps Oct-Dec peak"),
    "invoices_last_30d": ("seasonal/regime hypothesis: trailing 30d overlaps Nov-Dec peak"),
    "spend_last_30d": ("seasonal/regime hypothesis: trailing 30d overlaps Nov-Dec peak"),
}


def compute_psi(
    expected: pd.Series | np.ndarray | Sequence[float],
    actual: pd.Series | np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index using quantile bins fit on reference (`expected`).

    For binary/low-cardinality discrete features where unique quantile edges <= 2,
    compares exact unique category counts instead of continuous binning.

    Note:
        Uses a 1e-6 floor on bin proportions to avoid division-by-zero or log(0).
        Identical distributions yield exactly 0.0.
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    exp_clean = exp_arr[~np.isnan(exp_arr)]
    act_clean = act_arr[~np.isnan(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        raise ValueError(
            "Cannot compute PSI on empty reference or actual array after dropping NaNs."
        )

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(exp_clean, quantiles))

    if len(edges) <= 2:
        # Binary or constant/2-level discrete feature: exact category bins
        cats = np.unique(np.concatenate([exp_clean, act_clean]))
        exp_cnt = np.array([(exp_clean == c).sum() for c in cats], dtype=float)
        act_cnt = np.array([(act_clean == c).sum() for c in cats], dtype=float)
    else:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        exp_cnt, _ = np.histogram(exp_clean, bins=edges)
        act_cnt, _ = np.histogram(act_clean, bins=edges)

    exp_pct = np.clip(exp_cnt / len(exp_clean), 1e-6, None)
    act_pct = np.clip(act_cnt / len(act_clean), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Compute PSI across discrete categorical labels."""
    exp_clean = expected.dropna().astype(str)
    act_clean = actual.dropna().astype(str)
    if len(exp_clean) == 0 or len(act_clean) == 0:
        raise ValueError("Cannot compute categorical PSI on empty series.")

    cats = sorted(set(exp_clean.unique()).union(set(act_clean.unique())))
    exp_counts = exp_clean.value_counts()
    act_counts = act_clean.value_counts()

    exp_pct = np.clip(
        np.array([exp_counts.get(c, 0) for c in cats], dtype=float) / len(exp_clean), 1e-6, None
    )
    act_pct = np.clip(
        np.array([act_counts.get(c, 0) for c in cats], dtype=float) / len(act_clean), 1e-6, None
    )
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_ks(
    expected: pd.Series | np.ndarray | Sequence[float],
    actual: pd.Series | np.ndarray | Sequence[float],
) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov statistic and two-sided p-value."""
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    exp_clean = exp_arr[~np.isnan(exp_arr)]
    act_clean = act_arr[~np.isnan(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        raise ValueError(
            "Cannot compute KS test on empty reference or actual array after dropping NaNs."
        )

    res = ks_2samp(exp_clean, act_clean)
    return float(res.statistic), float(res.pvalue)


def classify_psi_status(
    psi_value: float,
    warn_threshold: float = PSI_WARN_THRESHOLD,
    alert_threshold: float = PSI_ALERT_THRESHOLD,
) -> str:
    """Classify PSI value into OK, WARN, or ALERT using initial starting thresholds."""
    if psi_value >= alert_threshold:
        return "ALERT"
    if psi_value >= warn_threshold:
        return "WARN"
    return "OK"


def classify_drifting_feature(feature: str, psi_val: float, ks_bonf_p: float) -> tuple[str, str]:
    """Assign a feature drift classification using a hard-coded domain hypothesis mapping.

    Note: `SEASONAL_REGIME_FEATURES` is a static heuristic list of feature names (not verified
    from post-cutoff ground-truth labels), so attributed shifts are labeled `seasonal/regime hypothesis`.
    """
    is_psi_drift = psi_val >= PSI_WARN_THRESHOLD
    is_ks_drift = ks_bonf_p < 0.01

    if not (is_psi_drift or is_ks_drift):
        return "stable", "PSI < 0.10 and Bonferroni-adjusted KS p >= 0.01"

    if feature in SEASONAL_REGIME_FEATURES:
        reason = SEASONAL_REGIME_FEATURES[feature]
        if not is_psi_drift:
            reason += " (detected by KS at large N; PSI remains < 0.10)"
        return "seasonal/regime hypothesis", reason

    if is_psi_drift:
        return (
            "needs investigation",
            f"Core non-subwindow feature exceeded PSI threshold ({psi_val:.4f} >= 0.10)",
        )
    return (
        "seasonal/regime hypothesis"
        if feature
        in {
            "n_unique_products",
            "avg_items_per_invoice",
            "avg_unit_price",
            "return_rate",
            "return_value_ratio",
            "monetary",
        }
        else "needs investigation",
        f"Small-effect distribution difference detected by high-power KS test (Bonferroni p={ks_bonf_p:.2e}) with stable PSI ({psi_val:.4f} < 0.10)",
    )


def compute_feature_drift_table(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    ref_label: str,
    cur_label: str,
    features: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI and KS drift diagnostics across numeric features between two cohorts.

    Includes Bonferroni-adjusted p-value (`min(1.0, p * m)`) to address multiple testing
    across ~25 correlated numeric features.
    """
    if features is None:
        features = NUMERIC_FEATURE_COLUMNS

    m_tests = len(features)
    rows = []
    for col in features:
        ref_s = ref_df[col].dropna()
        cur_s = cur_df[col].dropna()

        psi_val = compute_psi(ref_s, cur_s, n_bins=n_bins)
        ks_stat, ks_pval = compute_ks(ref_s, cur_s)
        bonf_pval = min(1.0, ks_pval * m_tests)
        classification, reason = classify_drifting_feature(col, psi_val, bonf_pval)

        rows.append(
            {
                "comparison": f"{cur_label} vs {ref_label}",
                "feature": col,
                "n_ref": int(len(ref_s)),
                "n_cur": int(len(cur_s)),
                "ref_mean": float(ref_s.mean()),
                "cur_mean": float(cur_s.mean()),
                "ref_median": float(ref_s.median()),
                "cur_median": float(cur_s.median()),
                "psi": psi_val,
                "psi_status": classify_psi_status(psi_val),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pval,
                "ks_bonferroni_p": bonf_pval,
                "classification": classification,
                "classification_reason": reason,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["psi", "ks_stat"], ascending=False).reset_index(drop=True)
    )


def compute_prediction_drift_table(
    ref_preds: pd.DataFrame,
    cur_preds: pd.DataFrame,
    model_cols: list[str],
    ref_label: str = "test_2011-09-10",
    cur_label: str = "deploy_2011-12-10_270d",
) -> pd.DataFrame:
    """Compute model-agnostic prediction PSI and tail/zero share shift across all model columns."""
    rows = []
    for col in model_cols:
        ref_arr = ref_preds[col].dropna().to_numpy(dtype=float)
        cur_arr = cur_preds[col].dropna().to_numpy(dtype=float)

        psi_val = compute_psi(ref_arr, cur_arr)
        ks_stat, ks_pval = compute_ks(ref_arr, cur_arr)

        rows.append(
            {
                "comparison": f"{cur_label} vs {ref_label}",
                "model_column": col,
                "n_ref": int(len(ref_arr)),
                "n_cur": int(len(cur_arr)),
                "ref_mean_gbp": float(np.mean(ref_arr)),
                "cur_mean_gbp": float(np.mean(cur_arr)),
                "ref_median_gbp": float(np.median(ref_arr)),
                "cur_median_gbp": float(np.median(cur_arr)),
                "psi": psi_val,
                "psi_status": classify_psi_status(psi_val),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pval,
                "ref_zero_pct": float(np.mean(ref_arr == 0.0) * 100.0),
                "cur_zero_pct": float(np.mean(cur_arr == 0.0) * 100.0),
                "ref_lt_10_gbp_pct": float(np.mean(ref_arr < 10.0) * 100.0),
                "cur_lt_10_gbp_pct": float(np.mean(cur_arr < 10.0) * 100.0),
                "ref_lt_50_gbp_pct": float(np.mean(ref_arr < 50.0) * 100.0),
                "cur_lt_50_gbp_pct": float(np.mean(cur_arr < 50.0) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def compute_segment_mix_drift(
    ref_segments: pd.Series,
    cur_segments: pd.Series,
    ref_label: str = "segments_t_seg_2011-09-10",
    cur_label: str = "segments_deploy_2011-12-10",
) -> tuple[pd.DataFrame, float]:
    """Compute segment share mix shift (percentage points) and overall categorical PSI."""
    ref_clean = ref_segments.dropna().astype(str)
    cur_clean = cur_segments.dropna().astype(str)

    seg_psi = compute_categorical_psi(ref_clean, cur_clean)
    ref_counts = ref_clean.value_counts()
    cur_counts = cur_clean.value_counts()
    ref_pct = ref_clean.value_counts(normalize=True) * 100.0
    cur_pct = cur_clean.value_counts(normalize=True) * 100.0

    all_segs = sorted(set(ref_counts.index).union(set(cur_counts.index)))
    rows = []
    for seg in all_segs:
        r_p = float(ref_pct.get(seg, 0.0))
        c_p = float(cur_pct.get(seg, 0.0))
        rows.append(
            {
                "comparison": f"{cur_label} vs {ref_label}",
                "segment": seg,
                "ref_count": int(ref_counts.get(seg, 0)),
                "cur_count": int(cur_counts.get(seg, 0)),
                "ref_share_pct": r_p,
                "cur_share_pct": c_p,
                "diff_pp": c_p - r_p,
                "segment_mix_psi": seg_psi,
                "psi_status": classify_psi_status(seg_psi),
            }
        )

    df = pd.DataFrame(rows).sort_values("cur_share_pct", ascending=False).reset_index(drop=True)
    return df, seg_psi
