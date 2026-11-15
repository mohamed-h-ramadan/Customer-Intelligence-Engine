"""Input validation and regime boundary checking for Customer Intelligence Engine.

Enforces data contracts, structural NaN rules, and emits explicit warnings for out-of-regime inputs.
"""

import numpy as np
import pandas as pd

from cie.config import AppConfig, load_config
from src.features import FEATURE_COLUMNS


def validate_dataframe_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame conforms to required feature columns and structural NaN rules.

    Raises:
        ValueError: If required columns are missing, unexpected NaNs occur, or structural NaN rule is breached.
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing_cols}")

    # Check structural NaN rule:
    # mean_interpurchase_days is allowed to be NaN iff is_single_invoice == 1
    single_invoice_mask = df["is_single_invoice"] == 1
    invalid_single_repeat = (~single_invoice_mask) & (df["mean_interpurchase_days"].isna())
    if invalid_single_repeat.any():
        bad_count = int(invalid_single_repeat.sum())
        raise ValueError(
            f"Found {bad_count} repeat customer records (is_single_invoice == 0) with null mean_interpurchase_days. "
            "mean_interpurchase_days may only be null for single-invoice buyers."
        )

    # Check that no other columns contain NaNs
    cols_to_check = [c for c in FEATURE_COLUMNS if c != "mean_interpurchase_days"]
    null_counts = df[cols_to_check].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Input DataFrame contains unexpected null values in columns: {cols_with_nulls.to_dict()}"
        )


def audit_regime_warnings(
    df: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int | None = None,
    config: AppConfig | None = None,
) -> list[list[str]]:
    """Inspect input features and cohort parameters, returning per-customer warnings for out-of-regime inputs.

    Never fails silently: flags feature extremes, non-Q4 seasons, and non-270d lookback windows.
    """
    if config is None:
        config = load_config()

    n_rows = len(df)
    row_warnings: list[list[str]] = [[] for _ in range(n_rows)]

    # 1. Seasonal Cutoff Validation
    seasonal_warning = None
    if as_of is not None:
        ts = pd.Timestamp(as_of)
        if ts.month not in config.regime.validated_cutoff_months:
            seasonal_warning = (
                f"Cutoff as_of '{ts.strftime('%Y-%m-%d')}' (month {ts.month}) is outside validated "
                f"Q4 seasonal regime {config.regime.validated_cutoff_months} (Sep-Dec). "
                "Tweedie champion model is calibrated for holiday expansion and may over-predict in spring/summer."
            )

    # 2. Lookback Window Validation
    lookback_warning = None
    if lookback_days is not None and lookback_days != config.regime.validated_lookback_days:
        lookback_warning = (
            f"lookback_days={lookback_days} is outside the validated {config.regime.validated_lookback_days}-day window. "
            "Input features are scaled to a 270-day cadence."
        )

    # Apply global cohort warnings to all rows
    if seasonal_warning or lookback_warning:
        for w_list in row_warnings:
            if seasonal_warning:
                w_list.append(seasonal_warning)
            if lookback_warning:
                w_list.append(lookback_warning)

    # 3. Feature Min/Max Range Validation
    ranges = config.regime.feature_ranges
    for col, r_bounds in ranges.items():
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            min_bound = r_bounds.min
            max_bound = r_bounds.max

            # Skip checking NaNs in mean_interpurchase_days (handled by structural rule)
            valid_mask = ~np.isnan(vals)

            below_min_idxs = np.where(valid_mask & (vals < (min_bound - 1e-4)))[0]
            for idx in below_min_idxs:
                row_warnings[idx].append(
                    f"Feature '{col}' value {vals[idx]:.2f} is below validated training minimum {min_bound:.2f}."
                )

            above_max_idxs = np.where(valid_mask & (vals > (max_bound + 1e-4)))[0]
            for idx in above_max_idxs:
                row_warnings[idx].append(
                    f"Feature '{col}' value {vals[idx]:.2f} exceeds validated training maximum {max_bound:.2f}."
                )

    return row_warnings
