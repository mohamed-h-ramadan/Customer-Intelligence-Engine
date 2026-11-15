"""Leak-free customer feature and target engineering utilities for Customer Intelligence Engine."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "aov",
    "avg_items_per_invoice",
    "avg_unit_price",
    "n_unique_products",
    "n_active_months",
    "n_active_weeks",
    "tenure_days_in_window",
    "is_tenure_at_edge",
    "mean_interpurchase_days",
    "is_single_invoice",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
    "invoices_last_30d",
    "invoices_last_60d",
    "invoices_last_90d",
    "spend_last_90d_ratio",
    "return_invoice_count",
    "return_value_abs",
    "return_value_ratio",
    "return_rate",
    "is_uk",
    "country",
]

TARGET_COLUMNS = ["target", "target_net", "target_active"]


def get_window_bounds(
    cutoff: pd.Timestamp | str,
    lookback_days: int | None = 270,
    horizon_days: int = 90,
    data_start: pd.Timestamp | str | None = None,
) -> dict:
    """Compute [lookback_start, cutoff) and [cutoff, target_end) window boundaries."""
    cutoff_ts = pd.Timestamp(cutoff)
    if lookback_days is not None:
        lookback_start = cutoff_ts - pd.Timedelta(days=lookback_days)
    elif data_start is not None:
        lookback_start = pd.Timestamp(data_start)
    else:
        lookback_start = None

    target_end = cutoff_ts + pd.Timedelta(days=horizon_days)
    return {
        "lookback_start": lookback_start,
        "cutoff": cutoff_ts,
        "target_end": target_end,
        "lookback_days": (cutoff_ts - lookback_start).days if lookback_start is not None else None,
        "horizon_days": (target_end - cutoff_ts).days,
    }


def build_features(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    lookback_days: int | None = 270,
) -> pd.DataFrame:
    """Build customer-level features strictly from [cutoff - lookback_days, cutoff).

    When lookback_days is None, uses all history strictly before cutoff (for descriptive segmentation).
    Returns a DataFrame indexed by Customer ID with NaNs intentionally preserved.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    lb_start = cutoff_ts - pd.Timedelta(days=lookback_days) if lookback_days is not None else None

    if lb_start is not None:
        s_win = sales[
            (sales["InvoiceDate"] >= lb_start) & (sales["InvoiceDate"] < cutoff_ts)
        ].copy()
        r_win = returns[
            (returns["InvoiceDate"] >= lb_start) & (returns["InvoiceDate"] < cutoff_ts)
        ].copy()
    else:
        s_win = sales[sales["InvoiceDate"] < cutoff_ts].copy()
        r_win = returns[returns["InvoiceDate"] < cutoff_ts].copy()

    # Mandatory internal temporal boundary assertions
    if len(s_win) > 0:
        assert s_win["InvoiceDate"].max() < cutoff_ts, (
            "TEMPORAL LEAKAGE: sales >= cutoff detected inside build_features!"
        )
        if lb_start is not None:
            assert s_win["InvoiceDate"].min() >= lb_start, (
                "WINDOW VIOLATION: sales older than cutoff - lookback_days detected!"
            )
    if len(r_win) > 0:
        assert r_win["InvoiceDate"].max() < cutoff_ts, (
            "TEMPORAL LEAKAGE: returns >= cutoff detected inside build_features!"
        )
        if lb_start is not None:
            assert r_win["InvoiceDate"].min() >= lb_start, (
                "WINDOW VIOLATION: returns older than cutoff - lookback_days detected!"
            )

    if len(s_win) == 0:
        empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        empty_df.index.name = "Customer ID"
        return empty_df

    s_win["_year_week"] = s_win["InvoiceDate"].dt.to_period("W")
    s_sorted = s_win.sort_values(["Customer ID", "InvoiceDate", "Invoice"])
    grouped = s_sorted.groupby("Customer ID")

    first_dt = grouped["InvoiceDate"].min()
    last_dt = grouped["InvoiceDate"].max()
    frequency = grouped["Invoice"].nunique().astype("int64")
    monetary = grouped["TotalPrice"].sum().astype("float64")
    total_qty = grouped["Quantity"].sum().astype("float64")

    recency_days = (cutoff_ts - last_dt).dt.total_seconds() / 86400.0
    tenure_days_in_window = (cutoff_ts - first_dt).dt.total_seconds() / 86400.0
    is_tenure_at_edge = (
        (tenure_days_in_window >= (lookback_days - 7)).astype("int64")
        if lookback_days is not None
        else pd.Series(0, index=frequency.index, dtype="int64")
    )

    inv_span_days = (last_dt - first_dt).dt.total_seconds() / 86400.0
    mean_interpurchase_days = np.where(frequency > 1, inv_span_days / (frequency - 1), np.nan)
    is_single_invoice = (frequency == 1).astype("int64")

    latest_country = grouped["Country"].last().astype("string")
    is_uk = (latest_country == "United Kingdom").astype("int64")

    feat = pd.DataFrame(index=frequency.index)
    feat.index.name = "Customer ID"

    feat["recency_days"] = recency_days.astype("float64")
    feat["frequency"] = frequency
    feat["monetary"] = monetary
    feat["aov"] = monetary / frequency
    feat["avg_items_per_invoice"] = total_qty / frequency
    feat["avg_unit_price"] = monetary / total_qty
    feat["n_unique_products"] = grouped["StockCode"].nunique().astype("int64")
    feat["n_active_months"] = grouped["year_month"].nunique().astype("int64")
    feat["n_active_weeks"] = grouped["_year_week"].nunique().astype("int64")

    feat["tenure_days_in_window"] = tenure_days_in_window.astype("float64")
    feat["is_tenure_at_edge"] = is_tenure_at_edge
    feat["mean_interpurchase_days"] = pd.Series(
        mean_interpurchase_days, index=frequency.index, dtype="float64"
    )
    feat["is_single_invoice"] = is_single_invoice

    for w in (30, 60, 90):
        sub_start = cutoff_ts - pd.Timedelta(days=w)
        s_sub = s_win[s_win["InvoiceDate"] >= sub_start]
        if len(s_sub) > 0:
            sub_g = s_sub.groupby("Customer ID")
            spend_w = (
                sub_g["TotalPrice"].sum().reindex(feat.index, fill_value=0.0).astype("float64")
            )
            inv_w = sub_g["Invoice"].nunique().reindex(feat.index, fill_value=0).astype("int64")
        else:
            spend_w = pd.Series(0.0, index=feat.index, dtype="float64")
            inv_w = pd.Series(0, index=feat.index, dtype="int64")
        feat[f"spend_last_{w}d"] = spend_w
        feat[f"invoices_last_{w}d"] = inv_w

    feat["spend_last_90d_ratio"] = feat["spend_last_90d"] / feat["monetary"]

    if len(r_win) > 0:
        r_g = r_win.groupby("Customer ID")
        ret_inv_cnt = r_g["Invoice"].nunique().reindex(feat.index, fill_value=0).astype("int64")
        ret_val_abs = (
            r_g["TotalPrice"].sum().abs().reindex(feat.index, fill_value=0.0).astype("float64")
        )
    else:
        ret_inv_cnt = pd.Series(0, index=feat.index, dtype="int64")
        ret_val_abs = pd.Series(0.0, index=feat.index, dtype="float64")

    feat["return_invoice_count"] = ret_inv_cnt
    feat["return_value_abs"] = ret_val_abs
    feat["return_value_ratio"] = ret_val_abs / feat["monetary"]
    feat["return_rate"] = ret_inv_cnt / (feat["frequency"] + ret_inv_cnt)

    feat["is_uk"] = is_uk
    feat["country"] = latest_country

    return feat[FEATURE_COLUMNS]


def build_target(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    horizon_days: int = 90,
    eligible_ids: Iterable | None = None,
) -> pd.DataFrame:
    """Build future CLV targets strictly inside [cutoff, cutoff + horizon_days).

    Left-joins onto `eligible_ids` (filling inactive customers with 0.0).
    Returns DataFrame indexed by Customer ID with: target, target_net, target_active.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    target_end = cutoff_ts + pd.Timedelta(days=horizon_days)

    s_tg = sales[(sales["InvoiceDate"] >= cutoff_ts) & (sales["InvoiceDate"] < target_end)]
    r_tg = returns[(returns["InvoiceDate"] >= cutoff_ts) & (returns["InvoiceDate"] < target_end)]

    if len(s_tg) > 0:
        assert s_tg["InvoiceDate"].min() >= cutoff_ts, (
            "TARGET WINDOW VIOLATION: sales < cutoff included!"
        )
        assert s_tg["InvoiceDate"].max() < target_end, (
            "TARGET WINDOW VIOLATION: sales >= target_end included!"
        )
    if len(r_tg) > 0:
        assert r_tg["InvoiceDate"].min() >= cutoff_ts, (
            "TARGET WINDOW VIOLATION: returns < cutoff included!"
        )
        assert r_tg["InvoiceDate"].max() < target_end, (
            "TARGET WINDOW VIOLATION: returns >= target_end included!"
        )

    gross_sales = (
        s_tg.groupby("Customer ID")["TotalPrice"].sum()
        if len(s_tg) > 0
        else pd.Series(dtype="float64")
    )
    net_returns = (
        r_tg.groupby("Customer ID")["TotalPrice"].sum()
        if len(r_tg) > 0
        else pd.Series(dtype="float64")
    )

    if eligible_ids is None:
        idx = pd.Index(sorted(set(gross_sales.index) | set(net_returns.index)), name="Customer ID")
    else:
        idx = pd.Index(eligible_ids, name="Customer ID")

    gross_aligned = gross_sales.reindex(idx, fill_value=0.0).astype("float64")
    returns_aligned = net_returns.reindex(idx, fill_value=0.0).astype("float64")

    out = pd.DataFrame(index=idx)
    out["target"] = gross_aligned
    out["target_net"] = gross_aligned + returns_aligned
    out["target_active"] = (out["target"] > 0).astype("int64")

    assert (out["target"] >= 0).all(), "Negative gross target detected!"
    assert ((out["target_active"] == 1) == (out["target"] > 0)).all(), (
        "target_active inconsistent with target!"
    )
    assert out.isna().sum().sum() == 0, "Unexpected NaNs inside target DataFrame!"

    return out[TARGET_COLUMNS]
