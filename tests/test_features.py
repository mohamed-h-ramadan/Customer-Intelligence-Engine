"""Unit tests for leak-free feature and target engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_features,
    build_target,
    get_window_bounds,
)


@pytest.fixture
def synthetic_retail_frames():
    """Create tiny deterministic sales and returns fixtures covering boundary edge cases."""
    sales = pd.DataFrame(
        [
            # Cust 101: pre-lookback row (ignored when lookback_days=270)
            {
                "Invoice": "900",
                "StockCode": "A0",
                "Quantity": 100,
                "InvoiceDate": "2019-09-01 00:00:00",
                "Price": 10.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2019-09",
            },
            # Cust 101: in-window sales
            {
                "Invoice": "1001",
                "StockCode": "A1",
                "Quantity": 2,
                "InvoiceDate": "2019-12-03 00:00:00",
                "Price": 10.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2019-12",
            },
            {
                "Invoice": "1002",
                "StockCode": "A2",
                "Quantity": 3,
                "InvoiceDate": "2020-06-11 00:00:00",
                "Price": 20.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2020-06",
            },
            # Cust 101: exact cutoff row (excluded from features, included in target!)
            {
                "Invoice": "1004",
                "StockCode": "A1",
                "Quantity": 5,
                "InvoiceDate": "2020-07-01 00:00:00",
                "Price": 10.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2020-07",
            },
            # Cust 101: exact target_end row (excluded from 90d target!)
            {
                "Invoice": "1009",
                "StockCode": "A1",
                "Quantity": 9,
                "InvoiceDate": "2020-09-29 00:00:00",
                "Price": 10.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2020-09",
            },
            # Cust 102: single-invoice customer near left lookback edge (267d before 2020-07-01)
            {
                "Invoice": "1005",
                "StockCode": "B1",
                "Quantity": 4,
                "InvoiceDate": "2019-10-08 00:00:00",
                "Price": 5.0,
                "Customer ID": 102,
                "Country": "France",
                "year_month": "2019-10",
            },
        ]
    )
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    sales["TotalPrice"] = sales["Quantity"] * sales["Price"]

    returns = pd.DataFrame(
        [
            # Cust 101: return inside lookback
            {
                "Invoice": "C1003",
                "StockCode": "A2",
                "Quantity": -1,
                "InvoiceDate": "2020-06-21 00:00:00",
                "Price": 20.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2020-06",
            },
            # Cust 101: return inside target window
            {
                "Invoice": "C1010",
                "StockCode": "A1",
                "Quantity": -1,
                "InvoiceDate": "2020-07-15 00:00:00",
                "Price": 15.0,
                "Customer ID": 101,
                "Country": "United Kingdom",
                "year_month": "2020-07",
            },
            # Cust 102: return inside target window without target sales
            {
                "Invoice": "C1011",
                "StockCode": "B1",
                "Quantity": -4,
                "InvoiceDate": "2020-08-01 00:00:00",
                "Price": 5.0,
                "Customer ID": 102,
                "Country": "France",
                "year_month": "2020-08",
            },
            # Cust 103: returns-only customer in lookback -> excluded from eligible features
            {
                "Invoice": "C1006",
                "StockCode": "B1",
                "Quantity": -2,
                "InvoiceDate": "2020-05-01 00:00:00",
                "Price": 5.0,
                "Customer ID": 103,
                "Country": "France",
                "year_month": "2020-05",
            },
        ]
    )
    returns["InvoiceDate"] = pd.to_datetime(returns["InvoiceDate"])
    returns["TotalPrice"] = returns["Quantity"] * returns["Price"]
    return sales, returns


class TestWindowBounds:
    def test_window_bounds_math(self):
        win = get_window_bounds("2020-07-01", lookback_days=270, horizon_days=90)
        assert win["lookback_start"] == pd.Timestamp("2019-10-05")
        assert win["cutoff"] == pd.Timestamp("2020-07-01")
        assert win["target_end"] == pd.Timestamp("2020-09-29")
        assert win["lookback_days"] == 270
        assert win["horizon_days"] == 90


class TestBuildFeatures:
    def test_windowed_features_exact_values_and_boundaries(self, synthetic_retail_frames):
        sales, returns = synthetic_retail_frames
        feat = build_features(sales, returns, cutoff="2020-07-01", lookback_days=270)

        assert list(feat.columns) == FEATURE_COLUMNS
        assert list(feat.index) == [101, 102]  # Cust 103 excluded (returns-only)

        c101 = feat.loc[101]
        assert np.isclose(c101["recency_days"], 20.0)
        assert c101["frequency"] == 2
        assert np.isclose(c101["monetary"], 80.0)
        assert np.isclose(c101["aov"], 40.0)
        assert np.isclose(c101["avg_items_per_invoice"], 2.5)
        assert np.isclose(c101["avg_unit_price"], 16.0)
        assert np.isclose(c101["tenure_days_in_window"], 211.0)
        assert c101["is_tenure_at_edge"] == 0
        assert np.isclose(c101["mean_interpurchase_days"], 191.0)
        assert c101["is_single_invoice"] == 0
        assert np.isclose(c101["spend_last_30d"], 60.0)
        assert np.isclose(c101["spend_last_90d_ratio"], 0.75)
        assert c101["return_invoice_count"] == 1
        assert np.isclose(c101["return_value_abs"], 20.0)
        assert np.isclose(c101["return_value_ratio"], 0.25)
        assert np.isclose(c101["return_rate"], 1 / 3)
        assert c101["is_uk"] == 1

        c102 = feat.loc[102]
        assert c102["frequency"] == 1
        assert c102["is_tenure_at_edge"] == 1
        assert np.isnan(c102["mean_interpurchase_days"])
        assert c102["is_single_invoice"] == 1
        assert c102["is_uk"] == 0 and c102["country"] == "France"

    def test_full_history_snapshot_mode(self, synthetic_retail_frames):
        sales, returns = synthetic_retail_frames
        feat = build_features(sales, returns, cutoff="2020-07-01", lookback_days=None)
        assert feat.loc[101, "frequency"] == 3
        assert np.isclose(feat.loc[101, "monetary"], 1080.0)
        assert (feat["is_tenure_at_edge"] == 0).all()

    def test_empty_window_returns_empty_schema(self, synthetic_retail_frames):
        sales, returns = synthetic_retail_frames
        feat = build_features(sales, returns, cutoff="2018-01-01", lookback_days=270)
        assert len(feat) == 0
        assert list(feat.columns) == FEATURE_COLUMNS


class TestBuildTarget:
    def test_target_left_join_and_bounds(self, synthetic_retail_frames):
        sales, returns = synthetic_retail_frames
        feat = build_features(sales, returns, cutoff="2020-07-01", lookback_days=270)
        y = build_target(
            sales, returns, cutoff="2020-07-01", horizon_days=90, eligible_ids=feat.index
        )

        assert list(y.columns) == TARGET_COLUMNS
        assert np.isclose(y.loc[101, "target"], 50.0)
        assert np.isclose(y.loc[101, "target_net"], 35.0)
        assert y.loc[101, "target_active"] == 1

        assert np.isclose(y.loc[102, "target"], 0.0)
        assert np.isclose(y.loc[102, "target_net"], -20.0)
        assert y.loc[102, "target_active"] == 0
