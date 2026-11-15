"""Unit tests for data utility and validation functions."""

import pandas as pd
import pytest

from src.data_utils import (
    calculate_total_amount,
    detect_outliers_iqr,
    filter_valid_customers,
    is_cancellation,
    scan_custom_null_patterns,
)


class TestCancellationDetection:
    """Tests for invoice cancellation identification."""

    def test_is_cancellation_identifies_c_prefix(self, sample_transaction_df):
        cancellations = is_cancellation(sample_transaction_df["Invoice"])
        expected = pd.Series([False, False, True, False, True, False])
        pd.testing.assert_series_equal(cancellations, expected, check_names=False)

    def test_is_cancellation_case_insensitive_and_whitespace(self):
        invoices = pd.Series([" c123456 ", "C999999", "123456", "inv-123"])
        result = is_cancellation(invoices)
        assert list(result) == [True, True, False, False]


class TestTotalAmountCalculation:
    """Tests for total monetary amount calculation."""

    def test_calculate_total_amount_success(self, sample_transaction_df):
        totals = calculate_total_amount(sample_transaction_df)
        expected = sample_transaction_df["Quantity"] * sample_transaction_df["Price"]
        pd.testing.assert_series_equal(totals, expected)

    def test_calculate_total_amount_handles_negative_values(self):
        df = pd.DataFrame({"Quantity": [-2, 5], "Price": [10.0, 4.0]})
        totals = calculate_total_amount(df)
        assert totals.tolist() == [-20.0, 20.0]

    def test_calculate_total_amount_missing_column_raises_error(self):
        df = pd.DataFrame({"Qty": [1, 2], "Price": [10, 20]})
        with pytest.raises(KeyError, match="Columns 'Quantity' and 'Price' must be present"):
            calculate_total_amount(df)


class TestOutlierDetection:
    """Tests for IQR-based outlier detection."""

    def test_detect_outliers_iqr_identifies_extreme_values(self, synthetic_numeric_series):
        mask, lower, upper = detect_outliers_iqr(synthetic_numeric_series, multiplier=1.5)

        # Verify outliers detected
        outliers = synthetic_numeric_series[mask].tolist()
        assert 100 in outliers
        assert -50 in outliers
        assert 15 not in outliers
        assert lower < upper

    def test_detect_outliers_iqr_empty_or_constant_series(self):
        constant_series = pd.Series([5.0, 5.0, 5.0, 5.0])
        mask, lower, upper = detect_outliers_iqr(constant_series)
        assert not mask.any()
        assert lower == 5.0
        assert upper == 5.0


class TestMissingValuesScanning:
    """Tests for custom and standard missing value detection."""

    def test_scan_custom_null_patterns(self, sample_transaction_df):
        report = scan_custom_null_patterns(sample_transaction_df)

        # Description has '?' placeholder
        desc_row = report[report["Column"] == "Description"].iloc[0]
        assert desc_row["Custom_Nulls"] == 1

        # Customer ID has 1 NaN
        cust_row = report[report["Column"] == "Customer ID"].iloc[0]
        assert cust_row["Standard_Nulls"] == 1


class TestCustomerFiltering:
    """Tests for isolating valid registered customers from guest checkouts."""

    def test_filter_valid_customers_excludes_nan_and_zero(self, sample_transaction_df):
        filtered = filter_valid_customers(sample_transaction_df)

        # Original has 6 rows: 1 NaN customer, 1 zero customer -> 4 valid rows remain
        assert len(filtered) == 4
        assert not filtered["Customer ID"].isna().any()
        assert (filtered["Customer ID"] != 0.0).all()

    def test_filter_valid_customers_missing_column_raises_error(self):
        df = pd.DataFrame({"User_ID": [1, 2]})
        with pytest.raises(KeyError, match="Column 'Customer ID' not found"):
            filter_valid_customers(df)
