"""Shared test fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    """Fixture providing a small synthetic retail transaction dataset."""
    data = {
        "Invoice": ["489434", "489434", "C489449", "489450", "C538164", "489451"],
        "StockCode": ["85048", "79323P", "22041", "21232", "22750", "POST"],
        "Description": [
            "15CM CHRISTMAS GLASS BALL",
            "PINK CHERRY LIGHTS",
            'RECORD FRAME 7" SINGLE',
            "?",
            "FELTCRAFT PRINCESS DOLL",
            "POSTAGE",
        ],
        "Quantity": [12, 6, -1, 24, -5, 1],
        "InvoiceDate": pd.date_range(start="2009-12-01 07:45", periods=6, freq="D"),
        "Price": [6.95, 6.75, 2.10, 1.25, 3.75, 18.00],
        "Customer ID": [13085.0, 13085.0, 13085.0, np.nan, 17530.0, 0.0],
        "Country": [
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "Germany",
            "United Kingdom",
            "France",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_numeric_series() -> pd.Series:
    """Fixture providing a series with predictable IQR and outliers."""
    # Q1 = 10, Median = 15, Q3 = 20 -> IQR = 10 -> Bounds = [-5, 35]
    # Outliers: 100, -50
    values = [10, 12, 14, 15, 16, 18, 20, 100, -50]
    return pd.Series(values, name="test_metric")
