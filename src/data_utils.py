"""Data utility functions for validation, outlier detection, and preprocessing."""

import pandas as pd

# Default non-standard missing string placeholders detected during EDA
DEFAULT_CUSTOM_NULLS = [
    "?",
    "-",
    "--",
    "---",
    ".",
    "..",
    "...",
    "n/a",
    "na",
    "null",
    "none",
    "nan",
    "missing",
    "unknown",
    "not available",
    "empty",
    "",
]


def is_cancellation(invoice_series: pd.Series) -> pd.Series:
    """Identify canceled transactions.

    In the Online Retail dataset, invoice codes starting with 'C' or 'c'
    indicate cancellations or customer returns.

    Args:
        invoice_series: Series containing invoice identifiers.

    Returns:
        Boolean Series where True indicates a canceled invoice.
    """
    return invoice_series.astype(str).str.strip().str.upper().str.startswith("C")


def calculate_total_amount(
    df: pd.DataFrame, qty_col: str = "Quantity", price_col: str = "Price"
) -> pd.Series:
    """Calculate the total line-item amount (Quantity * Price).

    Args:
        df: Input DataFrame.
        qty_col: Name of the quantity column.
        price_col: Name of the unit price column.

    Returns:
        Series with calculated total monetary amount.
    """
    if qty_col not in df.columns or price_col not in df.columns:
        raise KeyError(f"Columns '{qty_col}' and '{price_col}' must be present in DataFrame.")
    return df[qty_col] * df[price_col]


def detect_outliers_iqr(
    series: pd.Series, multiplier: float = 1.5
) -> tuple[pd.Series, float, float]:
    """Detect outliers using the Interquartile Range (IQR) method.

    Args:
        series: Numeric pandas Series.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Tuple of (boolean outlier mask, lower_bound, upper_bound).
    """
    clean_series = series.dropna()
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return outlier_mask, float(lower_bound), float(upper_bound)


def scan_custom_null_patterns(
    df: pd.DataFrame, custom_patterns: list[str] | None = None
) -> pd.DataFrame:
    """Scan DataFrame for standard nulls and non-standard string placeholders.

    Args:
        df: Input DataFrame.
        custom_patterns: List of string representations considered missing.

    Returns:
        Summary DataFrame containing missing value counts and percentages per column.
    """
    if custom_patterns is None:
        custom_patterns = DEFAULT_CUSTOM_NULLS

    custom_patterns_lower = {p.lower() for p in custom_patterns}
    report = []

    for col in df.columns:
        series = df[col]
        std_null_mask = series.isna()
        std_null_count = int(std_null_mask.sum())

        non_null = series[~std_null_mask]
        if non_null.dtype == "object" or pd.api.types.is_string_dtype(non_null):
            cleaned_str = non_null.astype(str).str.strip().str.lower()
            custom_null_mask = cleaned_str.isin(custom_patterns_lower)
            custom_null_count = int(custom_null_mask.sum())
        else:
            custom_null_count = 0

        total_missing = std_null_count + custom_null_count
        pct_missing = (total_missing / len(df)) * 100 if len(df) > 0 else 0.0

        report.append(
            {
                "Column": col,
                "Standard_Nulls": std_null_count,
                "Custom_Nulls": custom_null_count,
                "Total_Missing": total_missing,
                "Missing_Pct": round(pct_missing, 2),
            }
        )

    return pd.DataFrame(report)


def filter_valid_customers(df: pd.DataFrame, customer_id_col: str = "Customer ID") -> pd.DataFrame:
    """Filter out guest transactions with missing or zero Customer IDs.

    Args:
        df: Input DataFrame.
        customer_id_col: Column name representing Customer ID.

    Returns:
        Filtered DataFrame containing only registered, non-zero customer records.
    """
    if customer_id_col not in df.columns:
        raise KeyError(f"Column '{customer_id_col}' not found in DataFrame.")

    mask = df[customer_id_col].notna() & (df[customer_id_col] != 0) & (df[customer_id_col] != 0.0)
    return df[mask].copy()
