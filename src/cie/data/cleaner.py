"""Raw data ingestion, row-level conservation accounting, and cleaning pipeline.

Faithfully ports and encapsulates Phase 2 cleaning logic from Notebook 2.
"""

from pathlib import Path

import pandas as pd

from cie.config import get_project_root


def log_row_accounting(
    accounting_records: list[dict],
    step: str,
    rows_before: int,
    rows_after: int,
    reason: str,
) -> None:
    """Log row count transition and delta to row accounting records."""
    removed = rows_before - rows_after
    pct_removed = (removed / rows_before * 100.0) if rows_before > 0 else 0.0
    accounting_records.append(
        {
            "step": step,
            "rows_before": rows_before,
            "removed": removed,
            "pct_removed": round(pct_removed, 2),
            "rows_after": rows_after,
            "reason": reason,
        }
    )


def classify_non_normal_sku(code: str) -> str:
    """Classify non-5-digit StockCodes into operational business categories."""
    if code in {"POST", "DOT", "C2"}:
        return "postage"
    if code in {"BANK CHARGES", "AMAZONFEE", "CRUK"}:
        return "bank_charges"
    if code in {"M", "D", "ADJUST", "ADJUST2", "B"}:
        return "manual_adjustments"
    if code == "S":
        return "samples"
    if code.startswith("TEST"):
        return "test_codes"
    if code.startswith("GIFT"):
        return "gift_vouchers"
    if code == "C3":
        return "other (unknown <NA> adjustment)"
    if code.startswith("DCGS") or code in {"SP1002", "PADS"}:
        return "other (legacy SKU / add-on product)"
    return "other"


def add_minimal_tx_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """Derive TotalPrice, normalized invoice_date, and year_month period."""
    df_out = df_in.copy()
    df_out["TotalPrice"] = df_out["Quantity"] * df_out["Price"]
    df_out["invoice_date"] = df_out["InvoiceDate"].dt.normalize()
    df_out["year_month"] = df_out["InvoiceDate"].dt.to_period("M").astype("string")
    return df_out


def clean_raw_data(
    raw_path: Path | str,
    output_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end cleaning and row-accounting on raw Online Retail II Excel.

    Returns:
        tuple of (sales_clean, returns_clean, guest_transactions, admin_rows, accounting_df)
    """
    raw_path = Path(raw_path)
    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    output_dir = Path(output_dir)

    accounting_records: list[dict] = []

    # Step 0: Raw Multi-Sheet Ingestion
    with pd.ExcelFile(raw_path) as xl:
        df_2009 = pd.read_excel(xl, sheet_name="Year 2009-2010")
        df_2009["source_sheet"] = "Year 2009-2010"
        df_2010 = pd.read_excel(xl, sheet_name="Year 2010-2011")
        df_2010["source_sheet"] = "Year 2010-2011"

    df = pd.concat([df_2009, df_2010], ignore_index=True)
    raw_row_count = len(df)
    log_row_accounting(
        accounting_records,
        step="Step 0: Load & Concat",
        rows_before=raw_row_count,
        rows_after=len(df),
        reason=f"Loaded Year 2009-2010 ({len(df_2009):,}) + Year 2010-2011 ({len(df_2010):,})",
    )

    # Step 1: Type & Text Normalization
    rows_before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["source_sheet"] = df["source_sheet"].astype("string")
    df["is_cancellation"] = df["Invoice"].str.startswith("C").fillna(False)

    log_row_accounting(
        accounting_records,
        step="Step 1: Type & Text Normalization",
        rows_before=rows_before,
        rows_after=len(df),
        reason="Cast dtypes, stripped strings, uppercased StockCode, added is_cancellation",
    )

    # Step 2 & 3A: Cross-Sheet Overlap Removal
    rows_before = len(df)
    s1_mask = df["source_sheet"] == "Year 2009-2010"
    s2_mask = df["source_sheet"] == "Year 2010-2011"
    overlap_start = max(df.loc[s1_mask, "InvoiceDate"].min(), df.loc[s2_mask, "InvoiceDate"].min())
    overlap_end = min(df.loc[s1_mask, "InvoiceDate"].max(), df.loc[s2_mask, "InvoiceDate"].max())

    overlap_drop_mask = (
        (df["source_sheet"] == "Year 2009-2010")
        & (df["InvoiceDate"] >= overlap_start)
        & (df["InvoiceDate"] <= overlap_end)
    )
    removed_overlap = df[overlap_drop_mask].copy()
    df = df[~overlap_drop_mask].copy()

    log_row_accounting(
        accounting_records,
        step="Step 3A: Remove Cross-Sheet Overlap",
        rows_before=rows_before,
        rows_after=len(df),
        reason="Dropped Year 2009-2010 copy of identical Dec 1-9, 2010 overlap (kept Year 2010-2011)",
    )

    # Step 3B: Within-Sheet Exact Deduplication
    rows_before = len(df)
    full_dup_cols = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    df = df.drop_duplicates(subset=full_dup_cols, keep="first").copy()
    within_sheet_dups_removed = rows_before - len(df)
    total_removed_duplicates = len(removed_overlap) + within_sheet_dups_removed

    log_row_accounting(
        accounting_records,
        step="Step 3B: Remove Within-Sheet Duplicates",
        rows_before=rows_before,
        rows_after=len(df),
        reason="Dropped exact full-row duplicates (keep='first')",
    )

    # Step 4B: Partition Non-Product / Admin Rows
    rows_before = len(df)
    normal_sku_pattern = r"^\d{5}[A-Z]*$"
    legacy_sku_mask = df["StockCode"].str.startswith("DCGS") | df["StockCode"].isin(
        ["SP1002", "PADS"]
    )
    admin_mask = (~df["StockCode"].str.match(normal_sku_pattern, na=False)) & (~legacy_sku_mask)

    admin_rows = df[admin_mask].copy()
    admin_rows["admin_category"] = admin_rows["StockCode"].map(classify_non_normal_sku)
    df = df[~admin_mask].copy()

    log_row_accounting(
        accounting_records,
        step="Step 4B: Move Non-Product Rows to admin_rows",
        rows_before=rows_before,
        rows_after=len(df),
        reason=f"Moved {len(admin_rows):,} admin rows to admin_rows; kept {legacy_sku_mask.sum()} legacy product rows",
    )

    # Step 5: Partition Sales, Returns, and Anomalies
    rows_before = len(df)
    sales_mask = (~df["is_cancellation"]) & (df["Quantity"] > 0) & (df["Price"] > 0)
    returns_mask = df["is_cancellation"] & (df["Quantity"] < 0) & (df["Price"] > 0)
    anomalies_mask = ~(sales_mask | returns_mask)

    sales = df[sales_mask].copy()
    returns = df[returns_mask].copy()
    anomalies = df[anomalies_mask].copy()

    log_row_accounting(
        accounting_records,
        step="Step 5: Partition Sales, Returns, Anomalies",
        rows_before=rows_before,
        rows_after=len(sales) + len(returns),
        reason=f"Moved {len(anomalies):,} zero/negative price or unaligned transactions to anomalies",
    )

    # Step 6: Isolate Guest Transactions (Missing Customer ID)
    rows_before = len(sales) + len(returns)
    sales_guest_mask = sales["Customer ID"].isna() | (sales["Customer ID"] == 0)
    returns_guest_mask = returns["Customer ID"].isna() | (returns["Customer ID"] == 0)

    guest_sales = sales[sales_guest_mask].copy()
    guest_sales["guest_group"] = "sale"
    guest_returns = returns[returns_guest_mask].copy()
    guest_returns["guest_group"] = "return"
    guest_transactions = pd.concat([guest_sales, guest_returns], ignore_index=True)

    sales = sales[~sales_guest_mask].copy()
    returns = returns[~returns_guest_mask].copy()

    log_row_accounting(
        accounting_records,
        step="Step 6: Isolate Missing Customer ID",
        rows_before=rows_before,
        rows_after=len(sales) + len(returns),
        reason=f"Moved {len(guest_transactions):,} guest rows to guest_transactions",
    )

    # Step 7B: Isolate Aborted Mega-Order Errors
    rows_before = len(sales) + len(returns)
    aborted_sale_invoices = {"581483", "541431", "556444"}
    aborted_return_invoices = {"C581484", "C541433"}

    aborted_sales_mask = sales["Invoice"].isin(aborted_sale_invoices)
    aborted_returns_mask = returns["Invoice"].isin(aborted_return_invoices)

    aborted_rows = pd.concat(
        [sales[aborted_sales_mask], returns[aborted_returns_mask]], ignore_index=False
    )
    anomalies = pd.concat([anomalies, aborted_rows], ignore_index=False)
    sales = sales[~aborted_sales_mask].copy()
    returns = returns[~aborted_returns_mask].copy()

    log_row_accounting(
        accounting_records,
        step="Step 7B: Isolate Aborted Mega-Order Errors",
        rows_before=rows_before,
        rows_after=len(sales) + len(returns),
        reason="Moved 3 aborted sale errors and 2 return counterparts (5 rows) to anomalies",
    )

    # Step 8: Add Minimal Derived Transaction Columns
    rows_before = len(sales) + len(returns)
    sales = add_minimal_tx_columns(sales)
    returns = add_minimal_tx_columns(returns)
    guest_transactions = add_minimal_tx_columns(guest_transactions)
    admin_rows = add_minimal_tx_columns(admin_rows)
    anomalies = add_minimal_tx_columns(anomalies)

    log_row_accounting(
        accounting_records,
        step="Step 8: Add Minimal Transaction Columns",
        rows_before=rows_before,
        rows_after=len(sales) + len(returns),
        reason="Added TotalPrice, invoice_date, and year_month columns",
    )

    # Invariant Assertions
    reconciled_total = (
        len(sales)
        + len(returns)
        + len(anomalies)
        + len(admin_rows)
        + len(guest_transactions)
        + total_removed_duplicates
    )
    assert reconciled_total == raw_row_count == 1_067_371, (
        f"Row conservation failed! Sum={reconciled_total:,} vs Raw={raw_row_count:,}"
    )
    assert sales["Customer ID"].notna().all() and (sales["Customer ID"] != 0).all()
    assert (sales["Price"] > 0).all() and (sales["Quantity"] > 0).all()
    assert (returns["Price"] > 0).all() and (returns["Quantity"] < 0).all()

    accounting_df = pd.DataFrame(accounting_records)
    return sales, returns, guest_transactions, admin_rows, accounting_df
