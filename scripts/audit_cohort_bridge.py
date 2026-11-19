"""Compute and verify the customer bridge between 2011-09-10 (test) and 2011-12-10 (deploy) 270d cohorts."""

import pandas as pd

from cie.config import get_project_root, load_config


def compute_cohort_bridge() -> dict[str, int]:
    """Return exact customer counts bridging test (2011-09-10) and deploy (2011-12-10) 270d cohorts."""
    root = get_project_root()
    config = load_config()
    sales_path = config.pipeline.paths.resolve("sales_clean", root)
    test_path = config.pipeline.paths.resolve("features_test", root)
    deploy_path = (
        root / "data" / "processed" / "features" / "features_deploy_2011-12-10_270d.parquet"
    )

    sales = pd.read_parquet(sales_path)
    test_df = pd.read_parquet(test_path)
    deploy_df = pd.read_parquet(deploy_path)

    test_ids = set(test_df.index)
    deploy_ids = set(deploy_df.index)

    aged_out_of_window_ids = test_ids - deploy_ids
    entered_ids = deploy_ids - test_ids

    first_purchase = sales.groupby("Customer ID")["InvoiceDate"].min()
    new_buyer_cutoff = pd.Timestamp("2011-09-10 00:00:00")
    new_buyer_ids = {cid for cid in deploy_ids if first_purchase.loc[cid] >= new_buyer_cutoff}
    reactivated_lapsed_ids = entered_ids - new_buyer_ids

    assert new_buyer_ids.issubset(entered_ids)
    assert (first_purchase.loc[list(reactivated_lapsed_ids)] < pd.Timestamp("2010-12-14")).all()

    test_n = len(test_ids)
    deploy_n = len(deploy_ids)
    aged_out_of_window = len(aged_out_of_window_ids)
    entered = len(entered_ids)
    new_buyers = len(new_buyer_ids)
    reactivated_lapsed = len(reactivated_lapsed_ids)
    net_delta = deploy_n - test_n

    assert entered == new_buyers + reactivated_lapsed
    assert deploy_n == test_n - aged_out_of_window + entered

    return {
        "test_n": test_n,
        "deploy_n": deploy_n,
        "net_delta": net_delta,
        "aged_out_of_window": aged_out_of_window,
        "entered": entered,
        "new_buyers": new_buyers,
        "reactivated_lapsed": reactivated_lapsed,
    }


def main() -> None:
    counts = compute_cohort_bridge()
    new_pct = 100.0 * counts["new_buyers"] / counts["deploy_n"]

    print(
        f"test_n             = {counts['test_n']:,} "
        "(active in 270d test lookback [2010-12-14, 2011-09-10))"
    )
    print(
        f"deploy_n           = {counts['deploy_n']:,} "
        "(active in 270d deploy lookback [2011-03-15, 2011-12-10))"
    )
    print(
        f"net_delta          = {counts['net_delta']:,} "
        "(net cohort headcount change: deploy_n - test_n)"
    )
    print(
        f"aged_out_of_window = {counts['aged_out_of_window']:,} "
        "(had a sale in the test lookback window but none in [2011-03-15, 2011-12-10); may return later, so NOT churn)"
    )
    print(
        f"entered            = {counts['entered']:,} "
        "(in deploy 270d cohort, not in test 270d cohort: new + reactivated)"
    )
    print(
        f"new_buyers         = {counts['new_buyers']:,} ({new_pct:.2f}% of deploy_n) "
        "(first-ever purchase in sales_clean.parquet on/after 2011-09-10)"
    )
    print(
        f"reactivated_lapsed = {counts['reactivated_lapsed']:,} "
        "(first-ever purchase < 2010-12-14, inactive in test window, purchased in [2011-09-10, 2011-12-10))"
    )
    print(
        f"IDENTITY CHECK     : {counts['deploy_n']:,} = "
        f"{counts['test_n']:,} - {counts['aged_out_of_window']:,} + {counts['entered']:,} "
        f"(and {counts['entered']:,} = {counts['new_buyers']:,} + {counts['reactivated_lapsed']:,}) -> VERIFIED"
    )


if __name__ == "__main__":
    main()
