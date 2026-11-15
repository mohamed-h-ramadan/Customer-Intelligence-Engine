"""Unit and golden-file verification tests for cie.pipeline."""

import argparse

import joblib
import numpy as np
import pandas as pd

from cie.clv.evaluate import run_evaluate_stage
from cie.clv.train import run_train_stage
from cie.config import get_project_root, load_config
from cie.pipeline.cli import build_parser, cmd_evaluate, cmd_train, cmd_verify
from src.features import FEATURE_COLUMNS


def test_golden_prediction_parity():
    """Assert champion model exactly reproduces saved test predictions within 1e-9."""
    config = load_config()
    root = get_project_root()

    model_path = config.pipeline.paths.resolve("clv_champion_model", root)
    model = joblib.load(model_path)

    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    golden_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    golden_df = pd.read_parquet(golden_path)

    current_preds = model.predict(test_df[FEATURE_COLUMNS])
    max_diff = float(np.max(np.abs(current_preds - golden_df["pred_frozen_tweedie"].values)))

    assert max_diff <= 1e-9, f"Champion predictions diverge by {max_diff}"


def test_heuristic_run_rate_parity():
    """Assert heuristic run-rate exactly reproduces saved pred_run_rate within 1e-9."""
    config = load_config()
    root = get_project_root()

    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    golden_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    golden_df = pd.read_parquet(golden_path)

    calc_rr = test_df["monetary"].values * (90.0 / 270.0)
    max_diff = float(np.max(np.abs(calc_rr - golden_df["pred_run_rate"].values)))

    assert max_diff <= 1e-9, f"Run-rate predictions diverge by {max_diff}"


def test_cli_parser_commands():
    """Verify CLI parser accepts all required pipeline subcommands."""
    parser = build_parser()
    for cmd in ["clean", "features", "segment", "train", "evaluate", "verify", "all"]:
        parsed = parser.parse_args([cmd])
        assert parsed.command == cmd


def test_run_train_stage_execution():
    """Verify train stage executes and returns a fitted sklearn pipeline."""
    pipeline = run_train_stage()
    assert hasattr(pipeline, "predict")
    assert "model" in pipeline.named_steps


def test_run_evaluate_stage_execution():
    """Verify evaluate stage executes and returns expected diagnostics tables."""
    dfs = run_evaluate_stage()
    assert "test_predictions" in dfs
    assert "targeting" in dfs
    assert "whale_slices" in dfs
    assert len(dfs["test_predictions"]) == 3224


def test_cli_commands_execution():
    """Verify CLI dispatch functions execute without error."""
    args = argparse.Namespace(config=None)
    cmd_verify(args)
    cmd_evaluate(args)
    cmd_train(args)


def test_cleaner_helper_functions():
    """Unit test individual pure helper functions in cie.data.cleaner."""
    from cie.data.cleaner import add_minimal_tx_columns, classify_non_normal_sku, log_row_accounting

    # 1. Non-normal SKU classifications
    assert classify_non_normal_sku("POST") == "postage"
    assert classify_non_normal_sku("BANK CHARGES") == "bank_charges"
    assert classify_non_normal_sku("M") == "manual_adjustments"
    assert classify_non_normal_sku("TEST01") == "test_codes"
    assert classify_non_normal_sku("GIFT10") == "gift_vouchers"
    assert classify_non_normal_sku("SP1002") == "other (legacy SKU / add-on product)"
    assert classify_non_normal_sku("RANDOM") == "other"

    # 2. Derived transaction columns
    toy_df = pd.DataFrame(
        {
            "Quantity": [2, -1],
            "Price": [10.5, 5.0],
            "InvoiceDate": pd.to_datetime(["2011-09-10 14:30:00", "2011-09-11 09:15:00"]),
        }
    )
    derived = add_minimal_tx_columns(toy_df)
    assert "TotalPrice" in derived.columns
    assert np.allclose(derived["TotalPrice"].values, [21.0, -5.0])
    assert "invoice_date" in derived.columns
    assert "year_month" in derived.columns

    # 3. Row accounting logging
    records = []
    log_row_accounting(records, step="Step 1", rows_before=100, rows_after=90, reason="Filter")
    assert len(records) == 1
    assert records[0]["removed"] == 10
    assert records[0]["pct_removed"] == 10.0


def test_cli_main_entrypoint(monkeypatch):
    """Verify main entrypoint dispatches correctly from CLI invocation."""
    from cie.pipeline.cli import main

    monkeypatch.setattr("sys.argv", ["cie", "verify"])
    main()
