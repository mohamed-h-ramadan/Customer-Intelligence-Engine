"""Unit and golden-file verification tests for cie.pipeline."""

import joblib
import numpy as np
import pandas as pd

from cie.config import get_project_root, load_config
from cie.pipeline.cli import build_parser
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
