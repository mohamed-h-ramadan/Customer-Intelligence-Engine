"""Command-Line Interface (CLI) runner for Customer Intelligence Engine.

Provides modular, reproducible pipeline stages: clean, features, segment, train, evaluate, verify, and score.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from cie.clv.evaluate import run_evaluate_stage
from cie.clv.train import run_train_stage
from cie.config import get_project_root, load_config
from cie.data.cleaner import clean_raw_data
from cie.features.builder import run_features_stage
from cie.segmentation.pipeline import run_segmentation_stage


def cmd_clean(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = get_project_root()
    raw_path = config.pipeline.paths.resolve("raw_data", root)
    out_dir = root / "data" / "processed"
    print(f"[CLEAN] Ingesting {raw_path}...")
    sales, returns, guest, admin, accounting = clean_raw_data(raw_path, out_dir)
    # Save outputs
    sales.to_parquet(out_dir / "sales_clean.parquet", index=False)
    returns.to_parquet(out_dir / "returns_clean.parquet", index=False)
    guest.to_parquet(out_dir / "guest_transactions.parquet", index=False)
    admin.to_parquet(out_dir / "admin_rows.parquet", index=False)
    print(
        f"[CLEAN] Successfully processed {len(sales):,} sales rows and {len(returns):,} returns rows."
    )


def cmd_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("[FEATURES] Building train, test, and deployment feature sets...")
    dfs = run_features_stage(config)
    print(
        f"[FEATURES] Generated train ({len(dfs['features_train']):,} rows), "
        f"test ({len(dfs['features_test']):,} rows), and deploy ({len(dfs['rfm_snapshot_deploy']):,} rows)."
    )


def cmd_segment(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("[SEGMENT] Running unsupervised K=5 RFM segmentation...")
    dfs = run_segmentation_stage(config)
    print(
        f"[SEGMENT] Clustered t_seg ({len(dfs['segments_t_seg']):,} rows) and deploy ({len(dfs['segments_deploy']):,} rows)."
    )


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("[TRAIN] Fitting champion Tweedie GLM on train features...")
    pipeline = run_train_stage(config)
    print(f"[TRAIN] Champion model fitted successfully: {pipeline}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("[EVALUATE] Scoring holdout test set and generating diagnostic tables...")
    dfs = run_evaluate_stage(config)
    print(f"[EVALUATE] Scored {len(dfs['test_predictions']):,} test customers.")


def cmd_verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = get_project_root()
    print("[VERIFY] Asserting artifact reproducibility against golden records...")

    # 1. Test Predictions Parity
    golden_pred_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    if not golden_pred_path.is_file():
        print(f"[VERIFY] ERROR: Golden file missing at {golden_pred_path}")
        sys.exit(1)

    golden_preds = pd.read_parquet(golden_pred_path)
    import joblib

    model_path = config.pipeline.paths.resolve("clv_champion_model", root)
    model = joblib.load(model_path)
    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    from src.features import FEATURE_COLUMNS

    current_preds = model.predict(test_df[FEATURE_COLUMNS])
    max_pred_diff = float(
        np.max(np.abs(current_preds - golden_preds["pred_frozen_tweedie"].values))
    )

    print(f"[VERIFY] Champion Prediction Max Abs Error: {max_pred_diff:.2e}")
    assert max_pred_diff <= 1e-9, f"Prediction discrepancy {max_pred_diff} exceeds 1e-9 tolerance!"

    # 2. Heuristic Run-Rate Parity
    current_rr = test_df["monetary"].values * (90.0 / 270.0)
    max_rr_diff = float(np.max(np.abs(current_rr - golden_preds["pred_run_rate"].values)))
    print(f"[VERIFY] Run-Rate Heuristic Max Abs Error : {max_rr_diff:.2e}")
    assert max_rr_diff <= 1e-9, f"Run-rate discrepancy {max_rr_diff} exceeds 1e-9 tolerance!"

    print("[VERIFY] ALL GOLDEN PARITY ASSERTIONS PASSED (<= 1e-9).")


def cmd_all(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("RUNNING COMPLETE REPRODUCIBILITY PIPELINE (CLEAN -> VERIFY)")
    print("=" * 60)
    cmd_clean(args)
    cmd_features(args)
    cmd_segment(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_verify(args)
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cie",
        description="Customer Intelligence Engine CLI - Pipeline Runner and Decision Platform",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to params.yaml configuration file (defaults to config/params.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("clean", help="Clean raw Online Retail II Excel data into parquets")
    subparsers.add_parser("features", help="Generate windowed features and 90-day future targets")
    subparsers.add_parser(
        "segment", help="Train unsupervised RFM segmentation pipeline and export profiles"
    )
    subparsers.add_parser("train", help="Train and freeze champion Tweedie GLM pipeline")
    subparsers.add_parser("evaluate", help="Score holdout test set and export diagnostic tables")
    subparsers.add_parser(
        "verify", help="Assert exact 1e-9 prediction parity against saved golden records"
    )
    subparsers.add_parser("all", help="Run end-to-end pipeline from raw data through verification")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "clean": cmd_clean,
        "features": cmd_features,
        "segment": cmd_segment,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "verify": cmd_verify,
        "all": cmd_all,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
