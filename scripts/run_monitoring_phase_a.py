"""Generate all Step 8 Phase A monitoring artifacts deterministically.

Produces:
  1. data/processed/features/features_deploy_2011-12-10_270d.parquet (when clean parquets exist)
  2. data/processed/monitoring/features_deploy_2011-12-10_270d_provenance.json
  3. data/processed/monitoring/feature_drift_report.csv
  4. data/processed/monitoring/prediction_drift_report.csv
  5. data/processed/monitoring/segment_mix_drift.csv
  6. data/processed/monitoring/backtest_report_2011-09-10.json
  7. data/processed/monitoring/backtest_decile_calibration.csv
  8. data/processed/monitoring/backtest_cohort_slices.csv
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cie.config import load_config  # noqa: E402
from cie.monitoring.backtest import build_backtest_report  # noqa: E402
from cie.monitoring.drift import (  # noqa: E402
    compute_feature_drift_table,
    compute_prediction_drift_table,
    compute_segment_mix_drift,
)
from src.features import FEATURE_COLUMNS, build_features  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase_a(
    root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    rebuild_deploy_features: bool = True,
) -> dict[str, Any]:
    """Execute complete Phase A monitoring analysis and persist reports."""
    config = load_config()
    if output_dir is None:
        output_dir = root / "data" / "processed" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_feat_path = (
        root / "data" / "processed" / "features" / "features_deploy_2011-12-10_270d.parquet"
    )
    sales_path = config.pipeline.paths.resolve("sales_clean", root)
    returns_path = config.pipeline.paths.resolve("returns_clean", root)

    # 1. Build or load 270d deployment features as of 2011-12-10
    if rebuild_deploy_features and sales_path.is_file() and returns_path.is_file():
        sales_clean = pd.read_parquet(sales_path)
        returns_clean = pd.read_parquet(returns_path)
        cutoff = pd.Timestamp("2011-12-10 00:00:00")
        lookback = int(config.pipeline.windows.lookback_days)

        deploy_df = build_features(
            sales_clean,
            returns_clean,
            cutoff=cutoff,
            lookback_days=lookback,
        )
        deploy_feat_path.parent.mkdir(parents=True, exist_ok=True)
        deploy_df.to_parquet(deploy_feat_path, index=True)

        provenance = {
            "artifact": "data/processed/features/features_deploy_2011-12-10_270d.parquet",
            "generator": "src.features.build_features(sales_clean, returns_clean, cutoff='2011-12-10', lookback_days=270)",
            "as_of": "2011-12-10",
            "lookback_days": lookback,
            "window_start": (cutoff - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d"),
            "n_customers": int(len(deploy_df)),
            "n_columns": int(deploy_df.shape[1]),
            "columns": list(deploy_df.columns),
            "source_sales_sha256": _sha256_file(sales_path),
            "source_returns_sha256": _sha256_file(returns_path),
            "output_sha256": _sha256_file(deploy_feat_path),
        }
        with open(
            output_dir / "features_deploy_2011-12-10_270d_provenance.json",
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            json.dump(provenance, f, indent=2)
            f.write("\n")
    else:
        if not deploy_feat_path.is_file():
            raise FileNotFoundError(
                f"Neither cleaned parquets nor {deploy_feat_path} found to run monitoring analysis."
            )
        deploy_df = pd.read_parquet(deploy_feat_path)

    # 2. Feature Drift: Deploy (2011-12-10, 270d) vs Test (2011-09-10) & Train (2010-09-10)
    train_df = pd.read_parquet(config.pipeline.paths.resolve("features_train", root))
    test_df = pd.read_parquet(config.pipeline.paths.resolve("features_test", root))

    drift_vs_test = compute_feature_drift_table(
        test_df, deploy_df, ref_label="test_2011-09-10", cur_label="deploy_2011-12-10_270d"
    )
    drift_vs_train = compute_feature_drift_table(
        train_df, deploy_df, ref_label="train_2010-09-10", cur_label="deploy_2011-12-10_270d"
    )
    feature_drift_df = pd.concat([drift_vs_test, drift_vs_train], ignore_index=True)
    feature_drift_df.to_csv(
        output_dir / "feature_drift_report.csv", index=False, lineterminator="\n"
    )

    # 3. Prediction Drift across ALL 3 model columns
    champion_model = joblib.load(config.pipeline.paths.resolve("clv_champion_model", root))
    exp_factor = float(config.clv.challengers.calibrated_run_rate.expansion_factor)

    test_preds = pd.read_parquet(config.pipeline.paths.resolve("clv_test_predictions", root)).copy()
    test_preds["champion_tweedie"] = test_preds["pred_frozen_tweedie"]
    test_preds["heuristic_run_rate"] = test_preds["pred_run_rate"]
    test_preds["calibrated_run_rate"] = test_preds["pred_run_rate"] * exp_factor

    deploy_preds = pd.DataFrame(index=deploy_df.index)
    deploy_preds["champion_tweedie"] = champion_model.predict(deploy_df[FEATURE_COLUMNS])
    deploy_preds["heuristic_run_rate"] = deploy_df["monetary"].to_numpy(dtype=float) * (
        90.0 / 270.0
    )
    deploy_preds["calibrated_run_rate"] = deploy_preds["heuristic_run_rate"] * exp_factor

    pred_drift_df = compute_prediction_drift_table(
        test_preds,
        deploy_preds,
        model_cols=["champion_tweedie", "calibrated_run_rate", "heuristic_run_rate"],
    )
    pred_drift_df.to_csv(
        output_dir / "prediction_drift_report.csv", index=False, lineterminator="\n"
    )

    # 3b. Segment-mix drift from existing saved segment assignments (2011-09-10 vs 2011-12-10)
    seg_test_path = (
        root / "data" / "processed" / "segmentation" / "segments_t_seg_2011-09-10.parquet"
    )
    seg_deploy_path = (
        root / "data" / "processed" / "segmentation" / "segments_deploy_2011-12-10.parquet"
    )
    if seg_test_path.is_file() and seg_deploy_path.is_file():
        seg_test = pd.read_parquet(seg_test_path)
        seg_deploy = pd.read_parquet(seg_deploy_path)
        seg_mix_df, _ = compute_segment_mix_drift(seg_test["segment"], seg_deploy["segment"])
        seg_mix_df.to_csv(output_dir / "segment_mix_drift.csv", index=False, lineterminator="\n")
    else:
        existing_seg_mix = root / "data" / "processed" / "monitoring" / "segment_mix_drift.csv"
        if existing_seg_mix.is_file():
            seg_mix_df = pd.read_csv(existing_seg_mix)
            seg_mix_df.to_csv(
                output_dir / "segment_mix_drift.csv", index=False, lineterminator="\n"
            )
        else:
            seg_mix_df = pd.DataFrame()

    # 4. Cohort Backtest Report (2011-09-10)
    backtest_report = build_backtest_report(config=config, root=root, output_dir=output_dir)

    return {
        "feature_drift": feature_drift_df,
        "prediction_drift": pred_drift_df,
        "segment_mix_drift": seg_mix_df,
        "backtest_report": backtest_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Step 8 Phase A Monitoring Analysis.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Load existing features_deploy_2011-12-10_270d.parquet without rebuilding.",
    )
    args = parser.parse_args()
    results = run_phase_a(output_dir=args.output_dir, rebuild_deploy_features=not args.no_rebuild)
    print(f"Feature drift rows: {len(results['feature_drift'])}")
    print(f"Prediction drift rows: {len(results['prediction_drift'])}")
    print(f"Segment mix rows: {len(results['segment_mix_drift'])}")
    print("Backtest report generated successfully.")


if __name__ == "__main__":
    main()
