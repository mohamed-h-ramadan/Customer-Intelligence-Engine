"""MLflow tracking and model registry integration for Customer Intelligence Engine.

Logs parameters, honest metrics with 95% bootstrap CIs, model artifacts, dataset SHA-256 hashes,
Git commit SHA, and registers champion/challenger models with aliases.
"""

import hashlib
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from cie.config import AppConfig, get_project_root, load_config
from src.features import FEATURE_COLUMNS


def get_git_commit_sha() -> str:
    """Return current git commit hash, or 'unknown' if not in git."""
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file for data lineage and provenance tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def track_and_register_models(config: AppConfig | None = None) -> dict[str, str]:
    """Log champion and challenger models to MLflow, including metrics, CIs, and aliases.

    Returns:
        dict with run IDs and registered model version details.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    db_path = root / "mlflow.db"
    artifacts_dir = root / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

    experiment_name = "customer-intelligence-engine"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    git_sha = get_git_commit_sha()
    train_path = config.pipeline.paths.resolve("features_train", root)
    test_path = config.pipeline.paths.resolve("features_test", root)
    train_hash = compute_file_sha256(train_path)
    test_hash = compute_file_sha256(test_path)

    # Load frozen champion model and test set
    model_path = config.pipeline.paths.resolve("clv_champion_model", root)
    champion_pipeline = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    test_preds_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    test_preds_df = pd.read_parquet(test_preds_path)

    # Read research metrics table with bootstrap CIs
    metrics_ci_path = root / "data" / "processed" / "clv" / "test_metrics_with_95ci.csv"
    metrics_ci_df = pd.read_csv(metrics_ci_path)
    champ_ci = metrics_ci_df[metrics_ci_df["model"].str.startswith("B: Tweedie GLM")].set_index(
        "metric"
    )
    rr_ci = metrics_ci_df[
        metrics_ci_df["model"].str.startswith("Baseline (c): Run-Rate")
    ].set_index("metric")

    registered_model_name = "clv_decision_engine"

    # ==============================================================================
    # 1. RUN 1: Champion Model (Tweedie GLM [Tuned])
    # ==============================================================================
    with mlflow.start_run(run_name="champion_tweedie_glm_v1") as champ_run:
        champ_run_id = champ_run.info.run_id

        # Tags
        mlflow.set_tags(
            {
                "model_role": "champion",
                "model_name": config.clv.champion.name,
                "model_version": config.clv.champion.model_version,
                "git_commit": git_sha,
                "test_evaluation_counter": "1",
                "train_cutoff": config.pipeline.cutoffs.train_as_of,
                "test_cutoff": config.pipeline.cutoffs.test_as_of,
                "train_data_sha256": train_hash,
                "test_data_sha256": test_hash,
                "validated_regime": "270d_lookback_90d_horizon_q4",
            }
        )

        # Hyperparameters
        mlflow.log_params(
            {
                "model_family": "TweedieRegressor",
                "power": config.clv.champion.power,
                "alpha": config.clv.champion.alpha,
                "link": config.clv.champion.link,
                "top_n_countries": config.clv.champion.top_n_countries,
                "lookback_days": config.pipeline.windows.lookback_days,
                "horizon_days": config.pipeline.windows.horizon_days,
                "seed": config.pipeline.seed,
            }
        )

        # Honest Point Estimates and 95% Bootstrap CIs from saved artifacts
        champ_metrics = {
            "spearman": float(champ_ci.loc["spearman", "point_estimate"]),
            "spearman_ci_low": float(champ_ci.loc["spearman", "ci_low"]),
            "spearman_ci_high": float(champ_ci.loc["spearman", "ci_high"]),
            "normalized_gini": float(champ_ci.loc["normalized_gini", "point_estimate"]),
            "normalized_gini_ci_low": float(champ_ci.loc["normalized_gini", "ci_low"]),
            "normalized_gini_ci_high": float(champ_ci.loc["normalized_gini", "ci_high"]),
            "top_10_capture": float(champ_ci.loc["top_10_capture", "point_estimate"]),
            "top_10_capture_ci_low": float(champ_ci.loc["top_10_capture", "ci_low"]),
            "top_10_capture_ci_high": float(champ_ci.loc["top_10_capture", "ci_high"]),
            "lift_at_10": float(champ_ci.loc["lift_at_10", "point_estimate"]),
            "total_rev_error_pct": float(champ_ci.loc["total_rev_error_pct", "point_estimate"]),
            "total_rev_error_pct_ci_low": float(champ_ci.loc["total_rev_error_pct", "ci_low"]),
            "total_rev_error_pct_ci_high": float(champ_ci.loc["total_rev_error_pct", "ci_high"]),
            "mae": float(champ_ci.loc["mae", "point_estimate"]),
            "rmse": float(champ_ci.loc["rmse", "point_estimate"]),
            "rmse_excl_top1": float(champ_ci.loc["rmse_excl_top1", "point_estimate"]),
            "rmse_excl_top5": float(champ_ci.loc["rmse_excl_top5", "point_estimate"]),
            "medae": float(champ_ci.loc["medae", "point_estimate"]),
            "calibration_margin_pp_vs_run_rate": float(
                champ_ci.loc["total_rev_error_pct", "delta_vs_ref"]
            ),
        }
        mlflow.log_metrics(champ_metrics)

        # Artifacts
        clv_dir = root / "data" / "processed" / "clv"
        for art_name in [
            "test_metrics_with_95ci.csv",
            "cv_leaderboard_5x3.csv",
            "test_whale_sensitivity.csv",
            "test_segment_error_slices.csv",
            "targeting_simulation.csv",
            "coverage_ratio_2011q4.csv",
            "target_net_sensitivity.csv",
        ]:
            art_file = clv_dir / art_name
            if art_file.is_file():
                mlflow.log_artifact(str(art_file), artifact_path="evaluation_tables")

        # Log Model with Signature
        sample_input = test_df[FEATURE_COLUMNS].head(5)
        sample_output = champion_pipeline.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model=champion_pipeline,
            name="model",
            signature=signature,
            registered_model_name=registered_model_name,
            serialization_format="cloudpickle",
        )

    # Set champion alias in MLflow Model Registry
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    champ_model_version = str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=champ_model_version,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=champ_model_version,
        key="role",
        value="champion",
    )

    # ==============================================================================
    # 2. RUN 2: Challenger Model (Calibrated Run-Rate)
    # ==============================================================================
    with mlflow.start_run(run_name="challenger_calibrated_run_rate_v1") as chal_run:
        chal_run_id = chal_run.info.run_id

        mlflow.set_tags(
            {
                "model_role": "challenger",
                "model_name": config.clv.challengers.calibrated_run_rate.name,
                "model_version": config.clv.challengers.calibrated_run_rate.model_version,
                "git_commit": git_sha,
                "test_evaluation_counter": "1",
                "validated_regime": "270d_lookback_90d_horizon_q4",
            }
        )

        mlflow.log_params(
            {
                "model_family": "CalibratedRunRate",
                "lookback_days": config.clv.challengers.calibrated_run_rate.lookback_days,
                "horizon_days": config.clv.challengers.calibrated_run_rate.horizon_days,
                "expansion_factor": config.clv.challengers.calibrated_run_rate.expansion_factor,
            }
        )

        # Calibrated run-rate metrics
        actual_total = float(test_preds_df["target"].sum())
        cal_rr_preds = (
            test_preds_df["pred_run_rate"].to_numpy(dtype=float)
            * config.clv.challengers.calibrated_run_rate.expansion_factor
        )
        cal_rr_total = float(np.sum(cal_rr_preds))
        cal_rr_rev_error = (cal_rr_total - actual_total) / actual_total * 100.0

        chal_metrics = {
            "spearman": float(rr_ci.loc["spearman", "point_estimate"]),
            "spearman_ci_low": float(rr_ci.loc["spearman", "ci_low"]),
            "spearman_ci_high": float(rr_ci.loc["spearman", "ci_high"]),
            "normalized_gini": float(rr_ci.loc["normalized_gini", "point_estimate"]),
            "normalized_gini_ci_low": float(rr_ci.loc["normalized_gini", "ci_low"]),
            "normalized_gini_ci_high": float(rr_ci.loc["normalized_gini", "ci_high"]),
            "top_10_capture": float(rr_ci.loc["top_10_capture", "point_estimate"]),
            "total_rev_error_pct": cal_rr_rev_error,
            "raw_heuristic_rev_error_pct": float(
                rr_ci.loc["total_rev_error_pct", "point_estimate"]
            ),
            "mae": float(np.mean(np.abs(cal_rr_preds - test_preds_df["target"].to_numpy()))),
            "rmse": float(
                np.sqrt(np.mean((cal_rr_preds - test_preds_df["target"].to_numpy()) ** 2))
            ),
        }
        mlflow.log_metrics(chal_metrics)

        # Log artifact table
        cal_summary = pd.DataFrame(
            [
                {
                    "metric": k,
                    "value": v,
                }
                for k, v in chal_metrics.items()
            ]
        )
        cal_summary_path = artifacts_dir / "calibrated_run_rate_summary.csv"
        cal_summary.to_csv(cal_summary_path, index=False)
        mlflow.log_artifact(str(cal_summary_path), artifact_path="challenger_metrics")

    return {
        "champion_run_id": champ_run_id,
        "champion_version": champ_model_version,
        "challenger_run_id": chal_run_id,
        "experiment_name": experiment_name,
    }
