"""Model evaluation stage for Customer Intelligence Engine.

Computes one-shot holdout predictions, bootstrap CIs, error slices, and diagnostic tables.
"""

import joblib
import numpy as np
import pandas as pd

from cie.config import AppConfig, get_project_root, load_config
from src.clv import (
    compute_slice_error_table,
    compute_targeting_simulation,
    evaluate_whale_sensitivity_slices,
)
from src.features import FEATURE_COLUMNS


def run_evaluate_stage(config: AppConfig | None = None) -> dict[str, pd.DataFrame]:
    """Score holdout test set, evaluate metrics, and export evaluation tables.

    Returns:
        dict containing 'test_predictions' and evaluation summary tables.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    model_path = config.pipeline.paths.resolve("clv_champion_model", root)
    model = joblib.load(model_path)

    test_path = config.pipeline.paths.resolve("features_test", root)
    test_df = pd.read_parquet(test_path)

    train_path = config.pipeline.paths.resolve("features_train", root)
    train_df = pd.read_parquet(train_path)

    # 1. Predictions
    pred_tweedie = model.predict(test_df[FEATURE_COLUMNS])
    pred_run_rate = (test_df["monetary"] * (90.0 / 270.0)).to_numpy(dtype=float)
    train_mean_val = float(train_df["target"].mean())
    train_median_val = float(train_df["target"].median())

    y_test_gross = test_df["target"].to_numpy(dtype=float)
    y_test_net = test_df["target_net"].to_numpy(dtype=float)
    y_test_active = test_df["target_active"].to_numpy(dtype=int)

    # Decile assignment (1 = lowest predicted, 10 = highest predicted)
    decile_series = pd.qcut(pred_tweedie, q=10, labels=list(range(1, 11)), duplicates="drop")

    # Segment mapping from t_seg
    seg_path = config.pipeline.paths.resolve("segments_t_seg", root)
    seg_df = pd.read_parquet(seg_path)
    test_segments = seg_df.loc[test_df.index, "segment"].to_numpy()

    seen_in_train = test_df.index.isin(train_df.index)
    residuals = pred_tweedie - y_test_gross

    test_predictions = pd.DataFrame(
        {
            "target": y_test_gross,
            "target_net": y_test_net,
            "target_active": y_test_active,
            "pred_frozen_tweedie": pred_tweedie,
            "pred_run_rate": pred_run_rate,
            "pred_train_mean": np.full(len(test_df), train_mean_val),
            "pred_train_median": np.full(len(test_df), train_median_val),
            "pred_decile": decile_series.astype(int),
            "segment": test_segments,
            "seen_in_train": seen_in_train,
            "residual_pred_minus_true": residuals,
        },
        index=test_df.index,
    )
    test_predictions.index.name = "Customer ID"

    # Export test_predictions
    clv_dir = root / "data" / "processed" / "clv"
    clv_dir.mkdir(parents=True, exist_ok=True)
    out_pred_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    test_predictions.to_parquet(out_pred_path)

    # 2. Slice error table
    diag_df = test_df.join(
        test_predictions[["pred_frozen_tweedie", "pred_run_rate", "pred_decile", "segment"]],
        how="inner",
    )
    segment_slices = compute_slice_error_table(diag_df, "segment")
    segment_slices.to_parquet(clv_dir / "test_segment_error_slices.parquet")
    segment_slices.to_csv(clv_dir / "test_segment_error_slices.csv", lineterminator="\n")

    # 3. Whale sensitivity slices
    whale_slices = evaluate_whale_sensitivity_slices(
        y_true=y_test_gross,
        y_pred=pred_tweedie,
        exclude_top_ks=(0, 1, 5),
    )
    whale_slices.to_parquet(clv_dir / "test_whale_sensitivity.parquet")
    whale_slices.to_csv(clv_dir / "test_whale_sensitivity.csv", lineterminator="\n")

    # 4. Targeting simulation
    targeting = compute_targeting_simulation(
        y_true=y_test_gross,
        pred_map={
            "Champion (Tweedie)": pred_tweedie,
            "Baseline (Run Rate)": pred_run_rate,
        },
        top_fracs=[0.1, 0.2, 0.3],
    )
    targeting.to_parquet(clv_dir / "targeting_simulation.parquet")
    targeting.to_csv(clv_dir / "targeting_simulation.csv", lineterminator="\n")

    return {
        "test_predictions": test_predictions,
        "segment_slices": segment_slices,
        "whale_slices": whale_slices,
        "targeting": targeting,
    }
