"""Model training stage for Customer Intelligence Engine.

Fits the champion Tweedie GLM on the frozen 2010-09-10 training features.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cie.config import AppConfig, get_project_root, load_config
from src.clv import build_candidate_models
from src.features import FEATURE_COLUMNS


def run_train_stage(config: AppConfig | None = None) -> Pipeline:
    """Train and freeze champion Tweedie GLM pipeline on training features.

    Returns:
        Fitted sklearn Pipeline.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    train_path = config.pipeline.paths.resolve("features_train", root)
    train_df = pd.read_parquet(train_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"].to_numpy(dtype=float)

    # Instantiate candidate dictionary and pick the frozen champion
    candidates = build_candidate_models()
    champion_pipeline = candidates["B: Tweedie GLM (Tuned)"]

    champion_pipeline.fit(X_train, y_train)

    # Verify zero-training-serving skew roundtrip
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = config.pipeline.paths.resolve("clv_champion_model", root)

    # Check if a frozen model already exists
    if out_model_path.is_file():
        existing_model = joblib.load(out_model_path)
        existing_preds = existing_model.predict(X_train)
        new_preds = champion_pipeline.predict(X_train)
        max_diff = float(np.max(np.abs(existing_preds - new_preds)))
        assert max_diff <= 1e-9, (
            f"Trained model predictions diverge from frozen model! Max diff: {max_diff}"
        )
    else:
        joblib.dump(champion_pipeline, out_model_path)

    return champion_pipeline
