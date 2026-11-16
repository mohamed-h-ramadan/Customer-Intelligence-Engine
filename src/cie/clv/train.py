"""Model training stage for Customer Intelligence Engine.

Fits the champion Tweedie GLM on the frozen 2010-09-10 training features.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline

from cie.config import AppConfig, get_project_root, load_config
from src.clv import build_linear_clv_preprocessor
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ModelParityError(RuntimeError):
    """Raised when freshly retrained model diverges from frozen champion beyond configured tolerances."""

    pass


def check_retraining_parity(
    existing_model: Pipeline,
    new_model: Pipeline,
    X: pd.DataFrame,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, float]:
    """Verify that retrained pipeline predictions, coefficients, and intercept match frozen pipeline.

    Raises:
        ModelParityError: If difference exceeds tolerances (atol, rtol).
    """
    existing_preds = existing_model.predict(X)
    new_preds = new_model.predict(X)

    # 1. Predictions discrepancy
    abs_pred_diff = np.abs(existing_preds - new_preds)
    max_abs_pred = float(np.max(abs_pred_diff))
    max_rel_pred = float(np.max(abs_pred_diff / np.maximum(existing_preds, 1e-12)))

    # 2. Coefficients discrepancy
    existing_coef = existing_model.named_steps["model"].coef_
    new_coef = new_model.named_steps["model"].coef_
    abs_coef_diff = np.abs(existing_coef - new_coef)
    max_abs_coef = float(np.max(abs_coef_diff))
    max_rel_coef = float(np.max(abs_coef_diff / np.maximum(np.abs(existing_coef), 1e-12)))

    # 3. Intercept discrepancy
    existing_intercept = float(existing_model.named_steps["model"].intercept_)
    new_intercept = float(new_model.named_steps["model"].intercept_)
    abs_intercept_diff = abs(existing_intercept - new_intercept)

    logger.info(
        f"Retraining Parity Check: pred_max_abs={max_abs_pred:.2e}, "
        f"pred_max_rel={max_rel_pred:.2e}, coef_max_abs={max_abs_coef:.2e}, "
        f"coef_max_rel={max_rel_coef:.2e}, intercept_diff={abs_intercept_diff:.2e}"
    )

    pred_ok = np.allclose(existing_preds, new_preds, rtol=rtol, atol=atol)
    coef_ok = np.allclose(existing_coef, new_coef, rtol=rtol, atol=atol)
    intercept_ok = np.isclose(existing_intercept, new_intercept, rtol=rtol, atol=atol)

    if not (pred_ok and coef_ok and intercept_ok):
        raise ModelParityError(
            f"Retrained Tweedie model diverges from frozen champion beyond tolerance (atol={atol}, rtol={rtol})!\n"
            f"Observed: pred_max_abs={max_abs_pred:.6e}, pred_max_rel={max_rel_pred:.6e}, "
            f"coef_max_abs={max_abs_coef:.6e}, intercept_abs={abs_intercept_diff:.6e}"
        )

    return {
        "pred_max_abs": max_abs_pred,
        "pred_max_rel": max_rel_pred,
        "coef_max_abs": max_abs_coef,
        "coef_max_rel": max_rel_coef,
        "intercept_abs": abs_intercept_diff,
    }


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

    # Instantiate tuned champion pipeline (power=1.65, alpha=0.5, top_n_countries=3)
    prep = build_linear_clv_preprocessor(top_n_countries=3)
    champion_pipeline = Pipeline(
        [
            ("prep", prep),
            ("model", TweedieRegressor(power=1.65, alpha=0.5, link="log", max_iter=1000)),
        ]
    )

    champion_pipeline.fit(X_train, y_train)

    # Verify zero-training-serving skew roundtrip
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = config.pipeline.paths.resolve("clv_champion_model", root)

    # Check if a frozen model already exists
    if out_model_path.is_file():
        existing_model = joblib.load(out_model_path)
        atol = config.clv.retraining_parity.atol
        rtol = config.clv.retraining_parity.rtol
        check_retraining_parity(
            existing_model=existing_model,
            new_model=champion_pipeline,
            X=X_train,
            atol=atol,
            rtol=rtol,
        )
    else:
        joblib.dump(champion_pipeline, out_model_path)

    return champion_pipeline
