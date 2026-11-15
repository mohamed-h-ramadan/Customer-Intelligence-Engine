"""Framework-agnostic customer scoring core.

Evaluates champion CLV forecasts, benchmark/challenger models, and CRM segmentation tiers.
"""

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from cie.config import AppConfig, get_project_root, load_config
from cie.core.schemas import (
    BatchCustomerScoringResponse,
    ChallengersResponse,
    CustomerScoringResponse,
)
from cie.core.validation import audit_regime_warnings, validate_dataframe_schema
from src.features import FEATURE_COLUMNS


@lru_cache(maxsize=1)
def load_serving_models(
    config_path: str | None = None,
) -> tuple[Any, Any]:
    """Load and cache serialized champion CLV pipeline and segmentation pipeline."""
    config = load_config(config_path)
    root = get_project_root()

    champion_path = config.pipeline.paths.resolve("clv_champion_model", root)
    if not champion_path.is_file():
        raise FileNotFoundError(f"Champion model not found at {champion_path}")
    champion_model = joblib.load(champion_path)

    seg_path = config.pipeline.paths.resolve("segmentation_pipeline", root)
    if not seg_path.is_file():
        raise FileNotFoundError(f"Segmentation pipeline not found at {seg_path}")
    seg_pipeline = joblib.load(seg_path)

    return champion_model, seg_pipeline


def score_customers(
    features_df: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int | None = None,
    config: AppConfig | None = None,
) -> list[CustomerScoringResponse]:
    """Score a DataFrame of customer features using champion, challengers, and CRM segmentation.

    Args:
        features_df: DataFrame containing the 26 FEATURE_COLUMNS, with Customer ID as index or column.
        as_of: Optional cutoff date for seasonal regime checking (e.g. '2011-09-10').
        lookback_days: Optional lookback length (defaults to 270).
        config: Optional AppConfig instance.

    Returns:
        List of CustomerScoringResponse objects with expected CLV, challengers, CRM segment, and warnings.
    """
    if config is None:
        config = load_config()

    # Preserve customer IDs from index or column
    if "Customer ID" in features_df.columns:
        customer_ids = features_df["Customer ID"].tolist()
    elif "customer_id" in features_df.columns:
        customer_ids = features_df["customer_id"].tolist()
    else:
        customer_ids = features_df.index.tolist()

    # 1. Enforce Data Contract & Structural NaN Rules
    validate_dataframe_schema(features_df)

    # 2. Audit Regime Boundaries and Generate Explicit Warnings
    row_warnings = audit_regime_warnings(
        features_df, as_of=as_of, lookback_days=lookback_days, config=config
    )

    # 3. Load Cached Serving Models
    champion_model, seg_pipeline = load_serving_models()

    # 4. Compute Champion CLV Predictions
    X_score = features_df[FEATURE_COLUMNS]
    champion_preds = champion_model.predict(X_score)

    # 5. Compute CRM Segmentation (Context Only - Never Used as CLV Feature)
    rfm_input = features_df[config.segmentation.feature_cols]
    cluster_labels = seg_pipeline.predict(rfm_input)

    c_to_s = config.segmentation.cluster_to_segment
    crm_actions_map = config.segmentation.crm_actions

    # 6. Compute Challenger & Baseline Predictions
    monetary_vals = features_df["monetary"].to_numpy(dtype=float)
    run_rate_preds = monetary_vals * (90.0 / 270.0)
    expansion_factor = config.clv.challengers.calibrated_run_rate.expansion_factor
    calibrated_rr_preds = run_rate_preds * expansion_factor
    train_mean_val = config.clv.challengers.train_mean.value
    train_median_val = config.clv.challengers.train_median.value

    model_version = config.clv.champion.model_version

    # 7. Assemble Structured Responses
    responses: list[CustomerScoringResponse] = []
    for i in range(len(features_df)):
        cluster_id = int(cluster_labels[i])
        segment_name = c_to_s.get(cluster_id, "Unknown")
        crm_action = crm_actions_map.get(segment_name, "")

        challengers = ChallengersResponse(
            run_rate=float(run_rate_preds[i]),
            calibrated_run_rate=float(calibrated_rr_preds[i]),
            train_mean=float(train_mean_val),
            train_median=float(train_median_val),
        )

        resp = CustomerScoringResponse(
            customer_id=customer_ids[i] if i < len(customer_ids) else None,
            expected_clv=float(champion_preds[i]),
            challengers=challengers,
            segment=segment_name,
            crm_action=crm_action,
            model_version=model_version,
            p_active=None,
            warnings=row_warnings[i],
        )
        responses.append(resp)

    return responses


def score_customers_batch(
    features_df: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int | None = None,
    config: AppConfig | None = None,
) -> BatchCustomerScoringResponse:
    """Score a batch of customers and return aggregate summary metrics and warning counts."""
    scored = score_customers(features_df, as_of=as_of, lookback_days=lookback_days, config=config)
    total_clv = sum(r.expected_clv for r in scored)
    warned_count = sum(1 for r in scored if len(r.warnings) > 0)
    version = scored[0].model_version if scored else "unknown"

    return BatchCustomerScoringResponse(
        total_customers=len(scored),
        scored_customers=scored,
        model_version=version,
        total_expected_clv=round(total_clv, 2),
        warnings_count=warned_count,
    )
