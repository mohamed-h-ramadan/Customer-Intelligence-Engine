"""Versioned API route handlers (/v1)."""

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from cie.api.dependencies import get_precomputed_customer_store, get_settings, verify_api_key
from cie.api.settings import ApiSettings
from cie.config import load_config
from cie.core.schemas import (
    BatchCustomerScoringResponse,
    CustomerFeatures,
    CustomerScoringResponse,
)
from cie.core.scoring import score_customers, score_customers_batch

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])


@router.get(
    "/health",
    summary="Service Health Check",
    description="Returns service availability, model version, and precomputed store status.",
    tags=["System"],
)
def get_health(
    store: dict[int, dict[str, Any]] = Depends(get_precomputed_customer_store),
) -> dict[str, Any]:
    config = load_config()
    return {
        "status": "healthy",
        "service": "customer-intelligence-engine",
        "champion_model_version": config.clv.champion.model_version,
        "champion_model_name": config.clv.champion.name,
        "precomputed_cohort_as_of": "2011-09-10",
        "precomputed_customers_available": len(store),
    }


@router.get(
    "/model-card",
    summary="Model Card & Honest Metrics",
    description="Returns intended use, validated regime specifications, and honest test metrics with 95% bootstrap CIs.",
    tags=["Documentation"],
)
def get_model_card() -> dict[str, Any]:
    config = load_config()
    return {
        "model_name": config.clv.champion.name,
        "model_version": config.clv.champion.model_version,
        "problem_type": "Supervised 90-Day Gross Customer Lifetime Value (CLV) Regression",
        "primary_architecture": "Tweedie GLM (power=1.65, alpha=0.5, log link, 3-country OHE)",
        "fallback_architecture": "Hurdle Model (Stage-1 Logistic Regression x Stage-2 Ridge Regressor)",
        "intended_use": (
            "Quarterly B2B customer prioritization, budget calibration, and CRM segment routing "
            "for registered active buyers."
        ),
        "validated_regime": {
            "lookback_window_days": config.pipeline.windows.lookback_days,
            "horizon_window_days": config.pipeline.windows.horizon_days,
            "season_months": config.regime.validated_cutoff_months,
            "description": "Tested on September-to-December (Q4) customer transactions; excludes cold-start buyers.",
        },
        "holdout_test_metrics": {
            "cohort_size": 3224,
            "test_gross_spend_gbp": 2585609.91,
            "spearman_rank_correlation": {
                "point_estimate": 0.5289,
                "ci_95": [0.5036, 0.5566],
            },
            "normalized_gini": {
                "point_estimate": 0.7751,
                "ci_95": [0.6880, 0.8361],
            },
            "top_10_percent_capture": {
                "point_estimate": "56.12%",
                "lift": "5.61x",
                "ci_95": ["48.17%", "63.46%"],
                "active_buyer_hit_rate": "92.55%",
            },
            "top_30_percent_capture": {
                "point_estimate": "75.65%",
                "lift": "2.52x",
            },
            "total_revenue_calibration_error": {
                "overall": {
                    "point_estimate": "-13.45%",
                    "ci_95": ["-23.47%", "-1.44%"],
                },
                "excl_top_5_whales": "-0.96%",
                "paired_win_vs_run_rate_pp": "+23.93pp (95% CI: [+18.69pp, +29.88pp])",
            },
            "rmse_sensitivity_slices": {
                "all_customers": 3107.57,
                "excl_top_1_whale": 2454.36,
                "excl_top_5_whales": 1241.49,
            },
        },
        "crm_segmentation": {
            "framework": "Unsupervised RFM K-Means (K=5)",
            "clusters": config.segmentation.cluster_to_segment,
            "usage": "Contextual CRM action routing only; strictly NEVER fed as CLV features.",
        },
        "known_limitations": [
            "Valid exclusively for active registered buyers with recency <= 270 days.",
            "Calibrated for Q4 holiday acceleration; spring/summer deployment requires seasonal re-indexing.",
            "Extreme wholesale heavy tail: top 5 accounts drive 17.26% of spend and 84.1% of SSE.",
        ],
    }


@router.post(
    "/score",
    response_model=CustomerScoringResponse,
    summary="Score a Single Customer",
    description="Evaluates champion forecast, benchmark challengers, CRM segment, and regime warnings for a single customer payload.",
    tags=["Scoring"],
)
def score_single(
    features: CustomerFeatures,
    as_of: str | None = Query(
        default=None,
        description="Cutoff date in YYYY-MM-DD format (checked against validated Q4 season).",
        examples=["2011-09-10"],
    ),
    lookback_days: int | None = Query(
        default=None,
        description="Lookback length in days (checked against validated 270-day window).",
        examples=[270],
    ),
) -> CustomerScoringResponse:
    # Convert validated Pydantic model to a single-row DataFrame
    feat_dict = features.model_dump()
    cust_id = feat_dict.pop("customer_id", None)
    df = pd.DataFrame([feat_dict], index=[cust_id] if cust_id else [0])
    if cust_id:
        df["Customer ID"] = cust_id

    responses = score_customers(
        df,
        as_of=as_of,
        lookback_days=lookback_days,
    )
    return responses[0]


@router.post(
    "/score/batch",
    response_model=BatchCustomerScoringResponse,
    summary="Score a Batch of Customers",
    description="Batch scoring endpoint with aggregate summary metrics and regime warning counts.",
    tags=["Scoring"],
)
def score_batch(
    customers: list[CustomerFeatures],
    as_of: str | None = Query(
        default=None,
        description="Cutoff date in YYYY-MM-DD format.",
        examples=["2011-09-10"],
    ),
    lookback_days: int | None = Query(
        default=None,
        description="Lookback length in days.",
        examples=[270],
    ),
    settings: ApiSettings = Depends(get_settings),
) -> BatchCustomerScoringResponse:
    if len(customers) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty.",
        )
    if len(customers) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(customers)} exceeds configured maximum limit of {settings.max_batch_size}.",
        )

    records = []
    ids = []
    for c in customers:
        d = c.model_dump()
        cid = d.pop("customer_id", None)
        ids.append(cid)
        records.append(d)

    df = pd.DataFrame(records, index=ids if any(x is not None for x in ids) else None)
    if any(x is not None for x in ids):
        df["Customer ID"] = ids

    return score_customers_batch(
        df,
        as_of=as_of,
        lookback_days=lookback_days,
    )


@router.get(
    "/customers/{customer_id}",
    summary="Precomputed Customer Store Lookup & Backtesting Record",
    description="Retrieve a precomputed customer record as-of 2011-09-10, pairing champion prediction with actual 90-day gross spend.",
    tags=["Customers"],
)
def get_customer(
    customer_id: int,
    store: dict[int, dict[str, Any]] = Depends(get_precomputed_customer_store),
) -> dict[str, Any]:
    if customer_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Customer ID {customer_id} not found in precomputed 2011-09-10 test store. "
                "The test store contains the 3,224 active eligible buyers evaluated in Phase 4b."
            ),
        )
    return store[customer_id]
