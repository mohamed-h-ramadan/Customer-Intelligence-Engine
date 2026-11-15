"""FastAPI dependencies, authentication guardrails, and cached customer store."""

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from cie.api.settings import ApiSettings
from cie.config import get_project_root, load_config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    """Return cached ApiSettings instance."""
    return ApiSettings()


def verify_api_key(
    api_key_header_val: str | None = Security(api_key_header),
) -> str | None:
    """Verify API key if CIE_API_KEY environment variable is configured."""
    settings = get_settings()
    if settings.api_key:
        if not api_key_header_val or api_key_header_val != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key in 'X-API-Key' header.",
            )
    return api_key_header_val


@lru_cache(maxsize=1)
def get_precomputed_customer_store() -> dict[int, dict[str, Any]]:
    """Load and index the 3,224 holdout test customers as-of 2011-09-10 for O(1) backtesting lookups."""
    root = get_project_root()
    config = load_config()

    pred_path = config.pipeline.paths.resolve("clv_test_predictions", root)
    feat_path = config.pipeline.paths.resolve("features_test", root)

    if not pred_path.is_file() or not feat_path.is_file():
        return {}

    preds_df = pd.read_parquet(pred_path)
    feats_df = pd.read_parquet(feat_path)

    store: dict[int, dict[str, Any]] = {}
    expansion_factor = config.clv.challengers.calibrated_run_rate.expansion_factor
    crm_actions = config.segmentation.crm_actions

    for cust_id, row in preds_df.iterrows():
        int_id = int(cust_id)
        f_row = feats_df.loc[cust_id] if cust_id in feats_df.index else None

        rr = float(row["pred_run_rate"])
        cal_rr = rr * expansion_factor
        seg = str(row["segment"])

        store[int_id] = {
            "customer_id": int_id,
            "as_of": "2011-09-10",
            "expected_clv": float(row["pred_frozen_tweedie"]),
            "actual_90d_spend": float(row["target"]),
            "actual_90d_net_spend": float(row["target_net"]),
            "is_active_in_q4": int(row["target_active"]),
            "residual_error": float(row["residual_pred_minus_true"]),
            "decile": int(row["pred_decile"]),
            "segment": seg,
            "crm_action": crm_actions.get(seg, ""),
            "seen_in_train": bool(row["seen_in_train"]),
            "challengers": {
                "run_rate": rr,
                "calibrated_run_rate": round(cal_rr, 2),
                "train_mean": float(row["pred_train_mean"]),
                "train_median": float(row["pred_train_median"]),
            },
            "features_snapshot": {
                "recency_days": float(f_row["recency_days"]) if f_row is not None else None,
                "frequency": float(f_row["frequency"]) if f_row is not None else None,
                "monetary": float(f_row["monetary"]) if f_row is not None else None,
                "aov": float(f_row["aov"]) if f_row is not None else None,
                "country": str(f_row["country"]) if f_row is not None else None,
                "return_rate": float(f_row["return_rate"]) if f_row is not None else None,
            },
        }

    return store
