"""Comprehensive API test suite using FastAPI TestClient."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cie.api.app import app
from cie.api.dependencies import get_settings
from cie.config import get_project_root


@pytest.fixture
def client():
    """Create FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def valid_customer_payload():
    """Extract a real customer record from features_test for realistic payloads."""
    root = get_project_root()
    test_path = root / "data" / "processed" / "features" / "features_test_2011-09-10.parquet"
    df = pd.read_parquet(test_path)
    first_cust_id = df.index[0]
    row = df.iloc[0].to_dict()
    row["customer_id"] = int(first_cust_id)
    # Ensure float NaN becomes None for JSON serialization
    if pd.isna(row.get("mean_interpurchase_days")):
        row["mean_interpurchase_days"] = None
    return row


def test_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "customer-intelligence-engine"
    assert data["champion_model_version"] == "v1.0-frozen-2010-09-10"
    assert data["precomputed_customers_available"] == 3224


def test_model_card_endpoint(client):
    response = client.get("/v1/model-card")
    assert response.status_code == 200
    data = response.json()
    assert "Tweedie GLM" in data["primary_architecture"]
    assert data["holdout_test_metrics"]["spearman_rank_correlation"]["point_estimate"] == 0.5289
    assert data["holdout_test_metrics"]["normalized_gini"]["point_estimate"] == 0.7751
    assert data["holdout_test_metrics"]["top_10_percent_capture"]["point_estimate"] == "56.12%"
    assert (
        data["holdout_test_metrics"]["total_revenue_calibration_error"]["excl_top_5_whales"]
        == "-0.96%"
    )


def test_get_customer_lookup_success(client):
    # Customer 12347 is a well-known active buyer in the holdout test set
    response = client.get("/v1/customers/12347")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == 12347
    assert data["as_of"] == "2011-09-10"
    assert data["expected_clv"] > 0.0
    assert data["actual_90d_spend"] > 0.0
    assert data["segment"] in ["VIP Champions", "Loyal Regulars", "Promising Recent Buyers"]
    assert "run_rate" in data["challengers"]
    assert data["features_snapshot"]["monetary"] > 0.0


def test_get_customer_lookup_not_found(client):
    response = client.get("/v1/customers/99999999")
    assert response.status_code == 404
    data = response.json()
    assert "not found in precomputed 2011-09-10 test store" in data["detail"]


def test_score_single_customer_success(client, valid_customer_payload):
    response = client.post(
        "/v1/score?as_of=2011-09-10&lookback_days=270",
        json=valid_customer_payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == valid_customer_payload["customer_id"]
    assert data["expected_clv"] > 0.0
    assert data["model_version"] == "v1.0-frozen-2010-09-10"
    assert data["segment"] is not None
    assert data["challengers"]["calibrated_run_rate"] > data["challengers"]["run_rate"]
    assert len(data["warnings"]) == 0


def test_score_single_customer_with_regime_warnings(client, valid_customer_payload):
    # Alter payload to breach training bounds (monetary = 500k) and pass out-of-season cutoff
    payload = valid_customer_payload.copy()
    payload["monetary"] = 500000.0

    response = client.post(
        "/v1/score?as_of=2011-04-10&lookback_days=180",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["expected_clv"] > 0.0
    assert len(data["warnings"]) >= 3
    assert any("exceeds validated training maximum" in w for w in data["warnings"])
    assert any("outside validated Q4 seasonal regime" in w for w in data["warnings"])
    assert any("is outside the validated 270-day window" in w for w in data["warnings"])


def test_score_single_customer_validation_error(client, valid_customer_payload):
    # Breaching structural NaN: repeat customer (is_single_invoice=0) with mean_interpurchase_days=None
    bad_payload = valid_customer_payload.copy()
    bad_payload["is_single_invoice"] = 0
    bad_payload["mean_interpurchase_days"] = None

    response = client.post("/v1/score", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Validation Error" in data["error"]
    assert "mean_interpurchase_days" in data["detail"]


def test_score_batch_success(client, valid_customer_payload):
    batch_payload = [valid_customer_payload, valid_customer_payload.copy()]
    response = client.post("/v1/score/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["scored_customers"]) == 2
    assert data["total_expected_clv"] > 0.0


def test_score_batch_empty_fails(client):
    response = client.post("/v1/score/batch", json=[])
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_score_batch_size_limit_exceeded(client, valid_customer_payload, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_batch_size", 2)

    batch_payload = [valid_customer_payload, valid_customer_payload, valid_customer_payload]
    response = client.post("/v1/score/batch", json=batch_payload)
    assert response.status_code == 400
    assert "exceeds configured maximum limit" in response.json()["detail"]


def test_api_key_authentication(client, valid_customer_payload, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_key", "secret-test-key-12345")

    # Request without API key -> 401
    unauth_resp = client.get("/v1/health")
    assert unauth_resp.status_code == 401

    # Request with wrong API key -> 401
    bad_key_resp = client.get("/v1/health", headers={"X-API-Key": "wrong-key"})
    assert bad_key_resp.status_code == 401

    # Request with valid API key -> 200
    auth_resp = client.get("/v1/health", headers={"X-API-Key": "secret-test-key-12345"})
    assert auth_resp.status_code == 200
