"""Production container and service smoke-test script.

Validates service availability, model card, scoring, and precomputed customer store.
Usage:
    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --local
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from cie.config import get_project_root  # noqa: E402


def get_sample_customer_payload() -> dict:
    """Load a real customer from the holdout test set to ensure a realistic payload."""
    root = get_project_root()
    test_path = root / "data" / "processed" / "features" / "features_test_2011-09-10.parquet"
    if not test_path.is_file():
        raise FileNotFoundError(f"Holdout features not found at {test_path}")

    df = pd.read_parquet(test_path)
    cust_id = df.index[0]
    payload = df.iloc[0].to_dict()
    payload["customer_id"] = int(cust_id)
    if pd.isna(payload.get("mean_interpurchase_days")):
        payload["mean_interpurchase_days"] = None
    return payload


def run_http_smoke_test(base_url: str, max_retries: int = 10, delay: float = 2.0) -> bool:
    """Test running HTTP service with retries for startup grace period."""
    base_url = base_url.rstrip("/")
    print(f"Connecting to service at {base_url} (up to {max_retries} attempts)...")

    # 1. Healthcheck probe with retry
    health_url = f"{base_url}/v1/health"
    healthy = False
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                healthy = True
                print(f"[OK] Health check passed on attempt {attempt}: {resp.json()}")
                break
        except httpx.RequestError as e:
            print(
                f"  Attempt {attempt}/{max_retries}: service not ready ({e}). Retrying in {delay}s..."
            )
            time.sleep(delay)

    if not healthy:
        print("[FAIL] Service failed health check after maximum retries.")
        return False

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # 2. Model Card
        card_resp = client.get("/v1/model-card")
        assert card_resp.status_code == 200, f"Model card failed: {card_resp.text}"
        print(
            f"[OK] Model card retrieved successfully: {card_resp.json().get('primary_architecture')}"
        )

        # 3. Customer Scoring
        payload = get_sample_customer_payload()
        score_resp = client.post("/v1/score?as_of=2011-09-10&lookback_days=270", json=payload)
        assert score_resp.status_code == 200, f"Scoring failed: {score_resp.text}"
        score_data = score_resp.json()
        print(
            f"[OK] Customer scoring passed: CLV=£{score_data['expected_clv']:.2f}, "
            f"Segment='{score_data['segment']}', "
            f"Run-Rate=£{score_data['challengers']['run_rate']:.2f}"
        )

        # 4. Customer Lookup
        cust_resp = client.get(f"/v1/customers/{payload['customer_id']}")
        assert cust_resp.status_code == 200, f"Customer lookup failed: {cust_resp.text}"
        cust_data = cust_resp.json()
        print(
            f"[OK] Backtesting store lookup passed: ID={cust_data['customer_id']}, "
            f"Predicted=£{cust_data['expected_clv']:.2f}, "
            f"Actual 90d Spend=£{cust_data['actual_90d_spend']:.2f}"
        )

    print("\n[SMOKE TEST RESULT: ALL SERVICE CHECKS PASSED SUCCESSFULLY]")
    return True


def run_local_smoke_test() -> bool:
    """Run in-process smoke test using FastAPI TestClient."""
    print("Running in-process FastAPI TestClient smoke test...")
    from fastapi.testclient import TestClient

    from cie.api.app import app

    client = TestClient(app)

    # 1. Health
    h_resp = client.get("/v1/health")
    assert h_resp.status_code == 200
    print(f"[OK] Local health check passed: {h_resp.json()}")

    # 2. Model card
    c_resp = client.get("/v1/model-card")
    assert c_resp.status_code == 200
    print(f"[OK] Local model card passed: {c_resp.json().get('model_name')}")

    # 3. Scoring
    payload = get_sample_customer_payload()
    s_resp = client.post("/v1/score?as_of=2011-09-10&lookback_days=270", json=payload)
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    print(
        f"[OK] Local scoring passed: CLV=£{s_data['expected_clv']:.2f}, Segment='{s_data['segment']}'"
    )

    # 4. Lookup
    cust_resp = client.get(f"/v1/customers/{payload['customer_id']}")
    assert cust_resp.status_code == 200
    print(f"[OK] Local customer lookup passed: ID={cust_resp.json()['customer_id']}")

    print("\n[SMOKE TEST RESULT: ALL LOCAL CHECKS PASSED SUCCESSFULLY]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Customer Intelligence Engine Smoke Test")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Base URL of deployed service (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run local in-process TestClient check"
    )
    args = parser.parse_args()

    if args.url:
        success = run_http_smoke_test(args.url)
    else:
        success = run_local_smoke_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
