"""Production container and service smoke-test script.

Validates service availability, model card, customer scoring, batch scoring,
precomputed backtesting store consistency, out-of-regime warnings, and 422 validation errors.

Zero dependency on 'cie', pandas, pyarrow, or repository data files.
Uses only standard library and httpx.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import httpx  # noqa: E402

FIXTURE_PATH = PROJECT_ROOT / "scripts" / "smoke_fixture.json"


def load_fixture() -> dict:
    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError(f"Smoke test fixture not found at {FIXTURE_PATH}")
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_http_smoke_test(base_url: str, max_retries: int = 15, delay: float = 2.0) -> bool:
    """Execute end-to-end smoke test against deployed service."""
    base_url = base_url.rstrip("/")
    print(
        f"Connecting to Customer Intelligence Engine at {base_url} (up to {max_retries} attempts)..."
    )

    # 1. Healthcheck probe with startup retry
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

    fixture = load_fixture()
    customers = fixture["customers"]

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # 1. Health check details
        health_resp = client.get("/v1/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        h_data = health_resp.json()
        assert h_data["status"] == "healthy"
        assert h_data["champion_model_name"] == "tweedie_glm_tuned"
        assert h_data["precomputed_customers_available"] == 3224
        print(
            f"[OK] Health check details verified: {h_data['precomputed_customers_available']} customers indexed"
        )

        # 2. Model Card
        card_resp = client.get("/v1/model-card")
        assert card_resp.status_code == 200, f"Model card failed: {card_resp.text}"
        c_data = card_resp.json()
        assert c_data["model_name"] == "tweedie_glm_tuned"
        print(f"[OK] Model card verified: {c_data['primary_architecture']}")

        # 3. Individual Scoring Parity (/v1/score) for each fixture archetype
        score_cache = {}
        for arch_name, cdata in customers.items():
            cid = cdata["customer_id"]
            payload = dict(cdata["features"])
            payload["customer_id"] = cid

            score_resp = client.post("/v1/score?as_of=2011-09-10&lookback_days=270", json=payload)
            assert score_resp.status_code == 200, (
                f"Scoring failed for {arch_name} (ID {cid}): {score_resp.text}"
            )
            s_data = score_resp.json()
            score_cache[cid] = s_data

            # Verify predictions within atol=1e-6
            clv_diff = abs(s_data["expected_clv"] - cdata["expected_clv"])
            assert clv_diff <= 1e-6, (
                f"CLV mismatch for {arch_name} (ID {cid}): got {s_data['expected_clv']}, "
                f"expected {cdata['expected_clv']} (diff: {clv_diff:.2e})"
            )
            rr_diff = abs(s_data["challengers"]["run_rate"] - cdata["expected_run_rate"])
            assert rr_diff <= 1e-6, (
                f"Run-rate mismatch for {arch_name} (ID {cid}): got {s_data['challengers']['run_rate']}, "
                f"expected {cdata['expected_run_rate']} (diff: {rr_diff:.2e})"
            )
            assert s_data["segment"] == cdata["expected_segment"], (
                f"Segment mismatch for {arch_name} (ID {cid}): got '{s_data['segment']}', "
                f"expected '{cdata['expected_segment']}'"
            )
            print(
                f"[OK] Archetype '{arch_name}' (ID {cid}) score verified: "
                f"CLV=£{s_data['expected_clv']:.2f}, Segment='{s_data['segment']}', "
                f"Run-Rate=£{s_data['challengers']['run_rate']:.2f} (diff: {clv_diff:.1e})"
            )

        # 4. Batch Scoring (/v1/score/batch)
        batch_payload = []
        for cdata in customers.values():
            p = dict(cdata["features"])
            p["customer_id"] = cdata["customer_id"]
            batch_payload.append(p)

        batch_resp = client.post(
            "/v1/score/batch?as_of=2011-09-10&lookback_days=270", json=batch_payload
        )
        assert batch_resp.status_code == 200, f"Batch scoring failed: {batch_resp.text}"
        batch_data = batch_resp.json()
        assert batch_data["total_customers"] == len(batch_payload)
        for item in batch_data["results"]:
            cid = item["customer_id"]
            expected_clv = score_cache[cid]["expected_clv"]
            assert math.isclose(item["expected_clv"], expected_clv, abs_tol=1e-6), (
                f"Batch vs single score discrepancy for ID {cid}"
            )
        print(
            f"[OK] Batch scoring verified: {batch_data['total_customers']} customers scored simultaneously"
        )

        # 5. Customer Store Lookup Consistency (/v1/customers/{id})
        for _arch_name, cdata in customers.items():
            cid = cdata["customer_id"]
            lookup_resp = client.get(f"/v1/customers/{cid}")
            assert lookup_resp.status_code == 200, f"Lookup failed for ID {cid}: {lookup_resp.text}"
            l_data = lookup_resp.json()

            assert math.isclose(
                l_data["expected_clv"], score_cache[cid]["expected_clv"], abs_tol=1e-6
            ), f"Store CLV mismatch with score endpoint for ID {cid}"
            assert l_data["segment"] == score_cache[cid]["segment"], (
                f"Store segment mismatch with score endpoint for ID {cid}"
            )
            assert math.isclose(
                l_data["actual_90d_spend"], cdata["actual_90d_spend"], abs_tol=1e-2
            ), f"Store actual spend mismatch with fixture for ID {cid}"
            print(
                f"[OK] Customer store lookup verified: ID {cid} consistent with live scoring "
                f"(Actual 90d Spend: £{l_data['actual_90d_spend']:.2f})"
            )

        # 6. Out-of-Regime Input Warnings
        normal_payload = dict(customers["normal"]["features"])
        normal_payload["customer_id"] = customers["normal"]["customer_id"]
        # Pass out-of-season cutoff (June) and invalid lookback (180 days)
        regime_resp = client.post(
            "/v1/score?as_of=2011-06-01&lookback_days=180", json=normal_payload
        )
        assert regime_resp.status_code == 200, f"Out-of-regime request failed: {regime_resp.text}"
        r_data = regime_resp.json()
        assert len(r_data["warnings"]) > 0, "Expected regime warnings for out-of-regime parameters"
        print(f"[OK] Out-of-regime warnings verified: {r_data['warnings']}")

        # 7. Invalid Input Structured Error Handling (HTTP 422)
        invalid_payload = {"customer_id": 99999, "recency_days": "not-a-number"}
        invalid_resp = client.post("/v1/score", json=invalid_payload)
        assert invalid_resp.status_code == 422, (
            f"Expected 422 for invalid payload, got {invalid_resp.status_code}"
        )
        inv_data = invalid_resp.json()
        assert inv_data.get("error") == "Validation Error", (
            f"Unexpected validation response: {inv_data}"
        )
        assert "detail" in inv_data, "Missing 'detail' in 422 error response"
        print(
            f"[OK] Invalid input error handling verified (HTTP 422 structured response): {inv_data['error']}"
        )

    print("\n[SMOKE TEST RESULT: ALL SERVICE CHECKS PASSED SUCCESSFULLY]")
    return True


def run_local_smoke_test() -> bool:
    """Run in-process smoke test using FastAPI TestClient across all 7 verification steps."""
    print("Running in-process FastAPI TestClient smoke test...")
    from fastapi.testclient import TestClient

    from cie.api.app import app

    client = TestClient(app)
    fixture = load_fixture()
    customers = fixture["customers"]

    # 1. Health check
    h_resp = client.get("/v1/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert h_data["precomputed_customers_available"] == 3224
    print(
        f"[OK] Local health check passed: {h_data['precomputed_customers_available']} customers indexed"
    )

    # 2. Model Card
    c_resp = client.get("/v1/model-card")
    assert c_resp.status_code == 200
    print(f"[OK] Local model card passed: {c_resp.json().get('model_name')}")

    # 3. Individual Scoring Parity
    score_cache = {}
    for arch_name, cdata in customers.items():
        cid = cdata["customer_id"]
        payload = dict(cdata["features"])
        payload["customer_id"] = cid

        s_resp = client.post("/v1/score?as_of=2011-09-10&lookback_days=270", json=payload)
        assert s_resp.status_code == 200, f"Scoring failed for {arch_name}: {s_resp.text}"
        s_data = s_resp.json()
        score_cache[cid] = s_data

        clv_diff = abs(s_data["expected_clv"] - cdata["expected_clv"])
        assert clv_diff <= 1e-6, f"CLV mismatch for {arch_name}: {clv_diff:.2e}"
        rr_diff = abs(s_data["challengers"]["run_rate"] - cdata["expected_run_rate"])
        assert rr_diff <= 1e-6, f"Run-rate mismatch for {arch_name}: {rr_diff:.2e}"
        assert s_data["segment"] == cdata["expected_segment"]
        print(
            f"[OK] Local score verified for '{arch_name}' (ID {cid}): CLV=£{s_data['expected_clv']:.2f}"
        )

    # 4. Batch Scoring
    batch_payload = [dict(c["features"], customer_id=c["customer_id"]) for c in customers.values()]
    b_resp = client.post("/v1/score/batch?as_of=2011-09-10&lookback_days=270", json=batch_payload)
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert b_data["total_customers"] == len(batch_payload)
    print(f"[OK] Local batch scoring verified: {b_data['total_customers']} customers")

    # 5. Customer Store Lookup
    for _arch_name, cdata in customers.items():
        cid = cdata["customer_id"]
        l_resp = client.get(f"/v1/customers/{cid}")
        assert l_resp.status_code == 200
        l_data = l_resp.json()
        assert math.isclose(l_data["expected_clv"], score_cache[cid]["expected_clv"], abs_tol=1e-6)
        assert l_data["segment"] == score_cache[cid]["segment"]
        assert math.isclose(l_data["actual_90d_spend"], cdata["actual_90d_spend"], abs_tol=1e-2)
        print(f"[OK] Local store lookup verified: ID {cid}")

    # 6. Out-of-Regime Warnings
    norm_p = dict(customers["normal"]["features"], customer_id=customers["normal"]["customer_id"])
    reg_resp = client.post("/v1/score?as_of=2011-06-01&lookback_days=180", json=norm_p)
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert len(reg_data["warnings"]) > 0
    print(f"[OK] Local out-of-regime warnings verified: {reg_data['warnings']}")

    # 7. Invalid Input 422
    inv_resp = client.post("/v1/score", json={"customer_id": 99999, "recency_days": "invalid"})
    assert inv_resp.status_code == 422
    inv_data = inv_resp.json()
    assert inv_data.get("error") == "Validation Error"
    print(f"[OK] Local invalid input handling verified (422): {inv_data['error']}")

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
