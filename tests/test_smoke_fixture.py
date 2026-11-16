"""Test proving scripts/smoke_fixture.json matches the serving core (score_customers)."""

import json

import numpy as np
import pandas as pd

from cie.config import get_project_root
from cie.core.scoring import score_customers


def test_smoke_fixture_matches_serving_core():
    """Verify each customer archetype in smoke_fixture.json matches score_customers within atol=1e-6."""
    root = get_project_root()
    fixture_path = root / "scripts" / "smoke_fixture.json"
    assert fixture_path.is_file(), f"Fixture not found at {fixture_path}"

    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    customers = fixture["customers"]
    assert len(customers) == 3
    assert set(customers.keys()) == {"normal", "single_invoice", "whale"}

    # Build DataFrame matching the schema
    rows = []
    expected_values = {}
    for arch_name, cdata in customers.items():
        cid = cdata["customer_id"]
        feat_dict = dict(cdata["features"])
        feat_dict["customer_id"] = cid
        rows.append(feat_dict)
        expected_values[cid] = {
            "archetype": arch_name,
            "clv": cdata["expected_clv"],
            "run_rate": cdata["expected_run_rate"],
            "segment": cdata["expected_segment"],
        }

    df = pd.DataFrame(rows).set_index("customer_id")
    results = score_customers(df, as_of="2011-09-10", lookback_days=270)

    for r in results:
        exp = expected_values[r.customer_id]
        # Tweedie CLV forecast parity within atol=1e-6
        assert np.isclose(r.expected_clv, exp["clv"], atol=1e-6, rtol=1e-6), (
            f"Archetype {exp['archetype']} (ID {r.customer_id}) CLV mismatch: "
            f"got {r.expected_clv}, expected {exp['clv']}"
        )
        # Run-rate baseline parity within atol=1e-6
        assert np.isclose(r.challengers.run_rate, exp["run_rate"], atol=1e-6, rtol=1e-6), (
            f"Archetype {exp['archetype']} (ID {r.customer_id}) run-rate mismatch: "
            f"got {r.challengers.run_rate}, expected {exp['run_rate']}"
        )
        # CRM Segment exact match
        assert r.segment == exp["segment"], (
            f"Archetype {exp['archetype']} (ID {r.customer_id}) segment mismatch: "
            f"got '{r.segment}', expected '{exp['segment']}'"
        )
