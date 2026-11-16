"""Generate scripts/smoke_fixture.json from persisted test artifacts.

Selected customer archetypes:
1. normal: Customer 12352 (multi-invoice, moderate monetary, standard repeat buyer)
2. single_invoice: Customer 12350 (single-invoice, frequency=1, null mean_interpurchase_days)
3. whale: Customer 18102 (extreme top-tier spender, £123k+ gross spend)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cie.config import get_project_root  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402


def generate_fixture() -> Path:
    root = get_project_root()
    feat_path = root / "data" / "processed" / "features" / "features_test_2011-09-10.parquet"
    pred_path = root / "data" / "processed" / "clv" / "test_predictions_2011-09-10.parquet"

    feats_df = pd.read_parquet(feat_path)
    preds_df = pd.read_parquet(pred_path)

    archetypes = {
        "normal": 12352,
        "single_invoice": 12350,
        "whale": 18102,
    }

    fixture_data = {
        "metadata": {
            "description": "Smoke test customer fixtures generated from holdout test artifacts (as_of=2011-09-10)",
            "source_features": "data/processed/features/features_test_2011-09-10.parquet",
            "source_predictions": "data/processed/clv/test_predictions_2011-09-10.parquet",
            "archetypes": list(archetypes.keys()),
        },
        "customers": {},
    }

    for arch_name, cid in archetypes.items():
        feat_row = feats_df.loc[cid, FEATURE_COLUMNS].to_dict()
        pred_row = preds_df.loc[cid]

        # Convert numpy types to standard Python types; replace NaN with None for valid JSON null
        clean_features = {}
        for col, val in feat_row.items():
            if pd.isna(val) or val is None or (isinstance(val, float) and np.isnan(val)):
                clean_features[col] = None
            elif isinstance(val, (np.integer, int)):
                clean_features[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                clean_features[col] = float(val)
            else:
                clean_features[col] = str(val)

        fixture_data["customers"][arch_name] = {
            "customer_id": int(cid),
            "archetype": arch_name,
            "features": clean_features,
            "expected_clv": float(pred_row["pred_frozen_tweedie"]),
            "expected_run_rate": float(pred_row["pred_run_rate"]),
            "expected_segment": str(pred_row["segment"]),
            "actual_90d_spend": float(pred_row["target"]),
        }

    out_path = root / "scripts" / "smoke_fixture.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture_data, f, indent=2)
        f.write("\n")

    print(f"Generated smoke test fixture at {out_path}")
    return out_path


if __name__ == "__main__":
    generate_fixture()
