"""Feature engineering pipeline module for Customer Intelligence Engine.

Builds window-bounded feature sets and forward-looking CLV targets.
"""

import pandas as pd

from cie.config import AppConfig, get_project_root, load_config
from src.features import build_features, build_target


def run_features_stage(config: AppConfig | None = None) -> dict[str, pd.DataFrame]:
    """Execute feature generation for train, test, and deployment snapshots.

    Returns:
        dict mapping artifact name to generated DataFrame.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    sales_path = config.pipeline.paths.resolve("sales_clean", root)
    returns_path = config.pipeline.paths.resolve("returns_clean", root)

    sales_clean = pd.read_parquet(sales_path)
    returns_clean = pd.read_parquet(returns_path)

    lookback = config.pipeline.windows.lookback_days
    horizon = config.pipeline.windows.horizon_days

    # 1. Train Features & Target (2010-09-10)
    t_train = pd.Timestamp(config.pipeline.cutoffs.train_as_of)
    train_feat = build_features(sales_clean, returns_clean, cutoff=t_train, lookback_days=lookback)
    train_target = build_target(
        sales_clean,
        returns_clean,
        cutoff=t_train,
        horizon_days=horizon,
        eligible_ids=train_feat.index,
    )
    features_train = train_feat.join(train_target)

    # 2. Test Features & Target (2011-09-10)
    t_test = pd.Timestamp(config.pipeline.cutoffs.test_as_of)
    test_feat = build_features(sales_clean, returns_clean, cutoff=t_test, lookback_days=lookback)
    test_target = build_target(
        sales_clean,
        returns_clean,
        cutoff=t_test,
        horizon_days=horizon,
        eligible_ids=test_feat.index,
    )
    features_test = test_feat.join(test_target)

    # 3. Descriptive RFM Snapshot for Deployment (2011-12-10, Full History)
    t_deploy = pd.Timestamp(config.pipeline.cutoffs.deploy_as_of)
    rfm_snapshot_deploy = build_features(
        sales_clean,
        returns_clean,
        cutoff=t_deploy,
        lookback_days=None,
    )

    # Export to data/processed/features/
    feat_dir = root / "data" / "processed" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    features_train.to_parquet(
        feat_dir / f"features_train_{config.pipeline.cutoffs.train_as_of}.parquet"
    )
    features_test.to_parquet(
        feat_dir / f"features_test_{config.pipeline.cutoffs.test_as_of}.parquet"
    )
    rfm_snapshot_deploy.to_parquet(
        feat_dir / f"rfm_snapshot_{config.pipeline.cutoffs.deploy_as_of}.parquet"
    )

    return {
        "features_train": features_train,
        "features_test": features_test,
        "rfm_snapshot_deploy": rfm_snapshot_deploy,
    }
