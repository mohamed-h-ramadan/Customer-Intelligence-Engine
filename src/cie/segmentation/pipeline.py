"""Unsupervised RFM segmentation pipeline stage for Customer Intelligence Engine."""

import joblib
import pandas as pd

from cie.config import AppConfig, get_project_root, load_config
from src.features import build_features, build_target
from src.segmentation import (
    RFM_COLUMNS,
    build_rfm_pipeline,
    profile_clusters,
    score_rfm_rules,
)


def run_segmentation_stage(config: AppConfig | None = None) -> dict[str, pd.DataFrame]:
    """Execute unsupervised K=5 RFM segmentation, assign clusters, and generate profiles.

    Returns:
        dict containing 'segments_t_seg', 'segments_deploy', and 'segment_profiles'.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    sales_path = config.pipeline.paths.resolve("sales_clean", root)
    returns_path = config.pipeline.paths.resolve("returns_clean", root)

    sales_clean = pd.read_parquet(sales_path)
    returns_clean = pd.read_parquet(returns_path)

    # 1. Full history RFM at T_seg (2011-09-10)
    t_seg = pd.Timestamp(config.pipeline.cutoffs.test_as_of)
    rfm_seg_df = build_features(sales_clean, returns_clean, cutoff=t_seg, lookback_days=None)

    # Build and fit K=5 pipeline
    rfm_pipeline = build_rfm_pipeline(
        n_clusters=config.segmentation.n_clusters,
        feature_cols=config.segmentation.feature_cols,
        recency_transform=config.segmentation.recency_transform,
        n_init=config.segmentation.n_init,
        random_state=config.segmentation.random_state,
    )
    kmeans_labels = rfm_pipeline.fit_predict(rfm_seg_df[RFM_COLUMNS])

    # Cluster-to-segment mapping
    c_to_s = config.segmentation.cluster_to_segment
    rule_df = score_rfm_rules(rfm_seg_df)

    # T_seg assignment table with 90-day forward targets
    t_seg_targets = build_target(
        sales_clean,
        returns_clean,
        cutoff=t_seg,
        horizon_days=config.pipeline.windows.horizon_days,
        eligible_ids=rfm_seg_df.index,
    )

    t_seg_export = rfm_seg_df[
        [
            "recency_days",
            "frequency",
            "monetary",
            "aov",
            "return_rate",
            "is_single_invoice",
            "is_uk",
            "country",
        ]
    ].copy()
    t_seg_export["cluster"] = kmeans_labels
    t_seg_export["segment"] = t_seg_export["cluster"].map(c_to_s)
    t_seg_export["rule_segment"] = rule_df.loc[t_seg_export.index, "rule_segment"]
    t_seg_export = t_seg_export.join(t_seg_targets)

    # 2. Deploy Snapshot assignment (2011-12-10)
    deploy_path = (
        root
        / "data"
        / "processed"
        / "features"
        / f"rfm_snapshot_{config.pipeline.cutoffs.deploy_as_of}.parquet"
    )
    if deploy_path.is_file():
        rfm_deploy_df = pd.read_parquet(deploy_path)
    else:
        rfm_deploy_df = build_features(
            sales_clean,
            returns_clean,
            cutoff=pd.Timestamp(config.pipeline.cutoffs.deploy_as_of),
            lookback_days=None,
        )

    deploy_labels = rfm_pipeline.predict(rfm_deploy_df[RFM_COLUMNS])
    deploy_rule_df = score_rfm_rules(rfm_deploy_df)

    deploy_export = rfm_deploy_df[
        [
            "recency_days",
            "frequency",
            "monetary",
            "aov",
            "return_rate",
            "is_single_invoice",
            "is_uk",
            "country",
        ]
    ].copy()
    deploy_export["cluster"] = deploy_labels
    deploy_export["segment"] = deploy_export["cluster"].map(c_to_s)
    deploy_export["rule_segment"] = deploy_rule_df.loc[deploy_export.index, "rule_segment"]

    # 3. Cluster Profiling Table
    profiles = profile_clusters(
        rfm_deploy_df,
        deploy_labels,
        feature_cols=["recency_days", "frequency", "monetary", "return_rate"],
        relative_cols=["recency_days", "frequency", "monetary"],
    )
    profiles["segment"] = profiles.index.map(c_to_s)
    profiles["recommended_crm_action"] = profiles["segment"].map(config.segmentation.crm_actions)

    # Export models and parquets
    models_dir = root / "models"
    seg_dir = root / "data" / "processed" / "segmentation"
    models_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rfm_pipeline, models_dir / "rfm_kmeans_k5_pipeline.joblib")
    t_seg_export.to_parquet(
        seg_dir / f"segments_t_seg_{config.pipeline.cutoffs.test_as_of}.parquet"
    )
    deploy_export.to_parquet(
        seg_dir / f"segments_deploy_{config.pipeline.cutoffs.deploy_as_of}.parquet"
    )
    profiles.to_parquet(seg_dir / "segment_profiles.parquet")
    profiles.to_csv(seg_dir / "segment_profiles.csv")

    return {
        "segments_t_seg": t_seg_export,
        "segments_deploy": deploy_export,
        "segment_profiles": profiles,
    }
