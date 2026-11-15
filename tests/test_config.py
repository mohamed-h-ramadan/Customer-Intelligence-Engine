"""Unit tests for cie.config loading and path resolution."""

from pathlib import Path

from cie.config import AppConfig, get_project_root, load_config


def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "config" / "params.yaml").is_file()


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.pipeline.seed == 42
    assert cfg.pipeline.cutoffs.train_as_of == "2010-09-10"
    assert cfg.pipeline.cutoffs.test_as_of == "2011-09-10"
    assert cfg.pipeline.cutoffs.deploy_as_of == "2011-12-10"
    assert cfg.pipeline.windows.lookback_days == 270
    assert cfg.pipeline.windows.horizon_days == 90


def test_segmentation_config():
    cfg = load_config()
    assert cfg.segmentation.n_clusters == 5
    assert len(cfg.segmentation.cluster_to_segment) == 5
    assert cfg.segmentation.cluster_to_segment[1] == "VIP Champions"
    assert cfg.segmentation.cluster_to_segment[4] == "Loyal Regulars"


def test_clv_config_and_challengers():
    cfg = load_config()
    assert cfg.clv.champion.name == "tweedie_glm_tuned"
    assert cfg.clv.champion.power == 1.65
    assert cfg.clv.champion.alpha == 0.5
    assert "United Kingdom" in cfg.clv.champion.top_countries

    # Challenger definitions
    assert cfg.clv.challengers.run_rate.lookback_days == 270
    assert cfg.clv.challengers.calibrated_run_rate.expansion_factor > 1.4
    assert cfg.clv.challengers.train_mean.value > 700.0


def test_regime_and_feature_ranges():
    cfg = load_config()
    assert cfg.regime.validated_lookback_days == 270
    assert cfg.regime.validated_horizon_days == 90
    assert 9 in cfg.regime.validated_cutoff_months
    assert len(cfg.regime.feature_ranges) >= 20
    assert "recency_days" in cfg.regime.feature_ranges
    assert cfg.regime.feature_ranges["recency_days"].max > 260.0


def test_path_resolution():
    cfg = load_config()
    root = get_project_root()
    resolved_raw = cfg.pipeline.paths.resolve("raw_data")
    assert resolved_raw == (root / "data" / "raws" / "online_retail_II.xlsx").resolve()
