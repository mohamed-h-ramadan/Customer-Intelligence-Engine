"""Configuration loading and validation for Customer Intelligence Engine (CIE)."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


class CutoffsConfig(BaseModel):
    train_as_of: str = "2010-09-10"
    test_as_of: str = "2011-09-10"
    deploy_as_of: str = "2011-12-10"


class WindowsConfig(BaseModel):
    lookback_days: int = 270
    horizon_days: int = 90
    recency_max_days: int = 270


class PathsConfig(BaseModel):
    raw_data: str = "data/raws/online_retail_II.xlsx"
    sales_clean: str = "data/processed/sales_clean.parquet"
    returns_clean: str = "data/processed/returns_clean.parquet"
    admin_rows: str = "data/processed/admin_rows.parquet"
    guest_transactions: str = "data/processed/guest_transactions.parquet"
    features_train: str = "data/processed/features/features_train_2010-09-10.parquet"
    features_test: str = "data/processed/features/features_test_2011-09-10.parquet"
    rfm_snapshot_deploy: str = "data/processed/features/rfm_snapshot_2011-12-10.parquet"
    segments_t_seg: str = "data/processed/segmentation/segments_t_seg_2011-09-10.parquet"
    segments_deploy: str = "data/processed/segmentation/segments_deploy_2011-12-10.parquet"
    segment_profiles: str = "data/processed/segmentation/segment_profiles.parquet"
    clv_test_predictions: str = "data/processed/clv/test_predictions_2011-09-10.parquet"
    clv_champion_model: str = "models/clv_tweedie_tuned_2010_09_10.joblib"
    segmentation_pipeline: str = "models/rfm_kmeans_k5_pipeline.joblib"

    def resolve(self, key: str, root: Path | None = None) -> Path:
        """Resolve a relative path against the project root."""
        base = root or get_project_root()
        rel = getattr(self, key)
        return (base / rel).resolve()


class PipelineConfig(BaseModel):
    seed: int = 42
    cutoffs: CutoffsConfig = Field(default_factory=CutoffsConfig)
    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


class SegmentationConfig(BaseModel):
    n_clusters: int = 5
    recency_transform: str = "sqrt"
    n_init: int = 20
    random_state: int = 42
    feature_cols: list[str] = Field(
        default_factory=lambda: ["recency_days", "frequency", "monetary"]
    )
    cluster_to_segment: dict[int, str] = Field(
        default_factory=lambda: {
            0: "At-Risk Repeaters",
            1: "VIP Champions",
            2: "Hibernating / Lost",
            3: "Promising Recent Buyers",
            4: "Loyal Regulars",
        }
    )
    crm_actions: dict[str, str] = Field(default_factory=dict)


class ChampionConfig(BaseModel):
    name: str = "tweedie_glm_tuned"
    model_version: str = "v1.0-frozen-2010-09-10"
    power: float = 1.65
    alpha: float = 0.5
    link: str = "log"
    top_n_countries: int = 3
    top_countries: list[str] = Field(
        default_factory=lambda: ["United Kingdom", "Germany", "France"]
    )


class RunRateChallengerConfig(BaseModel):
    name: str = "run_rate_heuristic"
    model_version: str = "v1.0-run-rate"
    lookback_days: int = 270
    horizon_days: int = 90


class CalibratedRunRateChallengerConfig(BaseModel):
    name: str = "calibrated_run_rate"
    model_version: str = "v1.0-calibrated-run-rate"
    lookback_days: int = 270
    horizon_days: int = 90
    expansion_factor: float = 1.42746205196053


class BaselineValueConfig(BaseModel):
    name: str
    model_version: str
    value: float


class ChallengersConfig(BaseModel):
    run_rate: RunRateChallengerConfig = Field(default_factory=RunRateChallengerConfig)
    calibrated_run_rate: CalibratedRunRateChallengerConfig = Field(
        default_factory=CalibratedRunRateChallengerConfig
    )
    train_mean: BaselineValueConfig = Field(
        default_factory=lambda: BaselineValueConfig(
            name="train_mean_baseline",
            model_version="v1.0-train-mean",
            value=752.8095052664189,
        )
    )
    train_median: BaselineValueConfig = Field(
        default_factory=lambda: BaselineValueConfig(
            name="train_median_baseline",
            model_version="v1.0-train-median",
            value=222.08,
        )
    )


class ClvConfig(BaseModel):
    champion: ChampionConfig = Field(default_factory=ChampionConfig)
    challengers: ChallengersConfig = Field(default_factory=ChallengersConfig)


class FeatureRange(BaseModel):
    min: float
    max: float


class RegimeConfig(BaseModel):
    validated_lookback_days: int = 270
    validated_horizon_days: int = 90
    validated_cutoff_months: list[int] = Field(default_factory=lambda: [9, 10, 11, 12])
    feature_ranges: dict[str, FeatureRange] = Field(default_factory=dict)


class AppConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    clv: ClvConfig = Field(default_factory=ClvConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)


@lru_cache(maxsize=1)
def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from a YAML file, caching the parsed config."""
    if config_path is None:
        root = get_project_root()
        path = root / "config" / "params.yaml"
    else:
        path = Path(config_path)

    if not path.is_file():
        # Fallback to default configuration if YAML file is absent
        return AppConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig.model_validate(data)
