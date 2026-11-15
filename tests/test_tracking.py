"""Unit tests for MLflow experiment tracking and model registry integration."""

from cie.config import get_project_root
from cie.tracking.mlflow_tracker import (
    compute_file_sha256,
    get_git_commit_sha,
    track_and_register_models,
)


def test_git_commit_sha():
    sha = get_git_commit_sha()
    assert isinstance(sha, str)
    assert len(sha) == 40 or sha == "unknown"


def test_compute_file_sha256():
    root = get_project_root()
    params_file = root / "config" / "params.yaml"
    sha = compute_file_sha256(params_file)
    assert isinstance(sha, str)
    assert len(sha) == 64


def test_track_and_register_models():
    res = track_and_register_models()
    assert "champion_run_id" in res
    assert "challenger_run_id" in res
    assert res["experiment_name"] == "customer-intelligence-engine"
    assert res["champion_version"] is not None
