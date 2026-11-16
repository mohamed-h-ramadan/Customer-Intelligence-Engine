# Architectural Decision Records (ADR)

## ADR-001: Retention of Raw Dataset in Early Git History
- **Status**: Accepted
- **Date**: 2026-11-16
- **Context**: The raw Excel dataset (`data/raws/online_retail_II.xlsx`, 43.51 MiB) was committed in the repository's initial commit (`5d4ab69`) and already exists on `origin/main`.
- **Decision**: Keep history as-is. Do not rewrite history and do not force-push to `main`.
- **Rationale**:
  1. The file exists in early git history (`5d4ab69` through `3304215`), but has been completely untracked from the working tree since commit `d32f64a`.
  2. The raw dataset is strictly ignored going forward via [.gitignore](.gitignore) (`data/raws/*` with exception for `data/raws/.gitkeep`).
  3. Total repository size is 47.77 MiB, well below GitHub's 100 MB per-file and 2 GB repository limits.
  4. Preserves cryptographic and hash stability across all existing commits and references.

---

## ADR-002: Dual Contract for Model Verification (Serving Invariant vs. Retraining Parity)
- **Status**: Accepted
- **Date**: 2026-11-16
- **Context**:
  In GitHub Actions CI on Ubuntu Linux (`ubuntu-latest`), running `run_train_stage()` refits the Tweedie GLM using scikit-learn's iterative L-BFGS-B solver. Because the frozen champion pipeline (`models/clv_tweedie_tuned_2010_09_10.joblib`) was fitted on Windows x86_64, floating-point solver termination thresholds produced a maximum absolute prediction discrepancy of $4.2275 \times 10^{-7}$ (£0.00042 on a £1,000 order). The previous bare assertion `assert max_diff <= 1e-9` failed on Linux while passing with $0.00\text{e}+00$ on Windows.
- **Decision**:
  Implement two distinct, mathematically justified contracts:
  1. **Serving Contract (Golden Invariant)**: In serving tests and golden parity tests (`tests/test_clv_reconciliation.py`, `tests/test_scoring.py`), inferences from the frozen model are compared against persisted holdout outputs (`data/processed/clv/test_predictions_2011-09-10.parquet`) with strict tolerance (`atol=1e-9` / $0.00\text{e}+00$ error). This guarantees zero training-serving skew in production.
  2. **Retraining Reproducibility Contract**: In `src/cie/clv/train.py`, replace bare assertions with `check_retraining_parity()`. Validate predictions, coefficients, and intercept against tolerances configured in `config/params.yaml` (`rtol=1e-6`, `atol=1e-6`), which accommodates the empirically measured $4.22 \times 10^{-7}$ cross-platform solver divergence while catching meaningful model drift. Raise explicit `ModelParityError` upon divergence.
