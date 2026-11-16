# Metrics Provenance & Reporting Process Rules

## 1. The Core Integrity Mandate
**Every number in any report, model card, pull request, or architectural summary must cite a concrete source artifact and the exact CLI command or script that generates it. Any number lacking a verifiable source artifact and generation script is strictly prohibited and must be marked UNVERIFIED or removed immediately.**

## 2. Deterministic Generation Standard
- **No Hand-Typed Numbers**: Metrics tables in user-facing documentation must be rendered deterministically by [scripts/render_metrics.py](../scripts/render_metrics.py) directly from persisted CSV and Parquet artifacts.
- **CI Parity Enforcement**: The continuous integration pipeline runs `python scripts/render_metrics.py --check`. Any drift between committed documentation and underlying artifacts immediately fails the build with exit code 1.

## 3. Test-Set Honesty & Evaluation Accounting
- **Single-Shot Holdout Invariant**: The Champion model (`models/clv_tweedie_tuned_2010_09_10.joblib`) was evaluated exactly **once** on the holdout test cohort (`2011-09-10`).
- **Reference Benchmark Disclosure**: Any challenger models, heuristic baselines, or statistical benchmarks evaluated afterwards on the same test set are strictly **post-hoc reference comparisons** (`evaluation_counter >= 2`). They serve as directional comparisons and must never be used to tune or select model architectures.
- **Model Selection Protocol**: All model promotion decisions, hyperparameter tuning, and hypothesis testing (such as core 90% vs whale slices) must be performed strictly using Out-Of-Fold (OOF) cross-validation on the training dataset (`2010-09-10`), never on the holdout test set.

## 4. Artifact Verification & Audit Trail
- Every persisted artifact in `data/processed/clv/` must have its cryptographic SHA-256 hash recorded in `docs/METRICS.md` and `artifacts.sha256`.
- Pull requests modifying evaluation metrics must demonstrate exact numerical reproducibility using deterministic random seeds (`seed = 42`).
