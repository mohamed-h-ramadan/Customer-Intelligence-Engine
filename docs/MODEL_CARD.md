# Model Card: Customer Intelligence Engine — 90-Day Gross CLV Engine

## 1. Model Details & Architecture
- **Model Name**: Customer Decision Engine — Gross CLV Regressor
- **Current Candidate Champion**: Tweedie Generalized Linear Model (`TweedieRegressor(power=1.65, alpha=0.5, link="log")`)
- **Preprocessor**: `build_linear_clv_preprocessor(top_n_countries=3)` (StandardScaler on continuous RFM features, OneHotEncoder on country).
- **Target**: Gross 90-day forward customer spending (GBP).
- **Validated Operating Regime**:
  - Lookback Window: Exactly 270 days.
  - Forward Prediction Horizon: Exactly 90 days.
  - Seasonality: Q4 Holiday Regime (September 10 cutoff predicting September–December spending).

---

## 2. Test-Set Honesty & Benchmark Disclosure
- **Single-Shot Holdout Invariant**: The Champion model was evaluated exactly once on the holdout test cohort as-of `2011-09-10` ($n=3,224$).
- **Post-Hoc Motivation Disclosure**: The idea of comparing Tweedie against a calibrated run-rate was motivated by post-hoc observation of holdout test results (specifically the $-37.38\%$ holiday under-forecasting of the heuristic run-rate).
- **Benchmark Evaluation Counter (`evaluation_counter = 2`)**: All holdout test comparisons for challengers (heuristic run-rate, calibrated run-rate) and future benchmarks are post-hoc reference comparisons.
- **Strict Prohibition of Test-Set Tuning**: The holdout test set is locked and must never be used to select models, tune hyperparameters, or choose decision thresholds. All promotion and selection decisions are made strictly on training data via out-of-fold cross-validation.

---

## 3. Pre-Registered Champion Promotion Rule

To maintain scientific integrity and prevent benchmark hacking, the promotion criteria are pre-registered below prior to executing out-of-fold comparative evaluation:

### Primary Metrics (Decision Gate)
1. **Mean Absolute Error (MAE, £)**: General prediction accuracy on the customer population.
2. **Normalized Gini**: Ranking quality across the entire customer spend distribution.
3. **Top-10% Spend Capture Ratio**: Marketing targeting efficiency among top predicted spenders.
4. **Absolute Total Revenue Error ($|\text{total\_rev\_error\_pct}|$ %)**: Portfolio-level fiscal forecasting accuracy.

### Secondary Metrics (Explanatory Only)
- Root Mean Squared Error (Overall, Excl. Top-1 Whale, Excl. Top-5 Whales).
- Spearman Rank Correlation ($\rho$).
- Decile Mean Absolute Calibration Error (£).
- Core 90% vs. Whale Slice Error.
*(Secondary metrics provide diagnostic context but cannot overturn the primary decision gate).*

### The Promotion Decision Rule
> **The complex Champion (Tweedie GLM) stays Champion only if the paired 95% bootstrap / fold delta CI (Tweedie minus Calibrated Run-Rate) shows it is statistically significantly superior (paired CI strictly excludes zero in its favor) on at least one primary metric AND the simpler Calibrated Run-Rate is NOT statistically significantly superior on any primary metric.**
> 
> **If the Calibrated Run-Rate ties or outperforms Tweedie on primary metrics, the Calibrated Run-Rate becomes the default served prediction (`prediction`), and Tweedie GLM is served as a challenger (`challengers`).**

### Exploratory Hybrid Hypothesis
- **Hypothesis**: Tweedie GLM provides superior decile calibration on the core 90% customer base, while heuristic/calibrated run-rate is more resilient to extreme whale accounts ($> 99\text{th}$ percentile).
- **Protocol**: This hypothesis will be evaluated directionally on training out-of-fold slices for scientific understanding. In accordance with governance rules, this exploratory slice analysis **must not influence the promotion decision**, which is governed exclusively by the pre-registered primary metrics.

---

## 4. Serving Contract & Regime Guardrails
- Serving responses return the default prediction alongside all challenger predictions.
- If input features fall outside validated training ranges, lookback is not 270 days, or current date is outside September–December, the engine emits explicit structured warning codes (`OUT_OF_RANGE_*`, `OUT_OF_SEASON_CALIBRATED_RUN_RATE`).
- Serving models must originate from verified artifacts matching `artifacts.sha256`.

---

## 5. Model Reproducibility & Dual Contract Architecture
- **Frozen Serving Parity (Golden Invariant)**: In serving environments, the frozen candidate champion (`models/clv_tweedie_tuned_2010_09_10.joblib`) must reproduce persisted test predictions (`data/processed/clv/test_predictions_2011-09-10.parquet`) within `atol = 1e-9` ($0.00\text{e}+00$ observed on Windows local; passes at `atol = 1e-9` in Ubuntu Linux CI Run `43950135466` with exact Linux numeric diff not recorded).
- **Retraining Parity Across Platforms**: When refitting the model pipeline from training features (`data/processed/features/features_train_2010-09-10.parquet`), predictions, coefficients, and intercept are validated via `check_retraining_parity()` against configured tolerances (`rtol = 1e-6`, `atol = 1e-6`, per `config/params.yaml`). This accommodates cross-platform iterative optimization solver variations (e.g. `4.2275e-07` max abs prediction diff observed in CI Run `43946052472`; full Linux coefficient/intercept diffs and environment versions pending `retrain_parity_report.json` artifact verification) while raising `ModelParityError` on genuine architectural or data drift.
