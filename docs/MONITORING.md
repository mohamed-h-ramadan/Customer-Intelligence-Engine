# Customer Intelligence Engine — Monitoring Plan & Phase A Analysis (Draft)

> **Provenance Mandate ([docs/PROCESS_RULES.md](PROCESS_RULES.md))**: Every quantitative figure in this document cites its generating artifact and command (`python scripts/run_monitoring_phase_a.py`).
> **Scope**: This document covers **Step 8, Phase A (Offline Monitoring Analysis)**. No API runtime behavior has been changed. CI checks for the latest local monitoring commits are pending manual push (**UNVERIFIED in CI**; see [docs/STATUS.md](STATUS.md)).

---

## 1. Philosophy: Leading vs. Lagging Indicators & Model-Agnostic Scope

1. **90-Day Label Lag**:
   - For any prediction cohort scored at cutoff $T$, ground-truth 90-day gross spending labels ($y \in [T, T + 90\text{d})$) arrive **90 days late**.
   - Consequently, **Regime Compliance**, **Feature Drift (PSI/KS)**, and **Prediction/Segment-Mix Drift** are our **leading operational indicators** (available immediately at scoring time).
   - **Cohort Backtesting** is our **lagging ground-truth indicator** (available 90 days post-cutoff when full transaction settlement closes).
2. **Model-Agnostic Monitoring**:
   - Monitoring tracks **all three forecasting columns** identically (`champion_tweedie`, `calibrated_run_rate`, and `heuristic_run_rate`) alongside RFM segment proportions, ensuring monitoring continuity regardless of which model serves as the default prediction after the pending champion decision refactor.

---

## 2. Validated Regime Rule (Approved Rule — API Integration Pending Step 6 Refactor)

- **Limitation of Current Runtime Validation ([src/cie/core/validation.py](../src/cie/core/validation.py))**:
  - Currently checks `ts.month in [9, 10, 11, 12]`, which conflates prediction horizon months with cutoff months. Passing `as_of=2011-12-10` returns `[[]]` (`list[list[str]]` — one empty warning list `[]` for the 1 input customer row), meaning **zero warnings** are emitted even though `[2011-12-10, 2012-03-09)` has `0/90` (`0%`) overlap with `[Sep 10, Dec 9)`.
- **Integer-Day Horizon-Overlap Rule ([src/cie/monitoring/regime.py](../src/cie/monitoring/regime.py), API untouched)**:
  - Requires `lookback_days == 270`, `horizon_days == 90`, and **`overlap_days >= 76`** ($\ge 76/90$ days overlapping `[Sep 10, Dec 9)`, mathematically equivalent to `|as_of - Sep 10| <= 14 days`: Aug 27–Sep 24).
  - **Judgment-Call Caveat**: This $\pm 14$-day (`overlap_days >= 76`) tolerance is an operational **judgment call**. Only exact `Sep 10` cutoffs (`2010-09-10` and `2011-09-10`) were empirically validated; **no empirical support exists for other offsets.**
  - Evaluation across test dates:
    - `2011-09-10`: `90/90` days (`100.0%`) -> `in_regime = True`
    - `2011-09-24` (`+14d`): `76/90` days -> `in_regime = True`
    - `2011-09-25` (`+15d`): `75/90` days -> `in_regime = False`
    - `2011-11-01`: `38/90` days (`42.2%`) -> `in_regime = False` (52 days post-Dec 9)
    - `2011-12-10`: `0/90` days (`0.0%`) -> `in_regime = False` (forecasts Dec 10–Mar 9)

---

## 3. Initial Thresholds *(Initial, To Be Tuned)*

All statistical thresholds below are conventional starting baselines (*initial, to be tuned* once multiple seasonal cohorts accumulate):

| Signal | Metric | Warn Threshold *(Initial)* | Alert Threshold *(Initial)* | Multiple-Testing & Method Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Feature & Prediction Drift** | Population Stability Index (PSI) | `>= 0.10` | `>= 0.25` | 10 quantile bins fit on reference (`expected`); exact unique categories for binary/low-cardinality features (`len(edges) <= 2`); `1e-6` floor. |
| **Distribution Shift Significance** | Two-sample Kolmogorov-Smirnov (KS) | `p_bonf < 0.01` | `p_bonf < 0.01` & `PSI >= 0.10` | **Multiple-testing caveat**: Testing ~25 correlated numeric features simultaneously inflates false positives; Bonferroni correction (`min(1.0, p * 25)`) controls FWER, and at large $N > 3,200$ KS detects small shifts (`KS ~ 0.05–0.07`) even when `PSI < 0.03`. |
| **Out-of-Regime Share** | Share of scored requests with `in_regime == False` | `> 5%` | `> 20%` | Flags non-Q4 seasonality (`overlap_days < 76`) or `lookback_days != 270` or `horizon_days != 90`. |
| **Backtest Calibration Error** | Cohort MAE or Aggregate Total Revenue Error | Near 95% CI boundary | **Outside pre-registered 95% CI** | Checked at $T + 90\text{d}$ (`test_metrics_with_95ci.csv` / `calibrated_run_rate_test_metrics_with_95ci.csv`). |

---

## 4. Phase A Empirical Findings (Generated via `python scripts/run_monitoring_phase_a.py`)

### 4.1 Feature Drift (`2011-12-10` 270d Cohort, $n=3,945$ vs `2011-09-10` Test $n=3,224$ & `2010-09-10` Train $n=3,228$)
*Source Artifacts*: [`data/processed/features/features_deploy_2011-12-10_270d.parquet`](../data/processed/features/features_deploy_2011-12-10_270d.parquet), [`data/processed/monitoring/feature_drift_report.csv`](../data/processed/monitoring/feature_drift_report.csv), [`data/processed/monitoring/features_deploy_2011-12-10_270d_provenance.json`](../data/processed/monitoring/features_deploy_2011-12-10_270d_provenance.json).

- **Core 270-Day Scale & Loyalty Fundamentals — STABLE (`PSI < 0.025`)**:
  - `monetary`: `PSI = 0.0111` (vs Test), `0.0144` (vs Train) -> **OK**
  - `frequency`: `PSI = 0.0109` (vs Test), `0.0054` (vs Train) -> **OK**
  - `aov`: `PSI = 0.0094` (vs Test), `0.0082` (vs Train) -> **OK**
  - `is_uk`: `PSI = 0.0007` (vs Test), `0.0034` (vs Train) -> **OK**
  - `is_single_invoice`: `PSI = 0.0061` (vs Test), `0.0027` (vs Train) -> **OK**
- **Drifting Features (`PSI >= 0.10`) — Classified as `expected seasonal/regime effect`**:
  1. `tenure_days_in_window` (`PSI = 0.2257` WARN vs Test; `0.3322` ALERT vs Train): Lower tail (`10th` pct `52.4d -> 35.4d`) shifts downward due to ~720 newly onboarded Q4 holiday buyers ($n=3,945$ vs $3,224$), plus edge shift vs Dec 2009 dataset start.
  2. `spend_last_90d_ratio` (`PSI = 0.1544` WARN vs Test; `0.1765` WARN vs Train): Trailing 90d at `2011-12-10` spans Q4 peak (`Sep–Dec`), raising mean 90d spend share from `34–35%` (summer) to `50.0%`.
  3. `recency_days` (`PSI = 0.1266` WARN vs Test; `0.1534` WARN vs Train): Active buyers shop more recently during Q4 peak (`mean 86.5d -> 70.1d`).
  4. `invoices_last_90d` (`PSI = 0.0987` OK vs Test; `0.1028` WARN vs Train): Trailing 90d captures Q4 purchase frequency surge (`mean 1.22 -> 1.67`).

### 4.2 Prediction & Segment-Mix Drift (All Three Models)
*Source Artifacts*: [`data/processed/monitoring/prediction_drift_report.csv`](../data/processed/monitoring/prediction_drift_report.csv), [`data/processed/monitoring/segment_mix_drift.csv`](../data/processed/monitoring/segment_mix_drift.csv).

- **Prediction PSI (`deploy` vs `test`, all `OK < 0.10`)**:
  - `champion_tweedie`: `PSI = 0.0173` (`mean £694.11 -> £781.01`, `0.00%` exact zeros, `<£10` `0.03% -> 0.00%`, `<£50` `0.71% -> 0.58%`).
  - `calibrated_run_rate`: `PSI = 0.0111` (`mean £716.92 -> £811.19`, `0.00%` exact zeros, `<£10` `0.34% -> 0.38%`, `<£50` `4.90% -> 4.54%`).
  - `heuristic_run_rate`: `PSI = 0.0111` (`mean £502.23 -> £568.27`, `0.00%` exact zeros, `<£10` `0.50% -> 0.46%`, `<£50` `11.57% -> 9.63%`).
- **Segment-Mix Shift (`segments_t_seg_2011-09-10` $n=5,256$ vs `segments_deploy_2011-12-10` $n=5,852$)**:
  - Overall categorical `PSI = 0.0208` (**OK**): Q4 holiday activation lifted `Promising Recent Buyers` (`+2.68 pp`), `Loyal Regulars` (`+2.39 pp`), and `VIP Champions` (`+1.74 pp`), while reducing `At-Risk Repeaters` (`-4.25 pp`) and `Hibernating / Lost` (`-2.56 pp`).

### 4.3 Backtest Report (`2011-09-10` Cohort, $n=3,224$)
*Source Artifacts*: [`data/processed/monitoring/backtest_report_2011-09-10.json`](../data/processed/monitoring/backtest_report_2011-09-10.json), [`data/processed/monitoring/backtest_decile_calibration.csv`](../data/processed/monitoring/backtest_decile_calibration.csv), [`data/processed/monitoring/backtest_cohort_slices.csv`](../data/processed/monitoring/backtest_cohort_slices.csv), and persisted CSVs under `data/processed/clv/`.

- **Aggregate Metrics & 95% CIs**: Exact parity with `test_metrics_with_95ci.csv` & `calibrated_run_rate_test_metrics_with_95ci.csv` (e.g., Tweedie Total Rev Error `-13.45%` `[-23.47%, -1.44%]`, MAE `£600.1`; Calibrated Run-Rate Rev Error `-10.61%` `[-18.74%, -0.20%]`, MAE `£564.5`).
- **Whale Concentration Coincidence**:
  - **Top-5 by actual 90d spend**: `[18102, 17450, 14646, 14911, 14096]` (`17.26%` of total spend).
  - **Top-5 by Tweedie squared error**: `[18102, 17450, 14096, 14646, 14911]` (`84.06%` of total SSE).
  - **Coincidence**: Identical **as sets** (`True`), different **rank order** (`False` — customer `14096` jumps from #5 spend [`£53,054.54`] to #3 squared error because Tweedie predicted `£221.71`).
- **Sub-Cohort Slices**:
  - `Core Non-Whales (n=3,219)`: Tweedie achieves `-0.96%` total revenue error (vs Run-Rate `-31.17%`).
  - `Repeat (n=1,894)` vs `Single-Invoice (n=1,330)`: Tweedie `-16.71%` vs `+10.62%` revenue error (`61.80%` zero future spend among single-invoice buyers).
  - `UK (n=2,899)` vs `Export (n=325)`: Tweedie `-14.94%` (`Gini 0.7519`) vs `-6.10%` (`Gini 0.8900`).

---

## 5. Retraining & Recalibration Decision Table

| Trigger Condition | Threshold *(Initial, To Be Tuned)* | Verification Step | Prescribed Action |
| :--- | :--- | :--- | :--- |
| **Core Feature Drift Alert** | `PSI >= 0.25` on `>= 2` core non-seasonal features (`monetary`, `frequency`, `aov`, `is_uk`) | Audit upstream ETL schema (`sales_clean` row accounting) and currency/pricing changes | **Investigate** pipeline first; **Retrain** pipeline if underlying customer behavior shifted permanently |
| **Prediction Drift Alert** | `PSI >= 0.25` across champion or run-rate columns, or zero/near-zero share shifts `> 5 pp` | Cross-check against segment-mix shift and recent whale composition | **Investigate** scoring cohort mix; if persistent with degrading backtest, **Retrain** |
| **High Out-of-Regime Traffic** | Out-of-regime requests (`overlap_days < 76` or `lookback != 270d` or `horizon != 90d`) `> 20%` of traffic | Inspect `as_of` cutoff dates and API caller configurations | **Block Q4 calibrated predictions / Emit Regime Warnings**; calibrate/train a season-specific or multi-quarter model before serving off-season |
| **Lagging Backtest Degradation** | Backtest Aggregate Revenue Error or MAE falls outside pre-registered 95% CI, while ranking (`Gini` / `Spearman`) remains stable | Check expansion factor shift ($y / \text{run\_rate}$) across latest closed 90d window | **Recalibrate** the run-rate multiplier (`calibrated_run_rate`) on recent training windows |
| **Lagging Ranking Collapse** | Backtest Normalized Gini or Top-10% Capture falls below lower 95% CI bound | Run permutation feature importance & slice diagnostics (`segment`, `repeat` vs `single`) | **Retrain** CLV models and re-run CV selection protocol |

---

## 6. Known Limitations

1. **Strict `[Sep 10, Dec 9)` Calibration Regime**: Both the Tweedie GLM weights and the calibrated run-rate expansion factor (`1.42746205196053`) were derived and validated exclusively for the 90-day Q4 holiday window `[Sep 10, Dec 9)` using a 270-day lookback (`2010-09-10` and `2011-09-10`). Deploying at `2011-12-10` (`[Dec 10, Mar 9)`, 0% horizon overlap) applies Q4 expansion to winter post-Christmas demand without empirical backing.
2. **Threshold Tuning Pending History**: Because the dataset spans two Q4 cutoffs (`2010` and `2011`), empirical variance across non-Q4 quarters cannot yet be calibrated; initial PSI thresholds (`0.10` / `0.25`) must be revisited once 4+ rolling quarters are observed.

---

## 7. Planned Phase B Work *(Not Yet Implemented)*

The following runtime monitoring components belong to **Step 8, Phase B** and remain **unimplemented** pending Phase A review and regime approval:
- Structured JSON request/response telemetry logging in the FastAPI service (`/v1/score`, `/v1/score/batch`)
- Production log replay script aggregating real request payloads into rolling drift snapshots
- `cie monitor-report` CLI subcommand generating automated JSON & static HTML drift dashboards
- Automated CI publishing of static HTML monitoring reports as workflow artifacts
