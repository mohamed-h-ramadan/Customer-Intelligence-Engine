# Interview Playbook 03: Phase 3 — Cutoff Design & Feature Engineering Q&A

**Source Files**: [`3_cutoff_design_&_feature_engineering.ipynb`](file:///c:/Users/mosa/Desktop/Projects/Customer-Intelligence-Engine/Notebooks/3_cutoff_design_&_feature_engineering.ipynb), [`src/features.py`](file:///c:/Users/mosa/Desktop/Projects/Customer-Intelligence-Engine/src/features.py), [`tests/test_features.py`](file:///c:/Users/mosa/Desktop/Projects/Customer-Intelligence-Engine/tests/test_features.py), [`cutoff_design_and_feature_engineering_report.md`](file:///c:/Users/mosa/Desktop/Projects/Customer-Intelligence-Engine/reports/cutoff_design_and_feature_engineering_report.md)

---

### Q1: Why did you choose two seasonally aligned cutoffs (`T_train = 2010-09-10`, `T_test = 2011-09-10`) instead of a single chronological train/val/test split inside Year 2?
* **Direct Answer**: Retail spend surges more than **2x** during September–December (Phase 1 found Jan–Aug ~`£800k/mo` vs `>£1.5M/mo` Sep–Nov). Training on a Spring/Summer 90-day horizon and testing on Q4 (`Sep 10 – Dec 9`) confounds model calibration with massive seasonal expansion!
* **Aligned Architecture**:
  * **`T_train = 2010-09-10`** (`[2009-12-14, 2010-09-10)` lookback -> `[2010-09-10, 2010-12-09)` Q4 target): **`3,228` customers**, `£2.43M`, `41.60%` zeros.
  * **`T_test = 2011-09-10`** (`[2010-12-14, 2011-09-10)` lookback -> `[2011-09-10, 2011-12-09)` Q4 target): **`3,224` customers**, `£2.59M`, `42.25%` zeros.
  * Half-open target boundaries (`< 2011-12-09 00:00:00`) cleanly exclude **`604` partial-day transactions** on `2011-12-09 00:00–12:50`, keeping both target windows exactly **90.0 full days**.
* **Follow-up Trap**: *"Wait — don't you have customer overlap between Train 2010 and Test 2011?"*
  * **Counter**: **Yes (`1,855` customers = `57.54%` of test; `1,369 = 42.46%` new/reactivated in test)** — which mirrors **real production CLV**, where existing accounts repeat each Q4 alongside newly acquired accounts! Notice every feature in `test_df` is recomputed *strictly from scratch* inside `[2010-12-14, 2011-09-10)`, and in Phase 4b Step 10 we explicitly slice performance on `Seen in Train` vs `New to Train` (`Rev Err -14.00% vs -11.89%`)!

---

### Q2: Why fix the lookback window at `L = 270` days rather than using all available past history (`lookback_days = None`) for modeling?
* **Direct Answer**: At `T_train` (`2010-09-10`), only **`~283` days** of dataset history exist (`2009-12-01` start), whereas `T_test` (`2011-09-10`) has **`~648` days** (`2.3x` longer!). Using an expanding lookback gives Year-2 test buyers twice as much time to inflate `frequency`, `monetary`, and `tenure_days_in_window`, manufacturing severe covariate shift even if customer behavior never changed!
* **Empirical Proof (`Step 7 KS & PSI Drift Audit`)**: Fixing `L = 270` days yielded **`0 / 25` numeric features with notable drift** between `train` (`3,228`) and `test` (`3,224`) (`max PSI = 0.0199` << `0.10` threshold; `max KS = 0.0443`!), keeping active repeat rate (`58.40%` vs `57.75%`) and conditional median target (`£590.45` vs `£596.86`) tightly aligned YoY.
* **Contrast With `rfm_snapshot_2011-12-10.parquet`**: For descriptive CRM segmentation (`Phase 4a`), we intentionally passed `lookback_days = None` (`5,852` accounts, `24m`, `£16.78M`) because there is no cross-year train/test prediction contract there.

---

### Q3: Why predict Gross `target` instead of Net-of-Returns `target_net` as the primary CLV target?
* **Direct Answer**: Cross-window refunds (like Phase 2's `+28d` to `+97d` B2B reversals) mean a customer purchasing `£0` in Q4 can still post a return from Q3, causing **`32` negative values in `train` and `28` in `test`** (`target_net < 0`). Modeling negative CLV breaks non-negative Tweedie/Gamma/Poisson/log1p regressions and punishes future demand predictions for historical administrative reversals!
* **How We Still Captured Returns**:
  1. Engineered `4` return behavior features strictly inside `[T-270d, T)` (`return_invoice_count`, `return_value_abs`, `return_value_ratio`, `return_rate`).
  2. Exported `target_net` strictly as a **post-hoc sensitivity check (`Phase 4b Step 11.3`)**, proving rankings (`Gini 0.7721 vs 0.7751`) and model comparisons hold identically.

---

### Q4: Why did you keep structural `NaN`s in `mean_interpurchase_days` (`39.96%` train / `41.25%` test) AND leave all 26 features raw/unscaled in Phase 3?
* **Direct Answer**: Two critical reasons:
  1. **Semantic Honesty**: Single-invoice buyers (`frequency == 1`) have **no undefined interpurchase interval** (`(last_dt - first_dt) / (1 - 1)`). Imputing `0` falsely implies daily ordering; imputing median *before* Phase 4 leaks validation statistics across folds and forces tree ensembles (`HistGradientBoostingRegressor`), which exploit missingness natively, to lose the exact structural split. So Phase 3 keeps `mean_interpurchase_days = NaN` + explicit binary indicator `is_single_invoice`.
  2. **Deferring Preprocessing to Pipelines**: Imputers, `log1p`, `StandardScaler`, and `CountryGrouper` belong strictly inside `sklearn.pipeline.Pipeline` fit on training folds only (`Step 4` of Phase 4b). Note that **`28 / 29` columns have zero nulls**; only `mean_interpurchase_days` has nulls matching `is_single_invoice == 1` `1-for-1`.

---

### Q5: Who is excluded by `>= 1` purchase inside `[T - 270d, T)` and how many Q4 buyers sit outside model eligibility?
* **Exact Audit at `T_test = 2011-09-10`**: Out of `5,280` pre-cutoff registered accounts, **`3,224` (`61.06%`)** had `>= 1` sale in `[2010-12-14, 2011-09-10)` (vs `3,228 / 3,383 = 95.42%` in train because Year 1 only started 283 days earlier!). Excluded prior accounts split into `33` returns-only (`13` bought in Q4) and `2,023` dormant pre-lookback accounts (`413` bought in Q4 — annual Q4 seasonal buyers!), alongside **`594` cold-start brand-new buyers** appearing first in Q4 (`12.06%` of registered Q4 spend in Phase 4b Step 11.2).

---

### Q6: How did you verify zero temporal leakage inside `src/features.py`?
1. **Hard Runtime Assertions**: Both `build_features` and `build_target` assert `max(InvoiceDate) < cutoff` and `min(InvoiceDate) >= cutoff - lookback` (verified across exact timestamps: `train` features end `2010-09-09 20:03:00` vs targets start `2010-09-10 08:50:00`; `test` features end `2011-09-09 16:36:00` vs targets start `2011-09-11 10:35:00`).
2. **Spearman Target Correlation Scan**: Zero features exceeded `|rho| > 0.95` (`monetary` = `0.5065 / 0.5315`, `is_single_invoice` = `-0.3506 / -0.3749`).
3. **End-to-End Customer Transaction Trace**: Audited individual accounts (`15865`, `14957`, `16938`) invoice-by-invoice against raw Parquet, plus verified `pd.testing.assert_frame_equal` parity between notebook and `src/features.py` with `5` synthetic unit tests in `tests/test_features.py`.

