# Interview Playbook 00: End-to-End Project Pitch, Architecture & System Design

**Target Role**: Data Scientist / Machine Learning Engineer (Portfolio Defense)  
**Project**: Customer Intelligence Engine — Dual-Track Unsupervised RFM Segmentation + Supervised 90-Day CLV Regression  
**Dataset**: UCI Online Retail II (`1,067,371` raw rows, `2009-12-01` to `2011-12-09`, `£20.97M` GMV, `5,942` raw / `5,875` clean registered accounts)

---

## 1. The 60-Second Elevator Pitch

> *"I built the **Customer Intelligence Engine** on the 2-year Online Retail II ledger (`1.07M` transactions, `£20.97M` GMV) to solve two real e-commerce/B2B problems: **CRM customer segmentation** and **90-day future gross spend (CLV) prediction**.*  
> *I deliberately decoupled data cleaning, leak-free temporal feature engineering (`270d` lookback -> `90d` future Q4 horizon at aligned `Sep 10` cutoffs with `0/25` feature drift), unsupervised K-Means segmentation (`K=5`, `sqrt(R) + log1p(F, M)`, externally validated on `£2.81M` unseen future spend), and supervised CLV regression (`TweedieRegressor` tuned under a strict zero-leakage single-shot test lock).*  
> *On the unseen `2011-09-10` holdout cohort (`n = 3,224`, `£2.59M`), our frozen Tweedie GLM achieves **`0.7751` Normalized Gini**, **`56.12%` Top-10% revenue capture (`5.61x` lift, `92.55%` buyer hit rate)**, covers **`80.96%` of all registered Q4 store revenue**, and cuts aggregate budget error from `Run-Rate`'s `-37.38%` down to `-13.45%` (`-0.96%` across the 3,219 non-top-5 customers) — backed by `55` passing pytest unit tests and `joblib` production pipelines."*

---

## 2. End-to-End Pipeline Architecture

| Phase | Notebook / Code | Input -> Output | Core Validation / Leakage Guardrail | Key Quantitative Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: EDA** | `01_data_understanding_&_EDA.ipynb` | Raw Excel (`2` sheets) -> Diagnostics blueprint | Documented non-IID weekly lag-7, Q4 surge, right censoring, VIF>20 | Audited `1,067,371` rows (`738d`), `15.46%` cancellations, `22.77%` guests |
| **Phase 2: Cleaning** | `2_data_cleaning_&_data_preprocessing.ipynb` | Raw `1,067,371` -> `sales_clean` (`776,593`), `returns_clean` (`17,585`), `guest` (`227,089`), `admin` (`5,801`) | Zero customer aggregation, scaling, or deduplication assumptions hidden | `100.00%` row conservation (`1,067,371 == 1,067,371`); `0` null IDs in sales/returns |
| **Phase 3: Features & Cutoffs** | `3_cutoff_design_&_feature_engineering.ipynb`, `src/features.py` | Clean Parquet -> `train` (`3,228 x 29`), `test` (`3,224 x 29`), `rfm_snapshot` (`5,852 x 26`) | Half-open bounds `[T-270d, T)` vs `[T, T+90d)`, structural `NaN` kept | **`0/25`** features drifted (`PSI <= 0.0199`, `KS <= 0.0443`); `15` pytest checks |
| **Phase 4a: Segmentation** | `4_rfm_segmentation.ipynb`, `src/segmentation.py` | `T_seg=2011-09-10` (`5,256`) -> Live `2011-12-10` (`5,852`) + `rfm_kmeans_k5_pipeline.joblib` | Fit at `T_seg`, validated post-hoc on `£2.81M` unseen Q4, scored un-refitted | `K=5` beats `8`-rule RFM (`KW H = 1,636` vs `1,538`, `74.6pp` reactivation spread); `23` tests |
| **Phase 4b: CLV Modeling** | `5_clv_modeling_&_evaluation.ipynb`, `src/clv.py` | `train` (`2010`) -> One-shot `test` (`2011`) + `clv_tweedie_tuned_2010_09_10.joblib` | `TEST_EVALUATION_COUNT = 0 -> 1` single-shot lock; never uses cluster labels | `Gini = 0.7751`, `Cap@10 = 56.12%`, **`+23.93pp` calibrated revenue win** vs Run-Rate; `17` tests (`55` total) |

---

## 3. Most Asked Cross-Phase Interview Questions

### Q1: What makes your project structurally stronger than typical Kaggle/portfolio notebooks on Online Retail II?
**Direct Answer:** Three engineering disciplines: (1) **100% row-level conservation accounting** in Phase 2 instead of silent `.dropna()`/`drop_duplicates()`, (2) **strict temporal cutoffs with fixed lookback `L=270d`**, avoiding lookback length mismatch and leakage, and (3) **one-shot test discipline (`TEST_EVALUATION_COUNT`)** paired with quantile-calibrated business evaluation instead of chasing noisy unweighted RMSE.
* **Evidence**: Eliminated the `22,523` identical Dec 1–9, 2010 cross-sheet duplicates between Sheet 1 and Sheet 2, separated `5,801` admin rows (`POST`, `M`, `BANK CHARGES`, etc.) via data-driven regex auditing of 61 StockCodes, caught 3 same-day aborted mega-orders (`£168.5k`, `£77.2k`, `£39.0k`) using price+timestamp matching, and modularized all logic into `src/` with `55` unit tests.

### Q2: Why didn't you feed the Phase 4a K-Means segment labels into your Phase 4b CLV model as features?
**Direct Answer:** Two reasons: **temporal leakage** and **pure feature redundancy**.
* **Technical Details**:
  1. Our live Phase 4a pipeline `rfm_kmeans_k5_pipeline.joblib` was fit at `T_seg = 2011-09-10` on full `2009–2011` transactions; assigning those 2011 cluster IDs onto 2010 training rows leaks 2011 distribution statistics into 2010.
  2. Even within 2011, K-Means clusters are deterministic Voronoi regions computed solely from `(recency_days, frequency, monetary)`, which are already inside `X` (`26` features) without quantization loss.
  3. Thus, Phase 4a segment labels (`segments_t_seg_2011-09-10.parquet`) are strictly **banned as model inputs** and used exclusively **post-hoc in Step 10** for business error slicing (`VIP Champions` `-25.60%` vs `Loyal Regulars` `+0.14%`).

### Q3: If deployed into real production next quarter, how would you serve, monitor, and handle the 19% out-of-scope Q4 revenue?
**Direct Answer:** Two pipelines + explicit routing rules + scheduled drift/calibration monitoring.
* **Production Design**:
  1. **Batch Inference Split (`T_cutoff`)**:
     * Cohort 1 — *Active `<= 270d` Registered Accounts (`80.96%` Q4 gross revenue)* -> scored by frozen `clv_tweedie_tuned_2010_09_10.joblib` for budget forecasting & Top-30% CRM targeting (`75.65%` capture).
     * Cohort 2 — *VIP Key Accounts (`Lookback Top-1%`, e.g., `18102`, `14646`)* -> paired with key-account manager overrides to counteract log-log sub-linear compression.
     * Cohort 3 — *Cold-Start New Acquisitions (`12.06%` Q4 revenue)* -> routed to a separate onboarding propensity/basket-1 model triggered 14–30 days after signup instead of predicting `£0`.
     * Cohort 4 — *Dormant `> 270d` Accounts (`6.98%` Q4 revenue)* -> scored only ahead of major Q4 campaigns using a low-cost seasonal reactivation score (`426` Q4 returners out of `2,023+` dormant accounts).
  2. **Monitoring**: Track input `PSI` (baseline `0/25 >= 0.10`), monthly `Decile 1–10` `pred_to_actual_ratio`, and separate out-of-season retraining schedules because our `Sep 10 -> Dec 09` model specifically encodes Q4 peak uplift.

### Q4: Name the single hardest failure mode you found at the end of the project and what you would try in v2.
**Direct Answer:** Tweedie GLM's log-log elasticity (`log1p` features + log link) systematically compresses ultra-high-AOV moderate-frequency B2B whales (`ID 18102`: actual `£123.7k` -> pred `£15.5k`; `ID 17450`: actual `£108.1k` -> pred `£14.8k`), while simultaneously giving soft negative weight to high `recency_days` on `Hibernating / Lost` churners (`+178.82%` bias, `80.86%` zeros).
* **v2 Roadmap**:
  1. **Spline / Piecewise Linear Features or Untransformed Monetary Interaction for `Top 5%` Accounts**: Keep a dedicated high-tier whale linear/Poisson branch or `Hurdle (Calibrated Logistic Stage-1 × Uncompressed Gamma/Tweedie Stage-2)` so high-`recency` churners get hard propensity suppression (`P(active) -> 0`) while high-`AOV` VIPs scale proportionally rather than sub-linearly.
  2. **Separate Q1–Q3 Seasonal Cohorts**: Re-train once quarterly roll-forward windows exist across multi-year history rather than relying on one Q4 train/test pair.

