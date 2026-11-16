# Interview Playbook 01: Phase 1 — EDA & Statistical Diagnostics Q&A

**Source Files**: [`01_data_understanding_&_EDA.ipynb`](../Notebooks/01_data_understanding_&_EDA.ipynb), [`data_understanding_and_eda_report.md`](../reports/data_understanding_and_eda_report.md)

---

### Q1: Why did you concatenate both sheets (`Year 2009-2010` + `Year 2010-2011`) instead of modeling just Sheet 2?
* **Direct Answer**: A single 12-month sheet (`541,910` rows) doesn't leave enough history to build **two matched seasonal lookback + target windows (`T_train` and `T_test`)** for out-of-time generalization.
* **Evidence**: Concatenating Sheet 1 (`525,461` rows, `373d`) and Sheet 2 (`541,910` rows, `365d`) creates **1,067,371 rows spanning 738 consecutive days (`2009-12-01` to `2011-12-09`)**, giving `£20.97M` GMV across `5,942` raw customers and covering **two full Q4 holiday surges (`Sep–Dec 2010` and `Sep–Dec 2011`)** so train and test share identical seasonality.
* **Follow-up Trap**: *"Wait — didn't concatenating create duplicates between the two sheets?"*
  * **Counter**: **Yes!** Auditing the overlap in Phase 2 revealed an identical 9-day boundary (`2010-12-01` to `2010-12-09`) duplicated verbatim across both sheets (`22,523` rows). We proactively tagged `source_sheet` before concatenating so we could audit and safely remove Sheet 1's overlap copy.

---

### Q2: You ran a "21-Point Statistical & ML Diagnostics Audit" before modeling. What were the 4 most consequential statistical violations you found?
1. **Extreme Positive Skew & Leptokurtosis (`skew > 12`, `kurtosis > 150`)**: Most line items are under `£20`, yet bulk orders hit `> £10,000`. Standard Euclidean K-Means and unscaled linear OLS would collapse onto outliers — dictating `log1p` / `sqrt` transforms and robust/quantile metrics downstream.
2. **Heteroscedasticity (`Quantity ~ Price` OLS residual cone) & Non-Linear LOWESS Decay**: Error variance grows systematically with order value (`heteroscedastic`), and purchase volume drops non-linearly above `£10`. That ruled out homoscedastic Gaussian OLS assumptions (and later explained why **Tweedie GLM** with variance $V(\mu) = \mu^{1.65}$ fits natural heteroscedastic variance!).
3. **Mechanical Multicollinearity (`VIF > 20`, `r(Quantity, TotalAmount) ≈ 0.88`)**: Feeding raw `Quantity`, `Price`, and `TotalAmount = Quantity × Price` together destabilizes coefficients and importance attribution — shaping our orthogonal customer feature set (`Monetary`, `Units`, `AvgUnitCost`, `AOV`) and our **Step 8 grouped collinearity audit**.
4. **Zero-Inflation & Moderate Imbalance (`Price == 0` > 2k rows; Cancellations = `15.46%`)**: Motivated Two-Stage Hurdle / Tweedie models for spend and `PR-AUC` + Stratified CV rather than accuracy or MSE alone.

---

### Q3: What operational anomalies in Phase 1 proved this is NOT a simple B2C e-commerce store?
* **Direct Answer**: Three unmistakable **B2B wholesale fingerprints** directly guided downstream modeling decisions:
  1. **Zero Saturday Transactions**: Operations run Sun–Fri (`12:00 PM – 3:00 PM` peak), meaning Saturday is literally absent — classic commercial warehouse scheduling.
  2. **Extreme Negative Extremes & Stock Codes**: Observed `Price` down to `-£53,594.36` (`Adjust bad debt`) and `Quantity` between `-80,995` and `+80,995` alongside bookkeeping SKUs (`POST`, `M`, `BANK CHARGES`, `AMAZONFEE`).
  3. **Symmetric Q4 Surge (`~£800k/mo` Jan–Aug -> `> £1.5M/mo` Sep–Nov)**: Wholesale holiday pre-stocking causes explosive multi-fold seasonality, making random row-shuffled CV **invalid** (must enforce strict temporal cutoffs and matched Q4 horizons).

---

### Q4: How did Phase 1 handle missing `Customer ID` (`243,007` rows / `22.77%`) and why didn't you drop them during raw loading or impute a dummy ID?
* **Direct Answer**: Dropping immediately would hide revenue context, while imputing a single constant like `0` or `-1` would merge **227k+ anonymous checkout baskets** into one fake mega-customer in RFM/CLV aggregation.
* **Solution**: Treated `Customer ID == NaN` as a distinct cohort (`guest_transactions`) to monitor overall revenue coverage (`~13.11%` of post-clean gross revenue vs `22.6%` of rows) while **isolating** it cleanly before building customer-level models.

---

### Q5: You noticed `8,292` canceled invoices (`15.46%`, `-£1.53M`). Why keep returns instead of filtering `Quantity > 0` right at the start?
* **Direct Answer**: Filtering `Quantity > 0` blindly discards `490,992` returned units (`£1.53M` refunded revenue). In retail CLV, **historical return propensity** (`return_rate`, `return_value_ratio`) is a crucial churn/friction signal, and returns posted weeks/months later must be checked against gross targets (`target_net` sensitivity).

---

### Q6: What does "Right-Censored Data" at `2011-12-09 12:50:00` mean for CLV & Churn?
* **Direct Answer**: Any customer who bought inside our observation window still has unobserved lifetime purchases *after* `2011-12-09 12:50:00`. Moreover, `2011-12-09` only runs half a day (`12:50`).
* **Impact**: Phase 3 explicitly set half-open target cutoffs `< 2011-12-09 00:00:00`, automatically dropping `604` partial-day Dec 9 rows so `train` (`90.0` full days) and `test` (`90.0` full days) compare identical complete calendar spans.

