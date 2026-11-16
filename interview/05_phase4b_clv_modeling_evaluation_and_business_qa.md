# Interview Playbook 05: Phase 4b — CLV Modeling, Evaluation & Business Defense Q&A

**Source Files**: [`5_clv_modeling_&_evaluation.ipynb`](../Notebooks/5_clv_modeling_&_evaluation.ipynb), [`src/clv.py`](../src/clv.py), [`tests/test_clv.py`](../tests/test_clv.py), [`clv_modeling_and_evaluation_report.md`](../reports/clv_modeling_and_evaluation_report.md)

---

### Q1: Why couldn't you use standard Random K-Fold + RMSE as your primary CLV tuning and selection metric?
* **Direct Answer**: Two statistical pathologies break standard CV + RMSE here:
  1. **42% Zeros (`41.60%` train / `42.25%` test) + Extreme Skew (`16.82` / `20.95`)**: Under Random K-Fold, top whales like `14646` (`£79.2k`) or `18102` (`£78.2k`) clump randomly into one validation fold, exploding fold variance! So we built **`RepeatedStratifiedKFold(5 x 3)` on `make_stratification_bins`** (`Bin 0 = zeros`, `Bins 1–9 = positive quantiles`).
  2. **SSE Tyranny**: Step 10 proved just **5 test customers (`0.16%`) cause `84.1%` of total Test SSE** (Top 10 = `90.0%`), and excluding Top-5 test whales drops `RMSE` from `£3,107.6` to `£1,241.5`! Thus, business-aligned ranking metrics (**Normalized Gini**, **Top-10%/20% Spend Capture & Lift**, **Spearman**) and **portfolio revenue calibration (`Total Rev Error %`, Decile MAE)** come first, with **dual RMSE (`All`, `Excl Top-1`, `Excl Top-5`)** reported alongside.

---

### Q2: Test whale `ID 18102` spent `£123,737.11` (`4.79%` of test revenue) and `ID 14096` jumped from `£203.89` to `£53,054.54`. Why didn't you drop whales as outliers?
* **Direct Answer**: Because unlike the 3 same-day aborted entry errors quarantined in Phase 2 (`581483`, `541431`, `556444`), **`18102` (`32` invoices, `267` lines, `118` SKUs, `32,507` units, `0` returns) and `14096` (`14` invoices, `5,039` lines, `1,114` SKUs, `0` returns) are 100% real wholesale revenue!** Dropping real B2B accounts fakes low RMSE while blinding leadership to `17.26%` (`Top-5`) of company revenue. Instead, we kept every real buyer intact and audited **sensitivity slices (`All` vs `Excl Top-1` vs `Excl Top-5`)** cleanly in CV and Test.

---

### Q3: Your Fold 0 experiment showed **Duan Smearing** blew up `A1: Ridge (log1p)` by `+294.43%` (`smearing = 11.608x`). Why did Duan Smearing fail on single-stage CLV yet work cleanly (`1.064x`) inside the Hurdle model?
* **Mathematical Intuition**: Duan's smearing factor $\hat{s} = \frac{1}{n}\sum \exp(\hat{\epsilon}_i)$ assumes homoscedastic errors on the transformed $\log(1 + y)$ scale. When **`41.6%` of rows sit at exact `log1p(0) = 0`** while predictions hover around `~3–4`, half the sample has massive negative residuals and positive buyers have large positive residuals — causing $\mathbb{E}[\exp(\epsilon)]$ to explode to **`11.608x` (`Ridge`)** and **`4.425x` (`HistGBR`)**, multiplying positive spenders upward into `+280%` / `+98%` aggregate over-prediction!
* **How Hurdle & Tweedie Avoid This**: Inside [`HurdleCLVRegressor`](../src/clv.py#L492-L545), Stage 1 predicts $P(Y > 0 \mid X)$ and Stage 2 fits `log1p(y)` **strictly on positive buyers (`y > 0`)**, where errors are near-symmetric and Duan smearing drops to **`1.064x`**. Meanwhile **`TweedieRegressor` (`power=1.65`, log link)** directly models the compound Poisson-Gamma distribution ($0$ point mass + continuous skewed spend) on the native pound scale via MLE deviance — needing **zero retransformation heuristic**.

---

### Q4: On Step 7 CV leaderboard, `D2: Hurdle (Logit x Ridge) [Tuned]` scored slightly higher on Objective (`0.6478` vs `0.6467`), Spearman (`0.5068` vs `0.5013`), and Core `RMSE Excl Top-5` (`£790` vs `£834`). Why did you choose `B: Tweedie GLM [Tuned]` instead?
* **Direct Answer**: Paired across the identical `15` folds, `D2 [Tuned]`'s `Top-10% Capture` lead (`+0.36pp ± 1.04pp`) was **pure noise** (std is `~3x` mean diff) and `B [Tuned]` edged out `Normalized Gini` (`0.7691` vs `0.7677`, 9/15 folds) — **yet `B [Tuned]` dominated decile calibration!**
* **Hard Evidence (Step 7.2)**:
  * `B: Tweedie GLM [Tuned]`: Decile mean abs calibration error = **`£57.17`** (`pred/actual` tightly between **`0.90–1.16`** across all 10 deciles; Decile 10 **`0.917`**), overall `Total Rev Error = -2.50%`, lower `MAE` (`£554.4`), and lower `RMSE` (`£1,629.6` in `11/15` folds).
  * `D2 [Tuned]`: Heavy Stage-2 Ridge penalty (`alpha=200.0`) pulled predictions inward, doubling decile error (**`£121.32`**), pushing Decile 10 ratio down to **`0.783`**, and hurting `Total Rev Error` (`-8.40%`).
  * `D1: Shallow HistGB [Tuned]`: Hit the cleanest core `RMSE excl top-5 = £766.4`, yet shallow tree leaf-averaging flat-capped whales (`Decile 10 ratio = 0.589`, `-21.05%` total rev bias, `RMSE £2,420.8`).

---

### Q5: Why did `frequency` look smaller in single-feature permutation importance (`gini_drop = 0.0153`) than in grouped importance (`0.0674`), and why did `return_value_abs` have a POSITIVE GLM weight (`+0.1547`)?
1. **Collinearity Dilution (`3.36x`)**: Under `L2` regularization (`alpha=0.5`), weights spread across collinear twins (`|rho| = 0.70–0.98`: `frequency` `+0.1881`, `n_active_months` `+0.0581`, `n_active_weeks` `+0.0486`, `mean_interpurchase_days` `-0.0811`, `is_single_invoice` `-0.0482`). Permuting `frequency` alone leaves the other 4 proxies intact; shuffling **`Frequency & Cadence` jointly** causes a **`3.36x` surge** to `0.0674` Gini drop and makes it **#1 on `Top-10% Capture Drop` (`0.0670`), `Spearman Drop` (`0.0957`), and `MAE Increase` (`+£120.44`)** alongside **`Basket Size & Pricing` (`0.0775` Gini drop)**.
2. **Return Scale Confounding**: Absolute return pounds (`return_value_abs`) scale mechanically with wholesale volume, acting as a gross-spend proxy (`exp(beta) = 1.167x`), whereas **normalized return ratios** have expected **negative slopes (`return_rate = 0.950x`, `return_value_ratio = 0.960x`)**. Shuffled together, **`Returns & Refunds` nearly cancels out (`joint gini_drop = 0.0018 ± 0.0022`, `0.15x` ratio)**.

---

### Q6: WAIT — On Holdout Test, `B: Tweedie GLM [Tuned]` statistically tied `Run-Rate` (`monetary * 90/270`) on Normalized Gini (`0.7751` vs `0.7719`) and Top-10% Capture (`56.12%` vs `55.90%`). Why is your ML model still worth shipping?
* **Direct Honest Defense**: I never oversell ranking where a heuristic matches it — **paired 95% bootstrap CIs (`1,000` customer resamples) confirm ranking metrics statistically tie.** However, **Step 3 explicitly sets out-of-sample business utility as shipping criteria, and `Run-Rate` fails catastrophically on revenue calibration:**
  1. **`+23.93pp` Statistically Significant Calibration Win (`95% CI [+18.69pp, +29.88pp]`)**: `Run-Rate` linearly scales trailing 270-day spend by `1/3`, ignoring Q4 expansion and repeat-rate nonlinearity — under-predicting Q4 test revenue by **`-37.38%`** (`-31.17%` even without Top-5 whales!). Tweedie GLM cuts portfolio bias to `-13.45%` overall and **`-0.96%` across the 3,219 non-top-5 core customers** (`Loyal Regulars: +0.14%`, `Promising Recent: +2.98%`, `Lookback Bottom 90%: +0.30%`)!
  2. **Halves Test Decile Calibration Error**: `£145.69` (`0.91x–1.17x` across Deciles 1, 2, 4–9) vs `£299.53` (`Run-Rate` under-predicts every decile at `0.19x–0.74x`).
  3. **Higher Top-10% Buyer Hit Rate**: `92.55%` active buyer precision at Decile 10 vs `90.06%` for Run-Rate (`+2.49pp` less wasted campaign outreach on zero buyers).
* **Why Run-Rate Wins Raw `MedAE` (`£192.8` vs `£258.8`)**: `42.25%` of test targets equal exact `£0.00`; Run-Rate's severe low-decile under-prediction (`£32` mean in Decile 1) artificially sits closer to zero on non-buyers at the expense of ruining company revenue forecasts by `-37.4%`!

---

### Q7: Walk through Step 10 & Step 11 highlights: Where does Tweedie fail most, and how much of total store Q4 revenue does it actually cover?
* **Four Concrete Step 10 Failure Modes**:
  1. **Log-Log VIP Compression (`18102` `-£108.3k`, `17450` `-£93.2k`)**: Applying `log1p` features under a log link produces sub-linear elasticity on high-AOV medium-frequency (`freq=21–23`) accounts (`£107.6k -> £15.5k pred`). Tweedie still captures **`30 / 33` (`90.9%`)** of Actual Top-1% whales inside Decile 10, so ranking works while key-account budgets need manual floors.
  2. **Lapsed-Segment Over-Prediction (`At-Risk` `+36.07%`, `Hibernating / Lost` `+178.82%`)**: With `63%–81%` zeros and mild Tweedie `recency_days` weight (`std_beta = -0.0137`), lapsed high-historical accounts like `15749` (`£44.5k, return_rate=0.25 -> £0 actual vs £10.3k pred`) get over-predicted.
  3. **Unforeseeable B2B Step-Changes (`ID 14096`)**: New lookback account (`£203.89`, `3` orders -> Decile 3 `£221.71`) exploded into `£53,054.54` (`14` orders) — singlehandedly distorting Decile 3's mean (`£334.49` vs `£0.00` median) and contributing to **5 customers (`0.16%`) driving `84.1%` of Test SSE**.
  4. **Single-Invoice Sparsity (`freq = 1`)**: `61.80%` zeros, `Spearman = 0.16` due to missing cadence history.
* **Step 11 Coverage & Sensitivity**:
  * **`80.96%` Registered Coverage (`£2.59M / £3.19M` gross, `80.88%` net)** across `[2011-09-10, 2011-12-09)`: Remaining **`12.06%` (`594` buyers)** = Cold-Start new accounts; **`6.98%` (`426` buyers)** = Dormant-Reactivated pre-`2010-12-14` buyers; `6` returns-only (`0.00%` gross / `-£625` net).
  * **`target_net` Sensitivity**: All conclusions hold intact (`Gini = 0.7721`, `Top-10% = 56.01%`, `+24.60pp` paired calibration margin over Run-Rate, `Excl Top-5` net error `+2.15%`).

