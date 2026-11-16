# Interview Playbook 04: Phase 4a — Unsupervised RFM Segmentation Q&A

**Source Files**: [`4_rfm_segmentation.ipynb`](../Notebooks/4_rfm_segmentation.ipynb), [`src/segmentation.py`](../src/segmentation.py), [`tests/test_segmentation.py`](../tests/test_segmentation.py), [`rfm_segmentation_report.md`](../reports/rfm_segmentation_report.md)

---

### Q1: How can you "validate" an unsupervised clustering model without turning it into circular validation?
* **Direct Answer**: Fitting K-Means on `2011-12-10` and checking Silhouette only measures internal geometry, while testing segments against historical `monetary` used to create them is circular! Instead, we fit the **entire `sklearn` `Pipeline`** (`sqrt(R) + log1p(F, M) -> StandardScaler -> KMeans(K=5)`) strictly at **`T_seg = 2011-09-10`** (`5,256` buyers, `£13.57M` past spend), validated clusters post-hoc against **`£2.81M` of completely unseen future Q4 spend (`[2011-09-10, 2011-12-09)`)**, and then scored the live **`2011-12-10`** snapshot (`5,852` buyers, `£16.78M`) **without refitting**.
* **Follow-up Trap**: *"Why didn't you refit K-Means on `2011-12-10` once you saw it validated well?"*
  * **Counter**: Applying the frozen `T_seg` pipeline **without refitting** preserves exact segment centroid definitions across snapshots, allowing us to compute a true **`5 x 5` longitudinal customer migration matrix** (showing `97.7% = 469/480` VIP stickiness, `164` Loyal -> VIP graduations, and `535 / 596 = 89.77%` Q4 new-buyer intake into `Promising Recent Buyers`!). Refitting changes cluster IDs/boundaries and breaks quarter-over-quarter tracking.

---

### Q2: Why did you transform Recency with `sqrt(recency_days)` instead of `log1p(recency_days)` like `frequency` and `monetary`?
* **Direct Answer**: Inspecting exact skew on `T_seg` (`N = 5,256`) proved `recency_days` is only moderately right-skewed (`raw skew = 0.671`):
  * Applying **`log1p`** to `recency_days` over-compresses recent customers and flips skewness strongly **negative (`-1.021`, Silhouette drops `0.3421 -> 0.3360`)**, disproportionately exaggerating the difference between a 1-day buyer and a 10-day buyer compared to 100 vs 500 days!
  * Applying **`sqrt(recency_days)`** drops skew cleanly to **`0.008`** (essentially symmetric!), whereas **`log1p`** fixes extreme right-skew on `frequency` (`10.487 -> 1.058`) and `monetary` (`24.263 -> 0.271`).

---

### Q3: How did you handle monetary whales (`max = £456,780.49`, top 1% = `30.74%` of spend) inside K-Means?
* **Direct Answer**: Raw `StandardScaler` on untransformed RFM blows up (`max |z| = 37.73`, Silhouette `0.5472` from degenerate singleton `0.08% = ~4` whale clusters!). Instead of isolating whales into a separate rule (`Strategy C`) or `q99` winsorizing (`Strategy B1`), **`A1` (Keep all uncapped + `sqrt(R) + log1p(F, M)` + `StandardScaler`)** keeps `max |z| = 5.31`, balanced minimum cluster size **`9.13%` (`480` VIPs)**, and nearly identical Silhouette (`0.3421` vs `B1`'s `0.3419`) inside a **single stateless `sklearn` Pipeline (`joblib` safe)**.

---

### Q4: Why did standard `pd.qcut(frequency, 5)` fail in Rule-Based RFM scoring (`Step 2`), and how did you fix it?
* **Direct Answer**: Because **`29.97%` (`1,575 / 5,256`)** of `T_seg` customers placed **`frequency == 1`**, 20%-quintile edges collapse (`0%` and `20%` edges collide!). Using `rank(method="first")` arbitrarily splits identical `frequency == 1` buyers across `F=1` and `F=2` by row order!
* **Fix in `score_rfm_rules`**: Used **order-invariant average percentile rank (`rank(method="average", pct=True)`)** discretized cleanly into quintiles (`[0, 0.2..1.0]`), mapping **100% of `freq=1` (`1,575`) to `F=1`**, `freq=2` (`891`) to `F=2`, `freq=3` (`575`) to `F=3`, `4–7` (`1,170`) to `F=4`, and `8–284` (`1,045`) to `F=5`.

---

### Q5: How did you justify `K = 5` over `K = 4` (which had higher Silhouette `0.3609` vs `0.3421`) and `K = 6`?
* **Direct Answer**: Silhouette on smooth non-separated customer clouds is not the sole criterion — we evaluated a 4-metric sweep (`K = 2..10`) with **20 bootstrap resamples** in `evaluate_kmeans_k_range`:
  1. **`K = 5` wins Bootstrap Stability**: `ARI = 0.9515 ± 0.0214`, far more stable than `K = 4` (`0.8939 ± 0.0730`) and `K = 6` (`0.8857 ± 0.1066`).
  2. **`K = 5` achieves lowest Davies-Bouldin (`0.9148`) across all $K \ge 3$** (`K=4`: `0.9605`, `K=6`: `0.9757`).
  3. **Business Granularity**: Separates the `9.13%` (`480`) elite `VIP Champions` tier (`90.21%` Q4 reactivation, `£3,241` mean unseen Q4 spend, `55.41%` of total unseen future spend) cleanly from `Loyal Regulars` (`23.15%`) without degenerate `<3%` micro-clusters.

---

### Q6: Why did Gaussian Mixture Models (`GMM`, full covariance) perform so poorly compared to K-Means (`Silhouette = 0.1551`, `ARI = 0.3888`)?
* **Direct Answer**: Full-covariance GMM latches onto **zero-variance discrete integer hyperplanes** — specifically `frequency == 1` (`1,575` buyers) and `frequency == 2` (`891` buyers)! As a result, GMM collapsed **both 32-day recent `£425` shoppers and 410-day churned `£58` buyers** into the exact same integer-`1` pancake cluster because covariance collapsed along the `F` axis! Meanwhile **Ward Hierarchical Clustering (`ARI = 0.5347`, `NMI = 0.6331`)** confirmed K-Means' elite VIP cluster (`8.81%` vs `9.13%`). Also verified excluding derived `aov` avoided double-weighting basket size (`ARI = 0.3089`).

---

### Q7: Prove K-Means (`K = 5`) actually outperformed the simpler 8-segment heuristic Rule-Based RFM baseline (`Step 7`).
* **Unseen Q4 Future Spend Validation (`[2011-09-10, 2011-12-09)`, `£2.81M` total)**:
  * **Simpler & Stronger**: With **3 fewer segments (`5` vs `8`)**, K-Means achieves higher **Kruskal-Wallis `H = 1,636.14`** (vs `1,538.33`), rank **$\eta^2 = 0.3108$** (vs `0.2918`), ANOVA $\eta^2$ on `log1p(target)` = **`0.2894`** (vs `0.2769`), a **`74.57pp` unseen reactivation spread** (`VIP Champions 90.21%` down to `Hibernating / Lost 15.64%`), and **nearly 2x higher mean future spend per top-tier buyer (`£3,241.48` vs `£1,640`)** (`55.41%` of unseen Q4 revenue in just `9.13%` of buyers!).

