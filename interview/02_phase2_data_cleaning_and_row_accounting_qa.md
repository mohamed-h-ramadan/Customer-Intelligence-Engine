# Interview Playbook 02: Phase 2 — Data Cleaning & Row Accounting Q&A

**Source Files**: [`2_data_cleaning_&_data_preprocessing.ipynb`](../Notebooks/2_data_cleaning_&_data_preprocessing.ipynb), [`data_cleaning_and_preprocessing_report.md`](../reports/data_cleaning_and_preprocessing_report.md)

---

### Q1: Why did you refuse to compute RFM features, scalers (`StandardScaler`), or country encodings inside Phase 2?
* **Direct Answer**: Computing customer-level features or scalers before Phase 3's temporal split introduces both **temporal leakage** (future orders contaminate past RFM) and **preprocessing leakage** (test distributions leak into scaler means/IQRs).
* **Rule Enforced**: Phase 2 is strictly transaction-level only (`source_sheet`, `is_cancellation`, `TotalPrice`, `invoice_date`, `year_month`), guaranteeing `100%` row conservation (`1,067,371 == 1,067,371`).

---

### Q2: How did mixed types in Excel create subtle Python bugs, and how did you fix them?
* **Direct Answer**: `pd.read_excel()` mixed Python `int` and `str` inside `object` columns (`Invoice`: `1,047,871` ints + `19,500` strings; `StockCode`: `932,385` ints + `134,986` strings; `Description`: `4` raw ints!), and stored integer `Customer ID`s as `float64` because of `243,007` `NaN`s.
* **Engineering Fix**:
  * Cast text columns to pandas nullable `"string"` instead of `.astype(str)` — because `.astype(str)` silently turns genuine `<NA>` (`4,382` descriptions) into the literal string `"nan"`, while `.str.strip()` on raw mixed `object` would turn those `4` integer descriptions into `NaN`! Found and fixed `213,035` description whitespace anomalies and `3,471` lowercase `StockCode`s.
  * Cast `Customer ID` to pandas nullable **`Int64`** (zero fractional IDs verified) to preserve `<NA>` without `.0` float artifacts.

---

### Q3: How did you handle duplicates cleanly? What is the difference between the `22,523` overlap rows and the `11,812` within-sheet duplicates?
* **Direct Answer**: I separated **schema-level ETL overlap** from **behavioral cart/scan duplicates**:
  1. **Cross-Sheet Overlap (`22,523` rows, `2010-12-01` to `2010-12-09`)**: Audited both sheets with a multiset group count (`groupby(compare_cols).size()`), proving all **`22,523` rows were 100% identical (`0` diffs)**, including `321` genuine repeat line-items within those 9 days. Removed Sheet 1's copy (`-22,523`), keeping Sheet 2's copy (`1,067,371 -> 1,044,848`).
  2. **Within-Sheet Exact Duplicates (`11,812` rows, `1.13%` of remaining rows across `4,387` invoices, `£54,228.42` / `0.29%` of net revenue)**: Forensic inspection showed `85.7%` (`10,127`) were non-consecutive (`70.5%` with `Qty == 1`), consistent with unmerged web-cart additions, while `14.3%` (`1,685`) were consecutive double-scans (`idx_gap == 1`). Dropped exact full-row duplicates (`keep="first"`), **preserving `4` pairs (`8` rows)** sharing `(Invoice, StockCode, Quantity, Price)` that differed in 1-minute timestamps (`492807`, `567183`) or updated descriptions (`554084`, `575335`).
* **Follow-up Trap**: *"Could dropping the `11,812` cart additions bias revenue downward?"*
  * **Counter**: I quantified the exact maximum exposure (**`0.29%` / `£54.2k`** out of `£18.5M+`), chose transaction-row uniqueness, and documented deduplication sensitivity explicitly in Phase 2 & Phase 3 limitations.

---

### Q4: Instead of copying a hardcoded list of bad `StockCode`s from Kaggle, what data-driven check did you run?
* **Direct Answer**: I flagged and manually classified **all 61 unique StockCodes (`5,980` rows)** failing the standard catalog regex `^\d{5}[A-Z]*$`:
  * **Isolated `5,801` rows (`25` codes) to `admin_rows.parquet`** (`-£71.9k` net): Postage (`POST`, `DOT`, `C2`: `3,788` rows, `+£433.4k`), Manual/Adjustments (`M`, `D`, `ADJUST`, `ADJUST2`, `B`: `1,641` rows, `-£236.2k`), Bank/Fee (`BANK CHARGES`, `AMAZONFEE`, `CRUK`: `152` rows, `-£264.9k`), Gift vouchers (`101`), Samples `S` (`101`), Test codes (`17`), and `C3` (`1`).
  * **Retained `179` rows (`36` codes, `+£1,132.89`)** belonging to real physical catalog SKUs (`DCGS*` legacy items, `SP1002`, `PADS`). Standard blog blacklists often delete real legacy products (`PADS`/`DCGS*`) or miss newer codes like `CRUK` (`-£7.9k`+ UK charity commission) and `ADJUST2`.

---

### Q5: Why does the order of splitting (`sales` vs `returns` vs `anomalies`) BEFORE filtering `Quantity > 0` matter?
* **Direct Answer**: If you filter `Quantity > 0` first, you accidentally destroy all `17,585` valid return/cancellation rows (`-£464k`), forcing net CLV to look 100% optimistic!
* **Exact Partition (`1,027,235` rows)**:
  1. `sales` (`~is_cancellation & Quantity > 0 & Price > 0`): `1,003,357` rows (`97.68%`).
  2. `returns` (`is_cancellation & Quantity < 0 & Price > 0`): `17,915` rows (`1.74%`).
  3. `anomalies` (complement): **`5,963` rows (`0.58%`)** — all `100%` had **`Price == 0.0`** (`2,572` giveaways/found inventory vs `3,391` damaged/lost write-offs), proving zero unexplained negative-price or positive-qty C-rows remained!

---

### Q6: Walk me through the forensic mega-order audit in Step 7B. Why did checking `Price` save you from a subtle pairing error?
* **Direct Answer**: Inspecting the Top 20 `Quantity` and `TotalPrice` rows caught **3 same-day aborted entry errors (`5` product rows moved to `anomalies`)** while keeping legitimate multi-day wholesale re-invoices intact:
  * **Aborted Errors Quarantined**:
    1. `581483` (`Cust 16446`, `80,995 @ £2.08 = £168,469.60`) canceled **+12 min later** by `C581484`.
    2. `541431` (`Cust 12346`, `74,215 @ £1.04 = £77,183.60`) canceled **+16 min later** by `C541433`.
    3. `556444` (`Cust 15098`, `60 @ £649.50 = £38,970.00`) reversed **+3 min later** via Manual `'M'` credit `C556445` (already in `admin_rows`!). Crucially, checking **`Price`** proved `C556448` (`-60 @ £4.95`) balanced `15098`'s earlier order `556442` (`+60 @ £4.95`) — naive quantity-only matching would have mispaired `556444`!
  * **Kept Real Wholesale**: Multi-day cancellations/re-orders (`+97d` on `540815 -> C550456 -> 550461`) and high-volume repeat wholesalers (`13902`, `17450`, `14646`), resetting clean `max TotalPrice` to `£15,818.40`.

---

### Q7: What did weekly continuity and guest seasonality audits uncover?
* **Zero ETL Dropouts**: Across `106` weeks (median `315.5` invoices/wk), the only `2` weeks with `0` invoices (`2010-01-03` and `2011-01-02`) coincide cleanly with two **12-day Christmas/New Year store closures** (`Dec 23 – Jan 4`) plus 5-day Easter holidays — essential context before interpreting `recency_days` or `interpurchase_days`.
* **Non-Stationary Guest Spike**: Guest checkouts (`227,089` rows, `13.11%` of gross revenue vs `22.24%` of rows) are **not random**: guest row share spikes from `14%–24%` (Feb–Oct) up to **`37.37%` (Dec 2010, `27.13%` of revenue)** and **`38.36%` (Jan 2011)**. Any total-store revenue tracker must note registered-only CLV excludes this seasonal guest inflow.

---

### Q8: Prove your cleaning pipeline didn't silently lose a single row.
* **Row Accounting Conservation (`100.00%` verified via assertions)**:
  `776,593 (sales_clean) + 17,585 (returns_clean) + 227,089 (guest_transactions) + 5,801 (admin_rows) + 5,968 (anomalies) + 34,335 (removed_duplicates) == 1,067,371` raw input rows!
  Combined registered net merchandise revenue across `sales_clean + returns_clean` = **`£16,319,622.71`** across **`5,875` unique registered accounts** (`5,852` purchasers + `23` returns-only).

