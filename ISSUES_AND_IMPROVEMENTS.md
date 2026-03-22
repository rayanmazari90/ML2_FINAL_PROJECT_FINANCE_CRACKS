# Issues & Improvements Tracker

> Living document — add new findings below their respective section.
> Each issue has a severity tag, a status, and a precise code pointer so fixes can be tracked.

---

## Severity Legend

| Tag | Meaning |
|-----|---------|
| 🔴 CRITICAL | Structural failure; results are invalid without fix |
| 🟠 HIGH | Active leakage or bias that inflates reported performance |
| 🟡 MEDIUM | Currently dormant but will become active under realistic extensions |
| 🟢 LOW | Minor bias or code smell; worth fixing but not urgent |
| ✅ FIXED | Remediated; kept for audit trail |

---

## Section 1 — Data Leakage Audit

### [LEAK-01] 🟠 HIGH — OOF Meta-Label Generation Uses Future Data

**Status:** `OPEN`
**File:** `src/modeling/meta_labeling.py`, lines 141–171
**Function:** `_out_of_fold_primary_predictions()`

**Description:**
The out-of-fold (OOF) splitting loop constructs training indices as:
```python
tr_idx = np.concatenate([sorted_idx[:val_start], sorted_idx[val_end:]])
```
This includes **future observations** (`sorted_idx[val_end:]`) in the training set for folds 0 through 3.
For fold `k=0` (the earliest 20% of training data), the primary model is trained **exclusively on future data** before generating predictions for that early fold.

| Fold | Validation period | Future data in training? |
|------|------------------|--------------------------|
| 0 | Earliest 20% | ✅ YES (folds 1–4 = all future) |
| 1 | 20%–40% | ✅ YES (folds 2–4 = future) |
| 2 | 40%–60% | ✅ YES (folds 3–4 = future) |
| 3 | 60%–80% | ✅ YES (fold 4 = future) |
| 4 | Latest 20% | ❌ No (past only — correct) |

The meta-label targets (`meta_target_train`) for ~80% of training samples are derived from a model that saw the future, corrupting the Random Forest meta-model's training signal.
Note: the `KFold` import on line 155 is dead code — it is imported but never used.

**Impact:** Meta-model learns to be "correct" in ways that do not generalise. Position filter pass-rate and meta-conviction scores are inflated.

**Fix required:**
Replace the bidirectional split with a **strictly expanding window**:
```python
# Only train on data BEFORE the validation period
tr_idx = sorted_idx[:val_start]
# Skip fold k=0 if val_start == 0 (no history yet)
if len(tr_idx) == 0:
    continue
```

---

### [LEAK-02] 🔴 CRITICAL — Fundamentals Fetch Only the Latest Filing (Not a Time Series)

**Status:** `OPEN`
**File:** `src/data/pit_ingestion.py`, lines 272–276
**Function:** `fetch_pit_fundamentals()`

**Description:**
The function signature declares `n_periods: int = 8` but the implementation body ignores it entirely and calls only:
```python
latest_filing = company.get_filings(form="10-K").latest()
```
This produces **one row per ticker** — exclusively from the most recent 10-K.
The notebook output confirms all 617 fundamental rows have `sec_acceptance_date` between **2025-02-20 and 2026-03-20**.

Because `pit_asof_join` uses `direction="backward"`, any decision date before the earliest filing (2025-02-20) gets no fundamental match → NaN.

**Effect on the feature set:**

| Period | Fundamental features |
|--------|----------------------|
| 2010–2025-02 (entire training set) | All NaN |
| Late 2025 test window | Partially populated |

All 16 fundamental-derived features (`pe_trailing`, `pb`, `roe`, `roa`, `debt_equity`, `gross_margin`, `net_margin`, `ocf_to_assets`, `revenue_growth_yoy`, `net_income_growth_yoy`, raw `revenue`, `net_income`, `total_assets`, `total_liabilities`, `stockholders_equity`, `operating_cash_flow`) are **non-functional for the bulk of the backtest**.

The PIT compliance check passes trivially because NaN rows are dropped before the assertion — the compliance guarantee is vacuously true.

The `revenue_growth_yoy` / `net_income_growth_yoy` features use `.pct_change(4)` on a one-row-per-ticker frame and always return NaN regardless.

**Fix required:**
Use the EDGAR `MultiFinancials` API to retrieve a time series of historical filings:
```python
filings = company.get_filings(form=("10-K", "10-Q"))
# Iterate over the last n_periods filings, not just .latest()
```
Each filing's `filing_date` (SEC acceptance timestamp) becomes the PIT anchor for that row. The `pit_asof_join` then correctly aligns each decision date to the most recent available filing as of that date.

---

### [LEAK-03] 🟡 MEDIUM — Fractional Differencing d-Parameter Fitted on Full Dataset

**Status:** `OPEN` (currently inactive — all d = 0.0)
**File:** `src/features/feature_store.py`, lines 270–284
**Function:** `fractional_difference()`

**Description:**
Both `FracdiffStat` (fracdiff backend) and `FractionalDifferentiator` (tsfracdiff backend) call `fit_transform` on the **entire column** — train and test concatenated — before the train/test split.
The ADF test that determines the optimal differencing order `d` therefore sees future-period data.

**Current practical impact:** Zero. The notebook shows all 18 features received `d = 0.000`, meaning the ADF test found them already stationary (returns, z-scores, and ratios are stationary by construction). No transformation was applied.

**Future impact (HIGH):** If raw price levels, cumulative OBV, or book value levels are ever added as features, this becomes an active high-severity leakage vector. The d-parameter would encode information about the future volatility regime into the transformation applied to training data.

**Fix required:**
Fit the fracdiff estimator **inside each CPCV fold** on training data only:
```python
# Inside the CPCV loop or the train_primary_model call:
fds = FracdiffStat()
fds.fit(X_train[col].values.reshape(-1, 1))
X_train[col] = fds.transform(X_train[col].values.reshape(-1, 1)).flatten()
X_test[col]  = fds.transform(X_test[col].values.reshape(-1, 1)).flatten()
```

---

### [LEAK-04] 🟡 MEDIUM — Raw OBV (Unbounded Cumulative Sum) in Feature Set

**Status:** `OPEN`
**File:** `src/features/feature_store.py`, lines 121–125
**Function:** `compute_technical_features()`

**Description:**
```python
feat["obv"] = (direction * grp["volume"]).cumsum()
```
`obv` is an **unbounded, non-stationary cumulative sum** that starts from the first available date in each ticker's history.

- Its magnitude is entirely path-dependent and grows/shrinks indefinitely
- A ticker with 15 years of data has OBV in a completely different numerical range than one with 3 years
- Cross-sectional z-scoring partially masks this but does not make it stationary
- It encodes implicit look-back spanning the entire history of the stock (up to 2010), not just recent behaviour

The stationary version `obv_zscore` (60-day rolling z-score) is already computed and is in the feature set. The raw `obv` level should be removed.

**Fix required:**
Remove `obv` from the feature columns list, or apply fractional differencing to it (but this requires Fix LEAK-03 to be applied first):
```python
# In run_feature_pipeline, exclude raw obv from feature_cols:
feature_cols = [c for c in merged.columns
                if c not in exclude and c != 'obv'
                and merged[c].dtype in (...)]
```

---

### [LEAK-05] 🟢 LOW — Partial Survivorship Bias: 149 Bankrupt/Delisted Tickers Missing Price Data

**Status:** `OPEN`
**File:** `src/data/pit_ingestion.py`, lines 159–217
**Function:** `fetch_pricing()`

**Description:**
Of 801 historical tickers reconstructed by `build_historical_universe`, only 652 returned valid pricing data from Yahoo Finance. The 149 missing tickers are **disproportionately companies that were delisted due to bankruptcy or forced deregistration** — exactly the names whose prices approached zero and whose inclusion is most critical for anti-survivorship-bias training.

yfinance silently drops these tickers without warning. The pipeline has no fallback to an alternative source (CRSP, Compustat, or a third-party delisted-ticker database).

**Impact:** A subtle positive bias in the training universe. The model is never trained on a company whose price went to zero, leaving it unprepared to recognise the early warning signals of distress.

**Fix required:**
- Log which tickers returned no data and flag them explicitly
- Consider supplementing with CRSP or a service like Quandl/Polygon that retains delisted price history
- At minimum, add a warning in the survivorship report when `pricing.ticker.nunique() < universe.ticker.nunique()`

---

## Section 2 — Code Quality & Structural Issues

### [CODE-01] 🟡 MEDIUM — `log_market_cap` Proxy is Incorrect Formula

**Status:** `OPEN`
**File:** Notebook, Phase 5 cell (neutralization setup)

**Description:**
```python
test_df['log_market_cap'] = np.log1p(
    test_df[price_col].fillna(0) * test_df[vol_col].fillna(0)
)
```
`price × daily_volume` is **dollar turnover**, not market capitalisation.
Market cap = `price × shares_outstanding`.
This produces a proxy for liquidity/activity, not size.

**Impact:** The size neutralisation factor is noisy. The strategy is not truly size-neutral; it is turnover-neutral, which is a different thing.

**Fix required:**
Use shares outstanding from the EDGAR fundamentals once LEAK-02 is fixed, or use a dedicated market cap data source. Alternatively, use `log(adj_close)` as a rough size proxy (large-caps have higher absolute prices on average after adjustments).

---

### [CODE-02] 🟢 LOW — Dead Import in `_out_of_fold_primary_predictions`

**Status:** `OPEN`
**File:** `src/modeling/meta_labeling.py`, line 155

**Description:**
```python
from sklearn.model_selection import KFold
```
`KFold` is imported inside `_out_of_fold_primary_predictions` but is never instantiated or used anywhere in the function. The actual split is performed manually via `sorted_idx`.

**Fix required:** Remove the import.

---

### [CODE-03] 🟢 LOW — `n_periods` Parameter in `fetch_pit_fundamentals` Has No Effect

**Status:** `OPEN`
**File:** `src/data/pit_ingestion.py`, line 228

**Description:**
```python
def fetch_pit_fundamentals(
    tickers: List[str],
    filing_types: tuple = ("10-K", "10-Q"),
    n_periods: int = 8,          # ← declared but never used
) -> pd.DataFrame:
```
`n_periods` and `filing_types` are declared in the signature but neither is referenced anywhere in the function body. The function always fetches a single 10-K filing regardless of these parameters.

**Fix required:** Implement the multi-period retrieval logic (see LEAK-02), or remove the dead parameters until the implementation catches up.

---

### [CODE-04] 🟡 MEDIUM — `revenue_growth_yoy` / `net_income_growth_yoy` Always NaN

**Status:** `OPEN`
**File:** `src/features/feature_store.py`, lines 218–221
**Function:** `compute_fundamental_features()`

**Description:**
```python
for col in ["revenue", "net_income"]:
    if col in df.columns:
        df[f"{col}_growth_yoy"] = df.groupby("ticker")[col].pct_change(4)
```
`pct_change(4)` requires at least 5 sequential rows per ticker (4 quarters of filings).
Because LEAK-02 means each ticker has exactly one fundamental row, `.pct_change(4)` always returns NaN.
These two features contribute zero information and occupy feature space.

**Fix required:** Resolve LEAK-02 first (multi-period filings). Until then, optionally drop these columns from `feature_cols` to avoid misleading NaN features.

---

## Section 3 — Performance & Methodology

### [PERF-01] 🟡 MEDIUM — CPCV Purge Does Not Handle NaN `t_barrier`

**Status:** `OPEN`
**File:** `src/modeling/cpcv_pipeline.py`, lines 87–93

**Description:**
```python
if not pd.isna(tbar_vals[i]):
    tbar_np = np.datetime64(tbar_vals[i])
    if tbar_np >= ts_np:
        train_mask[i] = False
        break
```
If `t_barrier` is NaN for observation `i`, the observation is silently **kept in training** even if its `entry_date` is within `vertical_barrier_days` of the test period start. In theory the triple barrier always produces a `t_barrier`, but NaN rows can arise from float-conversion edge cases.

**Fix required:**
Add a secondary purge based on `entry_date + max_holding_period`:
```python
if pd.isna(tbar_vals[i]):
    # Conservatively purge if entry date is within vert_days of test start
    max_exit = np.datetime64(date_vals[i]) + np.timedelta64(vert_days, "D")
    if max_exit >= ts_np:
        train_mask[i] = False
        break
```

---

### [PERF-02] 🟢 LOW — Macro Features Fetched but Never Used as Model Inputs

**Status:** `OPEN` (design decision, not a bug)
**File:** Notebook Phase 1d; config `data.fred_series`

**Description:**
Five FRED macro series (VIX, 10Y-2Y spread, BBB credit spread, initial claims, CPI) are fetched and cached but are explicitly labelled as "not directly used as features in the current model — only used for regime decomposition analysis."

Macro regime features are well-documented alpha signals (especially VIX and the yield curve as conditioning variables for cross-sectional momentum).

**Suggested improvement:**
Merge macro series into the feature matrix via a daily left-join on `date`, then include them in `feature_cols`. This gives the model regime-conditioning capability without any leakage risk (FRED data is published with well-defined release lags that can be accounted for).

---

## Section 4 — Fixed Issues (Audit Trail)

### [FIXED-01] ✅ Early Stopping Validation on Test Set

**Fixed in:** `src/modeling/meta_labeling.py`, lines 227–237
**Description:** Previously, XGBoost early stopping used the final test set as `eval_set`, allowing the number of trees to be tuned against future data.
**Fix applied:** A validation slice is carved from the last 10% of the **training period** and used as the early-stopping monitor. The test set is never seen during training.

---

### [FIXED-02] ✅ t+0 Execution (Same-Day Signal and Return)

**Fixed in:** `src/backtest/execution_sim.py`, line 207
**Description:** Previously, signals computed from the closing price on rebalance date `t` could earn the same day's return (i.e., trading at t and being credited the close-to-close return from t-1 to t).
**Fix applied:** `exec_start = reb_loc + 1` enforces t+1 execution — positions are entered on the day *after* the signal date.

---

### [FIXED-03] ✅ Neutralization Over Full Test Window

**Fixed in:** `src/risk/neutralization.py`, lines 144–151
**Description:** Previously, `pinv(E)` was computed over the entire test window at once, allowing future exposure structure (which sectors were overweighted later in the test period) to influence neutralization of earlier scores.
**Fix applied:** Neutralization now runs independently per rebalance date via `groupby("date")`, so each date's pseudo-inverse uses only that date's cross-section.

---

### [FIXED-04] ✅ Meta-Label Target from In-Sample Predictions (Partial Fix)

**Fixed in:** `src/modeling/meta_labeling.py`, lines 247–251
**Description:** Previously, the meta-label target was built from the primary model's **in-sample** predictions on the training set. A model that perfectly memorises training data would produce a meta-target of "always correct", teaching the meta-model nothing.
**Fix applied:** OOF predictions are used via `_out_of_fold_primary_predictions()`. Note: this fix is **partially correct only** — see LEAK-01 for the residual future-data contamination in folds 0–3.

---

*Last updated: 2026-03-21*
*Add new issues below each section header, following the existing format.*
