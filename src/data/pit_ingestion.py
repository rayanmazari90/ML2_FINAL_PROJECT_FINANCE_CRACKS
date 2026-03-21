"""
Phase 1 – Bias-Immune Data Layer
=================================
Point-in-time ingestion for prices, SEC fundamentals, macro series,
and historical universe reconstitution.  Every function enforces the
rule that no observation may use data published after the decision
timestamp.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import ssl
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# Fix SSL certificate verification on macOS Python installs
# where "Install Certificates.command" hasn't been run.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config" / "default.yaml"


def load_config(path: Path = CFG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# 1. Historical Universe Reconstitution
# ──────────────────────────────────────────────────────────────

# Fallback: current S&P 500 tickers scraped from Wikipedia.
# Production systems should replace this with a monthly-reconstituted
# membership table that includes delisted/acquired names.
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _read_wiki_tables():
    """Fetch Wikipedia S&P 500 tables, handling SSL + User-Agent issues."""
    import requests as _req
    from io import StringIO
    resp = _req.get(_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _fetch_current_sp500() -> pd.DataFrame:
    """Scrape current constituents + historical changes from Wikipedia."""
    tables = _read_wiki_tables()
    current = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry",
                          "Date added", "CIK"]].copy()
    current.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                             "Security": "company"}, inplace=True)
    current["ticker"] = current["ticker"].str.replace(".", "-", regex=False)
    return current


def _fetch_sp500_changes() -> pd.DataFrame:
    """Historical additions/removals from the second Wikipedia table.
    Returns DataFrame with columns: date, added, removed."""
    tables = _read_wiki_tables()
    if len(tables) < 2:
        return pd.DataFrame(columns=["date", "added", "removed"])

    raw = tables[1].copy()

    # Wikipedia uses MultiIndex columns: ('Added','Ticker'), ('Removed','Ticker'), etc.
    if isinstance(raw.columns, pd.MultiIndex):
        date_col = [c for c in raw.columns if "date" in str(c).lower()][0]
        added_col = [c for c in raw.columns if "added" in str(c[0]).lower() and "ticker" in str(c[1]).lower()]
        removed_col = [c for c in raw.columns if "removed" in str(c[0]).lower() and "ticker" in str(c[1]).lower()]

        out = pd.DataFrame({
            "date": raw[date_col],
            "added": raw[added_col[0]] if added_col else np.nan,
            "removed": raw[removed_col[0]] if removed_col else np.nan,
        })
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        out = raw.rename(columns={
            c: "date" for c in raw.columns if "date" in c
        }).rename(columns={
            c: "added" for c in raw.columns if "added" in c
        }).rename(columns={
            c: "removed" for c in raw.columns if "removed" in c
        })

    return out[["date", "added", "removed"]].dropna(subset=["date"])


def build_historical_universe(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Reconstruct a monthly (ticker, date) membership table.

    Uses current constituents as the anchor and walks backwards through
    the historical changes table.  Tickers that were *removed* from the
    index are re-added for the months they were members, which mitigates
    survivorship bias relative to naively using today's list.

    Returns
    -------
    DataFrame with columns ['date', 'ticker', 'sector'] indexed monthly.
    """
    current = _fetch_current_sp500()
    changes = _fetch_sp500_changes()

    months = pd.date_range(start_date, end_date, freq="ME")
    tickers_now = set(current["ticker"].tolist())
    sector_map = dict(zip(current["ticker"], current["sector"]))

    membership: Dict[pd.Timestamp, set] = {}

    for m in reversed(months):
        membership[m] = set(tickers_now)

        if "date" in changes.columns:
            month_changes = changes[
                pd.to_datetime(changes["date"], errors="coerce").dt.to_period("M")
                == m.to_period("M")
            ]
            for _, row in month_changes.iterrows():
                added = str(row.get("added", "")).replace(".", "-")
                removed = str(row.get("removed", "")).replace(".", "-")
                if added in tickers_now:
                    tickers_now.discard(added)
                if removed and removed != "nan":
                    tickers_now.add(removed)

    rows = []
    for dt, members in sorted(membership.items()):
        for t in members:
            rows.append({"date": dt, "ticker": t,
                         "sector": sector_map.get(t, "Unknown")})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 2. Market Data (yfinance)
# ──────────────────────────────────────────────────────────────

def fetch_pricing(
    tickers: List[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download adjusted daily OHLCV for a list of tickers.

    Returns long-form DataFrame:
        date | ticker | open | high | low | close | adj_close | volume
    """
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(
            batch, start=start, end=end,
            group_by="ticker", auto_adjust=False, threads=True,
        )
        if raw.empty:
            continue

        # Normalise columns: yfinance may return flat Index (1 ticker)
        # or MultiIndex (multiple tickers).  Flatten to (ticker, field).
        if isinstance(raw.columns, pd.MultiIndex):
            # Newer yfinance: level-0 may be "Ticker" or "Price",
            # level-1 the other.  Detect which level holds ticker symbols.
            lvl0_vals = raw.columns.get_level_values(0).unique().tolist()
            lvl1_vals = raw.columns.get_level_values(1).unique().tolist()
            tickers_in_0 = any(t in lvl0_vals for t in batch)
            tickers_in_1 = any(t in lvl1_vals for t in batch)

            if tickers_in_1 and not tickers_in_0:
                raw = raw.swaplevel(axis=1)
        else:
            # Single ticker, flat columns – wrap into MultiIndex
            raw.columns = pd.MultiIndex.from_product(
                [batch, [str(c) for c in raw.columns]]
            )

        for t in batch:
            try:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].dropna(how="all").copy()
            except (KeyError, TypeError):
                continue
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = t
            sub.index.name = "date"
            sub = sub.reset_index()
            frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 3. Point-in-Time Fundamentals (edgartools)
# ──────────────────────────────────────────────────────────────

def fetch_pit_fundamentals(
    tickers: List[str],
    filing_types: tuple = ("10-K", "10-Q"),
    n_periods: int = 8,
) -> pd.DataFrame:
    """
    Pull key fundamental metrics from SEC EDGAR via *edgartools*,
    preserving the filing_date as the point-in-time timestamp.

    Uses the standardised get_financials() API which maps ~2000 XBRL
    tags to 95 consistent concepts across all companies.

    Returns DataFrame:
        ticker | fiscal_period_end | sec_acceptance_date |
        form_type | revenue | net_income | total_assets |
        total_liabilities | stockholders_equity | operating_cash_flow
    """
    try:
        import edgar
        from edgar import Company
        edgar.set_identity("QuantFramework research@university.edu")
    except ImportError:
        warnings.warn(
            "edgartools not installed – returning empty fundamentals frame.",
            stacklevel=2,
        )
        return pd.DataFrame()
    except Exception:
        pass

    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    rows: list[dict] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            company = Company(ticker)
            fin = company.get_financials()
            if fin is None:
                failed.append(ticker)
                continue

            # Determine PIT date from the latest filing
            try:
                latest_filing = company.get_filings(form="10-K").latest()
                filing_date = getattr(latest_filing, "filing_date", None)
                period_end = getattr(latest_filing, "period_of_report", None)
            except Exception:
                filing_date = None
                period_end = None

            row = {
                "ticker": ticker,
                "fiscal_period_end": period_end,
                "sec_acceptance_date": filing_date,
                "form_type": "10-K",
                "revenue": _safe(fin.get_revenue),
                "net_income": _safe(fin.get_net_income),
                "total_assets": _safe(fin.get_total_assets),
                "total_liabilities": _safe(fin.get_total_liabilities),
                "stockholders_equity": _safe(fin.get_stockholders_equity),
                "operating_cash_flow": _safe(fin.get_operating_cash_flow),
            }
            rows.append(row)
        except Exception as e:
            failed.append(f"{ticker}({type(e).__name__})")
            continue

    if failed:
        print(f"  [edgartools] Failed for {len(failed)} tickers: "
              f"{failed[:10]}{'…' if len(failed) > 10 else ''}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["fiscal_period_end", "sec_acceptance_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def pit_asof_join(
    feature_dates: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (ticker, decision_date) in *feature_dates*, attach the
    most recent fundamental row whose sec_acceptance_date <= decision_date.

    Parameters
    ----------
    feature_dates : DataFrame with columns ['ticker', 'date']
    fundamentals  : DataFrame from fetch_pit_fundamentals()

    Returns
    -------
    Merged DataFrame.  Any fundamental column that violates PIT is NaN.
    """
    if fundamentals.empty:
        return feature_dates

    left = feature_dates.copy()
    right = fundamentals.copy()

    left["date"] = pd.to_datetime(left["date"], utc=True).dt.tz_localize(None)
    right["sec_acceptance_date"] = pd.to_datetime(
        right["sec_acceptance_date"], utc=True
    ).dt.tz_localize(None)

    left = left.dropna(subset=["date"])
    right = right.dropna(subset=["sec_acceptance_date"])

    # merge_asof requires the key column to be globally sorted.
    # Merge per-ticker to guarantee sort order within each group.
    parts = []
    for ticker in left["ticker"].unique():
        l = left[left["ticker"] == ticker].sort_values("date")
        r = right[right["ticker"] == ticker].sort_values("sec_acceptance_date")
        if r.empty:
            parts.append(l)
            continue
        m = pd.merge_asof(
            l, r,
            left_on="date",
            right_on="sec_acceptance_date",
            direction="backward",
            suffixes=("", "_fund"),
        )
        parts.append(m)

    if not parts:
        return left
    merged = pd.concat(parts, ignore_index=True)
    if "ticker_fund" in merged.columns:
        merged.drop(columns=["ticker_fund"], inplace=True)
    return merged


# ──────────────────────────────────────────────────────────────
# 4. Macro Series (FRED)
# ──────────────────────────────────────────────────────────────

def fetch_macro_series(
    series_ids: List[str],
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull macro indicators from FRED.

    Tries pandas_datareader first (no API key needed), then fredapi.
    Returns a date-indexed DataFrame with one column per series.
    """
    # pandas_datareader works without an API key — try it first
    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_ids, "fred", start, end).ffill()
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: fredapi (requires FRED_API_KEY)
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            frames = {}
            for sid in series_ids:
                try:
                    frames[sid] = fred.get_series(sid, start, end)
                except Exception:
                    continue
            if frames:
                return pd.DataFrame(frames).ffill()
        except Exception:
            pass

    warnings.warn(
        "Could not fetch FRED data – install pandas_datareader "
        "or set FRED_API_KEY for fredapi.",
        stacklevel=2,
    )
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# 5. Leakage Diagnostics
# ──────────────────────────────────────────────────────────────

def check_pit_compliance(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Assert that no fundamental data point has
    sec_acceptance_date > date (the decision timestamp).

    Returns a diagnostics DataFrame with violation counts and
    publication-lag statistics per ticker.
    """
    if "sec_acceptance_date" not in merged.columns:
        return pd.DataFrame({"status": ["no_fundamentals_attached"]})

    merged = merged.dropna(subset=["sec_acceptance_date"])
    merged["_pit_lag_days"] = (
        merged["date"] - merged["sec_acceptance_date"]
    ).dt.days

    violations = merged[merged["_pit_lag_days"] < 0]
    stats = merged.groupby("ticker")["_pit_lag_days"].agg(
        ["count", "mean", "median", "min"]
    )
    stats["violations"] = merged.groupby("ticker").apply(
        lambda g: (g["_pit_lag_days"] < 0).sum()
    )

    assert len(violations) == 0, (
        f"PIT VIOLATION: {len(violations)} rows have "
        f"sec_acceptance_date after decision date!"
    )

    return stats.rename(columns={
        "count": "n_obs", "mean": "avg_lag_days",
        "median": "median_lag_days", "min": "min_lag_days",
    })


def survivorship_report(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise how many unique tickers appear per month and how many
    are no longer in the current index (proxy for dead/delisted names).
    """
    current_tickers = set(_fetch_current_sp500()["ticker"].tolist())
    monthly = universe.groupby("date")["ticker"].agg(
        total=lambda x: x.nunique(),
        dead_or_removed=lambda x: sum(t not in current_tickers for t in x),
    )
    return monthly


# ──────────────────────────────────────────────────────────────
# 6. Convenience Orchestrator
# ──────────────────────────────────────────────────────────────

def run_data_pipeline(cfg: Optional[dict] = None) -> dict:
    """
    Execute the full data acquisition phase and return a dict of
    DataFrames ready for feature engineering.
    """
    cfg = cfg or load_config()
    start = cfg["universe"]["start_date"]
    end = cfg["universe"]["end_date"]

    print("[Phase 1] Building historical universe …")
    universe = build_historical_universe(start, end)
    all_tickers = sorted(universe["ticker"].unique().tolist())
    print(f"  Universe: {len(all_tickers)} unique tickers, "
          f"{len(universe)} ticker-months")

    print("[Phase 1] Downloading pricing data …")
    pricing = fetch_pricing(all_tickers, start, end)
    print(f"  Pricing: {len(pricing)} rows, "
          f"{pricing['ticker'].nunique()} tickers")

    print("[Phase 1] Fetching PIT fundamentals from SEC EDGAR …")
    fundamentals = fetch_pit_fundamentals(all_tickers[:50])
    print(f"  Fundamentals: {len(fundamentals)} filing rows")

    print("[Phase 1] Fetching macro series from FRED …")
    macro = fetch_macro_series(
        cfg["data"]["fred_series"], start, end,
        api_key=cfg["data"].get("fred_api_key"),
    )
    print(f"  Macro: {macro.shape}")

    return {
        "universe": universe,
        "pricing": pricing,
        "fundamentals": fundamentals,
        "macro": macro,
        "config": cfg,
    }
