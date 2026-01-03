#!/usr/bin/env python3
"""
HH/HL (Higher High + Higher Low) Scanner - DAILY timeframe (LAST ~1 MONTH)
+ Correction fields (pullback started, stage, distance to support/breakout)

- Reads tickers from a text file (one per line)
- Downloads DAILY OHLC via yfinance (robust to MultiIndex return)
- Restricts to last N calendar days (default 35) and keeps last ~25 trading days
- Detects pivot (swing) highs/lows using a left/right window
- Declares HH/HL if the most recent 2 pivot highs are increasing (HH)
  AND the most recent 2 pivot lows are increasing (HL)
- Adds correction fields:
  - pullback_pct, above_support_pct
  - correction_started (between last_high_2 and last_low_2)
  - correction_stage: early/healthy/deep/invalid/none
  - near_support / near_breakout flags
  - SMA fast/slow and SMA direction
- Outputs strict JSON (no NaN) to --out (folders auto-created)

Example:
  python hh_hl_scanner.py --tickers_file data/nasdaq100.txt --out api/uptrend/nasdaq.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Utilities
# ----------------------------
def read_tickers_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s.split()[0].upper())
    return tickers


def extract_ticker_df(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Robust extraction for yfinance multi-ticker downloads.
    Handles MultiIndex columns or single ticker DataFrame.
    """
    if data is None:
        return None

    # MultiIndex columns: (field, ticker) OR (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        # (field, ticker)
        cols = []
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if (field, ticker) in data.columns:
                cols.append((field, ticker))
        if len(cols) >= 4:
            df = data[cols].copy()
            df.columns = [c[0] for c in df.columns]
            return df

        # (ticker, field)
        cols = []
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if (ticker, field) in data.columns:
                cols.append((ticker, field))
        if len(cols) >= 4:
            df = data[cols].copy()
            df.columns = [c[1] for c in df.columns]
            return df

        return None

    # Single ticker DataFrame
    cols = set(map(str, data.columns))
    if {"Open", "High", "Low", "Close"}.issubset(cols):
        return data.copy()

    # dict-like group_by='ticker' (rare)
    try:
        if ticker in data:
            return data[ticker].copy()
    except Exception:
        pass

    return None


def find_pivots(df: pd.DataFrame, left_right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Pivot High at index i if High[i] is strictly greater than highs in [i-n .. i+n]
    Pivot Low  at index i if Low[i]  is strictly lower  than lows  in [i-n .. i+n]
    Returns boolean series: pivot_high, pivot_low
    """
    h = df["High"]
    l = df["Low"]
    n = int(left_right)

    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)

    for i in range(n, len(df) - n):
        window_h = h.iloc[i - n : i + n + 1]
        window_l = l.iloc[i - n : i + n + 1]

        if h.iloc[i] == window_h.max() and (window_h == h.iloc[i]).sum() == 1:
            pivot_high.iloc[i] = True

        if l.iloc[i] == window_l.min() and (window_l == l.iloc[i]).sum() == 1:
            pivot_low.iloc[i] = True

    return pivot_high, pivot_low


def hh_hl_from_pivots(
    df: pd.DataFrame,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    min_pivots: int = 2,
) -> Dict[str, Any]:
    ph_idx = df.index[pivot_high]
    pl_idx = df.index[pivot_low]

    result: Dict[str, Any] = {
        "has_hh": False,
        "has_hl": False,
        "has_hh_hl": False,
        "last_high_1_date": None,
        "last_high_2_date": None,
        "last_low_1_date": None,
        "last_low_2_date": None,
        "last_high_1": None,
        "last_high_2": None,
        "last_low_1": None,
        "last_low_2": None,
        "pivot_high_count": int(len(ph_idx)),
        "pivot_low_count": int(len(pl_idx)),
    }

    if len(ph_idx) < min_pivots or len(pl_idx) < min_pivots:
        return result

    h1_date, h2_date = ph_idx[-2], ph_idx[-1]
    h1, h2 = float(df.loc[h1_date, "High"]), float(df.loc[h2_date, "High"])
    has_hh = h2 > h1

    l1_date, l2_date = pl_idx[-2], pl_idx[-1]
    l1, l2 = float(df.loc[l1_date, "Low"]), float(df.loc[l2_date, "Low"])
    has_hl = l2 > l1

    result.update(
        {
            "has_hh": bool(has_hh),
            "has_hl": bool(has_hl),
            "has_hh_hl": bool(has_hh and has_hl),
            "last_high_1_date": str(pd.to_datetime(h1_date).date()),
            "last_high_2_date": str(pd.to_datetime(h2_date).date()),
            "last_low_1_date": str(pd.to_datetime(l1_date).date()),
            "last_low_2_date": str(pd.to_datetime(l2_date).date()),
            "last_high_1": h1,
            "last_high_2": h2,
            "last_low_1": l1,
            "last_low_2": l2,
        }
    )
    return result


def classify_correction_stage(
    has_hh_hl: bool,
    last_close: float,
    last_high_2: Optional[float],
    last_low_2: Optional[float],
    early_max: float = 3.0,
    healthy_max: float = 6.0,
    deep_max: float = 10.0,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """
    Returns:
      correction_started, pullback_pct, above_support_pct, correction_stage
    """
    if not has_hh_hl or last_high_2 is None or last_low_2 is None:
        return False, None, None, "none"

    if last_high_2 <= 0 or last_low_2 <= 0:
        return False, None, None, "none"

    pullback_pct = (last_high_2 - last_close) / last_high_2 * 100.0
    above_support_pct = (last_close / last_low_2 - 1.0) * 100.0

    # Correction started if below HH but above HL support
    correction_started = (last_close < last_high_2) and (last_close > last_low_2)

    if not correction_started:
        # If already above HH -> breakout continuation
        if last_close >= last_high_2:
            return False, float(pullback_pct), float(above_support_pct), "none"
        # If below HL -> structure damaged
        if last_close <= last_low_2:
            return True, float(pullback_pct), float(above_support_pct), "invalid"
        return False, float(pullback_pct), float(above_support_pct), "none"

    # Stage by pullback depth (you can tune these)
    if pullback_pct <= early_max:
        stage = "early"
    elif pullback_pct <= healthy_max:
        stage = "healthy"
    elif pullback_pct <= deep_max:
        stage = "deep"
    else:
        stage = "invalid"  # too deep for a clean HH/HL continuation

    return True, float(pullback_pct), float(above_support_pct), stage


# ----------------------------
# Scanner
# ----------------------------
def scan_hh_hl(
    tickers: List[str],
    last_days: int = 35,
    max_trading_days: int = 25,
    pivot: int = 2,
    min_pivots: int = 2,
    sma_fast: int = 5,
    sma_slow: int = 10,
    near_support_pct: float = 3.0,
    near_breakout_pct: float = 2.0,
    threads: bool = True,
) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(last_days))

    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        threads=threads,
        progress=False,
        interval="1d",
    )

    rows: List[Dict[str, Any]] = []

    for t in tickers:
        try:
            df = extract_ticker_df(data, t)
            if df is None or df.empty:
                continue

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            # enforce last ~1 trading month
            df = df.tail(int(max_trading_days))

            if len(df) < (pivot * 2 + 6):
                continue

            # SMAs (for correction direction)
            df["sma_fast"] = df["Close"].rolling(int(sma_fast)).mean()
            df["sma_slow"] = df["Close"].rolling(int(sma_slow)).mean()

            ph, pl = find_pivots(df, left_right=pivot)
            info = hh_hl_from_pivots(df, ph, pl, min_pivots=min_pivots)

            last_close = float(df["Close"].iloc[-1])
            last_date = str(pd.to_datetime(df.index[-1]).date())

            # Correction fields
            last_high_2 = info.get("last_high_2")
            last_low_2 = info.get("last_low_2")

            corr_started, pullback_pct, above_support_pct, corr_stage = classify_correction_stage(
                has_hh_hl=bool(info.get("has_hh_hl")),
                last_close=last_close,
                last_high_2=float(last_high_2) if last_high_2 is not None else None,
                last_low_2=float(last_low_2) if last_low_2 is not None else None,
            )

            # Near zones (only meaningful if we have levels)
            near_support = None
            near_breakout = None
            if last_low_2 is not None and float(last_low_2) > 0:
                near_support = above_support_pct is not None and float(above_support_pct) <= float(near_support_pct)
            if last_high_2 is not None and float(last_high_2) > 0:
                dist_to_breakout_pct = (float(last_high_2) - last_close) / float(last_high_2) * 100.0
                near_breakout = dist_to_breakout_pct <= float(near_breakout_pct)

            sma_f = df["sma_fast"].iloc[-1]
            sma_s = df["sma_slow"].iloc[-1]
            sma_fast_gt_slow = None
            if np.isfinite(sma_f) and np.isfinite(sma_s):
                sma_fast_gt_slow = bool(float(sma_f) > float(sma_s))

            rows.append(
                {
                    "ticker": t,
                    "last_date": last_date,
                    "last_close": last_close,
                    "lookback_calendar_days": int(last_days),
                    "lookback_trading_days": int(len(df)),
                    "pivot_window": int(pivot),

                    **info,

                    # SMA / direction
                    "sma_fast_n": int(sma_fast),
                    "sma_slow_n": int(sma_slow),
                    "sma_fast": float(sma_f) if np.isfinite(sma_f) else None,
                    "sma_slow": float(sma_s) if np.isfinite(sma_s) else None,
                    "sma_fast_gt_slow": sma_fast_gt_slow,

                    # Correction fields
                    "correction_started": bool(corr_started),
                    "correction_stage": corr_stage,
                    "pullback_pct": pullback_pct,               # from last_high_2 to close
                    "above_support_pct": above_support_pct,     # from last_low_2 to close

                    # Zone helpers
                    "near_support": near_support,
                    "near_breakout": near_breakout,
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Rank: prefer HHHL true, and if correction started, prefer healthy/near support
    out["hhhl_rank"] = out["has_hh_hl"].astype(int)

    # Sort idea:
    # 1) HHHL first
    # 2) Correction started first (good watchlist for pullbacks)
    # 3) Higher quality: near_breakout or near_support
    out["_corr_sort"] = out["correction_started"].astype(int)
    out["_near_sup_sort"] = out["near_support"].astype(int) if "near_support" in out.columns else 0
    out["_near_brk_sort"] = out["near_breakout"].astype(int) if "near_breakout" in out.columns else 0

    out = out.sort_values(
        ["hhhl_rank", "_corr_sort", "_near_sup_sort", "_near_brk_sort", "last_date"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    out = out.drop(columns=["_corr_sort", "_near_sup_sort", "_near_brk_sort"], errors="ignore")
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan tickers for HH/HL pattern on DAILY timeframe (last ~1 month).")
    p.add_argument("--tickers_file", default="nifty.txt", help="Text file with one ticker per line.")
    p.add_argument("--last_days", type=int, default=35, help="Analyze last N calendar days (default 35).")
    p.add_argument("--max_trading_days", type=int, default=25, help="Keep last N trading days (default 25).")
    p.add_argument("--pivot", type=int, default=2, help="Pivot window size (left/right). 2 is best for 1M view.")
    p.add_argument("--min_pivots", type=int, default=2, help="Minimum pivot highs/lows required.")

    # Correction helpers
    p.add_argument("--sma_fast", type=int, default=5, help="Fast SMA length (default 5).")
    p.add_argument("--sma_slow", type=int, default=10, help="Slow SMA length (default 10).")
    p.add_argument("--near_support_pct", type=float, default=3.0, help="Near support if <= this % above last_low_2.")
    p.add_argument("--near_breakout_pct", type=float, default=2.0, help="Near breakout if within this % below last_high_2.")

    p.add_argument("--out", default="hh_hl_output.json", help="Output JSON path.")
    p.add_argument("--no_threads", action="store_true", help="Disable yfinance threading.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tickers = read_tickers_file(args.tickers_file)

    df = scan_hh_hl(
        tickers=tickers,
        last_days=args.last_days,
        max_trading_days=args.max_trading_days,
        pivot=args.pivot,
        min_pivots=args.min_pivots,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        near_support_pct=args.near_support_pct,
        near_breakout_pct=args.near_breakout_pct,
        threads=not args.no_threads,
    )

    # Strict JSON for Excel: NaN -> None
    if not df.empty:
        df = df.replace({np.nan: None})

    payload: Dict[str, Any] = {
        "updated_utc": pd.Timestamp.utcnow().isoformat(),
        "params": {
            "tickers_file": args.tickers_file,
            "last_days": args.last_days,
            "max_trading_days": args.max_trading_days,
            "pivot": args.pivot,
            "min_pivots": args.min_pivots,
            "sma_fast": args.sma_fast,
            "sma_slow": args.sma_slow,
            "near_support_pct": args.near_support_pct,
            "near_breakout_pct": args.near_breakout_pct,
        },
        "count": int(len(df)),
        "count_hhhl": int(df["has_hh_hl"].sum()) if not df.empty else 0,
        "count_corrections": int(df["correction_started"].sum()) if (not df.empty and "correction_started" in df.columns) else 0,
        "data": df.to_dict(orient="records") if not df.empty else [],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Wrote {payload['count']} rows to: {args.out}")
    if not df.empty:
        preview_cols = [
            "ticker", "has_hh_hl",
            "correction_started", "correction_stage",
            "pullback_pct", "above_support_pct",
            "near_support", "near_breakout",
            "sma_fast_gt_slow",
            "last_close",
        ]
        preview_cols = [c for c in preview_cols if c in df.columns]
        print(df[preview_cols].head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
